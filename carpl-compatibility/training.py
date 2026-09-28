import os, shutil, math, torch
import pandas as pd
import pickle as pkl
import logging

from sklearn.model_selection import train_test_split
from datetime import datetime

from src.config import Parser
from src.checkpoint import load_last_checkpoint
from src.utils import create_param_groups, identify_layer_number, assign_lr
from src.model import create_model, Trainer
from src.preprocessing import create_tabular_pipe, create_tabular_data
from src.dataloader import create_dataloader
from src.history import History
from src.logger import setup_logger

def run_training(conf, device):
    output_path = f"{conf.output_dir}{conf.run_name}"
    history = History()

    model = create_model(conf).to(device)
    param_groups = create_param_groups(model)
    layer_list = identify_layer_number(model)

    tabular_pipe = create_tabular_pipe(conf.age_bins)
    train_df = create_tabular_data(conf.image_dir, conf.tabular_train_path)
    val_df = create_tabular_data(conf.image_dir, conf.tabular_val_path)
    train_df = tabular_pipe.fit_transform(train_df)
    val_df = tabular_pipe.transform(val_df)

    # save the fitted tabular pipe for later use in inference
    with open(f"{output_path}/tabular_pipe.pkl", "wb") as f:
        pkl.dump(tabular_pipe, f)

    train_loader = create_dataloader(conf, conf.image_dir, train_df, type="train")
    val_loader = create_dataloader(conf, conf.image_dir, val_df, type="val")

    # lr scheduler and optimiser
    base_lr = conf.base_lr * conf.batch_size / 256
    lr_function = lambda epoch: base_lr * min((epoch + 1) / (conf.warmup_epochs + 1e-8),
                                           0.5 * (math.cos(epoch / conf.total_epochs * math.pi) + 1))
    optimiser = torch.optim.AdamW(params=param_groups, 
                                  lr=base_lr,
                                  betas=(0.9, 0.999),
                                  weight_decay=conf.weight_decay)
    
    # checck for existing checkpoint to resume training
    history, model, optimiser = load_last_checkpoint(conf, history, model, optimiser)

    trainer = Trainer(conf, model, history, optimiser, device)
    for epoch in range(history.latest_epoch, conf.total_epochs):
        epoch_lr = lr_function(epoch)
        assign_lr(param_groups, epoch_lr, layer_list, conf.layer_decay)
        trainer.train(train_loader, epoch)
        trainer.validate(val_loader, epoch)


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    config_file = "config/training.yaml"
    conf = Parser.from_yaml(config_file)

    conf.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"{conf.output_dir}{conf.run_name}"
    os.makedirs(output_dir, exist_ok=True)
    shutil.copy(config_file, f"{output_dir}/config.yaml")

    # initialise logger
    logger, log_path = setup_logger(run_name=f"{conf.run_name}_{conf.timestamp}")
    logger.info(f"Using device: {device}")
    logger.info(f"Starting run {conf.run_name}_{conf.timestamp}")

    try:
        logger.info("Running Retraining...")
        run_training(conf, device)
        logger.info("Completed Retraining...")
    except Exception as e:
        logger.info(f"Encountered Error: {e}")
    