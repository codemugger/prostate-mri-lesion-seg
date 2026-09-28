import os, shutil, torch
import pandas as pd

from datetime import datetime

from src.config import Parser
from src.dataloader import create_dataloader
from src.preprocessing import get_tabular_pipe, create_tabular_data, get_calibrator
from src.model import create_model, Trainer
from src.checkpoint import load_model
from src.logger import setup_logger

def run_inference(conf, model, device, input_dir=None):
    if input_dir:
        image_dir = os.path.join(conf.image_dir, input_dir)
    else:
        image_dir = os.path.join(conf.image_dir, "local")
    image_dir = f"{image_dir}/"

    tabular_pipe = get_tabular_pipe("components/tabular_pipeline.pkl")
    infer_df = create_tabular_data(image_dir)
    infer_df = tabular_pipe.transform(infer_df)
    inf_loader = create_dataloader(conf, image_dir, infer_df, type="inf")
    trainer = Trainer(conf, model, None, None, device)
    results = trainer.infer(inf_loader)

    calibrator_path = getattr(conf, "calibrator_path", None)
    if calibrator_path:
        calibrator = get_calibrator(calibrator_path)
        results["probability"] = calibrator.predict(results["probability"].values)
        results["prediction"] = (results["probability"] >= 0.5).astype(int)

    return results

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    config_file = "config/inference.yaml"
    conf = Parser.from_yaml(config_file)

    conf.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"{conf.output_dir}{conf.run_name}_{conf.timestamp}"
    os.makedirs(output_dir, exist_ok=True)
    shutil.copy(config_file, f"{output_dir}/config.yaml")

    # intialise logger
    logger, log_path = setup_logger(run_name=f"{conf.run_name}_{conf.timestamp}")
    logger.info(f"Using device: {device}")
    logger.info(f"Starting run {conf.run_name}_{conf.timestamp} with Source Code")

    # initialise model
    model = create_model(conf)
    model = load_model(model, conf.model_path).to(device)
    logger.info(f"Loaded model weights from {conf.model_path}")

    try:
        logger.info("Running Inference...")
        results = run_inference(conf, model, device)
        logger.info("Completed Inference...")
    except Exception as e:
        logger.info(f"Encountered Error: {e}")
    logger.info(results)

    try:
        results.to_csv(f"{output_dir}/results.csv")
        logger.info(f"Saved inference results to {output_dir}/results.csv")
    except Exception as e:
        logger.info(f"Encountered Error: {e}")