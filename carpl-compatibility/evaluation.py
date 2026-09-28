import os, shutil, math, torch, json
import pandas as pd
import pickle as pkl

from datetime import datetime
from src.config import Parser
from src.dataloader import create_dataloader
from src.metrics import get_confidence_interval_metrics, print_metrics, log_metrics
from src.model import create_model, Trainer
from src.checkpoint import load_model
from src.preprocessing import get_tabular_pipe, create_tabular_data, get_calibrator
from src.logger import setup_logger


def run_evaluation(conf, device, output_dir, logger):
    model = create_model(conf)
    tabular_pipe = get_tabular_pipe("components/tabular_pipeline.pkl")

    eval_df = create_tabular_data(conf.image_dir, conf.tabular_path)
    eval_df = tabular_pipe.transform(eval_df)
    eval_loader = create_dataloader(conf, conf.image_dir, eval_df, type="val")
    
    model = load_model(model, conf.model_path).to(device)
    logger.info(f"Loaded model weights from {conf.model_path}")
    
    trainer = Trainer(conf, model, None, None, device)
    results = trainer.evaluate(eval_loader)

    # change this part to apply isotonic calibration after checking if calibrator pickle exists at provided path in conf
    
    calibrator_path = getattr(conf, "calibrator_path", None)
    if calibrator_path:
        calibrator = get_calibrator(calibrator_path)
        results["probability"] = calibrator.predict(results["probability"].values)
        results["prediction"] = (results["probability"] >= 0.5).astype(int)
        logger.info(f"Applied isotonic calibration from {calibrator_path}")

    results.to_csv(f"{output_dir}/evaluation_results.csv", index=False)
    metrics = get_confidence_interval_metrics(
        n=10000,
        y_proba=results["probability"],
        y_pred=results["prediction"],
        y_true=results["target"]
    )
    with open(f"{output_dir}/evaluation_metrics.json", "w") as f:
        json.dump({"run": conf.output_dir, "metrics": metrics}, f, indent=4)
    print_metrics(metrics)
    return metrics

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    config_file = "config/evaluation.yaml"
    conf = Parser.from_yaml(config_file)
    output_dir = f"{conf.output_dir}{conf.run_name}_{timestamp}"
    os.makedirs(output_dir, exist_ok=True)
    shutil.copy(config_file, f"{output_dir}/config.yaml")

    logger, log_path = setup_logger(run_name=f"{conf.run_name}_{timestamp}")
    logger.info(f"Using device: {device}")
    logger.info(f"Starting run {conf.run_name}_{timestamp}")

    try:
        logger.info("Starting Evaluation...")
        metrics = run_evaluation(conf, device, output_dir, logger)
        logger.info("Completed Evaluation...")
        log_metrics(metrics)
    except Exception as e:
        logger.info(f"Encountered Error: {e}")