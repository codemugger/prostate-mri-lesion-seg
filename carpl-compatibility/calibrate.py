
import json, os, pickle, shutil
import numpy as np
import pandas as pd

from datetime import datetime
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import brier_score_loss, roc_auc_score

from src.config import Parser
from src.logger import setup_logger


def expected_calibration_error(y_true, y_prob, n_bins=10):
    """Compute Expected Calibration Error using equal-width bins."""
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        mask = (y_prob >= bins[i]) & (y_prob < bins[i + 1])
        if mask.sum() == 0:
            continue
        ece += (mask.sum() / len(y_true)) * abs(y_prob[mask].mean() - y_true[mask].mean())
    return ece


def fit_and_save_calibrator(conf, output_dir, logger):
    df = pd.read_csv(conf.results_csv)
    y_prob = df["probability"].values
    y_true = df["target"].values.astype(int)

    calibrator = IsotonicRegression(out_of_bounds="clip")
    calibrator.fit(y_prob, y_true)
    y_cal = calibrator.predict(y_prob)

    metrics = {
        "n_samples": int(len(y_true)),
        "results_csv": conf.results_csv,
        "before": {
            "ece":   round(float(expected_calibration_error(y_true, y_prob)), 4),
            "brier": round(float(brier_score_loss(y_true, y_prob)), 4),
            "auc":   round(float(roc_auc_score(y_true, y_prob)), 4),
        },
        "after": {
            "ece":   round(float(expected_calibration_error(y_true, y_cal)), 4),
            "brier": round(float(brier_score_loss(y_true, y_cal)), 4),
            "auc":   round(float(roc_auc_score(y_true, y_cal)), 4),
        },
    }

    logger.info(f"Fitting isotonic calibrator on {len(y_true)} samples from {conf.results_csv}")
    logger.info(f"  ECE   before: {metrics['before']['ece']:.4f}  ->  after: {metrics['after']['ece']:.4f}")
    logger.info(f"  Brier before: {metrics['before']['brier']:.4f}  ->  after: {metrics['after']['brier']:.4f}")
    logger.info(f"  AUC   before: {metrics['before']['auc']:.4f}  ->  after: {metrics['after']['auc']:.4f}  (preserved by monotonic transform)")

    with open(conf.calibrator_output_path, "wb") as f:
        pickle.dump(calibrator, f)
    logger.info(f"Saved isotonic calibrator to {conf.calibrator_output_path}")

    with open(f"{output_dir}/calibration_metrics.json", "w") as f:
        json.dump(metrics, f, indent=4)

    return metrics


if __name__ == "__main__":
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    config_file = "config/calibration.yaml"
    conf = Parser.from_yaml(config_file)
    output_dir = f"{conf.output_dir}{conf.run_name}_{timestamp}"
    os.makedirs(output_dir, exist_ok=True)
    shutil.copy(config_file, f"{output_dir}/config.yaml")

    logger, _ = setup_logger(run_name=f"{conf.run_name}_{timestamp}")
    logger.info(f"Starting calibration run {conf.run_name}_{timestamp}")

    metrics = fit_and_save_calibrator(conf, output_dir, logger)

    print(f"\n===== Calibration Report =====")
    print(f"Samples : {metrics['n_samples']}")
    print(f"{'Metric':<10} {'Before':>8} {'After':>8}")
    print(f"{'-' * 28}")
    print(f"{'ECE':<10} {metrics['before']['ece']:>8.4f} {metrics['after']['ece']:>8.4f}")
    print(f"{'Brier':<10} {metrics['before']['brier']:>8.4f} {metrics['after']['brier']:>8.4f}")
    print(f"{'AUC':<10} {metrics['before']['auc']:>8.4f} {metrics['after']['auc']:>8.4f}")
    print(f"\nCalibrator saved to: {conf.calibrator_output_path}")
