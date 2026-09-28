import numpy as np
import logging

from tqdm import tqdm
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score, f1_score, confusion_matrix

logger = logging.getLogger(__name__)

def get_confidence_interval_metrics(n, y_proba, y_pred, y_true):
    """
    Perform bootstrapping to calculate confidence intervals for evaluation metrics.
    
    Args:
        n (int): Number of bootstrap samples.
        y_proba (array-like): Predicted probabilities for the positive class.
        y_pred (array-like): Predicted class labels.
        y_true (array-like): True class labels.
    Returns:
        dict: A dictionary containing the mean and confidence intervals for each metric.
    """
    metrics = {
        "auc": [],
        "accuracy": [],
        "recall": [],
        "precision": [],
        "specificity": [],
        "macro_f1": [],
        "weighted_f1": []
    }
    for i in tqdm(range(n)):
        indices = np.random.randint(0, len(y_true), len(y_true))

        # Skip invalid resamples (all same class)
        if len(np.unique(y_true[indices])) < 2:
            continue

        yt = y_true.iloc[indices]
        yp = y_pred.iloc[indices]
        yp_prob = y_proba.iloc[indices]

        metrics["auc"].append(roc_auc_score(yt, yp_prob))
        metrics["accuracy"].append(accuracy_score(yt, yp))
        metrics["recall"].append(recall_score(yt, yp))
        metrics["precision"].append(precision_score(yt, yp))
        tn, fp, fn, tp = confusion_matrix(yt, yp, labels=[0,1]).ravel()
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        metrics["specificity"].append(specificity)
        metrics["macro_f1"].append(f1_score(yt, yp, average='macro'))
        metrics["weighted_f1"].append(f1_score(yt, yp, average='weighted'))
    
    for key, metric in metrics.items():
        metrics[key] = summarise(metric)

    return metrics

def summarise(scores):
    """
    Summarise bootstrap scores by calculating mean and confidence intervals.
    
    Args:
        scores (list): A list of metric scores from bootstrap samples.
    Returns:
        dict: A dictionary containing the mean and confidence intervals for the metric.
    """
    mean = np.mean(scores)
    lower = np.percentile(scores, 2.5)
    upper = np.percentile(scores, 97.5)
    plus_minus = (upper - lower) / 2
    return {"mean": round(mean, 4), "ci_lower": round(lower, 4), "ci_upper": round(upper, 4), "plus_minus": round(plus_minus, 4)}

def print_metrics(metrics):
    print(f"\n===== Results =====")
    print(f"AUC:         {metrics['auc']['mean']:.4f} (95% CI: {metrics['auc']['ci_lower']:.4f} to {metrics['auc']['ci_upper']:.4f})  ± {metrics['auc']['plus_minus']:.4f}")
    print(f"ACC:         {metrics['accuracy']['mean']:.4f} (95% CI: {metrics['accuracy']['ci_lower']:.4f} to {metrics['accuracy']['ci_upper']:.4f})  ± {metrics['accuracy']['plus_minus']:.4f}")
    print(f"Recall:      {metrics['recall']['mean']:.4f} (95% CI: {metrics['recall']['ci_lower']:.4f} to {metrics['recall']['ci_upper']:.4f})  ± {metrics['recall']['plus_minus']:.4f}")
    print(f"Precision:   {metrics['precision']['mean']:.4f} (95% CI: {metrics['precision']['ci_lower']:.4f} to {metrics['precision']['ci_upper']:.4f})  ± {metrics['precision']['plus_minus']:.4f}")
    print(f"Specificity: {metrics['specificity']['mean']:.4f} (95% CI: {metrics['specificity']['ci_lower']:.4f} to {metrics['specificity']['ci_upper']:.4f})  ± {metrics['specificity']['plus_minus']:.4f}")
    print(f"Macro F1:    {metrics['macro_f1']['mean']:.4f} (95% CI: {metrics['macro_f1']['ci_lower']:.4f} to {metrics['macro_f1']['ci_upper']:.4f})  ± {metrics['macro_f1']['plus_minus']:.4f}")
    print(f"Weighted F1: {metrics['weighted_f1']['mean']:.4f} (95% CI: {metrics['weighted_f1']['ci_lower']:.4f} to {metrics['weighted_f1']['ci_upper']:.4f})  ± {metrics['weighted_f1']['plus_minus']:.4f}")

def log_metrics(metrics):
    logger.info(f"\n===== Results =====")
    logger.info(f"AUC:         {metrics['auc']['mean']:.4f} (95% CI: {metrics['auc']['ci_lower']:.4f} to {metrics['auc']['ci_upper']:.4f})  ± {metrics['auc']['plus_minus']:.4f}")
    logger.info(f"ACC:         {metrics['accuracy']['mean']:.4f} (95% CI: {metrics['accuracy']['ci_lower']:.4f} to {metrics['accuracy']['ci_upper']:.4f})  ± {metrics['accuracy']['plus_minus']:.4f}")
    logger.info(f"Recall:      {metrics['recall']['mean']:.4f} (95% CI: {metrics['recall']['ci_lower']:.4f} to {metrics['recall']['ci_upper']:.4f})  ± {metrics['recall']['plus_minus']:.4f}")
    logger.info(f"Precision:   {metrics['precision']['mean']:.4f} (95% CI: {metrics['precision']['ci_lower']:.4f} to {metrics['precision']['ci_upper']:.4f})  ± {metrics['precision']['plus_minus']:.4f}")
    logger.info(f"Specificity: {metrics['specificity']['mean']:.4f} (95% CI: {metrics['specificity']['ci_lower']:.4f} to {metrics['specificity']['ci_upper']:.4f})  ± {metrics['specificity']['plus_minus']:.4f}")
    logger.info(f"Macro F1:    {metrics['macro_f1']['mean']:.4f} (95% CI: {metrics['macro_f1']['ci_lower']:.4f} to {metrics['macro_f1']['ci_upper']:.4f})  ± {metrics['macro_f1']['plus_minus']:.4f}")
    logger.info(f"Weighted F1: {metrics['weighted_f1']['mean']:.4f} (95% CI: {metrics['weighted_f1']['ci_lower']:.4f} to {metrics['weighted_f1']['ci_upper']:.4f})  ± {metrics['weighted_f1']['plus_minus']:.4f}")

def calculate_metrics(predictions, labels, metrics=["accuracy", "auc"]):
    """
    Calculate evaluation metrics for the given predictions and labels.

    Args:
        predictions (list): A list of predicted values.
        labels (list): A list of true values.
        metrics (list): A list of metrics to calculate. Supported metrics include "accuracy" and "auc".
    Returns:
        dict: A dictionary containing the calculated metrics.
    """
    results = {}
    if "accuracy" in metrics:
        correct = (predictions >= 0.5) == labels
        accuracy = correct.sum() / len(labels)
        results["accuracy"] = round(accuracy, 4)
    if "auc" in metrics:
        auc = roc_auc_score(labels, predictions)
        results["auc"] = round(auc, 4)
    if "precision" in metrics:
        precision = precision_score(labels, predictions >= 0.5)
        results["precision"] = round(precision, 4)
    if "recall" in metrics:
        recall = recall_score(labels, predictions >= 0.5)
        results["recall"] = round(recall, 4)
    if "weighted_f1" in metrics:
        f1 = f1_score(labels, predictions >= 0.5, average="weighted")
        results["weighted_f1"] = round(f1, 4)
    return results