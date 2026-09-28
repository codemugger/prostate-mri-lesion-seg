import pickle as pkl

class History:
    """
    Class to store training history, including current epoch and metrics.
    Metrics included for training and validation are: loss, accuracy, auc
    """
    def __init__(self, metrics_list = ["loss", "acc", "auc"]):
        self.latest_epoch = 0
        self.metrics_list = metrics_list
        self.history = {
            "train": {metric: [] for metric in self.metrics_list},
            "val": {metric: [] for metric in self.metrics_list}
        }
        self.best_val_auc = 0
    
    def update_best_val_auc(self, value):
        """Update the best metric value for a given metric name."""
        if value > self.best_val_auc:
            self.best_val_auc = value
            return True
        else:
            return False
    
    def update_history(self, split, metrics_dict):
        """
        Updates history class at the end of every training epoch.
        """
        for metric_name, value in metrics_dict.items():
            self.history[split][metric_name].append(value)
        if split == "val":
            self.latest_epoch += 1
    
    def log_history(self, conf):
        """
        Save history class in run's output folder together with checkpoint
        """
        history_path = f"{conf.output_dir}{conf.run_name}/history.pkl"
        with open(history_path, "wb") as f:
            pkl.dump(self, f)
    
    def retrieve_history(self, conf, epoch):
        """
        Return the history of metrics from beginning of training to specified epoch.
        """
        history_path = f"{conf.output_dir}{conf.run_name}/history.pkl"
        with open(history_path, "rb") as f:
            history = pkl.load(f)
        self.latest_epoch = epoch
        for split in history.history:
            for metric_name in history.history[split]:
                self.history[split][metric_name] = history.history[split][metric_name][:epoch]
        self.best_val_auc = max(history.history["val"]["auc"])