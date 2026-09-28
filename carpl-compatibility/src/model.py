import logging
import torch, math, os, itertools
import torch.nn as nn
import numpy as np
from tqdm import tqdm
import pandas as pd

from src.checkpoint import save_checkpoint

from src.components import ResNetClassifier
from src.metrics import calculate_metrics

logger = logging.getLogger(__name__)

def create_model(conf):
    """Instantiate the ResNetClassifier model with parameters from config."""
    return ResNetClassifier(tabular_dim=conf.tabular_dim,
                            tabular_proj=conf.tabular_proj,
                            num_classes=2, # binary classification
                            hidden_dim=conf.hidden_dim,
                            dropout=conf.dropout,
                            input_size=conf.img_size[0])

class Trainer:
    """
    Trainer class to handle the training loop, including model training, validation, and checkpointing.
    Handles the training and validation loops, checkpoint saving, early stopping,
    and metric tracking for a classification model. Uses automatic mixed precision
    (AMP) for memory efficiency and supports encoder freezing during initial epochs.

    Args:
        model (torch.nn.Module): The model to be trained.
        history (History): History object to track training metrics across epochs.
        conf: Configuration object containing training hyperparameters including
        total_epoch, frozen_epochs, and other training settings.
        optimiser (torch.optim.Optimizer): PyTorch optimizer for model parameters.
    """
    def __init__(self, conf, model, history, optimiser, device):
        self.device = device
        self.conf = conf
        self.model = model.to(device)
        self.history = history
        self.optimiser = optimiser
        if optimiser is not None:
            for state in optimiser.state.values():
                for k, v in state.items():
                    if isinstance(v, torch.Tensor):
                        state[k] = v.to(self.device)
        self.criterion = nn.CrossEntropyLoss() 
        self.scaler = torch.amp.GradScaler()
    
    def train(self, dataloader, epoch):
        """
        Train the model for one epoch, tracking loss and accuracy.
        Unfreezes the encoder layers after a specified number of epochs.
        Args:
            dataloader (torch.utils.data.DataLoader): DataLoader for the training dataset.
            epoch (int): Current epoch number, used to determine when to unfreeze encoder layers.
        """
        probabilities, targets, epoch_loss = [], [], []
        self.model.train()

        # unfreeze encoder after specified epochs
        if epoch == self.conf.frozen_epochs:
            logger.info(f"Unfreezing encoder layers at epoch {epoch + 1}...")
            for param in self.model.encoder.parameters():
                param.requires_grad = True
        
        with tqdm(dataloader, unit="batch") as one_epoch:
            one_epoch.set_description(f"Train Epoch {epoch+1}/{self.conf.total_epochs}")

            for batch, (x, demos, uid, y) in enumerate(one_epoch):
                x, demos, y = x.to(self.device), demos.to(self.device), y.to(self.device)
                self.optimiser.zero_grad()

                with torch.autocast(device_type='cuda', dtype=torch.float16):
                    logits = self.model(x, demos)
                    loss = self.criterion(logits, y.long())

                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimiser)
                self.scaler.update()
                
                probs = torch.softmax(logits, dim=1)[:, 1]  # class 1 probabilities
                probabilities.append(probs.detach().cpu())
                targets.append(y.detach().cpu())
                epoch_loss.append(loss.item())
            
            probabilities = torch.cat(probabilities).numpy()
            targets = torch.cat(targets).numpy()
            epoch_loss = np.mean(epoch_loss)
            metrics = calculate_metrics(probabilities, targets, metrics=["accuracy", "auc"])
            epoch_accuracy, epoch_auc = metrics["accuracy"], metrics["auc"]

            self.history.update_history("train", {"loss": epoch_loss, "acc": epoch_accuracy, "auc": epoch_auc})
            print(f"\nTrain Epoch {epoch+1}: Loss={epoch_loss:.3f}, Acc={epoch_accuracy*100:.1f}%, AUC={epoch_auc:.3f}")
    
    def validate(self, dataloader, epoch):
        """
        Validate the model on the validation dataset, tracking loss, accuracy, and AUC.
        Saves checkpoints at specified intervals and updates the best model based on validation AUC.
        Args:
            dataloader (torch.utils.data.DataLoader): DataLoader for the validation dataset.
            epoch (int): Current epoch number, used for checkpoint naming and tracking. 
        """
        probabilities, targets, epoch_loss = [], [], []

        self.model.eval()

        with torch.no_grad():
            with tqdm(dataloader, unit="batch") as one_epoch:
                one_epoch.set_description(f"Val Epoch {epoch+1}/{self.conf.total_epochs}")

                for batch, (x, demos, uid, y) in enumerate(one_epoch):
                    x, demos, y = x.to(self.device), demos.to(self.device), y.to(self.device)

                    logits = self.model(x, demos)
                    loss = self.criterion(logits, y.long())

                    probs = torch.softmax(logits, dim=1)[:, 1]  # class 1 probabilities
                    probabilities.append(probs.detach().cpu())
                    targets.append(y.detach().cpu())
                    epoch_loss.append(loss.item())
            
            probabilities = torch.cat(probabilities).numpy()
            targets = torch.cat(targets).numpy()
            epoch_loss = np.mean(epoch_loss)
            metrics = calculate_metrics(probabilities, targets, metrics=["accuracy", "auc"])
            epoch_accuracy, epoch_auc = metrics["accuracy"], metrics["auc"]

            self.history.update_history("val", {"loss": epoch_loss, "acc": epoch_accuracy, "auc": epoch_auc})
            print(f"\nVal Epoch {epoch+1}: Loss={epoch_loss:.3f}, Acc={epoch_accuracy*100:.1f}%, AUC={epoch_auc:.3f}")
            self.history.log_history(self.conf)

            if (epoch+1) % self.conf.checkpoint_freq == 0:
                if not os.path.exists(f"{self.conf.output_dir}{self.conf.run_name}/checkpoints"):
                    os.mkdir(f"{self.conf.output_dir}{self.conf.run_name}/checkpoints")
                checkpoint_path = f"{self.conf.output_dir}{self.conf.run_name}/checkpoints/model_epoch{epoch+1}.pth"
                save_checkpoint(epoch+1, self.model, self.optimiser, epoch_auc, checkpoint_path)
                print(f"Saved checkpoint at epoch {epoch+1} with AUC {epoch_auc:.3f}")
            if self.history.update_best_val_auc(epoch_auc):
                if not os.path.exists((f"{self.conf.output_dir}{self.conf.run_name}/best_model")):
                    os.mkdir(f"{self.conf.output_dir}{self.conf.run_name}/best_model")
                checkpoint_path = f"{self.conf.output_dir}{self.conf.run_name}/best_model/best_model_epoch{epoch+1}_auc{epoch_auc:.3f}.pth"
                save_checkpoint(epoch+1, self.model, self.optimiser, epoch_auc, checkpoint_path)
                print(f"Saved best model at epoch {epoch+1} with AUC {epoch_auc:.3f}")
    
    def infer(self, dataloader):
        """
        Run inference on the test dataset, returning predicted probabilities and corresponding UIDs.
        Args:
            dataloader (torch.utils.data.DataLoader): DataLoader for the test dataset.
        Returns:
            results (pd.DataFrame): DataFrame containing UIDs, predicted probabilities and binary predictions for each sample.
        """
        self.model.eval()
        probabilities, uids = [], []
        with torch.no_grad():
            for x, demos, uid, y in dataloader:
                x, demos, y = x.to(self.device), demos.to(self.device), y.to(self.device)
                logits = self.model(x, demos)
                probs = torch.softmax(logits, dim=1)[:, 1]  
                probabilities.append(probs.detach().cpu())
                uids.append(uid)
        probabilities = torch.cat(probabilities).numpy()
        uids = list(itertools.chain(*uids))
        results = pd.DataFrame({
            "uid": uids,
            "probability": probabilities,
            "prediction": (probabilities >= 0.5).astype(int)
        })
        return results

    def evaluate(self, dataloader):
        """
        Evaluate the model on a given dataloader, returning a DataFrame with predictions and targets.
        Args:
            dataloader (torch.utils.data.DataLoader): DataLoader for the dataset to evaluate on.
        Returns:
            results (pd.DataFrame): DataFrame containing UIDs, predicted probabilities, binary predictions, and true targets for each sample.
        """
        probabilities, targets, uids = [], [], []
        self.model.eval()
        with torch.no_grad():
            for x, demos, uid, y in dataloader:
                x, demos, y = x.to(self.device), demos.to(self.device), y.to(self.device)
                logits = self.model(x, demos)
                probs = torch.softmax(logits, dim=1)[:, 1]  
                probabilities.append(probs.detach().cpu())
                targets.append(y.detach().cpu())
                uids.append(uid)
        probabilities = torch.cat(probabilities).numpy()
        targets = torch.cat(targets).numpy()
        uids = list(itertools.chain(*uids))
        results = pd.DataFrame({
            "uid": uids,
            "probability": probabilities,
            "prediction": (probabilities >= 0.5).astype(int),
            "target": targets
        })
        return results