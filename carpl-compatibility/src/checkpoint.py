import torch
import os, re

from src.history import History

def save_checkpoint(epoch, model, optimizer, val_auc, checkpoint_path):
    """Saves the current model, optimizer state, and training history to the specified path in outputs/.

    Args:
        epoch (int):
        model (nn.)
        val_auc (float): 
        checkpoint_path (str):
    """
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "val_auc": val_auc
    }
    torch.save(checkpoint, checkpoint_path)

def load_checkpoint(checkpoint_path):
    """
    Loads checkpoint according to the specified path and returns the model and optimizer state.
    """
    return torch.load(checkpoint_path)

def _get_latest_epoch(folder_path):
    pattern = re.compile(r"model_epoch(\d+)\.pth$")
    epochs = [
        int(pattern.match(f).group(1))
        for f in os.listdir(folder_path)
        if pattern.match(f)
    ]
    return f"model_epoch{max(epochs)}" if epochs else None

def load_last_checkpoint(conf, history: History, model, optimizer):
    """
    Loads the most recent checkpoint from the checkpoints directory and
    returns the model state, optimizer state, and training history.
    """
    checkpoint_dir = f"{conf.output_dir}{conf.run_name}/checkpoints"
    if not os.path.exists(checkpoint_dir):
        print(f"No checkpoint directory found at {checkpoint_dir}. Starting training from scratch.")
        return history, model, optimizer
    else:
        checkpoint_files = [f for f in os.listdir(checkpoint_dir) if f.endswith(".pth")]
        if not checkpoint_files:
            print(f"No checkpoint files found in {checkpoint_dir}. Starting training from scratch.")
            return history, model, optimizer
        else:
            latest_checkpoint = _get_latest_epoch(checkpoint_dir) + ".pth"
            checkpoint_path = os.path.join(checkpoint_dir, latest_checkpoint)
            print(f"Loading checkpoint from {latest_checkpoint}")
            model.load_state_dict(torch.load(checkpoint_path)["model_state_dict"])
            optimizer.load_state_dict(torch.load(checkpoint_path)["optimizer_state_dict"])
            last_epoch = torch.load(checkpoint_path)["epoch"]
            history.retrieve_history(conf, last_epoch)
            return history, model, optimizer

def load_model(model, model_path):
    """
    Loads model checkpoint based on path and returns the model state.
    """
    if not os.path.exists(model_path):
        print(f"No model checkpoint found at {model_path}. Cannot load model.")
        return model
    else:
        print(f"Loading model from {model_path}")
        map_location = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        checkpoint = torch.load(model_path, map_location=map_location, weights_only=False)
        state_dict = checkpoint.get("model_state_dict", checkpoint)
        model.load_state_dict(state_dict)
        return model