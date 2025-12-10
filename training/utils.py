from __future__ import annotations

import csv
import os
import random
from typing import Any, Dict

import numpy as np
import torch
import yaml


def load_yaml(path: str) -> Dict[str, Any]:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def set_determinism(seed: int = 42, deterministic: bool = True):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    else:
        torch.backends.cudnn.benchmark = True


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def append_metrics_csv(csv_path: str, row: Dict[str, Any]):
    exists = os.path.exists(csv_path)
    with open(csv_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def get_next_experiment_number(exp_root: str) -> int:
    """
    Reads and increments a simple counter file in exp_root to generate a monotonic experiment number.
    Returns the next number (1-based).
    """
    ensure_dir(exp_root)
    counter_path = os.path.join(exp_root, ".exp_counter")
    current = 0
    if os.path.exists(counter_path):
        try:
            with open(counter_path, "r") as f:
                txt = f.read().strip()
                if txt:
                    current = int(txt)
        except Exception:
            current = 0
    next_num = current + 1
    with open(counter_path, "w") as f:
        f.write(str(next_num))
    return next_num

