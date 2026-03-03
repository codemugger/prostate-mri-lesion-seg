from __future__ import annotations

from typing import Dict, Any, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from monai.losses import DiceLoss, DiceCELoss


class DiceFocalLoss(nn.Module):
    """
    Dice loss plus focal term for class imbalance.
    Assumes channel-first logits; uses DiceLoss preprocessing (to_onehot_y/softmax).
    """
    def __init__(
        self,
        include_background: bool = False,
        to_onehot_y: bool = True,
        softmax: bool = True,
        gamma: float = 2.0,
        alpha: float = 0.75,
    ):
        super().__init__()
        self.softmax = softmax
        self.alpha = alpha
        self.gamma = gamma
        self.dice = DiceLoss(
            include_background=include_background,
            to_onehot_y=to_onehot_y,
            softmax=softmax,
        )

    def forward(self, logits: torch.Tensor, labels: torch.Tensor):
        dice = self.dice(logits, labels)
        probs = F.softmax(logits, dim=1) if self.softmax else torch.sigmoid(logits)

        # Ensure labels are one-hot for focal term
        target = labels
        if target.shape != probs.shape:
            # labels may be [B,1,H,W,D] or [B,H,W,D]
            if target.dim() == 4:
                target = target.unsqueeze(1)
            if target.shape[1] != probs.shape[1]:
                target = F.one_hot(target.long().squeeze(1), num_classes=probs.shape[1])
                target = target.permute(0, 4, 1, 2, 3).float()

        pt = (probs * target).sum(dim=1).clamp(min=1e-6)  # [B,H,W,D]
        focal = -(self.alpha * (1.0 - pt) ** self.gamma * torch.log(pt)).mean()
        return dice + focal


def get_loss(loss_cfg: Dict[str, Any]):
    """
    Returns a MONAI loss based on the config.
    Supported:
      - type: dice_ce with dice/ce kwargs
      - type: dice with kwargs
      - type: dice_focal with dice/focal kwargs
    """
    loss_type = loss_cfg.get("type", "dice_ce").lower()
    if loss_type == "dice_ce":
        dice_kwargs: Dict[str, Any] = loss_cfg.get("dice", {}) or {}
        return DiceCELoss(
            include_background=dice_kwargs.get("include_background", True),
            to_onehot_y=dice_kwargs.get("to_onehot_y", True),
            softmax=dice_kwargs.get("softmax", True),
        )
    if loss_type == "dice":
        dice_kwargs: Dict[str, Any] = loss_cfg.get("dice", {}) or {}
        return DiceLoss(
            include_background=dice_kwargs.get("include_background", True),
            to_onehot_y=dice_kwargs.get("to_onehot_y", True),
            softmax=dice_kwargs.get("softmax", True),
        )
    if loss_type == "dice_focal":
        dice_kwargs: Dict[str, Any] = loss_cfg.get("dice", {}) or {}
        focal_kwargs: Dict[str, Any] = loss_cfg.get("focal", {}) or {}
        return DiceFocalLoss(
            include_background=dice_kwargs.get("include_background", True),
            to_onehot_y=dice_kwargs.get("to_onehot_y", True),
            softmax=dice_kwargs.get("softmax", True),
            gamma=focal_kwargs.get("gamma", 2.0),
            alpha=focal_kwargs.get("alpha", 0.75),
        )
    raise ValueError(f"Unsupported loss type: {loss_type}")

