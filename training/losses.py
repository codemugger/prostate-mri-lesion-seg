from __future__ import annotations

from typing import Any, Mapping

import torch
import torch.nn as nn
from monai.losses import DiceCELoss


class ProbabilityDiceFocalLoss(nn.Module):
    """Dice + focal loss for RRUNet outputs that are already probabilities."""

    def __init__(
        self,
        *,
        gamma: float = 2.0,
        alpha: float = 0.75,
        dice_weight: float = 1.0,
        focal_weight: float = 1.0,
        epsilon: float = 1e-6,
    ):
        super().__init__()
        self.gamma = float(gamma)
        self.alpha = float(alpha)
        self.dice_weight = float(dice_weight)
        self.focal_weight = float(focal_weight)
        self.epsilon = float(epsilon)

    def forward(self, probabilities: torch.Tensor, labels: torch.Tensor):
        if probabilities.ndim != 5 or probabilities.shape[1] != 2:
            raise ValueError("Lesion probabilities must have shape [B,2,X,Y,Z]")
        target_index = labels.long()
        if target_index.ndim == 5:
            target_index = target_index[:, 0]
        target_index = target_index.clamp(0, 1)
        target = target_index.float()
        lesion_probability = probabilities[:, 1].clamp(
            self.epsilon, 1.0 - self.epsilon
        )

        intersection = (lesion_probability * target).sum(dim=(1, 2, 3))
        denominator = lesion_probability.sum(dim=(1, 2, 3)) + target.sum(
            dim=(1, 2, 3)
        )
        dice_loss = 1.0 - (
            (2.0 * intersection + self.epsilon) / (denominator + self.epsilon)
        )

        probability_true_class = torch.gather(
            probabilities.clamp(self.epsilon, 1.0 - self.epsilon),
            1,
            target_index.unsqueeze(1),
        ).squeeze(1)
        alpha = torch.where(
            target_index == 1,
            torch.full_like(probability_true_class, self.alpha),
            torch.full_like(probability_true_class, 1.0 - self.alpha),
        )
        focal = -alpha * (1.0 - probability_true_class).pow(self.gamma) * torch.log(
            probability_true_class
        )
        return self.dice_weight * dice_loss.mean() + self.focal_weight * focal.mean()


def get_loss(loss_cfg: Mapping[str, Any], *, task: str) -> nn.Module:
    if task == "organ":
        return DiceCELoss(
            include_background=bool(loss_cfg.get("include_background", False)),
            to_onehot_y=True,
            softmax=True,
            lambda_dice=float(loss_cfg.get("dice_weight", 1.0)),
            lambda_ce=float(loss_cfg.get("ce_weight", 1.0)),
        )
    if task == "lesion":
        return ProbabilityDiceFocalLoss(
            gamma=float(loss_cfg.get("gamma", 2.0)),
            alpha=float(loss_cfg.get("alpha", 0.75)),
            dice_weight=float(loss_cfg.get("dice_weight", 1.0)),
            focal_weight=float(loss_cfg.get("focal_weight", 1.0)),
        )
    raise ValueError(f"Unsupported task: {task}")
