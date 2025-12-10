from __future__ import annotations

from typing import Dict, Any, Optional
from monai.losses import DiceLoss
from monai.losses import DiceCELoss


def get_loss(loss_cfg: Dict[str, Any]):
    """
    Returns a MONAI loss based on the config.
    Supported:
      - type: dice_ce with dice/ce kwargs
      - type: dice with kwargs
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
    raise ValueError(f"Unsupported loss type: {loss_type}")

