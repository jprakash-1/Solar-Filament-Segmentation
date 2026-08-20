"""Loss functions for binary filament segmentation."""

from __future__ import annotations

import segmentation_models_pytorch as smp
import torch
import torch.nn as nn


class DiceBCELoss(nn.Module):
    """Dice + BCE combo loss, standard for imbalanced foreground/background segmentation."""

    def __init__(self, dice_weight: float = 1.0, bce_weight: float = 1.0) -> None:
        super().__init__()
        self.dice_weight = dice_weight
        self.bce_weight = bce_weight
        self.dice = smp.losses.DiceLoss(mode=smp.losses.BINARY_MODE, from_logits=True)
        self.bce = smp.losses.SoftBCEWithLogitsLoss()

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return self.dice_weight * self.dice(logits, target) + self.bce_weight * self.bce(logits, target)
