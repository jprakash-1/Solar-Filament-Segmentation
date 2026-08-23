"""Loss functions for binary filament segmentation."""

from __future__ import annotations

import segmentation_models_pytorch as smp
import torch
import torch.nn as nn


class DiceBCELoss(nn.Module):
    """Dice + BCE combo loss, standard for imbalanced foreground/background segmentation.

    Optional Lovasz term (weight 0 by default, so existing configs are unaffected):
    error_analysis.py found matched-instance IoU loosely distributed (mostly 0.55-0.75,
    almost none above 0.85) even for correctly-detected filaments -- Lovasz is a convex
    surrogate for the IoU/Jaccard index (tighter than plain soft-IoU), added here to push
    directly on that overlap-tightness problem rather than just region/pixel accuracy.
    Still named DiceBCELoss for backward compatibility with existing configs/checkpoints.
    """

    def __init__(self, dice_weight: float = 1.0, bce_weight: float = 1.0, lovasz_weight: float = 0.0) -> None:
        super().__init__()
        self.dice_weight = dice_weight
        self.bce_weight = bce_weight
        self.lovasz_weight = lovasz_weight
        self.dice = smp.losses.DiceLoss(mode=smp.losses.BINARY_MODE, from_logits=True)
        self.bce = smp.losses.SoftBCEWithLogitsLoss()
        self.lovasz = smp.losses.LovaszLoss(mode=smp.losses.BINARY_MODE, from_logits=True)

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        loss = self.dice_weight * self.dice(logits, target) + self.bce_weight * self.bce(logits, target)
        if self.lovasz_weight > 0:
            loss = loss + self.lovasz_weight * self.lovasz(logits, target)
        return loss
