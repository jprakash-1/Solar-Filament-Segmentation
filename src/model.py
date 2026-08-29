"""Tiny/fast U-Net for MVP1 -- pipeline validation, not accuracy. Swapping the
encoder or going deeper is explicitly a post-MVP1 iteration, not in scope here.
"""

from __future__ import annotations

import segmentation_models_pytorch as smp
import torch.nn as nn


def build_model(encoder_name: str = "resnet18", encoder_weights: str | None = "imagenet") -> nn.Module:
    return smp.Unet(
        encoder_name=encoder_name,
        encoder_weights=encoder_weights,
        in_channels=1,
        classes=1,
    )
