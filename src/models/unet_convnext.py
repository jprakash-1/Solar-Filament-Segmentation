"""U-Net with a ConvNeXt (timm) encoder, via segmentation_models_pytorch."""

from __future__ import annotations

import segmentation_models_pytorch as smp
import torch.nn as nn


def build_model(
    encoder_name: str = "tu-convnext_tiny",
    encoder_weights: str | None = "imagenet",
    in_channels: int = 3,
    classes: int = 1,
    decoder_use_norm: bool | str = "batchnorm",
) -> nn.Module:
    return smp.Unet(
        encoder_name=encoder_name,
        encoder_weights=encoder_weights,
        in_channels=in_channels,
        classes=classes,
        decoder_use_norm=decoder_use_norm,
    )
