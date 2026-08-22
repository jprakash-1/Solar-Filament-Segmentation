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
    gradient_checkpointing: bool = False,
) -> nn.Module:
    model = smp.Unet(
        encoder_name=encoder_name,
        encoder_weights=encoder_weights,
        in_channels=in_channels,
        classes=classes,
        decoder_use_norm=decoder_use_norm,
    )
    if gradient_checkpointing:
        # Recomputes encoder activations during backward instead of storing them --
        # trades compute time for VRAM headroom, so a larger image_size can fit. Only
        # covers the timm encoder (via its built-in, DDP-safe non-reentrant
        # checkpointing) -- smp's U-Net decoder has no checkpointing support.
        timm_model = getattr(model.encoder, "model", None)
        if timm_model is None or not hasattr(timm_model, "set_grad_checkpointing"):
            raise ValueError(f"encoder {encoder_name!r} does not support gradient_checkpointing")
        timm_model.set_grad_checkpointing(enable=True)
    return model
