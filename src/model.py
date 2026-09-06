"""U-Net (segmentation_models_pytorch) -- MVP1's architecture, extended here to
optionally load a domain-pretrained encoder checkpoint (e.g. the GONG Halpha BYOL
resnet50 from the jp-pretraining-data-prep branch's scripts/pretrain_resnet/) in
place of an ImageNet-init encoder.

Loading discipline for `encoder_checkpoint`: build the encoder with
`encoder_weights=None` (skip smp's own ImageNet stem-adaptation entirely, don't
layer the checkpoint on top of it) then `load_state_dict` the checkpoint directly.
This sidesteps a real, already-measured incompatibility: smp's `in_channels=1`
stem adaptation *sums* the 3 ImageNet channel filters and pulls its own
`smp-hub/resnet50.imagenet` checkpoint, whereas the BYOL pretraining's
`resnet50_1ch()` *averages* the filters and starts from
`torchvision.models.ResNet50_Weights.IMAGENET1K_V2` -- two independently
reasonable but numerically different stems (see RESNET_PRETRAIN_PLAN.md section
11 on the pretraining branch). Loading the checkpoint into a `None`-initialized
encoder means every encoder weight (stem included) comes from the checkpoint, so
that mismatch never enters the picture. Verified structurally (key sets, shapes,
and a real forward pass all match between `smp.Unet(encoder_name="resnet50",
encoder_weights=None, in_channels=1).encoder` and a standalone `resnet50_1ch()`)
in scripts/verify_encoder_checkpoint.py -- rerun that against a real exported
checkpoint before trusting a fine-tune run built on it.
"""

from __future__ import annotations

from pathlib import Path

import segmentation_models_pytorch as smp
import torch
import torch.nn as nn


def build_model(
    encoder_name: str = "resnet18",
    encoder_weights: str | None = "imagenet",
    encoder_checkpoint: str | Path | None = None,
) -> nn.Module:
    weights = None if encoder_checkpoint else encoder_weights
    model = smp.Unet(
        encoder_name=encoder_name,
        encoder_weights=weights,
        in_channels=1,
        classes=1,
    )
    if encoder_checkpoint:
        state = torch.load(encoder_checkpoint, map_location="cpu")["encoder_state_dict"]
        # smp's ResNetEncoder overrides load_state_dict to pop stray fc.* keys
        # before delegating to nn.Module's implementation, but doesn't return
        # its result -- always None, regardless of strict= or actual key
        # mismatches (confirmed directly: a real mismatched checkpoint raises
        # `TypeError: cannot unpack non-iterable NoneType object` here instead of
        # ever reaching a missing/unexpected check). So the missing/unexpected
        # check has to happen ourselves, against the target's own keys, before
        # calling load_state_dict at all.
        target_keys = set(model.encoder.state_dict().keys())
        state_keys = set(state.keys())
        missing, unexpected = target_keys - state_keys, state_keys - target_keys
        if missing or unexpected:
            raise RuntimeError(
                f"encoder checkpoint {encoder_checkpoint} doesn't line up with "
                f"{encoder_name}'s smp encoder: missing={missing}, unexpected={unexpected} "
                "-- run scripts/verify_encoder_checkpoint.py to diagnose before retrying."
            )
        model.encoder.load_state_dict(state, strict=True)  # raises on a shape mismatch even though it returns None
    return model
