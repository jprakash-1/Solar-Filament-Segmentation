#!/usr/bin/env python3
"""Verify a BYOL-exported ResNet50 encoder checkpoint (see the
jp-pretraining-data-prep branch's scripts/pretrain_resnet/export_encoder.py)
loads cleanly into this branch's smp.Unet(encoder_name="resnet50") encoder.

RESNET_PRETRAIN_PLAN.md section 11 already found smp's own in_channels=1
ImageNet stem adaptation does NOT match the BYOL pretraining's resnet50_1ch()
stem (sum vs. mean, different pretrained source). This script checks a
different, narrower thing: the *checkpoint-loading* path src/model.py actually
uses (encoder_weights=None, then load_state_dict), which sidesteps that stem
mismatch entirely as long as the two encoders' state dict keys and shapes line
up 1:1 -- confirmed once here (key sets, shapes, and a real forward pass with
random init), not merely assumed:

    ref_keys == smp_keys                              True  (318/318)
    per-key shape mismatches                          none
    max abs diff, pooled forward output, synced init   0.0

Usage:
    python scripts/verify_encoder_checkpoint.py
        # structural-only check: random-init resnet50_1ch() vs.
        # smp.Unet(encoder_name="resnet50", encoder_weights=None).encoder --
        # confirms the architectures stay compatible (e.g. after a
        # torchvision/smp version bump), without needing a real checkpoint.

    python scripts/verify_encoder_checkpoint.py --checkpoint path/to/resnet50_byol_encoder.pt
        # also loads the real exported checkpoint into the smp encoder and
        # confirms zero missing/unexpected keys -- run this once before trusting
        # a real fine-tune build on --encoder-checkpoint.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as tvm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.model import build_model  # noqa: E402

logger = logging.getLogger("verify_encoder_checkpoint")


def resnet50_1ch(pretrained: bool = False) -> nn.Module:
    """Same construction as the jp-pretraining-data-prep branch's
    scripts/pretrain_resnet/model.py:resnet50_1ch() -- duplicated here (not
    imported) since that branch's BYOL scaffolding isn't part of this branch's
    scope; only the encoder architecture itself is needed for this check."""
    m = tvm.resnet50(weights=tvm.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None)
    old_conv = m.conv1
    new_conv = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
    if pretrained:
        with torch.no_grad():
            new_conv.weight.copy_(old_conv.weight.mean(dim=1, keepdim=True))
    m.conv1 = new_conv
    m.fc = nn.Identity()
    return m


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", type=Path, default=None, help="export_encoder.py output; omit for a structural-only check with random init")
    return p.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
    args = parse_args()

    reference = resnet50_1ch(pretrained=False)
    target_encoder = build_model(encoder_name="resnet50", encoder_weights=None).encoder

    ref_keys = set(reference.state_dict().keys())
    tgt_keys = set(target_encoder.state_dict().keys())
    if ref_keys != tgt_keys:
        logger.error(f"key mismatch: only-in-reference={ref_keys - tgt_keys}, only-in-target={tgt_keys - ref_keys}")
        raise SystemExit(1)
    shape_mismatches = [k for k in ref_keys if reference.state_dict()[k].shape != target_encoder.state_dict()[k].shape]
    if shape_mismatches:
        logger.error(f"shape mismatches for keys: {shape_mismatches}")
        raise SystemExit(1)
    logger.info(f"key sets and shapes match: {len(ref_keys)} parameters/buffers")

    if args.checkpoint is not None:
        state = torch.load(args.checkpoint, map_location="cpu")["encoder_state_dict"]
        # smp's ResNetEncoder.load_state_dict doesn't return the missing/
        # unexpected keys tuple (always None) -- confirmed directly, so check
        # key sets ourselves first (same fix applied in src/model.build_model).
        state_keys = set(state.keys())
        missing, unexpected = tgt_keys - state_keys, state_keys - tgt_keys
        if missing or unexpected:
            logger.error(f"load_state_dict mismatch loading {args.checkpoint}: missing={missing}, unexpected={unexpected}")
            raise SystemExit(1)
        target_encoder.load_state_dict(state, strict=True)
        reference.load_state_dict(state)  # same weights into both, so the forward-pass diff below is a real equivalence check, not noise from two independent random inits
        logger.info(f"loaded real checkpoint {args.checkpoint} into smp's resnet50 encoder with zero key mismatch")
    else:
        # Sync a random init across both so the forward-pass check below is
        # meaningful without a real checkpoint on hand.
        target_encoder.load_state_dict(reference.state_dict())
        logger.info("no --checkpoint given -- synced random init across both encoders for the forward-pass check only")

    x = torch.randn(2, 1, 224, 224)
    with torch.no_grad():
        ref_out = reference(x)  # [B, 2048] -- plain torchvision forward (avgpool + Identity fc)
        # target_encoder is smp's ResNetEncoder -- forward() returns a list of
        # multi-scale feature maps (one per stage, features[0] is the input
        # itself), not a single pooled vector -- compare the final stage only.
        tgt_out = target_encoder(x)[-1]
        tgt_pooled = torch.flatten(F.adaptive_avg_pool2d(tgt_out, 1), 1)
    if ref_out.shape != tgt_pooled.shape:
        logger.error(f"forward-pass shape mismatch: reference={ref_out.shape}, target(pooled)={tgt_pooled.shape}")
        raise SystemExit(1)
    max_abs_diff = (ref_out - tgt_pooled).abs().max().item()
    logger.info(f"forward pass shapes align ({ref_out.shape}); max_abs_diff={max_abs_diff:.3e} given synced/loaded weights")
    logger.info("OK: a BYOL-exported resnet50_1ch() encoder checkpoint is structurally compatible with smp.Unet(encoder_name='resnet50').encoder")


if __name__ == "__main__":
    main()
