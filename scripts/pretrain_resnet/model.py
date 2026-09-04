"""BYOL model: 1-channel ResNet50 encoder + projector/predictor heads +
momentum target network. See RESNET_PRETRAIN_PLAN.md sections 2, 5, 6.
"""

from __future__ import annotations

import copy
import logging

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as tvm

logger = logging.getLogger("pretrain_resnet.model")

ENCODER_DIM = 2048  # ResNet50 Bottleneck final-stage width -- NOT 512
                     # (that would be ResNet34/18's BasicBlock width instead).


def resnet50_1ch(pretrained: bool = True) -> nn.Module:
    """Average the pretrained stem conv's 3 input-channel filters into 1, rather
    than re-initializing the stem randomly -- keeps the stem's learned edge/texture
    detectors as a meaningful starting point instead of throwing away exactly the
    layer ImageNet pretraining helps most. Backbone only (fc replaced with
    Identity) -- BYOL adds its own projector head. See section 2."""
    m = tvm.resnet50(weights=tvm.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None)
    old_conv = m.conv1  # [64, 3, 7, 7]
    new_conv = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
    if pretrained:
        with torch.no_grad():
            new_conv.weight.copy_(old_conv.weight.mean(dim=1, keepdim=True))
    m.conv1 = new_conv
    m.fc = nn.Identity()
    return m


def mlp(in_dim: int, hidden_dim: int, out_dim: int) -> nn.Module:
    return nn.Sequential(
        nn.Linear(in_dim, hidden_dim), nn.BatchNorm1d(hidden_dim), nn.ReLU(inplace=True),
        nn.Linear(hidden_dim, out_dim),
    )


class BYOL(nn.Module):
    def __init__(self, encoder_dim: int = ENCODER_DIM, proj_dim: int = 256, hidden_dim: int = 4096, pretrained: bool = True):
        super().__init__()
        self.online_encoder = resnet50_1ch(pretrained=pretrained)
        self.online_projector = mlp(encoder_dim, hidden_dim, proj_dim)
        self.online_predictor = mlp(proj_dim, hidden_dim, proj_dim)  # online branch only

        self.target_encoder = copy.deepcopy(self.online_encoder)
        self.target_projector = copy.deepcopy(self.online_projector)
        for p in self.target_encoder.parameters():
            p.requires_grad = False
        for p in self.target_projector.parameters():
            p.requires_grad = False

    @torch.no_grad()
    def update_target(self, tau: float) -> None:
        for online, target in [(self.online_encoder, self.target_encoder),
                                (self.online_projector, self.target_projector)]:
            for po, pt in zip(online.parameters(), target.parameters()):
                pt.data = tau * pt.data + (1 - tau) * po.data

    def forward(self, view1: torch.Tensor, view2: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        def online_out(v: torch.Tensor) -> torch.Tensor:
            return self.online_predictor(self.online_projector(self.online_encoder(v)))

        with torch.no_grad():
            def target_out(v: torch.Tensor) -> torch.Tensor:
                return self.target_projector(self.target_encoder(v))

            t1, t2 = target_out(view1), target_out(view2)
        o1, o2 = online_out(view1), online_out(view2)
        return o1, o2, t1.detach(), t2.detach()


def byol_loss(o1: torch.Tensor, o2: torch.Tensor, t1: torch.Tensor, t2: torch.Tensor) -> torch.Tensor:
    """Symmetrized normalized MSE -- predict each view's target from the *other*
    view's online output, average both directions. See section 6."""

    def d(o: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        o, t = F.normalize(o, dim=-1), F.normalize(t, dim=-1)
        return 2 - 2 * (o * t).sum(dim=-1)

    return (d(o1, t2) + d(o2, t1)).mean()


def verify_stem_averaging(atol: float = 1e-5) -> bool:
    """Resolves RESNET_PRETRAIN_PLAN.md section 11's open item: does this stem-
    averaging trick match what `segmentation_models_pytorch` already does under the
    hood for `in_channels=1` (the trick `jp-mvp1:src/model.py` relies on)?

    Measured result (run once locally against real pretrained weights): **they do
    not match** (max_abs_diff ~2.45, nowhere near atol). Two independent causes,
    confirmed by reading smp's own source
    (`segmentation_models_pytorch/encoders/_utils.py:patch_first_conv`):
      1. smp **sums** the 3 input-channel filters (`weight.sum(1, keepdim=True)`),
         not averages them -- a real 3x scale difference, not just a numerical
         wobble. This implementation deliberately uses `.mean()` instead: summing
         leaves the stem's output activation scale ~3x larger than what the
         pretrained BN/ReLU stack downstream was calibrated for, which averaging
         avoids.
      2. smp's `encoder_weights="imagenet"` pulls a *different* pretrained
         checkpoint (its own hub, `smp-hub/resnet50.imagenet` on HuggingFace) than
         `tvm.ResNet50_Weights.IMAGENET1K_V2` used here -- a different training
         recipe, not just a different stem.
    Net: `jp-mvp1:src/model.py`'s `smp.Unet(in_channels=1)` and this file's
    `resnet50_1ch()` are both reasonable, independently-justified choices, but are
    NOT numerically equivalent -- don't assume interchangeability between them.
    """
    import segmentation_models_pytorch as smp

    ours = resnet50_1ch(pretrained=True)
    ours_w = ours.conv1.weight.detach()

    smp_model = smp.Unet(encoder_name="resnet50", encoder_weights="imagenet", in_channels=1, classes=1)
    smp_w = smp_model.encoder.conv1.weight.detach()

    if ours_w.shape != smp_w.shape:
        logger.warning(f"shape mismatch: ours {tuple(ours_w.shape)} vs smp {tuple(smp_w.shape)}")
        return False

    max_abs_diff = (ours_w - smp_w).abs().max().item()
    matches = torch.allclose(ours_w, smp_w, atol=atol)
    logger.info(f"stem-averaging vs smp in_channels=1: max_abs_diff={max_abs_diff:.3e}, matches(atol={atol})={matches}")
    if not matches:
        logger.info("mismatch expected and explained -- see verify_stem_averaging() docstring (smp sums not averages, plus a different pretrained checkpoint source)")
    return matches


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
    verify_stem_averaging()
