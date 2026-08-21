"""Fast per-epoch semantic segmentation metrics (no instance splitting).

For the competition's actual instance-level Panoptic Quality metric, see
scripts/evaluate.py + src/utils/panoptic.py, which is too slow to run every
epoch (connected components + matching per image).
"""

from __future__ import annotations

import segmentation_models_pytorch as smp
import torch
from torchmetrics.segmentation import DiceScore


def compute_batch_iou_dice(logits: torch.Tensor, target: torch.Tensor, threshold: float = 0.5) -> tuple[float, float]:
    """Micro-averaged IoU and Dice (F1) over a batch of binary logits vs. 0/1 targets."""
    probs = torch.sigmoid(logits)
    tp, fp, fn, tn = smp.metrics.get_stats(probs, target.long(), mode="binary", threshold=threshold)
    iou = smp.metrics.iou_score(tp, fp, fn, tn, reduction="micro").item()
    dice = smp.metrics.f1_score(tp, fp, fn, tn, reduction="micro").item()
    return iou, dice


class EpochDiceScore:
    """Accumulates torchmetrics.segmentation.DiceScore (foreground-only) across an epoch."""

    def __init__(self, device: torch.device, threshold: float = 0.5) -> None:
        self.threshold = threshold
        # sync_on_compute=False: this only ever runs on rank 0 under DDP (validation is rank-0-only),
        # so compute() must not wait on a distributed all-gather the other ranks never join.
        self.metric = DiceScore(
            num_classes=2, include_background=False, average="macro", input_format="index", sync_on_compute=False
        ).to(device)

    def update(self, logits: torch.Tensor, target: torch.Tensor) -> None:
        preds = (torch.sigmoid(logits) > self.threshold).long().squeeze(1)
        self.metric.update(preds, target.long().squeeze(1))

    def compute(self) -> float:
        return self.metric.compute().item()

    def reset(self) -> None:
        self.metric.reset()
