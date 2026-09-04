"""Representation health checks for BYOL training -- see RESNET_PRETRAIN_PLAN.md
section 8. BYOL's training loss can keep dropping even while representations
collapse to a near-constant output (a known failure mode of negative-free SSL
methods if something's misconfigured), so loss alone isn't a trustworthy stopping
signal. These checks run on rank 0 only via `model.module` (never the DDP-wrapped
model) -- same lesson `jp-mvp1:src/train.py` already documents for validation:
calling forward() asymmetrically across ranks on a DDP-wrapped model can hang on
buffer-broadcast collectives the other ranks never join.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader


@torch.no_grad()
def embedding_std(online_encoder: torch.nn.Module, loader: DataLoader, device: torch.device, max_batches: int = 4) -> float:
    """Per-dimension standard deviation of L2-normalized online-encoder embeddings
    over a few held-out batches. A healthy run keeps this well above zero
    throughout; a value collapsing toward zero is the earliest, cheapest signal
    something is wrong (predictor/target update bug, LR too high, etc.)."""
    online_encoder.eval()
    embeddings = []
    for i, (view1, _view2) in enumerate(loader):
        if i >= max_batches:
            break
        view1 = view1.to(device)
        emb = F.normalize(online_encoder(view1), dim=-1)
        embeddings.append(emb.cpu())
    online_encoder.train()
    if not embeddings:
        return float("nan")
    all_emb = torch.cat(embeddings, dim=0)
    return all_emb.std(dim=0).mean().item()


@torch.no_grad()
def nearest_neighbors(
    online_encoder: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    k: int = 5,
    max_images: int = 64,
    grid_out: str | None = None,
) -> torch.Tensor:
    """Embeds held-out frames and returns the top-k cosine-nearest-neighbor index
    matrix [n, k] (excluding self) for manual inspection -- confirm the neighbors
    are astronomically similar (comparable filament density/activity level, not
    random frames). If `grid_out` is given, also saves a visual grid: each row is
    one query frame followed by its k nearest neighbors, for the manual sanity
    check in section 8."""
    online_encoder.eval()
    embeddings, images = [], []
    for view1, _view2 in loader:
        view1 = view1.to(device)
        emb = F.normalize(online_encoder(view1), dim=-1)
        embeddings.append(emb.cpu())
        images.append(view1.cpu())
        if sum(e.shape[0] for e in embeddings) >= max_images:
            break
    online_encoder.train()

    all_emb = torch.cat(embeddings, dim=0)[:max_images]
    all_img = torch.cat(images, dim=0)[:max_images]
    sims = all_emb @ all_emb.T  # [n, n] cosine similarity (already L2-normalized)
    sims.fill_diagonal_(-1.0)  # exclude self
    topk = sims.topk(k, dim=1).indices  # [n, k]

    if grid_out is not None:
        _save_neighbor_grid(all_img, topk, grid_out)
    return topk


def _save_neighbor_grid(images: torch.Tensor, topk: torch.Tensor, out_path: str, n_rows: int = 8) -> None:
    """Saves a PNG grid: row i = query image i followed by its k nearest neighbors."""
    from pathlib import Path

    import numpy as np
    from PIL import Image

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    n_rows = min(n_rows, images.shape[0])
    k = topk.shape[1]
    crop = images.shape[-1]
    grid = np.zeros((n_rows * crop, (k + 1) * crop), dtype=np.uint8)
    for row in range(n_rows):
        query = (images[row, 0].clamp(0, 1).numpy() * 255).astype(np.uint8)
        grid[row * crop : (row + 1) * crop, 0:crop] = query
        for j, nbr_idx in enumerate(topk[row].tolist()):
            nbr = (images[nbr_idx, 0].clamp(0, 1).numpy() * 255).astype(np.uint8)
            grid[row * crop : (row + 1) * crop, (j + 1) * crop : (j + 2) * crop] = nbr
    Image.fromarray(grid).save(out_path)
