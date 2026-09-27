"""Label-free pretraining: masked-view consistency (SimSiam + JEPA + VICReg ideas).

Labelled cancer images are scarce, unlabelled ones are plentiful. We therefore
learn representations with no labels at all, combining three ideas:

* JEPA (Assran et al. 2023): predict the *representation* of a full view from
  a view with random blocks masked out, so the encoder has to infer hidden
  structure from context instead of copying pixels;
* SimSiam (Chen & He 2021): a stop-gradient on the target branch removes the
  need for negative pairs or large batches, which suits a CPU;
* VICReg variance hinge (Bardes et al. 2022): keeps every embedding dimension's
  batch standard deviation above gamma, which guards against collapse.

    L = 1/2 [D(p(z_masked), sg(z_full)) + D(p(z_full), sg(z_masked))]
        + beta * sum_j max(0, gamma - std(z_j))

D is negative cosine similarity. Orientation augmentation is unnecessary
because the encoder is exactly D4-invariant (see ``equivariant.py``).
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def _augment(x: torch.Tensor, gen: torch.Generator, mask_ratio: float = 0.0, block: int = 8):
    b = x.shape[0]
    scale = 1 + 0.1 * (torch.rand(b, 1, 1, 1, generator=gen) - 0.5)
    shift = 0.05 * (torch.rand(b, 1, 1, 1, generator=gen) - 0.5)
    x = x * scale + shift + 0.02 * torch.randn(x.shape, generator=gen)
    if mask_ratio > 0:
        h, w = x.shape[-2:]
        keep = (torch.rand(b, 1, h // block, w // block, generator=gen) >= mask_ratio).float()
        keep = F.interpolate(keep, size=(h, w), mode="nearest")
        fill = x.mean((-2, -1), keepdim=True)
        x = x * keep + fill * (1 - keep)
    return x


def _vic_var(z, gamma=1.0):
    return F.relu(gamma - torch.sqrt(z.var(0) + 1e-4)).mean()


def pretrain_ssl(encoder: nn.Module, images: torch.Tensor, epochs: int = 3, batch_size: int = 32,
                 lr: float = 2e-3, mask_ratio: float = 0.35, beta: float = 1.0, seed: int = 0) -> list[float]:
    """Pretrain ``encoder`` in place on unlabelled ``images`` (N, 1, H, W). Returns loss history."""
    gen = torch.Generator().manual_seed(seed)
    d = encoder.out_dim
    proj = nn.Sequential(nn.Linear(d, 64), nn.BatchNorm1d(64), nn.ReLU(), nn.Linear(64, 64))
    pred = nn.Sequential(nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, 64))
    params = list(encoder.parameters()) + list(proj.parameters()) + list(pred.parameters())
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=1e-4)
    encoder.train()
    history = []
    n = images.shape[0]
    for _ in range(epochs):
        perm = torch.randperm(n, generator=gen)
        for i in range(0, n - 1, batch_size):
            xb = images[perm[i:i + batch_size]]
            if xb.shape[0] < 2:
                continue
            z1 = proj(encoder.embed(_augment(xb, gen, mask_ratio)))
            z2 = proj(encoder.embed(_augment(xb, gen)))
            p1, p2 = pred(z1), pred(z2)
            loss = -(F.cosine_similarity(p1, z2.detach()).mean() + F.cosine_similarity(p2, z1.detach()).mean()) / 2
            loss = loss + beta * (_vic_var(z1) + _vic_var(z2))
            opt.zero_grad()
            loss.backward()
            opt.step()
            history.append(loss.item())
    encoder.eval()
    return history
