"""Normality memory bank (PatchCore, Roth et al. CVPR 2022, from industrial defect inspection).

Factory inspection has the same problem as early cancer detection: defects are
rare, varied and tiny, while "good" parts are plentiful. PatchCore handles it
by never learning what a defect looks like. It stores what *normal* local
patches look like and scores every new patch by its distance to the nearest
stored normal patch:

    s(p) = min_{m in M} || p - m ||_2

M is a coreset of all normal patch embeddings picked by greedy k-center
selection, which minimises the covering radius

    max_{p in P} min_{m in M} || p - m ||

so a 10% coreset keeps nearly all of the full bank's coverage at 10% of the
memory and lookup cost. Selection runs in a random low-dimensional projection
(Johnson-Lindenstrauss) to stay cheap.

The bank needs no labels and no abnormal examples. That is what lets the
system flag patterns it has never seen, including ones outside any training
class.
"""
from __future__ import annotations

import torch


class NormalityMemory:
    def __init__(self, coreset_ratio: float = 0.1, max_size: int = 4096, proj_dim: int = 16,
                 k: int = 1, seed: int = 0):
        self.coreset_ratio = coreset_ratio
        self.max_size = max_size
        self.proj_dim = proj_dim
        self.k = k
        self.seed = seed
        self.bank: torch.Tensor | None = None

    @torch.no_grad()
    def fit(self, feats: torch.Tensor) -> "NormalityMemory":
        feats = feats.float()
        n = feats.shape[0]
        m = int(min(max(1, round(n * self.coreset_ratio)), self.max_size, n))
        gen = torch.Generator().manual_seed(self.seed)
        proj = torch.randn(feats.shape[1], min(self.proj_dim, feats.shape[1]), generator=gen)
        z = feats @ proj
        idx = [int(torch.randint(n, (1,), generator=gen))]
        dist = torch.cdist(z, z[idx[0]][None]).squeeze(1)
        for _ in range(m - 1):
            i = int(torch.argmax(dist))
            idx.append(i)
            dist = torch.minimum(dist, torch.cdist(z, z[i][None]).squeeze(1))
        self.bank = feats[idx].clone()
        self.covering_radius = float(dist.max())
        return self

    @torch.no_grad()
    def score(self, feats: torch.Tensor, chunk: int = 4096) -> torch.Tensor:
        if self.bank is None:
            raise RuntimeError("NormalityMemory.fit must be called first")
        out = []
        for part in feats.float().split(chunk):
            d = torch.cdist(part, self.bank)
            out.append(d.topk(self.k, largest=False).values.mean(1))
        return torch.cat(out)
