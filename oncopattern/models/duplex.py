"""Full-duplex streaming reasoner (from full-duplex speech models).

Full-duplex speech models such as Moshi (Defossez et al. 2024) listen and
speak at the same time: every incoming audio frame immediately updates the
outgoing stream, and the model can cut in once it knows enough. We apply the
same design to reading a scan:

* **Listen:** image patches arrive one token at a time, most suspicious first
  (the scan order comes from the label-free normality and concept maps, much
  like a radiologist's search pattern).
* **Speak:** after *every* token the model emits its current belief over
  hypotheses, so a running diagnosis stream is always available.
* **Barge in:** a halting head, trained CALM-style (Schuster et al. 2022, from
  early-exit language models), predicts whether the current belief already
  equals the final answer. Once it is confident, reading stops, and the rest of
  the image costs no compute.

The transformer is causal with a KV cache, so ``stream`` does O(t) work at step
t and matches ``forward`` exactly (this is tested). Training minimises

    L = sum_t w_t CE(belief_t, y) + lambda * sum_t BCE(halt_t, 1[argmax belief_t == y])

with w_t increasing in t, so early beliefs count but late ones count more.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class CausalBlock(nn.Module):
    def __init__(self, d: int, heads: int, mlp: int = 2):
        super().__init__()
        self.heads = heads
        self.ln1, self.ln2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.qkv = nn.Linear(d, 3 * d)
        self.proj = nn.Linear(d, d)
        self.mlp_in = nn.Linear(d, mlp * d)
        self.mlp_out = nn.Linear(mlp * d, d)

    def _split(self, t):
        b, n, d = t.shape
        return t.view(b, n, self.heads, d // self.heads).transpose(1, 2)

    def forward(self, x, cache=None):
        """x: (B, T, d). With ``cache`` (dict), x holds only new tokens and K/V are appended."""
        h = self.ln1(x)
        q, k, v = (self._split(t) for t in self.qkv(h).chunk(3, dim=-1))
        if cache is not None:
            if "k" in cache:
                k = torch.cat([cache["k"], k], 2)
                v = torch.cat([cache["v"], v], 2)
            cache["k"], cache["v"] = k, v
        tq, tk = q.shape[2], k.shape[2]
        mask = torch.ones(tq, tk, dtype=torch.bool, device=x.device).tril(tk - tq)
        att = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
        x = x + self.proj(att.transpose(1, 2).reshape(x.shape))
        return x + self.mlp_out(F.gelu(self.mlp_in(self.ln2(x))))


def spatial_encoding(rows: torch.Tensor, cols: torch.Tensor, grid: int, dim: int = 16) -> torch.Tensor:
    """2D sinusoidal position code for patch (row, col) on a grid x grid lattice."""
    freqs = torch.exp(torch.arange(dim // 4, dtype=torch.float32) * (-math.log(100.0) / max(dim // 4, 1)))
    out = []
    for p in (rows.float() / grid, cols.float() / grid):
        ang = p[..., None] * freqs * 2 * math.pi
        out += [torch.sin(ang), torch.cos(ang)]
    return torch.cat(out, -1)


class DuplexReasoner(nn.Module):
    def __init__(self, token_dim: int, n_classes: int, d: int = 48, heads: int = 4, layers: int = 2,
                 pos_dim: int = 16):
        super().__init__()
        self.pos_dim = pos_dim
        self.inp = nn.Linear(token_dim + pos_dim, d)
        self.blocks = nn.ModuleList([CausalBlock(d, heads) for _ in range(layers)])
        self.ln = nn.LayerNorm(d)
        self.belief = nn.Linear(d, n_classes)
        self.halt = nn.Linear(d, 1)

    def forward(self, tokens, pos):
        x = self.inp(torch.cat([tokens, pos], -1))
        for blk in self.blocks:
            x = blk(x)
        x = self.ln(x)
        return self.belief(x), self.halt(x).squeeze(-1)

    @torch.no_grad()
    def stream(self, tokens, pos, halt_threshold: float = 0.9, min_steps: int = 4):
        """Yield (t, belief_logits, halt_prob) one token at a time; stop on barge-in.

        tokens: (T, token_dim), pos: (T, pos_dim) for a single image.
        """
        caches = [{} for _ in self.blocks]
        for t in range(tokens.shape[0]):
            x = self.inp(torch.cat([tokens[t:t + 1], pos[t:t + 1]], -1))[None]
            for blk, c in zip(self.blocks, caches):
                x = blk(x, cache=c)
            x = self.ln(x)[0, 0]
            halt_p = float(torch.sigmoid(self.halt(x)))
            yield t, self.belief(x), halt_p
            if t + 1 >= min_steps and halt_p >= halt_threshold:
                return


def train_duplex(model: DuplexReasoner, tokens: torch.Tensor, pos: torch.Tensor, y: torch.Tensor,
                 epochs: int = 60, lr: float = 3e-3, batch_size: int = 32, halt_weight: float = 0.5,
                 seed: int = 0) -> list[float]:
    """tokens: (N, T, D), pos: (N, T, P), y: (N,)."""
    gen = torch.Generator().manual_seed(seed)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-3)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    n, T = tokens.shape[:2]
    w = torch.linspace(0.2, 1.0, T)
    w = w / w.sum()
    hist = []
    model.train()
    for _ in range(epochs):
        perm = torch.randperm(n, generator=gen)
        for i in range(0, n, batch_size):
            b = perm[i:i + batch_size]
            logits, halt = model(tokens[b], pos[b])
            yb = y[b][:, None].expand(-1, T)
            ce = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), yb.reshape(-1), reduction="none")
            ce = (ce.view(-1, T) * w).sum(1).mean()
            target = (logits.argmax(-1) == yb).float()
            hl = F.binary_cross_entropy_with_logits(halt, target)
            loss = ce + halt_weight * hl
            opt.zero_grad()
            loss.backward()
            opt.step()
            hist.append(loss.item())
        sched.step()
    model.eval()
    return hist
