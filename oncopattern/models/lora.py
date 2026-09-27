"""Low-rank adaptation (LoRA, Hu et al. 2021, from large language models).

To adapt a trained model to a new cancer type, scanner or hospital, freeze
W and learn only a rank-r update:

    y = W x + (alpha / r) * B A x,    A in R^{r x d_in}, B in R^{d_out x r}, B initialised to 0

This trains r (d_in + d_out) parameters instead of d_in * d_out (about 12x
fewer at r=2, d=48), and the adapter starts as an exact no-op. Each
site or cancer type can keep its own adapter and share one base model.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn


class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, r: int = 4, alpha: float = 8.0):
        super().__init__()
        self.base = base
        for p in self.base.parameters():
            p.requires_grad_(False)
        self.A = nn.Parameter(torch.empty(r, base.in_features))
        self.B = nn.Parameter(torch.zeros(base.out_features, r))
        nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))
        self.scale = alpha / r

    def forward(self, x):
        return self.base(x) + (x @ self.A.t() @ self.B.t()) * self.scale


def add_lora(module: nn.Module, r: int = 4, alpha: float = 8.0, targets: tuple[str, ...] = ()) -> nn.Module:
    """Freeze ``module`` and wrap every nn.Linear (or those whose name contains
    one of ``targets``) with a LoRA adapter. Returns the module for chaining."""
    for p in module.parameters():
        p.requires_grad_(False)
    for name, child in list(module.named_modules()):
        for cname, sub in list(child.named_children()):
            full = f"{name}.{cname}" if name else cname
            if isinstance(sub, nn.Linear) and (not targets or any(t in full for t in targets)):
                setattr(child, cname, LoRALinear(sub, r, alpha))
    return module


def trainable_parameters(module: nn.Module) -> int:
    return sum(p.numel() for p in module.parameters() if p.requires_grad)
