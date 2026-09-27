"""D4 group-equivariant convolutions (Cohen & Welling 2016, from symmetry physics).

A tissue section or an axial slice has no canonical orientation: rotating or
mirroring it does not change the diagnosis. An ordinary CNN has to *learn*
this from data (usually via 8x augmentation). A group convolution builds it in:
one learned kernel is applied in all 8 orientations of the dihedral group D4
(4 rotations x mirror), so

    f(T_u x) = T_u f(x)  (equivariance)      mean_g, mean_xy f(x) is invariant

holds *exactly*. The payoff for small datasets is that each weight is
effectively trained on 8x more views, at the same parameter count (Veeling et
al. 2018 showed this on PCam, one of the datasets in our catalogue).

Group elements are indexed g in 0..7 with r = g % 4 quarter-turns and f = g // 4
mirror, T_g(x) = rot90^r(flip^f(x)). The composition table is derived
numerically at import, so it cannot drift from the transform definition.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

G = 8


def d4_transform(x: torch.Tensor, g: int) -> torch.Tensor:
    r, f = g % 4, g // 4
    if f:
        x = torch.flip(x, dims=[-1])
    return torch.rot90(x, r, dims=[-2, -1]) if r else x


def _tables():
    a = torch.arange(9.0).reshape(3, 3)
    imgs = [d4_transform(a, g) for g in range(G)]
    compose = [[next(k for k in range(G) if torch.equal(d4_transform(imgs[j], i), imgs[k]))
                for j in range(G)] for i in range(G)]
    inverse = [next(j for j in range(G) if compose[i][j] == 0) for i in range(G)]
    return compose, inverse


COMPOSE, INVERSE = _tables()


class LiftingConv(nn.Module):
    """Image (B, C, H, W) -> group feature map (B, C_out, 8, H, W)."""

    def __init__(self, c_in: int, c_out: int, k: int = 5):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(c_out, c_in, k, k))
        self.bias = nn.Parameter(torch.zeros(c_out))
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        self.pad = k // 2

    def forward(self, x):
        w = torch.stack([d4_transform(self.weight, g) for g in range(G)], dim=1)
        c_out = w.shape[0]
        y = F.conv2d(x, w.flatten(0, 1), padding=self.pad)
        return y.view(x.shape[0], c_out, G, *y.shape[-2:]) + self.bias.view(1, -1, 1, 1, 1)


class GroupConv(nn.Module):
    """Group feature map (B, C_in, 8, H, W) -> (B, C_out, 8, H, W).

    out(g) = sum_h conv(in(h), T_g psi(g^-1 h)), which satisfies
    out[T_u in](g) = T_u out[in](u^-1 g).
    """

    def __init__(self, c_in: int, c_out: int, k: int = 3):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(c_out, c_in, G, k, k))
        self.bias = nn.Parameter(torch.zeros(c_out))
        nn.init.kaiming_uniform_(self.weight.view(c_out, c_in * G, k, k), a=math.sqrt(5))
        self.pad = k // 2
        perms = torch.tensor([[COMPOSE[INVERSE[g]][h] for h in range(G)] for g in range(G)])
        self.register_buffer("perms", perms, persistent=False)

    def full_weight(self):
        ws = [d4_transform(self.weight[:, :, self.perms[g]], g) for g in range(G)]
        w = torch.stack(ws, dim=1)  # (C_out, 8, C_in, 8, k, k)
        return w.flatten(0, 1).flatten(1, 2)

    def forward(self, x):
        b, c, g, h, w = x.shape
        y = F.conv2d(x.reshape(b, c * g, h, w), self.full_weight(), padding=self.pad)
        return y.view(b, -1, G, h, w) + self.bias.view(1, -1, 1, 1, 1)


def group_pool2d(x: torch.Tensor, k: int = 2) -> torch.Tensor:
    b, c, g, h, w = x.shape
    return F.avg_pool2d(x.reshape(b, c * g, h, w), k).view(b, c, g, h // k, w // k)


class PatternEncoder(nn.Module):
    """Small D4-equivariant encoder: image -> dense rotation-invariant-channel
    features at 1/4 resolution, plus a global embedding.

    ~50k parameters, runs comfortably on a CPU.
    """

    def __init__(self, in_ch: int = 1, widths=(8, 16, 32)):
        super().__init__()
        w0, w1, w2 = widths
        self.lift = LiftingConv(in_ch, w0, 5)
        self.bn0 = nn.BatchNorm3d(w0)
        self.g1 = GroupConv(w0, w1, 3)
        self.bn1 = nn.BatchNorm3d(w1)
        self.g2 = GroupConv(w1, w2, 3)
        self.bn2 = nn.BatchNorm3d(w2)
        self.out_dim = w2

    def forward(self, x):  # (B, 1, H, W) -> (B, C, H/4, W/4)
        x = group_pool2d(F.relu(self.bn0(self.lift(x))))
        x = group_pool2d(F.relu(self.bn1(self.g1(x))))
        x = F.relu(self.bn2(self.g2(x)))
        return x.mean(2)  # pool over orientations -> invariant channels, equivariant space

    def embed(self, x):
        return self.forward(x).mean((-2, -1))
