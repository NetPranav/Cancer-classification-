"""General Pattern Model (GPM): one transformer, one token stream, every task is a prompt.

Design in one paragraph
-----------------------
Code models are accurate because they (1) read vast amounts of unlabelled code
and learn its *grammar*, and (2) produce output that can be *checked*
(compilers, tests). GPM applies both to medical images. Tissue has a grammar:
cells tile space with regular size, spacing, orientation and texture. A
malignant change is a grammatical error, like a typo in code. GPM learns that
grammar by predicting hidden image patches from their context (masked-pattern
modelling). The same network also reads and writes text: prompts, answers,
numbers and coordinate tokens. Any task (detect, locate, measure, count,
compare timepoints, describe, abstain) is therefore a prompt with a text
answer, never a fixed classification head. Its surprise at each patch (how
badly it predicts the patch from context) is a label-free anomaly map, the
image equivalent of a code model flagging an unlikely token.

Token stream
------------
    <bos> prompt text <img>x N_1 [<img>x N_2 ...] question <sep> answer <eos>

* Text tokens are byte embeddings with 1D rotary positions (RoPE).
* Image tokens are patch embeddings from a D4 orientation-shared stem, plus a
  **physical-position code**: Fourier features of each patch's centre in
  millimetres, its pixel spacing (log2 mm/px) and the acquisition time
  (days). A 0.5 um/px slide and a 3 mm/px MRI therefore live in one
  coordinate system, and one model spans cells to organs, and months of
  follow-up.
* Attention mask: patches of the same image see each other bidirectionally
  (an image has no reading order); everything else is causal (prefix-LM).

Losses
------
    L = lm_weight * CE(next token | answer positions) + mim_weight * MSE(hidden patch pixels)

A batch mixes answer-generation examples and masked-pattern examples; a flag per
example chooses which one it is, so a single forward pass serves both.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from oncopattern.gpm.config import GPMConfig
from oncopattern.models.equivariant import GroupConv, LiftingConv, d4_transform

MODALITIES = ("other", "histology", "ct", "mri", "pet", "xray", "mammography", "ultrasound", "dermoscopy",
              "endoscopy", "cytology", "fundus", "phantom_tissue", "phantom_mri")


def modality_id(name: str) -> int:
    name = (name or "other").lower()
    return MODALITIES.index(name) if name in MODALITIES else 0


# --------------------------------------------------------------------------- embeddings
class D4PatchStem(nn.Module):
    """Each learned filter is applied in all 8 orientations (4 rotations x mirror).

    For a patch x and filter w: h_g = <T_g w, x>, g in D4. Rotating the patch
    permutes (h_g) instead of producing unrelated features, so orientation is
    explicit to the trunk, and every filter trains on 8x the views.
    """

    def __init__(self, in_ch: int, patch: int, channels: int, d_model: int):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(channels, in_ch, patch, patch) / math.sqrt(in_ch * patch * patch))
        self.proj = nn.Linear(channels * 8, d_model)

    def forward(self, patches):  # (P, C, p, p)
        w = torch.stack([d4_transform(self.weight, g) for g in range(8)], 1)  # (c, 8, C, p, p)
        h = patches.flatten(1) @ w.flatten(2).flatten(0, 1).t()  # (P, c*8)
        return self.proj(F.gelu(h))


class ConvPatchStem(nn.Module):
    """Two D4-equivariant conv layers *inside* each patch, then max + mean pooling.

    A single linear map per patch (the plain-ViT stem) cannot see local
    contrast, such as a blob against its surroundings, without huge amounts of
    data. Early convolutions fix that (Xiao et al. 2021). Convolutions never
    cross patch borders, so a hidden patch cannot leak into its neighbours'
    embeddings during masked-patch training. Orientation channels are kept (not
    pooled), so "which way a nucleus points" still reaches the trunk.
    """

    def __init__(self, in_ch: int, channels: int, d_model: int):
        super().__init__()
        c1 = max(4, channels // 4)
        # replicate padding: zero padding would draw a fake edge around every patch of bright tissue
        self.lift = LiftingConv(in_ch, c1, 3, padding_mode="replicate")
        self.gconv = GroupConv(c1, channels // 2, 3, padding_mode="replicate")
        self.proj = nn.Linear((channels // 2) * 8 * 2 + 5 * in_ch, d_model)

    def forward(self, patches, group: torch.Tensor | None = None):  # (P, C, p, p)
        flat = patches.flatten(2)
        # direct intensity statistics: brightness extremes are often the whole story (e.g. an enhancing
        # lesion) and random conv features at initialisation do not preserve them
        stats = torch.cat([flat.mean(-1), flat.std(-1), flat.amax(-1), flat.amin(-1)], -1)
        x = patches
        if group is not None:  # standardise each image (all its patches) so contrast, not offset, drives the convs
            n_g = int(group.max()) + 1
            s1 = torch.zeros(n_g, device=x.device, dtype=x.dtype).index_add_(0, group, flat.mean((1, 2)))
            s2 = torch.zeros_like(s1).index_add_(0, group, flat.pow(2).mean((1, 2)))
            cnt = torch.bincount(group, minlength=n_g).clamp(min=1).to(x.dtype)
            mu = s1 / cnt
            sd = (s2 / cnt - mu.pow(2)).clamp(min=1e-6).sqrt()
            x = (x - mu[group, None, None, None]) / (sd[group, None, None, None] + 1e-3)
            stats = torch.cat([stats, ((flat.amax(-1) - mu[group, None]) / (sd[group, None] + 1e-3))], -1)
        else:
            stats = torch.cat([stats, torch.zeros_like(stats[:, : patches.shape[1]])], -1)
        h = F.gelu(self.lift(x))
        h = F.gelu(self.gconv(h))  # (P, c, 8, p, p)
        pooled = torch.stack([h.amax((-2, -1)), h.mean((-2, -1))], -1)  # (P, c, 8, 2)
        return self.proj(torch.cat([pooled.flatten(1), stats], -1))


class LinearPatchStem(nn.Module):
    def __init__(self, in_ch: int, patch: int, d_model: int):
        super().__init__()
        self.proj = nn.Linear(in_ch * patch * patch, d_model)

    def forward(self, patches):
        return self.proj(patches.flatten(1))


class PhysicalPosition(nn.Module):
    """phys = [y_mm, x_mm, z_mm, log2(mm per px), t_days] -> d_model.

    Spatial bands span wavelengths from 2 um (sub-cellular) to 0.5 m (whole
    body), geometrically spaced; time bands span 1 week to 10 years.

    **Nyquist gating.** A band whose wavelength is shorter than twice the
    patch spacing (patch_size x mm/px) cannot be represented on that patch
    grid; sampled there it is pure aliasing noise, which would drown the image
    content. Each band is therefore weighted by a smooth gate that is ~1 above
    the patch grid's Nyquist wavelength and ~0 below it. A slide at 0.5 um/px
    keeps its micrometre bands; an MRI at 3 mm/px keeps only centimetre and
    longer ones. The same model is thus anti-aliased at every scale.
    """

    def __init__(self, d_model: int, n_freq: int = 16, patch_size: int = 8):
        super().__init__()
        wl = torch.logspace(math.log10(0.002), math.log10(500.0), n_freq)
        self.register_buffer("log_wl", wl.log(), persistent=False)
        self.register_buffer("space_freq", 2 * math.pi / wl, persistent=False)
        tw = torch.logspace(math.log10(7.0), math.log10(3650.0), 4)
        self.register_buffer("time_freq", 2 * math.pi / tw, persistent=False)
        self.patch_size = patch_size
        self.proj = nn.Linear(3 * 2 * n_freq + 1 + 2 * 4, d_model)

    def gate(self, log2_spacing):  # (P,) -> (P, F)
        nyquist = 2.0 * self.patch_size * torch.exp2(log2_spacing)
        return torch.sigmoid(4.0 * (self.log_wl[None] - torch.log(nyquist)[:, None]))

    def forward(self, phys):  # (P, 5)
        g = self.gate(phys[:, 3])[:, None, :]  # (P, 1, F)
        s = phys[:, :3, None] * self.space_freq  # (P, 3, F)
        t = phys[:, 4:5] * self.time_freq  # (P, 4)
        feats = [(torch.sin(s) * g).flatten(1), (torch.cos(s) * g).flatten(1), phys[:, 3:4] / 10.0,
                 torch.sin(t), torch.cos(t)]
        return self.proj(torch.cat(feats, 1))


# --------------------------------------------------------------------------- trunk
class RMSNorm(nn.Module):
    def __init__(self, d: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d))

    def forward(self, x):
        return x * torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + self.eps).to(x.dtype) * self.weight


def rope_tables(n: int, dim: int, theta: float, device=None):
    inv = 1.0 / theta ** (torch.arange(0, dim, 2, device=device).float() / dim)
    ang = torch.arange(n, device=device).float()[:, None] * inv[None]
    return torch.cos(ang), torch.sin(ang)


def apply_rope(x, cos, sin):  # x (B, H, T, D); cos/sin (T, D/2)
    x1, x2 = x[..., 0::2], x[..., 1::2]
    c, s = cos[None, None].to(x.dtype), sin[None, None].to(x.dtype)
    return torch.stack([x1 * c - x2 * s, x1 * s + x2 * c], -1).flatten(-2)


class Attention(nn.Module):
    def __init__(self, cfg: GPMConfig):
        super().__init__()
        self.h, self.dh = cfg.n_heads, cfg.d_model // cfg.n_heads
        self.qkv = nn.Linear(cfg.d_model, 3 * cfg.d_model, bias=False)
        self.out = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.dropout = cfg.dropout

    def forward(self, x, cos, sin, mask=None, cache=None):
        b, t, d = x.shape
        q, k, v = self.qkv(x).view(b, t, 3, self.h, self.dh).permute(2, 0, 3, 1, 4)
        q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
        if cache is not None:
            if "k" in cache:
                k, v = torch.cat([cache["k"], k], 2), torch.cat([cache["v"], v], 2)
            cache["k"], cache["v"] = k, v
        y = F.scaled_dot_product_attention(q, k, v, attn_mask=mask,
                                           dropout_p=self.dropout if self.training else 0.0)
        return self.out(y.transpose(1, 2).reshape(b, t, d))


class SwiGLU(nn.Module):
    def __init__(self, d: int, hidden: int):
        super().__init__()
        self.w1, self.w3 = nn.Linear(d, hidden, bias=False), nn.Linear(d, hidden, bias=False)
        self.w2 = nn.Linear(hidden, d, bias=False)

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class Block(nn.Module):
    def __init__(self, cfg: GPMConfig):
        super().__init__()
        self.n1, self.n2 = RMSNorm(cfg.d_model), RMSNorm(cfg.d_model)
        self.attn = Attention(cfg)
        self.mlp = SwiGLU(cfg.d_model, cfg.mlp_hidden)

    def forward(self, x, cos, sin, mask=None, cache=None):
        x = x + self.attn(self.n1(x), cos, sin, mask, cache)
        return x + self.mlp(self.n2(x))


# --------------------------------------------------------------------------- model
class GeneralPatternModel(nn.Module):
    def __init__(self, cfg: GPMConfig):
        super().__init__()
        self.cfg = cfg
        d = cfg.d_model
        self.tok = nn.Embedding(cfg.vocab_size, d)
        if cfg.stem == "d4conv":
            self.stem = ConvPatchStem(cfg.in_channels, cfg.stem_channels, d)
        elif cfg.stem == "d4":
            self.stem = D4PatchStem(cfg.in_channels, cfg.patch_size, cfg.stem_channels, d)
        else:
            self.stem = LinearPatchStem(cfg.in_channels, cfg.patch_size, d)
        self.phys = PhysicalPosition(d, cfg.n_freq, cfg.patch_size)
        # The position code is RMS-normalised (its magnitude means nothing). Content is NOT normalised:
        # its magnitude says how strong a pattern is, and normalising would blow faint background noise
        # up to the size of a real finding. A learned scalar gain sets content's overall scale.
        self.pos_norm = RMSNorm(d)
        self.content_gain = nn.Parameter(torch.tensor(1.0))
        self.modality = nn.Embedding(cfg.n_modalities, d)
        self.mask_emb = nn.Parameter(torch.zeros(d))
        self.summary = nn.Sequential(nn.Linear(2 * d, d), nn.GELU(), nn.Linear(d, d))
        self.surprise_proj = nn.Sequential(nn.Linear(1, d // 4), nn.GELU(), nn.Linear(d // 4, d))
        self.register_buffer("sur_mean", torch.tensor(0.0))
        self.register_buffer("sur_var", torch.tensor(1.0))
        self.summary_norm = RMSNorm(d)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layers)])
        self.norm = RMSNorm(d)
        self.lm_head = nn.Linear(d, cfg.vocab_size, bias=False)
        self.lm_head.weight = self.tok.weight  # tied
        self.patch_head = nn.Linear(d, cfg.in_channels * cfg.patch_size ** 2)
        self.apply(self._init)
        # image content must dominate its token: at GPT-style init (std 0.02) the image-dependent part of a
        # patch token was ~10x smaller than its fixed position part, and training collapsed to base rates
        proj = self.stem.proj
        nn.init.normal_(proj.weight, std=1.0 / math.sqrt(proj.in_features))
        nn.init.constant_(self.pos_norm.weight, 0.1)
        nn.init.constant_(self.summary_norm.weight, 0.1)
        nn.init.normal_(self.mask_emb, std=0.1)
        for n, p in self.named_parameters():  # scaled residual init (GPT-2)
            if n.endswith("out.weight") or n.endswith("w2.weight"):
                nn.init.normal_(p, std=0.02 / math.sqrt(2 * cfg.n_layers))

    @staticmethod
    def _init(m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, std=0.02)

    def n_params(self, trainable_only: bool = False) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad or not trainable_only)

    # ---- building the input ----------------------------------------------
    def embed(self, batch: dict, patch_mask: torch.Tensor | None = None, surprise: torch.Tensor | None = None):
        ids = batch["ids"]
        x = self.tok(ids)
        pidx = batch["patch_index"]  # (B, T) index into batch["patches"], -1 for text
        is_img = pidx >= 0
        if is_img.any():
            # grey <-> colour is adapted to the stem's channel count by _targets
            pos = self.pos_norm(self.phys(batch["phys"]) + self.modality(batch["modality"]))
            pt = self._targets(batch["patches"])
            content = self.content_gain * (self.stem(pt, batch.get("patch_group")) if isinstance(self.stem, ConvPatchStem)
                                           else self.stem(pt))
            if surprise is not None:
                z = ((surprise - self.sur_mean) / torch.sqrt(self.sur_var + 1e-6)).clamp(-5, 10)
                if patch_mask is not None:
                    z = torch.where(patch_mask, torch.zeros_like(z), z)  # a hidden patch's surprise would leak it
                content = content + self.surprise_proj(z[:, None].to(content.dtype))
            if patch_mask is not None and patch_mask.any():
                content = torch.where(patch_mask[:, None], self.mask_emb.to(content.dtype), content)
            pe = content + pos
            x = x.clone()
            x[is_img] = pe[pidx[is_img]].to(x.dtype)
            sg = batch.get("sum_group")
            if sg is not None and (sg >= 0).any():
                # per-image summary from *visible* content only (masked patches carry mask_emb), so no leak
                grp = batch["patch_group"]
                n_g = int(grp.max()) + 1
                idx = grp[:, None].expand(-1, content.shape[1])
                mx = torch.zeros(n_g, content.shape[1], dtype=content.dtype, device=content.device)
                mx = mx.scatter_reduce(0, idx, content, reduce="amax", include_self=False)
                mean = torch.zeros_like(mx).index_add_(0, grp, content)
                mean = mean / torch.bincount(grp, minlength=n_g).clamp(min=1)[:, None].to(mean.dtype)
                summ = self.summary_norm(self.summary(torch.cat([mx, mean], -1)))
                where = sg >= 0
                x[where] = x[where] + summ[sg[where]].to(x.dtype)
        return x

    @staticmethod
    def attention_mask(block: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
        """Allowed[b, i, j]: j <= i (causal) OR same image block; never attend to padding."""
        t = block.shape[1]
        causal = torch.ones(t, t, dtype=torch.bool, device=block.device).tril()
        same = (block[:, :, None] == block[:, None, :]) & (block[:, :, None] > 0)
        allowed = (causal[None] | same) & valid[:, None, :]
        allowed |= torch.eye(t, dtype=torch.bool, device=block.device)[None]  # padding rows attend to self
        return allowed[:, None]

    def trunk(self, x, mask, cache=None, offset: int = 0):
        t = x.shape[1]
        cos, sin = rope_tables(offset + t, self.cfg.d_model // self.cfg.n_heads, self.cfg.rope_theta, x.device)
        cos, sin = cos[offset:], sin[offset:]
        for i, blk in enumerate(self.blocks):
            c = None if cache is None else cache[i]
            if self.cfg.grad_checkpoint and self.training and c is None:
                x = checkpoint(blk, x, cos, sin, mask, use_reentrant=False)
            else:
                x = blk(x, cos, sin, mask, c)
        return self.norm(x)

    def _targets(self, pt):
        c = self.cfg.in_channels
        if pt.shape[1] == c:
            return pt
        return pt.expand(-1, c, -1, -1) if pt.shape[1] == 1 else pt.mean(1, keepdim=True).expand(-1, c, -1, -1)

    # ---- training forward -------------------------------------------------
    def forward(self, batch: dict, generator: torch.Generator | None = None) -> dict:
        patch_mask = None
        if "patches" in batch and batch["patches"].shape[0]:
            mim_ex = batch["mim"][batch["patch_example"]]  # patches belonging to masked-pattern examples
            r = torch.rand(batch["patches"].shape[0], generator=generator).to(mim_ex.device)
            patch_mask = mim_ex & (r < self.cfg.mim_ratio)
        x = self.embed(batch, patch_mask, self._surprise_input(batch, generator))
        h = self.trunk(x, self.attention_mask(batch["block"], batch["valid"]))

        out = {}
        labels = batch["labels"]
        logits = self.lm_head(h[:, :-1])
        tgt = labels[:, 1:]
        if (tgt != -100).any():
            out["lm_loss"] = F.cross_entropy(logits.reshape(-1, logits.shape[-1]).float(), tgt.reshape(-1),
                                             ignore_index=-100)
        else:
            out["lm_loss"] = logits.sum() * 0.0
        if patch_mask is not None and patch_mask.any():
            pidx = batch["patch_index"]
            flat_h = h[pidx >= 0]
            order = pidx[pidx >= 0]
            hp = torch.empty(batch["patches"].shape[0], h.shape[-1], dtype=h.dtype, device=h.device)
            hp[order] = flat_h
            pred = self.patch_head(hp[patch_mask]).float()
            out["mim_loss"] = F.mse_loss(pred, self._targets(batch["patches"])[patch_mask].flatten(1).float())
        else:
            out["mim_loss"] = h.sum() * 0.0
        out["loss"] = self.cfg.lm_weight * out["lm_loss"] + self.cfg.mim_weight * out["mim_loss"]
        return out

    # ---- inference ------------------------------------------------------------
    @torch.no_grad()
    def generate(self, batch: dict, max_new: int = 24, temperature: float = 0.0,
                 generator: torch.Generator | None = None, eos_id: int | None = None) -> list[int]:
        """Decode an answer for a single-example batch whose ids end at the prompt (after <sep>).

        The prompt is encoded once; new tokens reuse the KV cache (O(t) per token)."""
        assert batch["ids"].shape[0] == 1, "generate expects one example"
        cache = [{} for _ in self.blocks]
        x = self.embed(batch, None, self._surprise_input(batch))
        h = self.trunk(x, self.attention_mask(batch["block"], batch["valid"]), cache)
        pos = batch["ids"].shape[1]
        out: list[int] = []
        last = h[:, -1]
        for _ in range(max_new):
            logits = self.lm_head(last).float()
            if temperature > 0:
                probs = F.softmax(logits / temperature, -1).cpu()  # CPU generator works for any device
                nxt = torch.multinomial(probs, 1, generator=generator)[0, 0].to(logits.device)
            else:
                nxt = logits.argmax(-1)[0]
            out.append(int(nxt))
            if eos_id is not None and int(nxt) == eos_id:
                break
            x = self.tok(nxt.view(1, 1))
            last = self.trunk(x, None, cache, offset=pos)[:, -1]
            pos += 1
        return out

    def sequence_logprob(self, batch: dict) -> torch.Tensor:
        """Sum of log p(label tokens) per example (labels = -100 elsewhere). Differentiable."""
        x = self.embed(batch, None, self._surprise_input(batch))
        h = self.trunk(x, self.attention_mask(batch["block"], batch["valid"]))
        logp = F.log_softmax(self.lm_head(h[:, :-1]).float(), -1)
        tgt = batch["labels"][:, 1:]
        m = tgt != -100
        tok = logp.gather(-1, tgt.clamp(min=0)[..., None])[..., 0]
        return (tok * m).sum(1)

    @torch.no_grad()
    def _prediction_error(self, batch: dict, groups: int, generator: torch.Generator | None = None) -> torch.Tensor:
        """Per-patch MSE when the patch is hidden (each patch hidden exactly once over ``groups`` passes)."""
        n = batch["patches"].shape[0]
        dev = batch["patches"].device
        assign = torch.randperm(n, generator=generator).to(dev) % groups
        err = torch.zeros(n, device=dev)
        pidx = batch["patch_index"]
        mask = self.attention_mask(batch["block"], batch["valid"])
        target = self._targets(batch["patches"]).flatten(1).float()
        for k in range(groups):
            m = assign == k
            neutral = self.sur_mean.expand(n) if self.cfg.surprise_conditioning else None  # "average surprise"
            h = self.trunk(self.embed(batch, m, neutral), mask)
            hp = torch.empty(n, h.shape[-1], dtype=h.dtype, device=h.device)
            hp[pidx[pidx >= 0]] = h[pidx >= 0]
            err[m] = (self.patch_head(hp[m]).float() - target[m]).pow(2).mean(1)
        return err

    def _surprise_input(self, batch: dict, generator: torch.Generator | None = None):
        """log prediction error per patch for surprise conditioning (None if disabled or no images)."""
        if not self.cfg.surprise_conditioning or batch["patches"].shape[0] == 0:
            return None
        was = self.training
        self.eval()
        if generator is None and not was:
            generator = torch.Generator().manual_seed(0)  # deterministic at inference
        s = torch.log(self._prediction_error(batch, self.cfg.surprise_groups, generator) + 1e-6)
        self.train(was)
        if was:  # running statistics used to standardise surprise
            with torch.no_grad():
                self.sur_mean.lerp_(s.mean(), 0.01)
                self.sur_var.lerp_(s.var(unbiased=False), 0.01)
        return s

    @torch.no_grad()
    def patch_surprise(self, batch: dict, groups: int = 4, seed: int = 0) -> torch.Tensor:
        """Per-patch prediction error when that patch is hidden: the label-free 'typo detector'."""
        return self._prediction_error(batch, groups, torch.Generator().manual_seed(seed)).cpu()
