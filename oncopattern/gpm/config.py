"""Model configuration and size presets, from a CPU smoke test up to 7B.

Same code, different numbers. Which preset you can *train* depends on your
compute and data: see ``compute.py`` (``python -m oncopattern gpm-plan``).
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field


@dataclass
class GPMConfig:
    d_model: int = 256
    n_layers: int = 6
    n_heads: int = 8
    mlp_hidden: int | None = None  # SwiGLU hidden size; default ~8/3 * d_model rounded to 64
    vocab_size: int = 326  # set from the tokenizer
    patch_size: int = 8
    in_channels: int = 1
    stem: str = "d4conv"  # "d4conv" (equivariant convs inside each patch) | "d4" (one shared filter bank) | "linear" (ViT)
    stem_channels: int = 32  # filters per orientation for the D4 stem
    max_seq_len: int = 1024
    rope_theta: float = 10000.0
    n_freq: int = 16  # physical-position Fourier bands (micrometres ... decimetres)
    n_modalities: int = 16
    dropout: float = 0.0
    mim_ratio: float = 0.5  # fraction of patches hidden in masked-pattern examples
    mim_weight: float = 1.0
    lm_weight: float = 1.0
    grad_checkpoint: bool = False
    surprise_conditioning: bool = True  # feed each patch's context-prediction error back into its token
    surprise_groups: int = 2  # no-grad pre-passes; each hides 1/groups of the patches
    extra: dict = field(default_factory=dict)

    def __post_init__(self):
        if self.mlp_hidden is None:
            self.mlp_hidden = int(64 * round(self.d_model * 8 / 3 / 64))
        assert self.d_model % self.n_heads == 0 and (self.d_model // self.n_heads) % 2 == 0

    def to_dict(self):
        return asdict(self)


PRESETS: dict[str, dict] = {
    # runs on a laptop CPU in minutes; for tests and debugging only
    "tiny": dict(d_model=192, n_layers=4, n_heads=6, stem_channels=16),
    # Kaggle, one T4, a few hours
    "small": dict(d_model=384, n_layers=8, n_heads=6),
    # Kaggle 2x T4 within a 30 GPU-hour week (see compute.py)
    "base": dict(d_model=768, n_layers=12, n_heads=12, patch_size=16, stem_channels=48),
    # needs A100/H100-class GPUs
    "1b": dict(d_model=2048, n_layers=18, n_heads=16, patch_size=16, stem_channels=96),
    # ~7.15B, LLaMA-7B-shaped trunk (35 layers); needs a multi-GPU node and billions of image tokens
    "7b": dict(d_model=4096, n_layers=35, n_heads=32, mlp_hidden=11008, patch_size=16, stem_channels=128),
}


def preset(name: str, **overrides) -> GPMConfig:
    if name not in PRESETS:
        raise ValueError(f"unknown preset {name!r}; choose from {list(PRESETS)}")
    return GPMConfig(**{**PRESETS[name], **overrides})
