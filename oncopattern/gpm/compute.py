"""Compute planner: what model size can a given GPU budget and dataset actually train?

Uses the standard estimates:

* training FLOPs C = 6 N D (N parameters, D training tokens)
* compute-optimal allocation (Hoffmann et al. 2022, "Chinchilla"): D = 20 N, so N = sqrt(C / 120)
* data-limited regime (Muennighoff et al. 2023): repeating data up to about 4 epochs is nearly as good as
  fresh data, so with U unique tokens, N_data = 4U / 20
* memory to train with AdamW in mixed precision: about 16 bytes per parameter
  (fp32 master weights + two Adam moments + half-precision weights and gradients),
  before activations. Plain data parallelism keeps a full copy on every GPU.

Text-model rules of thumb are only a first guide for image tokens, which
carry more redundancy than words. Measure the real throughput with
``train.py`` and re-plan.
"""
from __future__ import annotations

import math

import torch

from oncopattern.gpm.config import PRESETS, preset

# dense fp16/bf16 tensor-core peak FLOP/s and memory per device
GPUS = {
    "t4": (65e12, 16e9),
    "p100": (18.7e12, 16e9),
    "v100": (125e12, 32e9),
    "l4": (121e12, 24e9),
    "a100": (312e12, 80e9),
    "h100": (989e12, 80e9),
    "tpu_v3_8": (420e12, 128e9),
    "cpu": (2e11, 16e9),
}


def count_params(name: str, **overrides) -> int:
    """Exact parameter count without allocating memory (meta device)."""
    from oncopattern.gpm.model import GeneralPatternModel
    from oncopattern.gpm.tokenizer import ByteTokenizer

    cfg = preset(name, vocab_size=ByteTokenizer().vocab_size, **overrides)
    with torch.device("meta"):
        m = GeneralPatternModel(cfg)
    return m.n_params()


def tokens_per_example(image_size: int = 64, patch: int = 8, n_images: float = 1.0, text_tokens: int = 70) -> int:
    return int(n_images * (image_size // patch) ** 2 + text_tokens)


def plan(gpu: str = "t4", n_gpus: int = 2, hours: float = 30.0, mfu: float = 0.3,
         unique_tokens: float | None = None, epochs: float = 4.0) -> dict:
    peak, mem = GPUS[gpu]
    C = peak * n_gpus * mfu * hours * 3600
    n_compute = math.sqrt(C / 120)
    n_data = unique_tokens * epochs / 20 if unique_tokens else float("inf")
    n_memory = mem * 0.6 / 16  # leave 40% for activations
    n_opt = min(n_compute, n_data, n_memory)
    limit = {n_compute: "compute", n_data: "data", n_memory: "GPU memory"}[n_opt]
    sizes = {k: count_params(k) for k in PRESETS}
    fits = {k: {"params": v,
                "trainable_on_one_device": v <= n_memory,
                "tokens_reachable": C / (6 * v),
                "tokens_per_param": C / (6 * v) / v} for k, v in sizes.items()}
    feasible = [k for k, v in sizes.items() if v <= n_memory]
    best = min(feasible, key=lambda k: abs(math.log(sizes[k] / n_opt))) if feasible else None
    return {"gpu": gpu, "n_gpus": n_gpus, "hours": hours, "assumed_mfu": mfu, "total_flops": C,
            "optimal_params": n_opt, "limited_by": limit, "optimal_tokens": 20 * n_opt,
            "recommended_preset": best, "presets": fits}


def format_plan(p: dict) -> str:
    lines = [f"Budget: {p['n_gpus']}x {p['gpu']} for {p['hours']:g} h at {p['assumed_mfu']:.0%} utilisation "
             f"= {p['total_flops']:.2e} FLOPs",
             f"Compute-optimal model: ~{p['optimal_params'] / 1e6:.0f}M parameters on ~{p['optimal_tokens'] / 1e9:.2f}B "
             f"tokens (limited by {p['limited_by']})",
             f"Recommended preset: {p['recommended_preset']}", "",
             f"{'preset':<6} {'params':>10} {'fits 1 GPU':>10} {'tokens reachable':>17} {'tokens/param':>13}"]
    for k, v in p["presets"].items():
        lines.append(f"{k:<6} {v['params'] / 1e6:>9.1f}M {str(v['trainable_on_one_device']):>10} "
                     f"{v['tokens_reachable'] / 1e9:>16.2f}B {v['tokens_per_param']:>13.2f}")
    lines.append("")
    lines.append("tokens/param << 20 means the model is under-trained on this budget; the 7b preset needs sharded "
                 "training (FSDP) across many 80 GB GPUs.")
    return "\n".join(lines)
