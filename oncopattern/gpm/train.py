"""Resumable training loop, built for Kaggle notebooks (and any other GPU box).

Kaggle specifics this handles:

* **Session limit**: a GPU session ends after at most 12 h (about 30 h per
  week in total). ``time_budget_h`` stops cleanly with a checkpoint before the
  cut-off, and ``resume`` continues from it in the next session, including
  optimiser, scaler and data position.
* **T4 GPUs have no bf16**, so fp16 autocast with a gradient scaler is used;
  bf16 is used where supported.
* **2x T4**: launch with ``torchrun --nproc_per_node 2`` for data parallelism.
  A single process also works.

    python -m oncopattern gpm-train --preset base --out /kaggle/working/gpm --time-budget-h 11.5 \\
        --sources phantom_tissue folder:/kaggle/input/<dataset>/train,modality=histology,mm=0.0005,rgb=1
"""
from __future__ import annotations

import json
import math
import os
import re
import time
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, IterableDataset, get_worker_info

from oncopattern.gpm.config import preset
from oncopattern.gpm.data import build, collate
from oncopattern.gpm.model import GeneralPatternModel
from oncopattern.gpm.sources import (CSVSource, ImageFolderSource, MaskFolderSource, Mixture, MRIPhantomSource,
                                     TissuePhantomSource)
from oncopattern.gpm.tokenizer import ByteTokenizer


def parse_spec(spec: str) -> tuple[str, str, dict]:
    """``kind[:path][,key=value,...]`` -> (kind, path, options). Paths must not contain commas."""
    m = re.match(r"^([A-Za-z_]+)(?::([^,]*))?(?:,(.*))?$", spec)
    if not m:
        raise ValueError(f"bad source spec {spec!r}")
    opts = dict(p.split("=", 1) for p in (m.group(3) or "").split(",") if p)
    return m.group(1), m.group(2) or "", opts


def build_sources(specs, tok: ByteTokenizer, image_size: int = 64, seed: int = 0, mim_fraction: float = 0.3) -> Mixture:
    """Source specs (options after commas; ``weight`` sets the sampling weight):

    * ``phantom_tissue`` / ``phantom_mri``
    * ``folder:<root>,modality=histology,mm=0.0005,rgb=1[,normal=lung_n|colon_n]``: class sub-folders
    * ``csv:<labels.csv>,images=<dir>,ext=.tif,id=id,label=label,names=0=normal|1=metastasis,modality=...,mm=...``
    * ``masks:<images>,masks=<dir>`` or ``masks:<images>,suffix=_mask``, plus ``modality=,mm=,finding=``
    """
    srcs, weights = [], []
    for spec in specs:
        kind, path, o = parse_spec(spec)
        w = float(o.pop("weight", 1.0))
        common = dict(size=image_size)
        if kind == "phantom_tissue":
            src = TissuePhantomSource(tok, seed=seed, **common)
        elif kind == "phantom_mri":
            src = MRIPhantomSource(tok, seed=seed, **common)
        elif kind == "folder":
            extra = {"normal_classes": tuple(o["normal"].split("|"))} if "normal" in o else {}
            src = ImageFolderSource(path, tok, o.get("modality", "other"), float(o.get("mm", 1.0)),
                                    rgb=o.get("rgb", "0") == "1", description=o.get("text", ""), **common, **extra)
        elif kind == "csv":
            names = dict(kv.split("=", 1) for kv in o["names"].split("|")) if "names" in o else None
            src = CSVSource(path, o["images"], tok, o.get("modality", "other"), float(o.get("mm", 1.0)),
                            id_col=o.get("id", "id"), label_col=o.get("label", "label"), ext=o.get("ext", ""),
                            names=names, rgb=o.get("rgb", "0") == "1", description=o.get("text", ""),
                            max_rows=int(o["max_rows"]) if "max_rows" in o else None, **common)
        elif kind == "masks":
            src = MaskFolderSource(path, o.get("masks"), tok, o.get("modality", "other"), float(o.get("mm", 1.0)),
                                   o.get("finding", "lesion"), mask_suffix=o.get("suffix", ""),
                                   rgb=o.get("rgb", "0") == "1", description=o.get("text", ""), **common)
        else:
            raise ValueError(f"unknown source kind {kind!r} in {spec!r}")
        srcs.append(src)
        weights.append(w)
    return Mixture(srcs, weights, mim_fraction)


class StreamDataset(IterableDataset):
    """Endless stream of built examples; each (rank, worker, resume point) gets its own seed."""

    def __init__(self, mixture: Mixture, tok: ByteTokenizer, patch: int, seed: int):
        self.mixture, self.tok, self.patch, self.seed = mixture, tok, patch, seed

    def __iter__(self):
        info = get_worker_info()
        wid = info.id if info else 0
        rng = np.random.default_rng([self.seed, wid])
        while True:
            yield build(self.mixture.sample(rng), self.tok, self.patch)


def _setup_distributed():
    world = int(os.environ.get("WORLD_SIZE", "1"))
    if world > 1 and not dist.is_initialized():
        dist.init_process_group("nccl" if torch.cuda.is_available() else "gloo")
    rank = dist.get_rank() if world > 1 else 0
    local = int(os.environ.get("LOCAL_RANK", "0"))
    if torch.cuda.is_available():
        torch.cuda.set_device(local)
        device = torch.device("cuda", local)
    else:
        device = torch.device("cpu")
    return world, rank, device


def lr_at(step: int, total: int, warmup: int, base: float, floor: float = 0.1) -> float:
    if step < warmup:
        return base * (step + 1) / warmup
    prog = min(1.0, (step - warmup) / max(1, total - warmup))
    return base * (floor + (1 - floor) * 0.5 * (1 + math.cos(math.pi * prog)))


def train(preset_name: str = "tiny", steps: int = 1000, batch_size: int = 16, lr: float = 1e-3, warmup: int = 100,
          out: str = "outputs/gpm", sources=("phantom_tissue", "phantom_mri"), time_budget_h: float | None = None,
          ckpt_minutes: float = 20.0, resume: bool = True, seed: int = 0, grad_accum: int = 1, image_size: int = 64,
          num_workers: int = 0, weight_decay: float = 0.05, log_every: int = 10, overrides: dict | None = None,
          mim_fraction: float = 0.3) -> dict:
    t_start = time.time()
    world, rank, device = _setup_distributed()
    torch.manual_seed(seed)
    out = Path(out)
    if rank == 0:
        out.mkdir(parents=True, exist_ok=True)
    tok = ByteTokenizer()
    cfg = preset(preset_name, vocab_size=tok.vocab_size, **(overrides or {}))
    model = GeneralPatternModel(cfg).to(device)

    decay = [p for n, p in model.named_parameters() if p.dim() >= 2 and "tok." not in n]
    no_decay = [p for n, p in model.named_parameters() if p.dim() < 2 or "tok." in n]
    opt = torch.optim.AdamW([{"params": decay, "weight_decay": weight_decay},
                             {"params": no_decay, "weight_decay": 0.0}], lr=lr, betas=(0.9, 0.95),
                            fused=device.type == "cuda")
    use_bf16 = device.type == "cuda" and torch.cuda.is_bf16_supported()
    use_fp16 = device.type == "cuda" and not use_bf16
    amp_dtype = torch.bfloat16 if use_bf16 else torch.float16
    scaler = torch.amp.GradScaler("cuda", enabled=use_fp16)

    start, tokens_seen, history = 0, 0, []
    ckpt = out / "checkpoint.pt"
    if resume and ckpt.exists():
        state = torch.load(ckpt, map_location=device, weights_only=False)
        model.load_state_dict(state["model"])
        opt.load_state_dict(state["opt"])
        scaler.load_state_dict(state["scaler"])
        start, tokens_seen = state["step"], state["tokens_seen"]
        if rank == 0:
            print(f"[gpm] resumed from step {start}", flush=True)

    net = model
    if world > 1:
        # some batches leave a head unused (e.g. no masked patches), so let DDP tolerate that
        net = torch.nn.parallel.DistributedDataParallel(model, device_ids=[device.index] if device.type == "cuda" else None,
                                                        find_unused_parameters=True)

    mixture = build_sources(sources, tok, image_size, seed, mim_fraction)
    ds = StreamDataset(mixture, tok, cfg.patch_size, seed * 1_000_003 + rank * 7919 + start)
    dl = iter(DataLoader(ds, batch_size=batch_size, num_workers=num_workers,
                         collate_fn=lambda b: collate(b, tok), pin_memory=device.type == "cuda"))

    def save(step):
        if rank != 0:
            return
        torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "scaler": scaler.state_dict(),
                    "step": step, "tokens_seen": tokens_seen, "config": cfg.to_dict(), "preset": preset_name},
                   str(ckpt) + ".tmp")
        os.replace(str(ckpt) + ".tmp", ckpt)  # atomic: a killed session never leaves a half-written file

    last_ckpt = time.time()
    log = open(out / "train_log.jsonl", "a") if rank == 0 else None
    step = start
    stop_reason = "steps"
    if rank == 0:
        print(f"[gpm] {preset_name}: {model.n_params() / 1e6:.2f}M params, device={device}, world={world}, "
              f"amp={'bf16' if use_bf16 else 'fp16' if use_fp16 else 'off'}", flush=True)
    model.train()
    t_log = time.time()
    while step < steps:
        for g in opt.param_groups:
            g["lr"] = lr_at(step, steps, warmup, lr)
        opt.zero_grad(set_to_none=True)
        agg = {"loss": 0.0, "lm_loss": 0.0, "mim_loss": 0.0}
        for _ in range(grad_accum):
            batch = {k: v.to(device, non_blocking=True) for k, v in next(dl).items()}
            tokens_seen += int(batch["valid"].sum()) * world
            with torch.autocast(device.type, dtype=amp_dtype, enabled=use_bf16 or use_fp16):
                res = net(batch)
            scaler.scale(res["loss"] / grad_accum).backward()
            for k in agg:
                agg[k] += res[k].item() / grad_accum
        scaler.unscale_(opt)
        gn = float(torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0))
        scaler.step(opt)
        scaler.update()
        step += 1
        if rank == 0 and (step % log_every == 0 or step == steps):
            dt = time.time() - t_log
            t_log = time.time()
            rec = {"step": step, **{k: round(v, 5) for k, v in agg.items()}, "grad_norm": round(gn, 3),
                   "lr": opt.param_groups[0]["lr"], "tokens_seen": tokens_seen,
                   "sec_per_step": round(dt / log_every, 3), "elapsed_h": round((time.time() - t_start) / 3600, 3)}
            history.append(rec)
            log.write(json.dumps(rec) + "\n")
            log.flush()
            print(f"[gpm] step {step} loss {agg['loss']:.4f} (lm {agg['lm_loss']:.4f}, mim {agg['mim_loss']:.4f}) "
                  f"{rec['sec_per_step']}s/step", flush=True)
        if time.time() - last_ckpt > ckpt_minutes * 60:
            save(step)
            last_ckpt = time.time()
        if time_budget_h is not None and time.time() - t_start > time_budget_h * 3600:
            stop_reason = "time_budget"
            break
    save(step)
    if rank == 0:
        torch.save({"model": model.state_dict(), "config": cfg.to_dict(), "preset": preset_name, "step": step},
                   out / "model.pt")
        log.close()
    if world > 1:
        dist.barrier()
    return {"step": step, "stop_reason": stop_reason, "tokens_seen": tokens_seen, "history": history,
            "params": model.n_params(), "out": str(out)}


def load_model(path: str, device="cpu") -> GeneralPatternModel:
    from oncopattern.gpm.config import GPMConfig
    state = torch.load(path, map_location=device, weights_only=False)
    model = GeneralPatternModel(GPMConfig(**state["config"])).to(device)
    model.load_state_dict(state["model"])
    return model.eval()
