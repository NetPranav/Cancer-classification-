"""Examples, sequence building and batching for the General Pattern Model.

An ``Example`` is images plus text. Each image has a pixel spacing, a
modality and an acquisition time. The text is a prompt, a question and an
answer. ``build`` turns it into one token stream; ``collate`` pads a list of
them into a batch the model consumes.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import torch

from oncopattern.gpm.model import modality_id
from oncopattern.gpm.tokenizer import ByteTokenizer


@dataclass
class ImageItem:
    pixels: np.ndarray  # (H, W) or (C, H, W), float in [0, 1]
    spacing_mm: float  # physical size of one pixel
    modality: str = "other"
    t_days: float = 0.0  # acquisition time relative to the patient's first scan


@dataclass
class Example:
    images: list
    prompt: str
    question: str = ""
    answer: str = ""
    task: str = "mim"
    meta: dict = field(default_factory=dict)
    mim: bool = False  # masked-pattern example (no answer loss)


def patchify(item: ImageItem, p: int):
    x = item.pixels.astype(np.float32)
    if x.ndim == 2:
        x = x[None]
    c, h, w = x.shape
    gh, gw = h // p, w // p
    x = x[:, : gh * p, : gw * p]
    patches = x.reshape(c, gh, p, gw, p).transpose(1, 3, 0, 2, 4).reshape(gh * gw, c, p, p)
    r, q = np.divmod(np.arange(gh * gw), gw)
    s = item.spacing_mm
    phys = np.stack([(r + 0.5) * p * s, (q + 0.5) * p * s, np.zeros(gh * gw),
                     np.full(gh * gw, math.log2(s)), np.full(gh * gw, item.t_days)], 1).astype(np.float32)
    return patches, phys, (gh, gw)


def build(ex: Example, tok: ByteTokenizer, patch: int, with_answer: bool = True) -> dict:
    # Order: context + question FIRST, then images, then <sep>. Image tokens can therefore attend to the
    # question while they are encoded (task-conditioned reading), and each image's <sum> token sits right
    # before the answer. With the question after the image, the answer position is surrounded by
    # identical text and training collapses to the answer's base rate (observed; see docs/GPM.md).
    ids = [tok.bos_id] + tok.encode(ex.prompt + ex.question)
    block = [0] * len(ids)
    pidx = [-1] * len(ids)
    patches, phys, mods, grids, pblock = [], [], [], [], []
    n = 0
    for k, im in enumerate(ex.images, 1):
        pt, ph, grid = patchify(im, patch)
        patches.append(pt)
        phys.append(ph)
        mods.append(np.full(len(pt), modality_id(im.modality)))
        grids.append(grid)
        ids += [tok.img_id] * len(pt) + [tok.sum_id]
        block += [k] * (len(pt) + 1)
        pidx += list(range(n, n + len(pt))) + [-1]
        pblock.append(np.full(len(pt), k))
        n += len(pt)
    ids.append(tok.sep_id)
    block.append(0)
    pidx.append(-1)
    labels = [-100] * len(ids)
    if with_answer and not ex.mim:
        a = tok.encode(ex.answer) + [tok.eos_id]
        ids += a
        labels += a
        block += [0] * len(a)
        pidx += [-1] * len(a)
    return {"ids": ids, "labels": labels, "block": block, "pidx": pidx,
            "patches": np.concatenate(patches) if patches else np.zeros((0, 1, patch, patch), np.float32),
            "phys": np.concatenate(phys) if phys else np.zeros((0, 5), np.float32),
            "modality": np.concatenate(mods) if mods else np.zeros(0, np.int64),
            "patch_block": np.concatenate(pblock) if pblock else np.zeros(0, np.int64),
            "grids": grids, "mim": ex.mim}


def collate(built: list[dict], tok: ByteTokenizer, device=None) -> dict:
    T = max(len(b["ids"]) for b in built)
    B = len(built)
    ids = torch.full((B, T), tok.pad_id, dtype=torch.long)
    labels = torch.full((B, T), -100, dtype=torch.long)
    block = torch.zeros((B, T), dtype=torch.long)
    pidx = torch.full((B, T), -1, dtype=torch.long)
    valid = torch.zeros((B, T), dtype=torch.bool)
    patches, phys, mods, pex, pgroup = [], [], [], [], []
    sum_group = torch.full((B, T), -1, dtype=torch.long)
    off, goff = 0, 0
    chans = max((b["patches"].shape[1] for b in built if len(b["patches"])), default=1)
    for i, b in enumerate(built):
        L = len(b["ids"])
        ids[i, :L] = torch.tensor(b["ids"])
        labels[i, :L] = torch.tensor(b["labels"])
        block[i, :L] = torch.tensor(b["block"])
        pi = torch.tensor(b["pidx"])
        pidx[i, :L] = torch.where(pi >= 0, pi + off, pi)
        valid[i, :L] = True
        n = len(b["patches"])
        pt = torch.from_numpy(b["patches"])
        if pt.shape[1] != chans:  # grey sources in a colour batch: repeat the channel
            pt = pt.expand(-1, chans, -1, -1) if pt.shape[1] == 1 else pt.mean(1, keepdim=True).expand(-1, chans, -1, -1)
        patches.append(pt.contiguous())
        phys.append(torch.from_numpy(b["phys"]))
        mods.append(torch.from_numpy(b["modality"].astype(np.int64)))
        pex.append(torch.full((n,), i, dtype=torch.long))
        # one group per (example, image): summary tokens pool their own image's patches
        pgroup.append(torch.from_numpy(b["patch_block"].astype(np.int64)) - 1 + goff)
        ids_i = torch.tensor(b["ids"])
        blk_i = torch.tensor(b["block"])
        is_sum = ids_i == tok.sum_id
        sum_group[i, :L][is_sum] = blk_i[is_sum] - 1 + goff
        goff += int(b["patch_block"].max()) if len(b["patch_block"]) else 0
        off += n
    batch = {"ids": ids, "labels": labels, "block": block, "patch_index": pidx, "valid": valid,
             "patches": torch.cat(patches), "phys": torch.cat(phys), "modality": torch.cat(mods),
             "patch_example": torch.cat(pex), "patch_group": torch.cat(pgroup), "sum_group": sum_group,
             "mim": torch.tensor([b["mim"] for b in built])}
    if device is not None:
        batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
    return batch


def prompt_batch(ex: Example, tok: ByteTokenizer, patch: int, device=None) -> dict:
    """A single-example batch ending right after <sep>, ready for ``model.generate``."""
    return collate([build(ex, tok, patch, with_answer=False)], tok, device)


def answer_batch(ex: Example, answers: list[str], tok: ByteTokenizer, patch: int, device=None) -> dict:
    """One row per candidate answer, for scoring options by log-likelihood."""
    rows = []
    for a in answers:
        e = Example(ex.images, ex.prompt, ex.question, a, ex.task, ex.meta, False)
        rows.append(build(e, tok, patch))
    return collate(rows, tok, device)


def build_with_answer_ids(ex: Example, answer_ids: list[int], tok: ByteTokenizer, patch: int) -> dict:
    """Like ``build``, but with the answer given as exact token ids (e.g. sampled ones), so the
    log-probability is computed for precisely the tokens that were generated."""
    b = build(ex, tok, patch, with_answer=False)
    b["ids"] = b["ids"] + list(answer_ids)
    b["labels"] = b["labels"] + list(answer_ids)
    b["block"] = b["block"] + [0] * len(answer_ids)
    b["pidx"] = b["pidx"] + [-1] * len(answer_ids)
    b["mim"] = False
    return b
