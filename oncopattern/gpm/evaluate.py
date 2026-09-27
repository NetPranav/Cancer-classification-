"""Evaluation: every task graded by the verifier; choice tasks also scored by answer likelihood
(for calibration and abstention); surprise maps scored against true lesion masks."""
from __future__ import annotations

import numpy as np
import torch

from oncopattern.calibration.conformal import SelectiveRiskController, softmax
from oncopattern.data.phantoms import TISSUE_PATTERNS, tissue_patch
from oncopattern.gpm.data import Example, ImageItem, answer_batch, collate, build, prompt_batch
from oncopattern.gpm.model import GeneralPatternModel
from oncopattern.gpm.sources import Source, TissuePhantomSource
from oncopattern.gpm.tokenizer import ByteTokenizer
from oncopattern.gpm.verify import verify
from oncopattern.pipeline import auroc


@torch.no_grad()
def option_probs(model: GeneralPatternModel, ex: Example, tok: ByteTokenizer, device=None) -> np.ndarray:
    """P(option | image, prompt) by normalised sequence likelihood: classification without a classifier head."""
    batch = answer_batch(ex, ex.meta["options"], tok, model.cfg.patch_size, device)
    return softmax(model.sequence_logprob(batch).cpu().numpy())


@torch.no_grad()
def evaluate_tasks(model: GeneralPatternModel, tok: ByteTokenizer, sources: list[Source], n_per_task: int = 20,
                   seed: int = 1234, device=None, max_new: int = 24) -> dict:
    model.eval()
    out = {}
    for src in sources:
        for task in src.task_weights:
            rng = np.random.default_rng(seed)
            rewards, correct, probs, ys, samples = [], [], [], [], []
            for i in range(n_per_task):
                ex = src.sample(rng, task)
                text = tok.decode(model.generate(prompt_batch(ex, tok, model.cfg.patch_size, device), max_new,
                                                 eos_id=tok.eos_id))
                v = verify(ex.meta, text, tok)
                rewards.append(v["reward"])
                correct.append(bool(v["correct"]))
                if i < 3:
                    samples.append({"question": ex.question.strip(), "truth": ex.answer, "output": text})
                if "options" in ex.meta:
                    p = option_probs(model, ex, tok, device)
                    probs.append(p)
                    ys.append(ex.meta["options"].index(ex.answer))
            res = {"mean_reward": float(np.mean(rewards)), "accuracy": float(np.mean(correct)), "samples": samples}
            if probs:
                P, y = np.stack(probs), np.array(ys)
                res["accuracy_by_likelihood"] = float((P.argmax(1) == y).mean())
                res["mean_confidence"] = float(P.max(1).mean())
            out[f"{src.name}/{task}"] = res
    return out


@torch.no_grad()
def surprise_auroc(model: GeneralPatternModel, tok: ByteTokenizer, n: int = 24, seed: int = 99, size: int = 64,
                   device=None) -> dict:
    """Does patch surprise (context-prediction error) single out the abnormal cells? No labels are used."""
    model.eval()
    rng = np.random.default_rng(seed)
    p = model.cfg.patch_size
    scores, labels, img_scores, img_labels = [], [], [], []
    for i in range(n):
        pat = TISSUE_PATTERNS[i % len(TISSUE_PATTERNS)]
        ph = tissue_patch(rng, pat, size=size)
        ex = Example([ImageItem(ph.image, TissuePhantomSource.spacing_mm, "phantom_tissue")], "Study this image.",
                     mim=True)
        batch = collate([build(ex, tok, p)], tok, device)
        s = model.patch_surprise(batch).numpy()
        g = size // p
        lab = ph.mask[: g * p, : g * p].reshape(g, p, g, p).any((1, 3)).ravel()
        scores.append(s)
        labels.append(lab)
        img_scores.append(s.max())
        img_labels.append(pat != "normal")
    return {"patch_auroc": auroc(np.concatenate(scores), np.concatenate(labels)),
            "image_auroc": auroc(img_scores, img_labels)}


def selective_report(probs: np.ndarray, y: np.ndarray, target_risk: float = 0.05, delta: float = 0.1) -> dict:
    sel = SelectiveRiskController(target_risk, delta).fit(probs, y)
    return sel.result.__dict__
