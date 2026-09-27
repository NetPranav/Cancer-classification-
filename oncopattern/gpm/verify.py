"""Verifier: grade a free-text answer against checkable facts (the model's "unit tests").

Rewards lie in [0, 1]:

* choice answers (class, yes/no): 1 if they match after normalisation, else 0
* boxes: intersection-over-union with the true box ("none" is right only when there is none)
* numbers: exp(-|pred - true| / tolerance)
* ``<abstain>``: a fixed ``abstain_reward`` r_a, whatever the truth

The abstention payment turns the reward into a decision rule. If the model
believes it is right with probability p, answering earns p on average and
abstaining earns r_a, so a reward-maximising policy answers exactly when
p > r_a. Choosing r_a = 1 - (target accuracy on answered cases) makes the
training objective match the certified-abstention layer used at deployment.
"""
from __future__ import annotations

import math
import re

from oncopattern.gpm.tokenizer import ByteTokenizer


def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9 ]", "", s.lower().replace("_", " ")).strip()


def box_iou(a, b) -> float:
    iy = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    ix = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = iy * ix
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def _first_number(s: str):
    m = re.search(r"-?\d+(?:\.\d+)?", s)
    return float(m.group()) if m else None


def verify(meta: dict, output: str, tok: ByteTokenizer, abstain_reward: float = 0.3) -> dict:
    task = meta.get("task")
    if "<abstain>" in output:
        return {"reward": abstain_reward, "abstained": True, "correct": None}
    if task in ("classify", "normal", "lesion", "compare", "present"):
        truth = _norm(meta["answer"])
        got = _norm(output.split(".")[0])
        ok = got == truth
        return {"reward": float(ok), "abstained": False, "correct": ok}
    if task in ("locate", "describe"):
        truth, pred = meta.get("box"), tok.parse_box(output)
        pattern_ok = True
        if task == "describe":  # the answer must also name the right pattern first
            pattern_ok = _norm(output).startswith(_norm(meta["label"]))
        if truth is None:  # nothing to find: must say so ("none" / the normal description), with no box
            ok = pattern_ok and pred is None and (task == "describe" or "none" in output.lower())
            return {"reward": float(ok), "abstained": False, "correct": ok}
        iou = box_iou(pred, truth) * pattern_ok if pred is not None else 0.0
        return {"reward": iou, "abstained": False, "correct": iou >= 0.3, "iou": iou}
    if task in ("measure", "count"):
        v = _first_number(output)
        if v is None:
            return {"reward": 0.0, "abstained": False, "correct": False}
        err = abs(v - float(meta["value"]))
        return {"reward": math.exp(-err / float(meta["tolerance"])), "abstained": False,
                "correct": err <= float(meta["tolerance"]), "abs_error": err}
    return {"reward": 0.0, "abstained": False, "correct": None}
