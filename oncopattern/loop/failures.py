"""Failure ledger: learn from every test the model fails (from test-driven engineering).

Mature software teams turn every fixed bug into a permanent regression test.
We do the same for the model:

1. **Record** every error, abstention and low-margin answer with its metadata,
   embedding and concept profile.
2. **Find failing slices**: error rate per metadata value (modality, organ,
   site, scanner, true label), each with a Wilson 95% interval. A slice is
   flagged when its lower bound is above the overall error rate, which avoids
   chasing noise in tiny slices.
3. **Cluster failures** in embedding space (k-means++), so failure *modes*
   ("large dark nuclei called hypercellularity") show up even without metadata.
4. **Estimate how much data is needed**: fit the learning-curve power law
   err(n) = a n^-b + c (Hestness et al. 2017) and solve for the n that reaches
   the target. c is the irreducible floor: if c > target, more of the *same*
   data will not help and a new modality or label is required.
5. **Recommend acquisitions** from the dataset catalogue that match each
   failing slice's organ and modality.
6. **Regression suite**: every failure case is saved and must pass before a
   new model version ships.
"""
from __future__ import annotations

import json
import math
from collections import defaultdict
from dataclasses import asdict, dataclass, field

import numpy as np

from oncopattern.data.catalogue import MODALITY_ALIASES, search


@dataclass
class CaseRecord:
    case_id: str
    true_label: str
    predicted: str
    confidence: float
    answered: bool
    in_set: bool  # true label inside the conformal prediction set
    metadata: dict = field(default_factory=dict)
    concepts: dict = field(default_factory=dict)
    embedding: list = field(default_factory=list)

    @property
    def kind(self) -> str:
        if not self.answered:
            return "abstained"
        return "correct" if self.predicted == self.true_label else "error"


def wilson_interval(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return 0.0, 1.0
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return max(0.0, c - h), min(1.0, c + h)


def fit_learning_curve(ns, errs) -> tuple[float, float, float]:
    """Fit err = a n^-b + c by profiling c on a grid and regressing log(err - c) on log n."""
    ns, errs = np.asarray(ns, float), np.asarray(errs, float)
    best = (np.inf, 0.0, 0.0, 0.0)
    for c in np.linspace(0, max(errs.min() - 1e-6, 0), 50):
        r = errs - c
        if np.any(r <= 0):
            continue
        slope, icpt = np.polyfit(np.log(ns), np.log(r), 1)
        pred = np.exp(icpt) * ns ** slope + c
        sse = float(((pred - errs) ** 2).sum())
        if sse < best[0]:
            best = (sse, float(np.exp(icpt)), float(-slope), float(c))
    return best[1], best[2], best[3]


def samples_needed(a: float, b: float, c: float, target: float) -> float | None:
    """Solve a n^-b + c = target. None if the floor c makes the target unreachable."""
    if target <= c or a <= 0 or b <= 0:
        return None
    return float(((target - c) / a) ** (-1 / b))


def _kmeans(X: np.ndarray, k: int, iters: int = 50, seed: int = 0):
    rng = np.random.default_rng(seed)
    centres = [X[rng.integers(len(X))]]
    for _ in range(1, k):  # k-means++ seeding
        d = np.min([((X - c) ** 2).sum(1) for c in centres], axis=0)
        centres.append(X[rng.choice(len(X), p=d / d.sum())] if d.sum() > 0 else X[rng.integers(len(X))])
    C = np.stack(centres)
    for _ in range(iters):
        lab = ((X[:, None] - C[None]) ** 2).sum(-1).argmin(1)
        C = np.stack([X[lab == j].mean(0) if (lab == j).any() else C[j] for j in range(k)])
    return lab


class FailureLedger:
    def __init__(self):
        self.records: list[CaseRecord] = []

    def add(self, rec: CaseRecord):
        self.records.append(rec)

    @property
    def failures(self) -> list[CaseRecord]:
        return [r for r in self.records if r.kind != "correct"]

    def summary(self) -> dict:
        n = len(self.records)
        ans = [r for r in self.records if r.answered]
        err = [r for r in ans if r.kind == "error"]
        return {
            "cases": n,
            "answered": len(ans),
            "abstained": n - len(ans),
            "errors_on_answered": len(err),
            "error_rate_on_answered": len(err) / max(len(ans), 1),
            "error_rate_on_answered_95ci": wilson_interval(len(err), len(ans)),
            "set_coverage": float(np.mean([r.in_set for r in self.records])) if n else 0.0,
        }

    def slices(self, keys=("true_label", "modality", "organ", "site", "scanner")) -> list[dict]:
        """Rate of *failures* (errors + abstentions) per slice; flag significant excess."""
        n = len(self.records)
        if not n:
            return []
        overall = len(self.failures) / n
        out = []
        for key in keys:
            groups = defaultdict(list)
            for r in self.records:
                v = r.true_label if key == "true_label" else r.metadata.get(key)
                if v is not None:
                    groups[str(v)].append(r)
            for v, rs in groups.items():
                f = sum(r.kind != "correct" for r in rs)
                lo, hi = wilson_interval(f, len(rs))
                out.append({"key": key, "value": v, "cases": len(rs), "failures": f,
                            "failure_rate": f / len(rs), "ci95": [lo, hi], "flagged": lo > overall})
        return sorted(out, key=lambda s: -s["failure_rate"])

    def clusters(self, k: int = 3) -> list[dict]:
        fails = [r for r in self.failures if r.embedding]
        if len(fails) < 2:
            return []
        k = min(k, len(fails))
        X = np.asarray([r.embedding for r in fails], float)
        X = (X - X.mean(0)) / (X.std(0) + 1e-9)
        lab = _kmeans(X, k)
        out = []
        for j in range(k):
            members = [fails[i] for i in np.flatnonzero(lab == j)]
            if not members:
                continue
            confusions = defaultdict(int)
            for m in members:
                confusions[f"{m.true_label} -> {m.predicted if m.answered else 'abstain'}"] += 1
            concept_means = {}
            if members[0].concepts:
                for c in members[0].concepts:
                    concept_means[c] = float(np.mean([m.concepts[c] for m in members]))
            out.append({"cluster": j, "size": len(members), "patterns": dict(confusions),
                        "mean_concepts": concept_means, "cases": [m.case_id for m in members]})
        return sorted(out, key=lambda c: -c["size"])

    def recommend(self, top: int = 3) -> list[dict]:
        recs = []
        for s in self.slices(("organ", "modality", "true_label")):
            if not s["flagged"] and s["failure_rate"] < 0.25:
                continue
            example = next((r for r in self.failures
                            if (r.true_label if s["key"] == "true_label" else r.metadata.get(s["key"])) == s["value"]), None)
            if example is None:
                continue
            organ = example.metadata.get("organ")
            mod = str(example.metadata.get("modality", "")).lower()
            cands = []
            for m in MODALITY_ALIASES.get(mod, (mod,)):
                cands += [d for d in search(organ=organ, modality=m) if d not in cands]
            if not cands:
                cands = search(organ=organ)
            cands.sort(key=lambda d: (not d.longitudinal, "localisation" not in d.tags))
            recs.append({"slice": f"{s['key']}={s['value']}", "failure_rate": s["failure_rate"],
                         "reason": f"{s['failures']}/{s['cases']} failed (95% CI {s['ci95'][0]:.2f}-{s['ci95'][1]:.2f})",
                         "acquire": [d.name for d in cands[:top]]})
        return recs

    def save_regression_suite(self, path: str):
        with open(path, "w") as fh:
            json.dump([{k: v for k, v in asdict(r).items() if k != "embedding"} | {"kind": r.kind}
                       for r in self.failures], fh, indent=2)

    def report(self) -> dict:
        return {"summary": self.summary(), "slices": self.slices(), "clusters": self.clusters(),
                "recommendations": self.recommend()}
