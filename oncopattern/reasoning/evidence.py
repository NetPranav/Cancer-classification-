"""Evidence ledger in decibans (Turing & Good's weight of evidence, from wartime cryptanalysis).

At Bletchley Park, Turing and I.J. Good scored competing hypotheses by adding
up "weights of evidence", each clue contributing

    WoE(H1 : H2 | e) = 10 * log10( P(e | H1) / P(e | H2) )    [decibans]

which they described as roughly the smallest change in belief a person can
notice. Because the weights add, the final log-odds break down exactly into
per-clue contributions:

    log-odds(H1 : H2 | e_1..e_J) = prior log-odds + sum_j WoE_j        (naive-Bayes form)

We apply this to the named concepts. For the winning hypothesis H1 and its
strongest rival H2, every concept is listed as

* **supporting** (WoE >= +threshold),
* **opposing**, i.e. counter-evidence that was weighed and outweighed (WoE <= -threshold),
* **uninformative**, i.e. considered and set aside (|WoE| < threshold),

so the output shows every fact it weighed, including the ones it discounted.
Class-conditional densities are 1D Gaussians on asinh-compressed concept
peaks, with Bayesian shrinkage of means and variances toward pooled
estimates, so the ledger stays stable with only a few labelled cases.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

_LN10 = np.log(10.0)


@dataclass
class EvidenceItem:
    concept: str
    value: float  # peak deviation, in multiples of the largest deviation typical of normal images
    woe_db: float  # weight of evidence for H1 over H2, decibans
    verdict: str  # supporting | opposing | uninformative
    description: str = ""

    def to_dict(self):
        return asdict(self)


def _t(x):
    return np.arcsinh(np.asarray(x, np.float64))


class ConceptEvidenceModel:
    def __init__(self, concepts: list[str], shrink: float = 3.0, threshold_db: float = 3.0):
        self.concepts = list(concepts)
        self.shrink = shrink
        self.threshold_db = threshold_db

    def fit(self, X: np.ndarray, y: np.ndarray, n_classes: int) -> "ConceptEvidenceModel":
        X = _t(X)
        y = np.asarray(y)
        self.n_classes = n_classes
        mu_all, var_all = X.mean(0), X.var(0) + 1e-3
        mus, vars_ = [], []
        for k in range(n_classes):
            xk = X[y == k]
            nk = len(xk)
            m = xk.mean(0) if nk else mu_all
            v = xk.var(0) if nk > 1 else var_all
            a = nk / (nk + self.shrink)
            mus.append(a * m + (1 - a) * mu_all)
            vars_.append((nk * v + self.shrink * var_all) / (nk + self.shrink) + 1e-3)
        self.mu, self.var = np.stack(mus), np.stack(vars_)
        counts = np.array([(y == k).sum() for k in range(n_classes)])
        self.log_prior = np.log((counts + 1) / (counts.sum() + n_classes))
        return self

    def loglik(self, X: np.ndarray) -> np.ndarray:
        """Per-concept log-likelihoods, shape (N, K, J)."""
        X = _t(np.atleast_2d(X))[:, None, :]
        return -0.5 * (np.log(2 * np.pi * self.var)[None] + (X - self.mu[None]) ** 2 / self.var[None])

    def logits(self, X: np.ndarray) -> np.ndarray:
        return self.loglik(X).sum(-1) + self.log_prior

    def explain(self, x: np.ndarray, h1: int, h2: int, descriptions: dict[str, str] | None = None) -> list[EvidenceItem]:
        ll = self.loglik(x)[0]
        woe = 10 * (ll[h1] - ll[h2]) / _LN10
        items = []
        for j, c in enumerate(self.concepts):
            w = float(woe[j])
            verdict = "supporting" if w >= self.threshold_db else "opposing" if w <= -self.threshold_db else "uninformative"
            items.append(EvidenceItem(c, float(np.asarray(x).ravel()[j]), w, verdict, (descriptions or {}).get(c, "")))
        return sorted(items, key=lambda e: -abs(e.woe_db))

    def prior_db(self, h1: int, h2: int) -> float:
        return float(10 * (self.log_prior[h1] - self.log_prior[h2]) / _LN10)
