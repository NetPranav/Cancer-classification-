"""Calibration, conformal prediction sets and risk-controlled abstention.

This module is how we pursue "100% correct" without making a false promise.

1. **Temperature scaling** (Guo et al. 2017): rescale logits by a single T,
   fitted by minimum NLL, so that "80%" means right about 80% of the time.

2. **Split conformal prediction** (Vovk; Angelopoulos & Bates 2021): with
   nonconformity score s = 1 - p_y on n calibration cases and

       q_hat = the ceil((n+1)(1-alpha))/n empirical quantile of s,
       C(x) = {k : 1 - p_k(x) <= q_hat},

   we get P(y in C(x)) >= 1 - alpha for exchangeable data, with no assumption
   about the model. A set with several labels is an honest "it is one of these".

3. **Selective risk control** (Learn-then-Test, Angelopoulos et al. 2021): the
   model answers only when max_k p_k >= lambda and abstains otherwise, i.e.
   refers the case to a specialist. Candidate lambdas are tested from strictest
   to loosest (fixed-sequence testing) with the exact binomial test of

       H0: P(error | answered) > alpha,
       p-value = P(Bin(n_answered, alpha) <= errors).

   Thresholds that answer fewer than n_min cases (below) cannot be rejected
   whatever happens, so they are skipped as untestable (Tarone 1990). After
   that, testing stops at the first lambda that fails. With probability about
   >= 1 - delta, the error rate on answered cases is <= alpha. The guarantee is
   approximate because the answered count is itself random. This makes the path to
   "100%" concrete: push alpha toward 0 and more cases get referred, never
   silently misdiagnosed. Certifying alpha needs at least

       n_min = ceil( log(delta) / log(1 - alpha) )

   consecutive error-free answered cases, e.g. 230 for 1% at delta=0.1 and
   2,302 for 0.1%. That number is the data requirement for any "near-100%" claim.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy import stats


def softmax(z: np.ndarray, T: float = 1.0) -> np.ndarray:
    z = np.asarray(z, np.float64) / T
    z = z - z.max(-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(-1, keepdims=True)


def nll(logits: np.ndarray, y: np.ndarray, T: float = 1.0) -> float:
    p = softmax(logits, T)
    return float(-np.mean(np.log(p[np.arange(len(y)), y] + 1e-12)))


def fit_temperature(logits: np.ndarray, y: np.ndarray) -> float:
    grid = np.exp(np.linspace(np.log(0.05), np.log(20), 200))
    losses = [nll(logits, y, t) for t in grid]
    return float(grid[int(np.argmin(losses))])


def min_samples_for_guarantee(alpha: float, delta: float) -> int:
    """Smallest n such that n error-free answers certify risk <= alpha at confidence 1 - delta."""
    return int(math.ceil(math.log(delta) / math.log(1 - alpha)))


class ConformalClassifier:
    def __init__(self, alpha: float = 0.1):
        self.alpha = alpha
        self.qhat = 1.0

    def fit(self, probs: np.ndarray, y: np.ndarray) -> "ConformalClassifier":
        n = len(y)
        scores = 1 - probs[np.arange(n), y]
        level = min(1.0, math.ceil((n + 1) * (1 - self.alpha)) / n)
        self.qhat = float(np.quantile(scores, level, method="higher"))
        return self

    def predict_set(self, probs: np.ndarray) -> np.ndarray:
        s = np.atleast_2d(probs) >= 1 - self.qhat - 1e-12
        top = np.atleast_2d(probs).argmax(-1)
        s[np.arange(len(top)), top] = True  # never return an empty set
        return s


@dataclass
class SelectiveResult:
    threshold: float
    coverage: float  # fraction of calibration cases answered
    errors: int
    answered: int
    certified: bool


class SelectiveRiskController:
    def __init__(self, target_risk: float = 0.05, delta: float = 0.1):
        self.target_risk = target_risk
        self.delta = delta
        self.result = SelectiveResult(threshold=1.01, coverage=0.0, errors=0, answered=0, certified=False)

    def fit(self, probs: np.ndarray, y: np.ndarray) -> "SelectiveRiskController":
        conf = probs.max(-1)
        wrong = probs.argmax(-1) != y
        for lam in np.unique(conf)[::-1]:  # strictest first
            acc = conf >= lam
            n, e = int(acc.sum()), int(wrong[acc].sum())
            pval = stats.binom.cdf(e, n, self.target_risk)
            if pval > self.delta:
                if n >= min_samples_for_guarantee(self.target_risk, self.delta):
                    break  # a real failure of the test
                continue  # too few answered cases to decide yet; keep extending
            self.result = SelectiveResult(float(lam), n / len(y), e, n, True)
        return self

    def accept(self, probs: np.ndarray) -> np.ndarray:
        return np.atleast_2d(probs).max(-1) >= self.result.threshold
