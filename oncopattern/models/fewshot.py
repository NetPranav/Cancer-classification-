"""Few-shot classification with shrinkage prototypes (Stein's paradox, from statistics).

With n_k support examples, the class mean mu_k is a noisy estimate. James &
Stein (1961) proved that shrinking several means toward a common centre
always lowers total squared error in 3+ dimensions. As a Bayesian posterior
mean under the prior mu_k ~ N(mu_0, sigma^2 / tau) this is

    mu_k_hat = mu_0 + n_k / (n_k + tau) * (mean_k - mu_0)

so rare classes are pulled toward the grand mean, and the pull fades as data
arrives. The same holds for the diagonal variances:

    v_hat = (n v_within + nu v_total) / (n + nu)

Logits are Gaussian log-likelihoods with the shared diagonal variance:

    logit_k(z) = -1/2 sum_j (z_j - mu_k_hat_j)^2 / v_hat_j + log pi_k

The nearest support cases are returned too, so each prediction can point to
similar, already-labelled cases.
"""
from __future__ import annotations

import numpy as np


class ShrinkagePrototypes:
    def __init__(self, tau: float = 2.0, nu: float = 4.0):
        self.tau, self.nu = tau, nu

    def fit(self, Z: np.ndarray, y: np.ndarray, n_classes: int) -> "ShrinkagePrototypes":
        Z = np.asarray(Z, np.float64)
        y = np.asarray(y)
        self.n_classes = n_classes
        mu0 = Z.mean(0)
        protos, resid = [], []
        counts = np.array([(y == k).sum() for k in range(n_classes)])
        for k in range(n_classes):
            zk = Z[y == k]
            if len(zk) == 0:
                protos.append(mu0)
                continue
            lam = len(zk) / (len(zk) + self.tau)
            protos.append(mu0 + lam * (zk.mean(0) - mu0))
            resid.append(zk - zk.mean(0))
        v_within = np.concatenate(resid).var(0) if resid else Z.var(0)
        v_total = Z.var(0) + 1e-6
        n = len(Z)
        self.var = (n * v_within + self.nu * v_total) / (n + self.nu) + 1e-6
        self.protos = np.stack(protos)
        self.log_prior = np.log((counts + 1) / (counts.sum() + n_classes))
        self.support_Z, self.support_y = Z, y
        return self

    def logits(self, Z: np.ndarray) -> np.ndarray:
        Z = np.atleast_2d(np.asarray(Z, np.float64))
        d2 = (((Z[:, None, :] - self.protos[None]) ** 2) / self.var).sum(-1)
        return -0.5 * d2 / Z.shape[1] + self.log_prior

    def nearest_support(self, z: np.ndarray, k: int = 3) -> list[tuple[int, int, float]]:
        d = np.sqrt((((self.support_Z - z) ** 2) / self.var).mean(1))
        idx = np.argsort(d)[:k]
        return [(int(i), int(self.support_y[i]), float(d[i])) for i in idx]
