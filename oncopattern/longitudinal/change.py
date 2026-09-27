"""Longitudinal change detection: Kalman trend + CUSUM (from control engineering and
industrial quality control), plus tumour volume doubling time (from oncology).

A single scan shows a state; a series of scans shows a *trajectory*, and the
trajectory often carries the diagnosis (a stable nodule is benign, a growing
one is suspicious). Given a per-timepoint measurement y_t, e.g. the
normality-deviation score or the lesion area:

* **Local-linear-trend Kalman filter** (Kalman 1960; used in aircraft
  navigation):

      level_t = level_{t-1} + slope_{t-1} + w1,   slope_t = slope_{t-1} + w2,
      y_t = level_t + v

  gives a growth rate with an uncertainty, handles irregular gaps between
  scans, and never over-reacts to one noisy scan.

* **One-sided CUSUM** (Page 1954; factory quality control):

      S_t = max(0, S_{t-1} + (y_t - mu_0)/sigma_0 - k),   alarm when S_t > h

  catches small persistent drifts faster than any single-scan threshold.

* **Doubling time** under exponential growth V(t) = V_0 2^{t/DT}:

      DT = ln 2 / b,   b = slope of ln V against t
"""
from __future__ import annotations

import numpy as np


def kalman_trend(times, y, obs_var: float = 0.25, q_level: float = 1e-3, q_slope: float = 1e-4):
    times = np.asarray(times, float)
    y = np.asarray(y, float)
    x = np.array([y[0], 0.0])
    P = np.diag([obs_var, 1.0])
    H = np.array([[1.0, 0.0]])
    out = []
    for i, (t, yt) in enumerate(zip(times, y)):
        if i:
            dt = t - times[i - 1]
            F = np.array([[1.0, dt], [0.0, 1.0]])
            Q = np.diag([q_level * dt, q_slope * dt])
            x, P = F @ x, F @ P @ F.T + Q
        S = H @ P @ H.T + obs_var
        K = P @ H.T / S
        x = x + (K * (yt - H @ x)).ravel()
        P = (np.eye(2) - K @ H) @ P
        out.append((x[0], x[1], np.sqrt(P[1, 1])))
    return np.array(out)  # columns: level, slope, slope_sd


def cusum(y, mu0: float, sigma0: float, k: float = 0.5, h: float = 4.0):
    s, path, alarm = 0.0, [], None
    for i, v in enumerate(y):
        s = max(0.0, s + (v - mu0) / sigma0 - k)
        path.append(s)
        if alarm is None and s > h:
            alarm = i
    return np.array(path), alarm


def doubling_time(times, volumes) -> float | None:
    t, v = np.asarray(times, float), np.asarray(volumes, float)
    ok = v > 0
    if ok.sum() < 2:
        return None
    b = np.polyfit(t[ok], np.log(v[ok]), 1)[0]
    return float(np.log(2) / b) if b > 1e-9 else None


def longitudinal_report(times, scores, mu0: float, sigma0: float, volumes=None) -> dict:
    """Summarise a patient's trajectory. ``mu0``/``sigma0`` describe the score on normal scans."""
    kt = kalman_trend(times, scores, obs_var=max(sigma0, 1e-3) ** 2)
    path, alarm = cusum(scores, mu0, sigma0)
    level, slope, sd = kt[-1]
    rep = {
        "times": list(map(float, times)),
        "scores": list(map(float, scores)),
        "trend_per_unit_time": float(slope),
        "trend_95ci": [float(slope - 1.96 * sd), float(slope + 1.96 * sd)],
        "trend_significant": bool(slope - 1.96 * sd > 0),
        "cusum": path.tolist(),
        "change_detected_at": None if alarm is None else float(times[alarm]),
    }
    if volumes is not None:
        rep["doubling_time"] = doubling_time(times, volumes)
    return rep
