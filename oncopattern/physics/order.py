"""Nematic order parameter for tissue architecture (borrowed from liquid-crystal physics).

Epithelial nuclei, like the rod-shaped molecules of a nematic liquid crystal,
line up along a local director field. Malignant transformation often starts
with loss of polarity: a few nuclei stop aligning with their neighbours. In
condensed-matter terms this is a *topological defect* in the director field,
and there are standard tools to measure it cheaply.

For an image I, the structure tensor at scale (sigma_d, sigma_i) is

    J = G_{sigma_i} * [[I_x^2, I_x I_y], [I_x I_y, I_y^2]],   I_x = d/dx (G_{sigma_d} * I)

with eigenvalues l1 >= l2. We work with the doubled-angle vector
(nematic Q-tensor components), which removes the 180-degree ambiguity of an
orientation:

    q = ((Jxx - Jyy), 2 Jxy) / (Jxx + Jyy)      |q| = coherence = (l1 - l2) / (l1 + l2)

Averaging q over a larger context window gives the context director Q. From
that:

    order S = |<q>_ctx| / <|q|>_ctx                         (1 = perfectly aligned, 0 = isotropic)
    misalignment m = |q| * (1 - cos(2 (phi - Phi))) / 2     (0 = aligned, 1 = orthogonal)

``m`` is large exactly where a strongly oriented structure (a nucleus) points
across the local tissue flow. It needs one Gaussian filter per term, so it
costs O(HW) with no learning, which keeps it cheap when data is scarce.
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage as ndi

_EPS = 1e-8


def structure_tensor(img: np.ndarray, sigma_d: float = 1.0, sigma_i: float = 1.5):
    img = img.astype(np.float64)
    ix = ndi.gaussian_filter(img, sigma_d, order=(0, 1))
    iy = ndi.gaussian_filter(img, sigma_d, order=(1, 0))
    jxx = ndi.gaussian_filter(ix * ix, sigma_i)
    jxy = ndi.gaussian_filter(ix * iy, sigma_i)
    jyy = ndi.gaussian_filter(iy * iy, sigma_i)
    return jxx, jxy, jyy


def orientation_fields(img: np.ndarray, sigma_d: float = 1.0, sigma_i: float = 1.5,
                       sigma_ctx: float = 6.0) -> dict[str, np.ndarray]:
    """Return per-pixel ``orientation`` (radians), ``coherence``, ``order`` and ``misalignment``."""
    jxx, jxy, jyy = structure_tensor(img, sigma_d, sigma_i)
    tr = jxx + jyy + _EPS
    qx, qy = (jxx - jyy) / tr, 2 * jxy / tr
    coh = np.hypot(qx, qy)
    Qx, Qy = ndi.gaussian_filter(qx, sigma_ctx), ndi.gaussian_filter(qy, sigma_ctx)
    Qn = np.hypot(Qx, Qy)
    order = Qn / (ndi.gaussian_filter(coh, sigma_ctx) + _EPS)
    cos2 = (qx * Qx + qy * Qy) / (coh * Qn + _EPS)
    mis = coh * 0.5 * (1 - cos2)
    return {
        "orientation": 0.5 * np.arctan2(qy, qx),
        "coherence": coh.astype(np.float32),
        "order": np.clip(order, 0, 1).astype(np.float32),
        "misalignment": mis.astype(np.float32),
    }
