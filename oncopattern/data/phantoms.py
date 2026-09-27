"""Procedural phantoms with exact ground truth.

Real cohorts (TCIA, IDC, TCGA, Camelyon, ...) need registration or data-use
agreements and cannot ship with the repository. Phantoms let every phase of
the pipeline run and be tested end to end, with *known* answers, including
anomalies at single-cell scale:

* ``tissue_patch`` renders an H&E-like field of nuclei on a jittered hexagonal
  lattice whose orientation follows a smooth tissue "flow" field. Anomalies are
  the classic cytological signs of malignancy, confined to 1-4 cells:

  - ``architectural_disorder``: nuclei rotated ~90 degrees against the local
    flow (loss of polarity / a "misaligned cell"),
  - ``nuclear_atypia``: enlarged, hyperchromatic (darker) nuclei,
  - ``hypercellularity``: extra nuclei crowded into the lattice.

* ``mri_slice`` renders an axial head-MRI-like slice; an optional mass lesion
  with irregular margin and oedema halo can be grown across timepoints for the
  longitudinal phase.

Phantoms are a test harness, not evidence of clinical performance.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy import ndimage as ndi

TISSUE_PATTERNS = ("normal", "architectural_disorder", "nuclear_atypia", "hypercellularity")
MRI_PATTERNS = ("normal", "mass_lesion")


@dataclass
class Phantom:
    image: np.ndarray  # float32, HxW, [0, 1]
    mask: np.ndarray  # bool, HxW, ground-truth abnormal pixels
    label: str
    meta: dict = field(default_factory=dict)


def _render_nuclei(size, centres, a, b, ang, dark):
    """Flat-topped elliptical nuclei; overlapping nuclei take the darker value."""
    nuc = np.zeros((size, size), np.float32)
    ids = np.full((size, size), -1, np.int32)
    for i, ((cx, cy), ai, bi, t, d) in enumerate(zip(centres, a, b, ang, dark)):
        r = int(np.ceil(ai * 1.6)) + 1
        x0, x1 = max(int(cx) - r, 0), min(int(cx) + r + 1, size)
        y0, y1 = max(int(cy) - r, 0), min(int(cy) + r + 1, size)
        if x0 >= x1 or y0 >= y1:
            continue
        yy, xx = np.mgrid[y0:y1, x0:x1].astype(np.float32)
        dx, dy = xx - cx, yy - cy
        c, s = np.cos(t), np.sin(t)
        u, v = c * dx + s * dy, -s * dx + c * dy
        d2 = (u / ai) ** 2 + (v / bi) ** 2
        val = d * np.exp(-(d2 ** 2))
        win = nuc[y0:y1, x0:x1]
        upd = val > win
        win[upd] = val[upd]
        ids[y0:y1, x0:x1][upd & (d2 < 1.3)] = i
    return nuc, ids


def tissue_patch(rng: np.random.Generator, pattern: str = "normal", size: int = 64,
                 spacing: float = 8.0, noise: float = 0.02) -> Phantom:
    if pattern not in TISSUE_PATTERNS:
        raise ValueError(f"unknown tissue pattern {pattern!r}")
    theta0 = rng.uniform(0, np.pi)
    kx, ky = rng.normal(0, 0.04, size=2)
    amp = rng.uniform(0.2, 0.6)

    row_h = spacing * np.sqrt(3) / 2
    pts = []
    for r in range(-1, int(size / row_h) + 2):
        off = (r % 2) * spacing / 2
        for c in range(-1, int(size / spacing) + 2):
            pts.append((c * spacing + off, r * row_h))
    centres = np.asarray(pts, np.float32) + rng.uniform(0, spacing, size=2).astype(np.float32)
    # rotate the whole lattice so it is not axis aligned
    rot = rng.uniform(0, np.pi)
    ctr = np.array([size / 2, size / 2], np.float32)
    R = np.array([[np.cos(rot), -np.sin(rot)], [np.sin(rot), np.cos(rot)]], np.float32)
    centres = (centres - ctr) @ R.T + ctr + rng.normal(0, 0.5, size=centres.shape).astype(np.float32)
    keep = np.all((centres > -4) & (centres < size + 4), axis=1)
    centres = centres[keep]
    n = len(centres)

    a = 2.6 * rng.normal(1, 0.05, n)
    b = 1.35 * rng.normal(1, 0.05, n)
    dark = np.clip(0.55 * rng.normal(1, 0.04, n), 0, 0.95)
    flow = theta0 + amp * np.sin(kx * centres[:, 0] + ky * centres[:, 1])
    ang = flow + rng.normal(0, 0.1, n)

    affected = np.zeros(n, bool)
    site = rng.uniform(14, size - 14, size=2).astype(np.float32)
    meta = {"site": site.tolist(), "modality": "histology", "organ": "breast"}
    if pattern != "normal":
        d = np.linalg.norm(centres - site, axis=1)
        order = np.argsort(d)
        if pattern == "architectural_disorder":
            k = int(rng.integers(1, 4))
            idx = order[:k]
            ang[idx] = flow[idx] + np.pi / 2 + rng.normal(0, 0.15, k)
        elif pattern == "nuclear_atypia":
            k = int(rng.integers(1, 3))
            idx = order[:k]
            scale = rng.uniform(1.45, 1.8, k)
            a[idx] *= scale
            b[idx] *= scale * rng.uniform(1.0, 1.25, k)
            dark[idx] = np.clip(dark[idx] * rng.uniform(1.25, 1.45, k), 0, 0.95)
        else:  # hypercellularity
            k = int(rng.integers(3, 5))
            extra = site + rng.normal(0, spacing * 0.45, size=(k, 2)).astype(np.float32)
            centres = np.vstack([centres, extra])
            a = np.concatenate([a, 2.6 * rng.normal(1, 0.05, k)])
            b = np.concatenate([b, 1.35 * rng.normal(1, 0.05, k)])
            dark = np.concatenate([dark, np.clip(0.55 * rng.normal(1, 0.04, k), 0, 0.95)])
            ang = np.concatenate([ang, theta0 + amp * np.sin(kx * extra[:, 0] + ky * extra[:, 1])
                                  + rng.normal(0, 0.1, k)])
            affected = np.concatenate([affected, np.zeros(k, bool)])
            idx = np.arange(n, n + k)
        affected[idx] = True
        meta["n_affected_cells"] = int(len(idx))

    nuc, ids = _render_nuclei(size, centres, a, b, ang, dark)
    yy, xx = np.mgrid[0:size, 0:size].astype(np.float32)
    bg = 0.9 + 0.03 * np.sin(xx * rng.uniform(0.05, 0.15) + rng.uniform(0, 6)) \
        * np.cos(yy * rng.uniform(0.05, 0.15))
    img = bg * (1 - nuc) + rng.normal(0, noise, (size, size))
    img = np.clip(img, 0, 1).astype(np.float32)

    mask = np.isin(ids, np.flatnonzero(affected)) if affected.any() else np.zeros_like(ids, bool)
    mask = ndi.binary_dilation(mask, iterations=1)
    return Phantom(img, mask, pattern, meta)


def mri_slice(rng: np.random.Generator, size: int = 64, lesion_radius: float = 0.0,
              lesion_center=None, anatomy_seed: int | None = None, noise: float = 0.025) -> Phantom:
    """Axial T2-FLAIR-like head slice. ``anatomy_seed`` fixes the anatomy so a
    lesion can be grown on the *same* head across timepoints."""
    arng = np.random.default_rng(anatomy_seed) if anatomy_seed is not None else rng
    yy, xx = np.mgrid[0:size, 0:size].astype(np.float32)
    c = size / 2 + arng.normal(0, 0.5, 2)
    ry, rx = size * arng.uniform(0.40, 0.44), size * arng.uniform(0.33, 0.37)
    r = np.sqrt(((xx - c[0]) / rx) ** 2 + ((yy - c[1]) / ry) ** 2)
    img = np.zeros((size, size), np.float32)
    img[r < 1.0] = 0.85  # scalp / skull ring
    img[r < 0.92] = 0.45  # grey matter
    img[r < 0.7] = 0.6  # white matter
    for side in (-1, 1):  # lateral ventricles
        vx, vy = c[0] + side * size * 0.07, c[1] - size * 0.02
        rv = np.sqrt(((xx - vx) / (size * 0.04)) ** 2 + ((yy - vy) / (size * 0.12)) ** 2)
        img[rv < 1] = 0.15
    tex = ndi.gaussian_filter(arng.normal(0, 1, (size, size)), 2.0)
    img = img + 0.03 * tex * (r < 0.92)

    mask = np.zeros((size, size), bool)
    label = "normal"
    meta = {"modality": "MRI", "organ": "brain", "lesion_radius": float(lesion_radius)}
    if lesion_radius > 0:
        if lesion_center is None:
            ang = arng.uniform(0, 2 * np.pi)
            lesion_center = (c[0] + 0.45 * rx * np.cos(ang), c[1] + 0.45 * ry * np.sin(ang))
        lx, ly = lesion_center
        phi = np.arctan2(yy - ly, xx - lx)
        rr = np.hypot(xx - lx, yy - ly)
        harmonics = sum(arng.uniform(0.05, 0.15) * np.cos(k * phi + arng.uniform(0, 6)) for k in (2, 3, 5))
        edge = lesion_radius * (1 + harmonics)
        oedema = rr < edge * 1.8
        core = rr < edge
        img[oedema & (r < 0.92)] = np.maximum(img[oedema & (r < 0.92)], 0.75)
        img[core] = 0.95
        mask = oedema & (r < 0.92)
        label = "mass_lesion"
        meta["lesion_center"] = [float(lx), float(ly)]
    img = np.abs(img + rng.normal(0, noise, (size, size)) + 1j * rng.normal(0, noise, (size, size)))
    return Phantom(np.clip(img, 0, 1).astype(np.float32), mask, label, meta)


def make_tissue_set(rng: np.random.Generator, per_class: int, patterns=TISSUE_PATTERNS, size: int = 64):
    out = [tissue_patch(rng, p, size=size) for p in patterns for _ in range(per_class)]
    order = rng.permutation(len(out))
    return [out[i] for i in order]


def longitudinal_mri(rng: np.random.Generator, radii=(0.0, 0.0, 1.5, 2.5, 3.5), size: int = 64):
    """Same head scanned at several timepoints while a lesion appears and grows."""
    seed = int(rng.integers(0, 2 ** 31))
    base = np.random.default_rng(seed)
    ang = base.uniform(0, 2 * np.pi)
    centre = (size / 2 + 0.45 * size * 0.35 * np.cos(ang), size / 2 + 0.45 * size * 0.42 * np.sin(ang))
    return [mri_slice(rng, size, r, centre if r > 0 else None, anatomy_seed=seed) for r in radii]
