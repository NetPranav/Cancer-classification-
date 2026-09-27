"""Named, human-readable pattern concepts computed as dense maps.

Every concept is a per-pixel map, so it can be *highlighted*, and its peak value
is an image-level feature, so it can be *weighed* in the evidence ledger. The
concepts mirror what pathologists and radiologists actually look for. Concepts
that do not apply to a modality (e.g. nuclear size on an MRI) are still
computed: the ledger learns they are uninformative there and lists them as
"considered and set aside", which is the behaviour we want.
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage as ndi

from oncopattern.physics.order import orientation_fields

CONCEPTS: dict[str, str] = {
    "orientation_misalignment": "nuclei oriented against the local tissue flow (loss of polarity)",
    "order_loss": "local loss of nematic order in tissue architecture",
    "nuclear_enlargement": "nucleus area larger than the normal reference",
    "hyperchromasia": "nuclei darker (denser chromatin) than the normal reference",
    "crowding": "local fraction of area occupied by nuclei above the normal reference (cellularity)",
    "focal_hyperintensity": "focal bright signal relative to its neighbourhood",
    "asymmetry": "left-right asymmetry about the image midline",
}

NUCLEUS_THRESHOLD = 0.6


def concept_maps(img: np.ndarray) -> dict[str, np.ndarray]:
    img = img.astype(np.float32)
    f = orientation_fields(img)
    dark = img < NUCLEUS_THRESHOLD
    nuc_soft = ndi.gaussian_filter(dark.astype(np.float32), 1.0)

    lab, n = ndi.label(dark)
    area_map = np.zeros_like(img)
    chrom_map = np.zeros_like(img)
    if n:
        idx = np.arange(1, n + 1)
        areas = ndi.sum(np.ones_like(img), lab, idx)
        mins = ndi.minimum(img, lab, idx)
        area_lut = np.concatenate([[0.0], np.sqrt(areas)])  # linear size, not area
        chrom_lut = np.concatenate([[0.0], 1.0 - np.asarray(mins)])
        area_map = area_lut[lab].astype(np.float32)
        chrom_map = chrom_lut[lab].astype(np.float32)
    # cellularity: local nuclear area fraction. Robust to touching nuclei merging into one blob,
    # which defeats centroid counting exactly when cells crowd together.
    density = ndi.gaussian_filter(dark.astype(np.float32), 3.0)

    focal = img - ndi.median_filter(img, size=9)
    asym = np.abs(ndi.gaussian_filter(img, 1.0) - ndi.gaussian_filter(img[:, ::-1], 1.0))

    return {
        "orientation_misalignment": ndi.gaussian_filter(f["misalignment"] * nuc_soft, 1.5),
        "order_loss": ndi.gaussian_filter((1.0 - f["order"]) * nuc_soft, 1.5),
        "nuclear_enlargement": ndi.gaussian_filter(area_map, 1.0),
        "hyperchromasia": ndi.gaussian_filter(chrom_map, 1.0),
        "crowding": density,
        "focal_hyperintensity": ndi.gaussian_filter(np.clip(focal, 0, None), 1.0),
        "asymmetry": ndi.gaussian_filter(asym, 1.5),
    }


class NormalReference:
    """Robust per-concept statistics of *normal* tissue, turning raw maps into
    sigma-deviation maps: z = (x - median) / (1.4826 * MAD).

    Image-level concept value = a high quantile of the z-map, i.e. "how far the
    most abnormal part of the image deviates from normal, in sigmas".
    """

    def __init__(self, peak_quantile: float = 99.5):
        self.peak_quantile = peak_quantile
        self.stats: dict[str, tuple[float, float]] = {}
        self.scales: dict[str, float] = {}
        self.region_threshold: float = 1.0

    def fit(self, maps_list: list[dict[str, np.ndarray]]) -> "NormalReference":
        for k in maps_list[0]:
            vals = np.concatenate([m[k].ravel() for m in maps_list])
            med = float(np.median(vals))
            mad = float(np.median(np.abs(vals - med))) * 1.4826
            spread = max(mad, float(np.std(vals)) * 0.25, 1e-6)
            self.stats[k] = (med, spread)
        zs = [self.zmaps(m) for m in maps_list]
        # typical *normal* peak per concept, so concepts with heavier tails do not dominate highlights
        self.scales = {k: max(float(np.quantile([np.percentile(z[k], self.peak_quantile) for z in zs], 0.95)), 1.0)
                       for k in self.stats}
        peaks = [self.combined(z).max() for z in zs]
        self.normal_peaks = [float(v) for v in peaks]
        # candidate-region threshold: the *median* normal image's strongest deviation. Candidates are
        # then ranked by how much they drive the decision; weak ones are shown as considered-and-dismissed.
        self.region_threshold = float(max(np.median(peaks), 1.0))
        return self

    def zmaps(self, maps: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        return {k: ((v - self.stats[k][0]) / self.stats[k][1]).astype(np.float32) for k, v in maps.items()}

    def peaks(self, zmaps: dict[str, np.ndarray]) -> dict[str, float]:
        return {k: float(np.percentile(v, self.peak_quantile)) for k, v in zmaps.items()}

    def combined(self, zmaps: dict[str, np.ndarray]) -> np.ndarray:
        """Pixel-wise max over concepts of the deviation, in units of "the largest
        deviation this concept typically reaches on a normal image"."""
        return np.max([np.clip(v / self.scales.get(k, 1.0), 0, None) for k, v in zmaps.items()], axis=0)
