"""Load real images (PNG/JPEG, NumPy, DICOM, NIfTI) and prepare them for the pipeline.

Format readers are optional dependencies, imported only when needed:
``pillow`` for PNG/JPEG, ``pydicom`` for DICOM, ``nibabel`` for NIfTI.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy import ndimage as ndi


def load(path: str | Path) -> np.ndarray:
    """Return a float32 array: 2D for images and single DICOM frames, 3D for volumes."""
    p = Path(path)
    name = p.name.lower()
    if name.endswith(".npy"):
        return np.load(p).astype(np.float32)
    if name.endswith((".nii", ".nii.gz")):
        import nibabel as nib
        return np.asarray(nib.load(str(p)).get_fdata(), np.float32)
    if name.endswith(".dcm") or p.is_dir():
        import pydicom
        files = sorted(p.glob("*.dcm")) if p.is_dir() else [p]
        ds = [pydicom.dcmread(str(f)) for f in files]
        ds.sort(key=lambda d: float(getattr(d, "InstanceNumber", 0)))
        arr = [d.pixel_array.astype(np.float32) * float(getattr(d, "RescaleSlope", 1))
               + float(getattr(d, "RescaleIntercept", 0)) for d in ds]
        return arr[0] if len(arr) == 1 else np.stack(arr)
    from PIL import Image
    im = Image.open(p)
    return np.asarray(im.convert("L"), np.float32)


def normalise(img: np.ndarray, lo: float = 0.5, hi: float = 99.5) -> np.ndarray:
    a, b = np.percentile(img, [lo, hi])
    return np.clip((img - a) / max(b - a, 1e-6), 0, 1).astype(np.float32)


def prepare(img: np.ndarray, size: int = 64, invert: bool = False) -> np.ndarray:
    """Normalise to [0, 1] and resample to size x size.

    For H&E use the haematoxylin channel or grayscale with nuclei dark
    (invert=False). Real whole-slide images should be *tiled* at a fixed
    microns-per-pixel, not downsampled: see ``tiles``.
    """
    img = normalise(np.asarray(img, np.float32))
    if invert:
        img = 1 - img
    zoom = (size / img.shape[0], size / img.shape[1])
    return np.clip(ndi.zoom(img, zoom, order=1), 0, 1).astype(np.float32)[:size, :size]


def tiles(img: np.ndarray, size: int = 64, stride: int | None = None):
    """Yield (y, x, tile) covering a large image at native resolution."""
    stride = stride or size
    img = normalise(img)
    for y in range(0, max(img.shape[0] - size, 0) + 1, stride):
        for x in range(0, max(img.shape[1] - size, 0) + 1, stride):
            yield y, x, img[y:y + size, x:x + size]


def axial_slices(volume: np.ndarray, size: int = 64, step: int = 1):
    """Yield (index, prepared 2D slice) from a 3D volume (slices along axis 0)."""
    for i in range(0, volume.shape[0], step):
        yield i, prepare(volume[i], size)
