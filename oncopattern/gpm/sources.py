"""Data sources that turn images into *prompted tasks*, many of them auto-labelled by verifiers.

Why this matters for learning from little data: a labelled cancer image
gives one label, but *any* image, labelled or not, gives dozens of checkable
facts about itself. How many nuclei? How large is the largest, relative to
normal? Where is the deviation strongest? Deterministic tools (``VerifierTools``)
compute these, so each image yields many supervised tasks for free, and the
same tools later grade the model's answers (``verify.py``). That is the
compiler-and-tests loop that makes code models reliable, applied to images.

Sources:
* ``TissuePhantomSource``, ``MRIPhantomSource``: synthetic, exact ground truth
* ``ImageFolderSource``: ``root/<class>/*.png``, the layout of most Kaggle image datasets
* ``MaskFolderSource``: ``images/`` + ``masks/`` with matching names (segmentation datasets)
* ``Mixture``: weighted mixture, plus masked-pattern (label-free) examples
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy import ndimage as ndi

from oncopattern.data.phantoms import TISSUE_PATTERNS, mri_slice, tissue_patch
from oncopattern.features.concepts import NUCLEUS_THRESHOLD, NormalReference, concept_maps
from oncopattern.gpm.data import Example, ImageItem
from oncopattern.gpm.tokenizer import ByteTokenizer

MEASURABLE = {
    "orientation_misalignment": "nuclear orientation misalignment",
    "nuclear_enlargement": "nuclear enlargement",
    "hyperchromasia": "hyperchromasia",
    "crowding": "cellularity",
    "focal_hyperintensity": "focal hyperintensity",
    "asymmetry": "left-right asymmetry",
}


def words(label: str) -> str:
    return label.replace("_", " ")


def mask_box(mask: np.ndarray):
    if mask is None or not mask.any():
        return None
    ys, xs = np.nonzero(mask)
    h, w = mask.shape
    return ys.min() / h, xs.min() / w, (ys.max() + 1) / h, (xs.max() + 1) / w


class VerifierTools:
    """Deterministic measurements used both to auto-label tasks and to grade answers."""

    def __init__(self, normal_images: list[np.ndarray]):
        self.reference = NormalReference().fit([concept_maps(_grey(i)) for i in normal_images])

    def measure(self, img: np.ndarray) -> dict[str, float]:
        z = self.reference.zmaps(concept_maps(_grey(img)))
        pk = self.reference.peaks(z)
        return {k: round(float(pk[k] / self.reference.scales[k]), 1) for k in MEASURABLE}

    @staticmethod
    def count_nuclei(img: np.ndarray) -> int:
        return int(ndi.label(img < NUCLEUS_THRESHOLD)[1])

    def strongest_box(self, img: np.ndarray, size: float = 0.25):
        """Box around the pixel of strongest combined deviation (a checkable 'where')."""
        comb = self.reference.combined(self.reference.zmaps(concept_maps(img)))
        y, x = np.unravel_index(int(comb.argmax()), comb.shape)
        h, w = comb.shape
        cy, cx = (y + 0.5) / h, (x + 0.5) / w
        return (max(cy - size / 2, 0), max(cx - size / 2, 0), min(cy + size / 2, 1), min(cx + size / 2, 1))


class Source:
    name = "source"
    task_weights: dict[str, float] = {}

    def sample(self, rng: np.random.Generator, task: str | None = None) -> Example:
        raise NotImplementedError

    def image_only(self, rng: np.random.Generator) -> list[ImageItem]:
        raise NotImplementedError

    def normal_images(self, rng: np.random.Generator) -> list[ImageItem] | None:
        """A *healthy* image from this source, or None if it has no known-normal pool. Normative
        training learns what healthy anatomy looks like only from these."""
        return None

    def _pick(self, rng, task):
        if task:
            return task
        names = list(self.task_weights)
        w = np.array([self.task_weights[n] for n in names], float)
        return names[rng.choice(len(names), p=w / w.sum())]


class TissuePhantomSource(Source):
    name = "tissue_phantom"
    spacing_mm = 0.0005  # 0.5 um per pixel, like a 20x slide
    task_weights = {"classify": 3, "normal": 1, "locate": 2, "measure": 2, "count": 1, "describe": 1}

    def __init__(self, tok: ByteTokenizer, tools: VerifierTools | None = None, size: int = 64, seed: int = 0):
        self.tok, self.size = tok, size
        rng = np.random.default_rng(seed + 991)
        self.tools = tools or VerifierTools([tissue_patch(rng, size=size).image for _ in range(24)])

    def _item(self, img):
        return ImageItem(img, self.spacing_mm, "phantom_tissue")

    def image_only(self, rng):
        return [self._item(tissue_patch(rng, TISSUE_PATTERNS[rng.integers(4)], size=self.size).image)]

    def normal_images(self, rng):
        return [self._item(tissue_patch(rng, "normal", size=self.size).image)]

    def sample(self, rng, task=None):
        task = self._pick(rng, task)
        pat = TISSUE_PATTERNS[rng.integers(len(TISSUE_PATTERNS))]
        p = tissue_patch(rng, pat, size=self.size)
        head = "H&E-like tissue, 0.5 um/px."
        box = mask_box(p.mask)
        meta = {"task": task, "label": pat, "mask": p.mask}
        if task == "classify":
            opts = [words(x) for x in rng.permutation(TISSUE_PATTERNS)]
            q = f" Which pattern is present? Options: {', '.join(opts)}."
            ans, meta["options"] = words(pat), opts
        elif task == "normal":
            q, ans = " Is this tissue normal? Answer yes or no.", "yes" if pat == "normal" else "no"
            meta["options"] = ["yes", "no"]
        elif task == "locate":
            q = " Locate the most abnormal cells."
            ans = self.tok.box_text(*box) if box else "none"
            meta["box"] = box
        elif task == "measure":
            concept = ["orientation_misalignment", "nuclear_enlargement", "hyperchromasia", "crowding"][rng.integers(4)]
            v = self.tools.measure(p.image)[concept]
            q, ans = f" Measure {MEASURABLE[concept]} relative to normal.", f"{v:.1f}"
            meta.update(value=v, tolerance=0.3, concept=concept)
        elif task == "count":
            v = self.tools.count_nuclei(p.image)
            q, ans = " Count the nuclei.", str(v)
            meta.update(value=v, tolerance=max(2.0, 0.1 * v))
        else:  # describe
            q = " Describe the findings."
            ans = (f"{words(pat)}; abnormal cells at {self.tok.box_text(*box)}." if box
                   else "normal tissue; no abnormal cells.")
            meta.update(box=box)
        meta["answer"] = ans
        return Example([self._item(p.image)], head, q, ans, task, meta)


class MRIPhantomSource(Source):
    name = "mri_phantom"
    spacing_mm = 3.0
    task_weights = {"lesion": 3, "locate": 2, "compare": 2, "measure": 1}

    def __init__(self, tok: ByteTokenizer, size: int = 64, seed: int = 0):
        self.tok, self.size = tok, size
        rng = np.random.default_rng(seed + 997)
        self.tools = VerifierTools([mri_slice(rng, size=size).image for _ in range(24)])

    def image_only(self, rng):
        r = rng.uniform(1.2, 4.0) if rng.random() < 0.5 else 0.0
        return [ImageItem(mri_slice(rng, self.size, r).image, self.spacing_mm, "phantom_mri")]

    def normal_images(self, rng):
        return [ImageItem(mri_slice(rng, self.size).image, self.spacing_mm, "phantom_mri")]

    def sample(self, rng, task=None):
        task = self._pick(rng, task)
        head = "Axial head MRI, 3 mm/px."
        meta = {"task": task}
        if task == "compare":
            seed = int(rng.integers(0, 2 ** 31))
            r0 = 0.0 if rng.random() < 0.5 else rng.uniform(1.2, 2.5)
            grows = rng.random() < 0.5
            r1 = r0 + rng.uniform(1.0, 2.0) if grows else r0
            ang = np.random.default_rng(seed).uniform(0, 2 * np.pi)
            centre = (self.size / 2 + 0.16 * self.size * np.cos(ang), self.size / 2 + 0.19 * self.size * np.sin(ang))
            a = mri_slice(rng, self.size, r0, centre if r0 else None, anatomy_seed=seed)
            b = mri_slice(rng, self.size, r1, centre if r1 else None, anatomy_seed=seed)
            dt = float(rng.integers(3, 13) * 30)
            ans = "yes" if grows else "no"
            meta.update(label=ans, options=["yes", "no"], answer=ans)
            return Example([ImageItem(a.image, self.spacing_mm, "phantom_mri", 0.0),
                            ImageItem(b.image, self.spacing_mm, "phantom_mri", dt)],
                           head + f" Two scans {int(dt)} days apart.", " Has the lesion grown? Answer yes or no.",
                           ans, task, meta)
        r = rng.uniform(1.2, 4.0) if rng.random() < 0.5 else 0.0
        p = mri_slice(rng, self.size, r)
        meta["mask"] = p.mask
        if task == "lesion":
            q, ans = " Is there a mass lesion? Answer yes or no.", "yes" if r else "no"
            meta.update(label=ans, options=["yes", "no"])
        elif task == "locate":
            box = mask_box(p.mask)
            q, ans = " Locate the lesion.", self.tok.box_text(*box) if box else "none"
            meta["box"] = box
        else:
            v = self.tools.measure(p.image)["asymmetry"]
            q, ans = f" Measure {MEASURABLE['asymmetry']} relative to normal.", f"{v:.1f}"
            meta.update(value=v, tolerance=0.3, concept="asymmetry")
        meta["answer"] = ans
        return Example([ImageItem(p.image, self.spacing_mm, "phantom_mri")], head, q, ans, task, meta)


def _load(path: Path, size: int, rgb: bool = False) -> np.ndarray:
    from oncopattern.data.io import load, load_rgb, prepare, prepare_rgb
    if rgb:
        arr = load(path) if path.suffix.lower() == ".npy" else load_rgb(path)
        if arr.ndim == 3:
            return prepare_rgb(arr if arr.shape[0] == 3 else arr.transpose(2, 0, 1), size)
        return prepare(arr, size)
    return prepare(load(path), size)


def in_split(path, split: str | None) -> bool:
    """Deterministic, name-based three-way split: test 10% | calib 10% | train 80%.

    The same file lands in the same split on every machine and every session. ``calib`` holds
    images the model never trains on, used to calibrate the normative atlas; ``test`` is only for
    final evaluation."""
    if split in (None, "all"):
        return True
    import zlib
    h = zlib.crc32(str(Path(path).name).encode()) / 2 ** 32
    return {"test": h < 0.1, "calib": 0.1 <= h < 0.2, "train": h >= 0.2}[split]


def _grey(img: np.ndarray) -> np.ndarray:
    return img.mean(0) if img.ndim == 3 else img


class ImageFolderSource(Source):
    """``root/<class>/*.(png|jpg|tif)``: classification-as-text, normality, auto-labelled measurements."""

    task_weights = {"classify": 3, "normal": 1, "measure": 1}

    def __init__(self, root: str, tok: ByteTokenizer, modality: str, spacing_mm: float, size: int = 64,
                 description: str = "", normal_classes=("normal", "benign", "negative", "notumor", "no_tumor", "0"),
                 max_per_class=None, rgb: bool = False, split: str | None = None):
        self.root, self.tok, self.modality, self.spacing, self.size = Path(root), tok, modality, spacing_mm, size
        self.rgb = rgb
        self.name = f"folder:{self.root.name}"
        self.description = description or f"{modality} image, {spacing_mm * 1000:g} um/px."
        exts = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".npy"}
        self.classes = sorted(d.name for d in self.root.iterdir() if d.is_dir())
        self.files = {c: sorted(f for f in (self.root / c).rglob("*")
                                if f.suffix.lower() in exts and in_split(f, split))[:max_per_class]
                      for c in self.classes}
        self.classes = [c for c in self.classes if self.files[c]]
        self.normal = [c for c in self.classes if c.lower() in normal_classes]
        self.tools = None
        if self.normal:
            fs = self.files[self.normal[0]][:24]
            self.tools = VerifierTools([_load(f, size, rgb) for f in fs]) if len(fs) >= 4 else None
        if not self.normal:
            self.task_weights = {"classify": 1}

    def _rand(self, rng):
        c = self.classes[rng.integers(len(self.classes))]
        f = self.files[c][rng.integers(len(self.files[c]))]
        return c, _load(f, self.size, self.rgb)

    def image_only(self, rng):
        return [ImageItem(self._rand(rng)[1], self.spacing, self.modality)]

    def normal_images(self, rng):
        if not self.normal:
            return None
        c = self.normal[rng.integers(len(self.normal))]
        f = self.files[c][rng.integers(len(self.files[c]))]
        return [ImageItem(_load(f, self.size, self.rgb), self.spacing, self.modality)]

    def sample(self, rng, task=None):
        task = self._pick(rng, task)
        c, img = self._rand(rng)
        meta = {"task": task, "label": c}
        if task == "measure" and self.tools is not None:
            concept = list(MEASURABLE)[rng.integers(len(MEASURABLE))]
            v = self.tools.measure(img)[concept]
            q, ans = f" Measure {MEASURABLE[concept]} relative to normal.", f"{v:.1f}"
            meta.update(value=v, tolerance=0.3, concept=concept)
        elif task == "normal" and self.normal:
            ans = "yes" if c in self.normal else "no"
            q = " Is this image normal? Answer yes or no."
            meta.update(options=["yes", "no"])
        else:
            meta["task"] = task = "classify"
            opts = [words(x) for x in rng.permutation(self.classes)]
            q, ans = f" Which class is this? Options: {', '.join(opts)}.", words(c)
            meta["options"] = opts
        meta["answer"] = ans
        return Example([ImageItem(img, self.spacing, self.modality)], self.description, q, ans, task, meta)


class MaskFolderSource(Source):
    """Images with binary masks: localisation and presence.

    Masks either live in a separate folder with the same file names
    (``masks=``), or beside the images with a name suffix (``mask_suffix="_mask"``,
    as in BUSI). Folders are searched recursively."""

    task_weights = {"locate": 2, "present": 1}
    EXTS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".npy"}

    def __init__(self, images: str, masks: str | None, tok: ByteTokenizer, modality: str, spacing_mm: float,
                 finding: str = "lesion", size: int = 64, description: str = "", mask_suffix: str = "",
                 rgb: bool = False, split: str | None = None):
        self.tok, self.modality, self.spacing, self.size, self.finding = tok, modality, spacing_mm, size, finding
        self.rgb, self.suffix = rgb, mask_suffix
        root = Path(images)
        files = sorted(f for f in root.rglob("*") if f.suffix.lower() in self.EXTS and in_split(f, split))
        if mask_suffix:
            self.pairs = [(f, f.with_name(f.stem + mask_suffix + f.suffix)) for f in files
                          if mask_suffix not in f.stem and f.with_name(f.stem + mask_suffix + f.suffix).exists()]
        else:
            self.pairs = [(f, Path(masks) / f.relative_to(root)) for f in files
                          if (Path(masks) / f.relative_to(root)).exists()]
        if not self.pairs:
            raise ValueError(f"no image/mask pairs found under {images}")
        self.name = f"masks:{root.name}"
        self.description = description or f"{modality} image, {spacing_mm * 1000:g} um/px."

    def _rand(self, rng):
        f, mf = self.pairs[rng.integers(len(self.pairs))]
        from oncopattern.data.io import load
        m = load(mf)
        m = m.max(-1) if m.ndim == 3 and m.shape[-1] in (3, 4) else m
        m = ndi.zoom((m > m.max() / 2).astype(np.float32), (self.size / m.shape[0], self.size / m.shape[1]), order=0) > 0.5
        return _load(f, self.size, self.rgb), m

    def image_only(self, rng):
        return [ImageItem(self._rand(rng)[0], self.spacing, self.modality)]

    def normal_images(self, rng):
        for _ in range(12):  # images with an empty mask are the healthy pool
            img, m = self._rand(rng)
            if not m.any():
                return [ImageItem(img, self.spacing, self.modality)]
        return None

    def sample(self, rng, task=None):
        task = self._pick(rng, task)
        img, m = self._rand(rng)
        box = mask_box(m)
        meta = {"task": task, "mask": m}
        if task == "locate":
            q, ans = f" Locate the {self.finding}.", self.tok.box_text(*box) if box else "none"
            meta["box"] = box
        else:
            q, ans = f" Is there a {self.finding}? Answer yes or no.", "yes" if box else "no"
            meta.update(label=ans, options=["yes", "no"])
        meta["answer"] = ans
        return Example([ImageItem(img, self.spacing, self.modality)], self.description, q, ans, task, meta)


class CSVSource(ImageFolderSource):
    """Labels in a CSV (``id,label``), images in a folder: e.g. PCam / Kaggle
    histopathologic-cancer-detection. ``names`` maps raw labels to words."""

    def __init__(self, csv_path: str, images: str, tok: ByteTokenizer, modality: str, spacing_mm: float,
                 id_col: str = "id", label_col: str = "label", ext: str = "", names: dict | None = None,
                 size: int = 64, rgb: bool = False, description: str = "", max_rows: int | None = None,
                 normal_classes=("normal", "benign", "negative", "0"), split: str | None = None):
        import csv
        self.tok, self.modality, self.spacing, self.size, self.rgb = tok, modality, spacing_mm, size, rgb
        self.root = Path(images)
        self.name = f"csv:{Path(csv_path).stem}"
        self.description = description or f"{modality} image, {spacing_mm * 1000:g} um/px."
        names = names or {}
        self.files: dict[str, list[Path]] = {}
        with open(csv_path, newline="") as fh:
            for i, row in enumerate(csv.DictReader(fh)):
                if max_rows and i >= max_rows:
                    break
                f = self.root / f"{row[id_col]}{ext}"
                if in_split(f, split):
                    self.files.setdefault(names.get(row[label_col], row[label_col]), []).append(f)
        self.classes = sorted(self.files)
        self.normal = [c for c in self.classes if c.lower() in normal_classes]
        self.tools = None
        if self.normal:
            fs = self.files[self.normal[0]][:24]
            self.tools = VerifierTools([_load(f, size, rgb) for f in fs]) if len(fs) >= 4 else None
        self.task_weights = {"classify": 3, "normal": 1, "measure": 1} if self.normal else {"classify": 1}


class HealthyFolderSource(Source):
    """A folder of images from healthy people only (e.g. IXI or OASIS brains, GTEx normal tissue).

    Its main job is the normative objective: teaching the model, in detail, what healthy anatomy
    looks like. It also contributes 'is this normal? -> yes' examples at a low weight."""

    task_weights = {"normal": 1}
    EXTS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".npy"}

    def __init__(self, root: str, tok: ByteTokenizer, modality: str, spacing_mm: float, size: int = 64,
                 rgb: bool = False, description: str = "", split: str | None = None):
        self.tok, self.modality, self.spacing, self.size, self.rgb = tok, modality, spacing_mm, size, rgb
        self.files = sorted(f for f in Path(root).rglob("*") if f.suffix.lower() in self.EXTS and in_split(f, split))
        if not self.files:
            raise ValueError(f"no images under {root}")
        self.name = f"healthy:{Path(root).name}"
        self.description = description or f"{modality} image, {spacing_mm * 1000:g} um/px."

    def _img(self, rng):
        return _load(self.files[rng.integers(len(self.files))], self.size, self.rgb)

    def image_only(self, rng):
        return [ImageItem(self._img(rng), self.spacing, self.modality)]

    normal_images = image_only

    def sample(self, rng, task=None):
        meta = {"task": "normal", "label": "normal", "options": ["yes", "no"], "answer": "yes"}
        return Example(self.image_only(rng), self.description, " Is this image normal? Answer yes or no.", "yes",
                       "normal", meta)


class Mixture:
    """Weighted mixture of sources.

    ``mim_fraction`` of examples are label-free masked-pattern examples. With ``normative=True``
    (the default) those images are drawn **only from healthy pools**, so the model's notion of
    "what fits" is healthy anatomy, and its surprise measures deviation *from health* rather than
    from a mixture that already contains tumours. Question-answer examples still use every image.
    """

    def __init__(self, sources: list[Source], weights: list[float] | None = None, mim_fraction: float = 0.3,
                 normative: bool = True):
        self.sources = sources
        w = np.asarray(weights or [1.0] * len(sources), float)
        self.p = w / w.sum()
        self.mim_fraction = mim_fraction
        self.normative = normative

    def sample(self, rng: np.random.Generator) -> Example:
        if rng.random() < self.mim_fraction:
            imgs = None
            if self.normative:
                for _ in range(len(self.sources) * 2):
                    src = self.sources[rng.choice(len(self.sources), p=self.p)]
                    imgs = src.normal_images(rng)
                    if imgs:
                        break
            if not imgs:  # no healthy pool anywhere (or normative off): use any image
                imgs = self.sources[rng.choice(len(self.sources), p=self.p)].image_only(rng)
            return Example(imgs, "Study this image.", "", "", "mim", {"task": "mim"}, mim=True)
        src = self.sources[rng.choice(len(self.sources), p=self.p)]
        return src.sample(rng)
