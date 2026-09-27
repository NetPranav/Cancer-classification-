import numpy as np
from scipy import ndimage as ndi

from oncopattern.data.phantoms import TISSUE_PATTERNS, longitudinal_mri, mri_slice, tissue_patch
from oncopattern.features.concepts import CONCEPTS, NormalReference, concept_maps
from oncopattern.physics.order import orientation_fields


def test_phantoms_have_ground_truth():
    rng = np.random.default_rng(0)
    for pat in TISSUE_PATTERNS:
        p = tissue_patch(rng, pat)
        assert p.image.shape == (64, 64) and p.image.dtype == np.float32
        assert 0 <= p.image.min() and p.image.max() <= 1
        assert p.mask.any() == (pat != "normal")
    assert mri_slice(rng, lesion_radius=3).mask.any()
    assert not mri_slice(rng).mask.any()


def test_longitudinal_series_shares_anatomy():
    series = longitudinal_mri(np.random.default_rng(0), radii=(0, 0, 3))
    diff = np.abs(series[0].image - series[1].image)
    assert diff.mean() < 0.05  # only noise differs between the two lesion-free scans
    assert series[2].mask.any()


def test_misalignment_peaks_on_the_rotated_cells():
    rng = np.random.default_rng(1)
    hits = 0
    for _ in range(20):
        p = tissue_patch(rng, "architectural_disorder")
        m = concept_maps(p.image)["orientation_misalignment"]
        y, x = np.unravel_index(m.argmax(), m.shape)
        hits += bool(ndi.binary_dilation(p.mask, iterations=4)[y, x])
    assert hits >= 16


def test_order_parameter_is_one_for_parallel_stripes():
    yy, xx = np.mgrid[0:64, 0:64]
    stripes = (np.sin(xx * 0.8) + 1) / 2
    f = orientation_fields(stripes)
    assert f["order"][16:48, 16:48].mean() > 0.95
    assert f["misalignment"][16:48, 16:48].mean() < 0.01


def test_normal_reference_separates_atypia():
    rng = np.random.default_rng(2)
    ref = NormalReference().fit([concept_maps(tissue_patch(rng).image) for _ in range(20)])
    assert set(ref.stats) == set(CONCEPTS)
    norm = [ref.peaks(ref.zmaps(concept_maps(tissue_patch(rng).image)))["nuclear_enlargement"] for _ in range(15)]
    aty = [ref.peaks(ref.zmaps(concept_maps(tissue_patch(rng, "nuclear_atypia").image)))["nuclear_enlargement"]
           for _ in range(15)]
    assert np.median(aty) > np.max(norm) * 0.9 and np.median(aty) > np.median(norm) * 1.5
