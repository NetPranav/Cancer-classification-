import numpy as np
import pytest

from oncopattern.data.phantoms import make_tissue_set, tissue_patch
from oncopattern.pipeline import Config, OncoPattern


@pytest.fixture(scope="session")
def trained():
    """A small model trained through every phase once, shared by the pipeline tests."""
    rng = np.random.default_rng(0)
    m = OncoPattern(Config(ssl_epochs=1, duplex_epochs=15, duplex_augment=False))
    m.pretrain([p.image for p in make_tissue_set(rng, 6)])
    m.fit_normal([tissue_patch(rng).image for _ in range(30)])
    sup = make_tissue_set(rng, 8)
    m.fit([p.image for p in sup], [p.label for p in sup])
    cal = make_tissue_set(rng, 25)
    info = m.calibrate([p.image for p in cal], [p.label for p in cal])
    test = make_tissue_set(rng, 8)
    return m, info, test
