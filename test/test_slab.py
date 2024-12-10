# %%
import numpy as np
import pytest as pt
import remsol
from remsol import Polarization as pol

slab = remsol.MultiLayer(
    [
        remsol.Layer(1.0, 1.0),
        remsol.Layer(2.0, 0.6),
        remsol.Layer(1.0, 1.0),
    ]
)

coupled_slab = remsol.MultiLayer(
    [
        remsol.Layer(1.0, 1.0),
        remsol.Layer(2.0, 0.6),
        remsol.Layer(1.0, 2.0),
        remsol.Layer(2.0, 0.6),
        remsol.Layer(1.0, 1.0),
    ]
)


om = 2.0 * np.pi / 1.55


# %%
def test_slab():
    assert slab.neff(om, pol.TE, 0) == pt.approx(1.804297363)
    assert slab.neff(om, pol.TE, 1) == pt.approx(1.191174978)
    assert slab.neff(om, pol.TM, 0) == pt.approx(1.657017474)
    assert slab.neff(om, pol.TM, 1) == pt.approx(1.028990635)


def test_coupled_slab():
    assert coupled_slab.neff(om, pol.TE, 0) == pt.approx(1.804297929)
    assert coupled_slab.neff(om, pol.TE, 1) == pt.approx(1.804296798)
    assert coupled_slab.neff(om, pol.TE, 2) == pt.approx(1.192052932)
    assert coupled_slab.neff(om, pol.TE, 3) == pt.approx(1.190270579)

    assert coupled_slab.neff(om, pol.TM, 0) == pt.approx(1.657019473)
    assert coupled_slab.neff(om, pol.TM, 1) == pt.approx(1.657015474)
    assert coupled_slab.neff(om, pol.TM, 2) == pt.approx(1.035192425)
    assert coupled_slab.neff(om, pol.TM, 3) == pt.approx(1.019866805)


if __name__ == "__main__":
    from time import time

    class Timer:
        def __init__(self):
            self.start = None
            self.end = None

        @property
        def elapsed(self):
            return self.end - self.start

        def __enter__(self):
            self.start = time()
            return self

        def __exit__(self, exc_type, exc_value, traceback):
            self.end = time()

    with Timer() as t:
        print(coupled_slab.neff(om, pol.TE, 2))
    print(t.elapsed)
