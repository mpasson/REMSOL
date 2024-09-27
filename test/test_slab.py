import multilayer_solver as msl
import numpy as np
import pytest as pt
from multilayer_solver import Polarization as pol

slab = msl.MultiLayer(
    [
        msl.Layer(1, 1),
        msl.Layer(2, 0.6),
        msl.Layer(1, 1),
    ]
)

coupled_slab = msl.MultiLayer(
    [
        msl.Layer(1, 1),
        msl.Layer(2, 0.6),
        msl.Layer(1, 2.0),
        msl.Layer(2, 0.6),
        msl.Layer(1, 1),
    ]
)


om = 2.0 * np.pi / 1.55


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
    print(slab.neff(om, pol.TE, 0))
    print(slab.neff(om, pol.TM, 0))
