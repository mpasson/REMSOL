"""Integration tests for PEC (Perfect Electric Conductor) boundary conditions.

Multi-layer validation
-----------------------
The pec_slab.py example uses a 4-element structure with an intermediate cladding
layer between the PEC wall and the guiding core.  This exercises the full
physical transfer-matrix path (propagation through L0 → interface L0/L1 →
propagation through L1 → interface L1/L2 …) and was the original source of the
bug that was fixed.  We validate:

  1. neff symmetry: PEC-left and PEC-right mirror structures must give the same neff.
  2. Field continuity: Ey (TE) and Ez (TM) must be continuous at every dielectric
     interface.
  3. PEC boundary condition: the tangential E field must vanish at the PEC wall.

Physical validation strategy
-----------------------------
A PEC wall at x = 0 enforces E_tangential = 0.  For both TE (Ey = 0) and TM
(Ez = 0) polarisations this means the primary field component is
antisymmetric, i.e. it follows a sine-like profile.

This is identical to the *odd* modes of a symmetric slab whose full width is
twice the half-slab thickness.  Therefore:

    neff( PEC | Layer(n2, d) | Layer(n1) , mode 0 )
    == neff( Layer(n1) | Layer(n2, 2*d) | Layer(n1) , mode 1 )

This relationship lets us validate PEC against already-verified standard-slab
results without needing any additional analytical reference.
"""

import numpy as np
import pytest

import remsol
from remsol import BoundaryCondition as BC
from remsol import Polarization as pol

om = 2.0 * np.pi / 1.55  # vacuum wavevector for lambda = 1.55 µm

# ---------------------------------------------------------------------------
# Structures
# ---------------------------------------------------------------------------

# Standard symmetric slab used as the reference (d = 0.6 µm, twice the half-slab)
full_slab = remsol.MultiLayer(
    [remsol.Layer(1.0, 1.0), remsol.Layer(2.0, 0.6), remsol.Layer(1.0, 1.0)]
)

# PEC on the left: equivalent to the odd modes of full_slab
pec_left = remsol.MultiLayer(
    [remsol.PEC(), remsol.Layer(2.0, 0.3), remsol.Layer(1.0, 1.0)]
)

# PEC on the right: mirror image, must give the same neff as pec_left
pec_right = remsol.MultiLayer(
    [remsol.Layer(1.0, 1.0), remsol.Layer(2.0, 0.3), remsol.PEC()]
)


# ---------------------------------------------------------------------------
# neff correctness — PEC-left matches odd mode of full slab
# ---------------------------------------------------------------------------


def test_pec_left_te_matches_odd_mode_of_full_slab():
    """PEC-left mode 0 TE == full-slab mode 1 TE (the odd mode)."""
    assert pec_left.neff(om, pol.TE, 0) == pytest.approx(
        full_slab.neff(om, pol.TE, 1), rel=1e-6
    )


def test_pec_left_tm_matches_even_mode_of_full_slab():
    """PEC-left mode 0 TM == full-slab mode 0 TM (the even TM mode).

    For TM, the (a, b) transfer-matrix amplitudes represent Ez (not Hy).
    The PEC condition Ez = 0 at the wall produces a sine-like Ez profile,
    which corresponds to a cosine-like Hy profile — i.e. the *even* TM mode
    of the symmetric full slab.  This is the opposite parity to TE, where
    PEC selects the *odd* mode.

    Equivalently: the PEC half-slab eigenvalue condition
        (kappa2/n2^2) * tan(kappa2 * d) = gamma1 / n1^2
    is identical to the *even* TM condition of the full slab with width 2d:
        (kappa2/n2^2) * tan(kappa2 * d_full/2) = gamma1 / n1^2
    so the two structures share the same effective indices.
    """
    assert pec_left.neff(om, pol.TM, 0) == pytest.approx(
        full_slab.neff(om, pol.TM, 0), rel=1e-6
    )


# ---------------------------------------------------------------------------
# Symmetry — PEC-left and PEC-right give the same neff
# ---------------------------------------------------------------------------


def test_pec_left_right_symmetry_te():
    """PEC-left and PEC-right of the same half-slab must give identical TE neff."""
    assert pec_left.neff(om, pol.TE, 0) == pytest.approx(
        pec_right.neff(om, pol.TE, 0), rel=1e-9
    )


def test_pec_left_right_symmetry_tm():
    """PEC-left and PEC-right of the same half-slab must give identical TM neff."""
    assert pec_left.neff(om, pol.TM, 0) == pytest.approx(
        pec_right.neff(om, pol.TM, 0), rel=1e-9
    )


# ---------------------------------------------------------------------------
# set_left_boundary / set_right_boundary API (alternative to PEC in list)
# ---------------------------------------------------------------------------


def test_set_left_boundary_api():
    """set_left_boundary() must give the same result as placing PEC() first."""
    ml = remsol.MultiLayer([remsol.Layer(2.0, 0.3), remsol.Layer(1.0, 1.0)])
    ml.set_left_boundary(BC.PEC)
    assert ml.neff(om, pol.TE, 0) == pytest.approx(
        pec_left.neff(om, pol.TE, 0), rel=1e-9
    )


def test_set_right_boundary_api():
    """set_right_boundary() must give the same result as placing PEC() last."""
    ml = remsol.MultiLayer([remsol.Layer(1.0, 1.0), remsol.Layer(2.0, 0.3)])
    ml.set_right_boundary(BC.PEC)
    assert ml.neff(om, pol.TM, 0) == pytest.approx(
        pec_right.neff(om, pol.TM, 0), rel=1e-9
    )


# ---------------------------------------------------------------------------
# Field boundary condition — E must vanish at PEC wall
# ---------------------------------------------------------------------------


def test_pec_left_field_ey_zero_at_wall():
    """For PEC-left TE mode, Ey must be zero at x = 0 (the PEC wall)."""
    ml = remsol.MultiLayer(
        [remsol.PEC(), remsol.Layer(2.0, 0.3), remsol.Layer(1.0, 1.0)]
    )
    ml.plot_step = 1e-4
    field = ml.field(om, pol.TE, 0)
    # x[0] is the first grid point at x = 0 (PEC wall)
    assert abs(field.Ey[0]) == pytest.approx(0.0, abs=1e-6)


def test_pec_left_field_ez_zero_at_wall():
    """For PEC-left TM mode, Ez must be zero at x = 0 (the PEC wall)."""
    ml = remsol.MultiLayer(
        [remsol.PEC(), remsol.Layer(2.0, 0.3), remsol.Layer(1.0, 1.0)]
    )
    ml.plot_step = 1e-4
    field = ml.field(om, pol.TM, 0)
    assert abs(field.Ez[0]) == pytest.approx(0.0, abs=1e-6)


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------


def test_pec_in_middle_raises():
    """PEC placed in the middle of the layer list must raise ValueError."""
    with pytest.raises(Exception):
        remsol.MultiLayer(
            [remsol.Layer(1.0, 1.0), remsol.PEC(), remsol.Layer(1.0, 1.0)]
        )


def test_pec_both_ends_raises():
    """PEC on both ends simultaneously must raise ValueError."""
    with pytest.raises(Exception):
        remsol.MultiLayer([remsol.PEC(), remsol.Layer(2.0, 0.6), remsol.PEC()])


# ---------------------------------------------------------------------------
# Multi-layer PEC: [PEC, n=1(d=0.2), n=2(d=0.5), n=1] and its mirror
# This exercises the full propagation path through an intermediate cladding
# layer between the PEC wall and the guiding core.
# ---------------------------------------------------------------------------

_plot_step = 1e-4  # fine enough to resolve field gradients


@pytest.fixture(scope="module")
def multi_pec_left():
    ml = remsol.MultiLayer(
        [
            remsol.PEC(),
            remsol.Layer(n=1.0, d=0.2),
            remsol.Layer(n=2.0, d=0.5),
            remsol.Layer(n=1.0, d=2.0),
        ]
    )
    ml.plot_step = _plot_step
    return ml


@pytest.fixture(scope="module")
def multi_pec_right():
    ml = remsol.MultiLayer(
        [
            remsol.Layer(n=1.0, d=2.0),
            remsol.Layer(n=2.0, d=0.5),
            remsol.Layer(n=1.0, d=0.2),
            remsol.PEC(),
        ]
    )
    ml.plot_step = _plot_step
    return ml


def test_multi_pec_left_right_symmetry_te(multi_pec_left, multi_pec_right):
    """Mirror PEC-left / PEC-right 4-layer structures must give identical TE neff."""
    assert multi_pec_left.neff(om, pol.TE, 0) == pytest.approx(
        multi_pec_right.neff(om, pol.TE, 0), rel=1e-6
    )


def test_multi_pec_left_right_symmetry_tm(multi_pec_left, multi_pec_right):
    """Mirror PEC-left / PEC-right 4-layer structures must give identical TM neff."""
    assert multi_pec_left.neff(om, pol.TM, 0) == pytest.approx(
        multi_pec_right.neff(om, pol.TM, 0), rel=1e-6
    )


def test_multi_pec_left_te_ey_zero_at_wall(multi_pec_left):
    """TE Ey must vanish at the PEC wall (x = 0) for the multi-layer structure."""
    field = multi_pec_left.field(om, pol.TE, 0)
    assert abs(field.Ey[0]) == pytest.approx(0.0, abs=1e-6)


def test_multi_pec_left_tm_ez_zero_at_wall(multi_pec_left):
    """TM Ez must vanish at the PEC wall (x = 0) for the multi-layer structure."""
    field = multi_pec_left.field(om, pol.TM, 0)
    assert abs(field.Ez[0]) == pytest.approx(0.0, abs=1e-6)


def _field_jump(values, x_arr, x_interface, n_steps=1):
    """Return |field[i+n] - field[i-n]| at the grid point closest to x_interface."""
    import numpy as np

    idx = int(np.argmin(np.abs(np.asarray(x_arr) - x_interface)))
    return abs(values[idx + n_steps] - values[idx - n_steps])


def test_multi_pec_left_te_ey_continuous_at_interfaces(multi_pec_left):
    """TE Ey must be continuous at both dielectric interfaces for PEC-left."""
    field = multi_pec_left.field(om, pol.TE, 0)
    # Allowed jump = field_magnitude * (d_field/dx) * 2*step ~ small vs field itself
    # We use a loose absolute tolerance of 0.1 (field is O(10)).
    assert _field_jump(field.Ey, field.x, 0.2) == pytest.approx(0.0, abs=0.1)
    assert _field_jump(field.Ey, field.x, 0.7) == pytest.approx(0.0, abs=0.1)


def test_multi_pec_left_tm_ez_continuous_at_interfaces(multi_pec_left):
    """TM Ez must be continuous at both dielectric interfaces for PEC-left."""
    field = multi_pec_left.field(om, pol.TM, 0)
    assert _field_jump(field.Ez, field.x, 0.2) == pytest.approx(0.0, abs=0.1)
    assert _field_jump(field.Ez, field.x, 0.7) == pytest.approx(0.0, abs=0.1)


def test_multi_pec_right_te_ey_near_zero_at_wall(multi_pec_right):
    """TE Ey must approach 0 near the PEC wall for the PEC-right structure.

    The grid endpoint is one step before the physical PEC wall, so Ey[-1]
    should be small (proportional to the step size).
    """
    field = multi_pec_right.field(om, pol.TE, 0)
    # Ey[-1] is one plot_step away from the wall; allow abs tolerance = 0.05
    assert abs(field.Ey[-1]) == pytest.approx(0.0, abs=0.05)


def test_multi_pec_right_tm_ez_near_zero_at_wall(multi_pec_right):
    """TM Ez must approach 0 near the PEC wall for the PEC-right structure."""
    field = multi_pec_right.field(om, pol.TM, 0)
    assert abs(field.Ez[-1]) == pytest.approx(0.0, abs=0.05)


def test_multi_pec_right_te_ey_continuous_at_interfaces(multi_pec_right):
    """TE Ey must be continuous at both dielectric interfaces for PEC-right.

    For the PEC-right structure [n=1(d=2), n=2(d=0.5), n=1(d=0.2), PEC] with
    xstart = -2.0, the interfaces are at x = 0.0 and x = 0.5.
    """
    field = multi_pec_right.field(om, pol.TE, 0)
    assert _field_jump(field.Ey, field.x, 0.0) == pytest.approx(0.0, abs=0.1)
    assert _field_jump(field.Ey, field.x, 0.5) == pytest.approx(0.0, abs=0.1)


def test_multi_pec_right_tm_ez_continuous_at_interfaces(multi_pec_right):
    """TM Ez must be continuous at both dielectric interfaces for PEC-right."""
    field = multi_pec_right.field(om, pol.TM, 0)
    assert _field_jump(field.Ez, field.x, 0.0) == pytest.approx(0.0, abs=0.1)
    assert _field_jump(field.Ez, field.x, 0.5) == pytest.approx(0.0, abs=0.1)
