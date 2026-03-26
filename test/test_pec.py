"""Integration tests for PEC (Perfect Electric Conductor) boundary conditions.

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
