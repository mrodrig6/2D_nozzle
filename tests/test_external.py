"""Closed-form external wave structure at the nozzle lip.

These relations are classical, so the tests pin them against textbook values and
against each other's inverses rather than against the solver.
"""

from __future__ import annotations

import numpy as np
import pytest

from src import solve_nozzle
from src.external import (
    exit_wave_structure,
    mach_angle,
    max_deflection_angle,
    oblique_shock_angle,
    prandtl_meyer,
    prandtl_meyer_inverse,
    pressure_ratio_across_oblique_shock,
)

GAMMA = 1.4


# --------------------------------------------------------------------------
# the relations themselves
# --------------------------------------------------------------------------


def test_prandtl_meyer_vanishes_at_mach_one():
    assert prandtl_meyer(1.0, GAMMA) == pytest.approx(0.0, abs=1e-12)


@pytest.mark.parametrize(
    ("mach", "nu_deg"),
    [(1.5, 11.905), (2.0, 26.380), (3.0, 49.757), (4.0, 65.785)],
)
def test_prandtl_meyer_matches_the_tables(mach, nu_deg):
    """Textbook values for gamma = 1.4, to three decimals of a degree."""
    assert np.degrees(prandtl_meyer(mach, GAMMA)) == pytest.approx(nu_deg, abs=5e-3)


@pytest.mark.parametrize("mach", [1.2, 2.0, 3.5, 6.0])
def test_prandtl_meyer_inverse_round_trips(mach):
    assert prandtl_meyer_inverse(prandtl_meyer(mach, GAMMA), GAMMA) == pytest.approx(mach, rel=1e-9)


def test_prandtl_meyer_refuses_subsonic_flow():
    with pytest.raises(ValueError, match="supersonic"):
        prandtl_meyer(0.8, GAMMA)


def test_the_vacuum_limit_is_reported_rather_than_silently_clipped():
    with pytest.raises(ValueError, match="vacuum limit"):
        prandtl_meyer_inverse(10.0, GAMMA)


@pytest.mark.parametrize("mach", [1.5, 2.5, 4.0])
def test_oblique_shock_angle_inverts_its_own_pressure_relation(mach):
    """Sampled below the normal-shock limit, which is the strongest shock that M.

    At ``beta = 90`` the oblique shock *is* a normal shock, so
    ``1 + 2g/(g+1)(M^2-1)`` is the largest pressure ratio that Mach number can
    produce -- 2.46 at M=1.5.  Asking for more is not a near-miss, it is a
    different flow: the shock detaches.
    """
    strongest = 1.0 + 2.0 * GAMMA / (GAMMA + 1.0) * (mach**2 - 1.0)
    for frac in (0.1, 0.5, 0.95):
        ratio = 1.0 + frac * (strongest - 1.0)
        beta = oblique_shock_angle(mach, ratio, GAMMA)
        back = pressure_ratio_across_oblique_shock(mach, beta, GAMMA)
        assert back == pytest.approx(ratio, rel=1e-10)


@pytest.mark.parametrize("mach", [1.5, 2.5, 4.0])
def test_the_normal_shock_is_the_strongest_attached_shock(mach):
    """One step past it must be refused, and at it beta must be 90 degrees."""
    strongest = 1.0 + 2.0 * GAMMA / (GAMMA + 1.0) * (mach**2 - 1.0)
    assert oblique_shock_angle(mach, strongest, GAMMA) == pytest.approx(np.pi / 2, rel=1e-6)
    with pytest.raises(ValueError, match="detach"):
        oblique_shock_angle(mach, strongest * 1.01, GAMMA)


def test_a_vanishing_shock_is_a_mach_wave():
    """As the pressure ratio approaches 1, beta must approach the Mach angle.

    This is what makes the design point come out right: the oblique shock does
    not disappear discontinuously, it degenerates into a zero-strength Mach wave.
    """
    mach = 2.5
    beta = oblique_shock_angle(mach, 1.0 + 1e-12, GAMMA)
    assert beta == pytest.approx(mach_angle(mach), rel=1e-6)


def test_a_turn_past_detachment_is_refused():
    """M=1.5 cannot be turned through an attached shock of arbitrary strength."""
    with pytest.raises(ValueError, match="detach"):
        oblique_shock_angle(1.5, 50.0, GAMMA)


def test_max_deflection_is_between_the_mach_angle_and_normal():
    for mach in (1.5, 2.0, 4.0):
        theta, beta = max_deflection_angle(mach, GAMMA)
        assert 0.0 < theta < np.pi / 2
        assert mach_angle(mach) < beta < np.pi / 2


# --------------------------------------------------------------------------
# reading them off a converged solve
# --------------------------------------------------------------------------


def _solved(pb, order=1, refine=0):
    return solve_nozzle(
        contour="smooth",
        area_ratio=2.5,
        back_pressure_ratio=pb,
        order=order,
        refine=refine,
        geometry_order=2,
        verbose=False,
    )


@pytest.mark.slow
@pytest.mark.parametrize(
    ("pb", "regime"),
    [(0.0300, "under-expanded"), (0.1500, "over-expanded")],
)
def test_the_regime_is_read_off_the_exit_state(pb, regime):
    w = exit_wave_structure(_solved(pb), ambient_pressure_ratio=pb)
    assert w.regime == regime
    assert w.exit_mach > 1.0
    assert np.isfinite(w.turn_angle)


@pytest.mark.slow
def test_the_two_dimensional_design_point_is_not_the_quasi_1d_one():
    """A measured property of the nozzle, not of the solver's accuracy.

    Quasi-1D theory puts the design back pressure at 0.0640 for AR=2.5, but the
    computed exit pressure is 0.0669 -- a 4.5% offset that does *not* shrink with
    refinement, because quasi-1D assumes parallel exit streamlines and the real
    flow is still diverging.  So the jet is slightly under-expanded at the
    quasi-1D design value, and genuinely at design near 0.0669.
    """
    at_quasi1d = exit_wave_structure(_solved(0.0640), ambient_pressure_ratio=0.0640)
    assert at_quasi1d.regime == "under-expanded"
    assert at_quasi1d.pressure_mismatch == pytest.approx(1.045, abs=0.01)

    at_true = exit_wave_structure(_solved(0.0669), ambient_pressure_ratio=0.0669)
    assert at_true.regime == "design"


@pytest.mark.slow
def test_a_non_converged_solve_is_refused_rather_than_read():
    from src.config import SolverOptions

    r = solve_nozzle(
        contour="smooth",
        area_ratio=2.5,
        back_pressure_ratio=0.15,
        order=1,
        refine=0,
        options=SolverOptions(max_iterations=5),
        verbose=False,
    )
    assert not r.converged
    with pytest.raises(ValueError, match="converged"):
        exit_wave_structure(r)
