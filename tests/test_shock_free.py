"""The operating set a nozzle-design exercise actually uses.

A student sizing a nozzle is trying to *avoid* a shock in the diverging
section.  The wave structure they want to see -- oblique shocks when
over-expanded, a Prandtl-Meyer fan when under-expanded -- forms **outside** the
exit plane, downstream of the computed domain.  So the useful operating set is
everything below the second critical ratio, plus the unchoked branch above the
first.

This file asserts that the solver converges across that whole set.  It is the
counterpart to the known limitation: inside the excluded band a shock stands in
the diverging section and the march does not reach a steady state, which
``dgnozzle.is_shock_free`` reports and :func:`dgnozzle.solve_nozzle` warns about.
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest

from dgnozzle import (
    Discretization,
    SolverOptions,
    critical_ratios,
    is_shock_free,
    performance,
    shock_free_range,
    solve_nozzle,
)

AREA_RATIO = 2.5
CRIT = critical_ratios(AREA_RATIO)


#: The unchoked branch needs a far larger budget, and not because anything is
#: wrong with it.  At ``pb = 0.98`` the exit Mach number is 0.17 -- a nearly
#: stagnant venturi flow -- while the residual is scaled by stagnation
#: conditions, so the march creeps: p=2 converges at 47,901 iterations where the
#: choked points take 1,000-4,000.  Measured, not guessed.
MAX_ITERATIONS = 80_000


def _solve(pb, order, refine, flux="roe", **kw):
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)   # no shocked-band warning
        return solve_nozzle(
            contour="smooth", area_ratio=AREA_RATIO, back_pressure_ratio=pb,
            discretization=Discretization(order=order, refine=refine,
                                          geometry_order=2),
            options=SolverOptions(max_iterations=MAX_ITERATIONS, tolerance=1e-6,
                                  **kw),
            flux=flux, verbose=False,
        )


# --------------------------------------------------------------------------
# the classification itself
# --------------------------------------------------------------------------


def test_the_shock_free_range_matches_the_critical_ratios():
    lo, hi = shock_free_range(AREA_RATIO)
    assert lo == (0.0, CRIT.second)
    assert hi == (CRIT.first, 1.0)


@pytest.mark.parametrize(
    ("pb", "free"),
    [
        (0.03, True),            # under-expanded
        (CRIT.third, True),      # design
        (0.15, True),            # over-expanded
        (CRIT.second, False),    # shock exactly on the exit plane
        (0.50, False),           # shock in the diverging section
        (0.70, False),
        (0.95, False),
        (0.99, True),            # unchoked, subsonic throughout
    ],
)
def test_is_shock_free_classifies_each_regime(pb, free):
    assert is_shock_free(AREA_RATIO, pb) is free


@pytest.mark.parametrize("pb", [0.50, 0.70, 0.95, CRIT.second])
def test_the_shocked_band_is_refused(pb):
    """Refusing beats running: the run costs minutes and cannot converge.

    ``CRIT.second`` is included deliberately -- a normal shock sitting exactly
    on the exit plane is both inside the domain and on the outflow condition's
    branch switch, so it is refused with the rest of the band.
    """
    with pytest.raises(ValueError, match="shock inside the diverging section"):
        solve_nozzle(
            area_ratio=AREA_RATIO, back_pressure_ratio=pb, order=0,
            options=SolverOptions(max_iterations=5), verbose=False,
        )


def test_the_shocked_band_can_still_be_opted_into():
    """Refusal must be a default, not a wall: the band is worth exploring."""
    r = solve_nozzle(
        area_ratio=AREA_RATIO, back_pressure_ratio=0.70, order=0,
        allow_shock_in_nozzle=True,
        options=SolverOptions(max_iterations=20), verbose=False,
    )
    assert not r.converged


def test_shock_free_points_are_accepted():
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        solve_nozzle(
            area_ratio=AREA_RATIO, back_pressure_ratio=0.15, order=0,
            options=SolverOptions(max_iterations=5), verbose=False,
        )


def test_a_sweep_across_the_band_records_it_instead_of_aborting():
    """A back-pressure sweep runs straight through the band.

    Aborting the whole sweep over one refused point would make the refusal
    worse than the non-convergence it replaced, so the point is recorded as a
    failure with the reason and the sweep carries on.
    """
    from dgnozzle import sweep

    table = sweep(
        back_pressure_ratio=[0.15, 0.70, 0.30],
        area_ratio=AREA_RATIO, order=0,
        options=SolverOptions(max_iterations=200, tolerance=1e-3),
    )
    failures = table.failures()
    assert len(failures) == 1, "expected the 0.70 point to be the only failure"
    point, why = failures[0]
    assert point["back_pressure_ratio"] == pytest.approx(0.70)
    assert "shock inside the diverging section" in why


# --------------------------------------------------------------------------
# does the whole shock-free set converge?
# --------------------------------------------------------------------------

#: spread across the choked supersonic-exit branch, plus the unchoked branch.
#: 0.42 sits just below the second critical ratio (0.4348), which is the tightest
#: point in the set -- the shock is nearly on the exit plane.
SHOCK_FREE_PB = (0.02, 0.0640, 0.15, 0.30, 0.42, 0.98)


@pytest.mark.slow
@pytest.mark.parametrize("pb", SHOCK_FREE_PB)
@pytest.mark.parametrize("order", [0, 1, 2])
def test_every_shock_free_point_converges(pb, order):
    r = _solve(pb, order, 0)
    assert r.converged, f"pb={pb} p={order}: {r.message}"
    assert r.min_density > 0.0
    assert r.min_pressure > 0.0
    assert r.mean_repairs == 0, "a shock-free point should need no repairs"


@pytest.mark.slow
@pytest.mark.parametrize("pb", (0.0640, 0.15, 0.42))
def test_the_shock_free_set_converges_on_a_finer_mesh(pb):
    r = _solve(pb, 1, 1)
    assert r.converged, f"pb={pb}: {r.message}"


@pytest.mark.slow
@pytest.mark.parametrize("pb", (0.0640, 0.15))
def test_both_fluxes_converge_and_agree_on_the_shock_free_set(pb):
    """Roe and HLLC must reach the same answer to discretisation error."""
    roe = _solve(pb, 1, 0, flux="roe")
    hllc = _solve(pb, 1, 0, flux="hllc")
    assert roe.converged and hllc.converged
    a, b = performance(roe), performance(hllc)
    assert a.thrust == pytest.approx(b.thrust, rel=0.02)
    assert a.mass_flow_in == pytest.approx(b.mass_flow_in, rel=0.02)


@pytest.mark.slow
def test_mass_flow_is_frozen_across_the_choked_shock_free_branch():
    """Choking, checked only where the solver converges.

    Once the throat is sonic, mass flow cannot depend on back pressure.  Across
    the choked part of the shock-free set that is a check the solver was never
    tuned to pass.
    """
    flows = []
    for pb in (0.02, 0.0640, 0.15, 0.30, 0.42):
        r = _solve(pb, 1, 0)
        assert r.converged, f"pb={pb}: {r.message}"
        flows.append(performance(r).mass_flow_in)
    spread = (max(flows) - min(flows)) / np.mean(flows)
    assert spread < 1e-4, f"mass flow varies by {spread:.2e} across the branch"
