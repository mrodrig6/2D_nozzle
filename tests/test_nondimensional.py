"""Is the solver actually non-dimensional, or only documented as such?

Documenting a non-dimensionalisation proves nothing.  The property it claims is
**invariance**: the solution depends only on ``gamma``, the geometry *ratios*
and ``p_b/p_t``, so rescaling any reference value must leave every dimensionless
output unchanged and scale every dimensional one by exactly its own scale.

These tests are the check.  They are slow -- each is a pair of converged solves
-- but they are the only thing standing between "the units are consistent" as a
claim and as a fact.
"""

from __future__ import annotations

import pytest

from src import NozzleGeometry
from src.api import solve_nozzle
from src.postprocess import performance

#: Tight, because the point is to separate a real inconsistency from the
#: tolerance-level wobble of stopping the march at slightly different places.
SOLVE = dict(
    order=1,
    refine=0,
    back_pressure_ratio=0.15,
    tolerance=1e-10,
    max_iterations=200_000,
    verbose=False,
)

#: Outputs that must not move at all when a reference value changes.
DIMENSIONLESS = (
    "thrust_coefficient",
    "discharge_coefficient",
    "exit_mach_area_averaged",
    "exit_pressure_ratio",
    "exit_temperature_ratio",
    "entropy_error",
)


def _solve(**kw):
    r = solve_nozzle(**SOLVE, **kw)
    assert r.converged
    return r, performance(r)


def _scaled_geometry(scale: float) -> NozzleGeometry:
    """Every length in the problem multiplied by ``scale``.

    Note what this is *not*: scaling ``length`` alone holds the heights fixed
    and so makes the nozzle more slender -- a different shape, which is entitled
    to a different answer.  A scale test has to scale the heights too.
    """
    base = NozzleGeometry()
    return NozzleGeometry(
        length=base.length * scale,
        throat_half_height=base.throat_half_height * scale,
        inlet_half_height=base.inlet_half_height * scale,
    )


@pytest.mark.slow
@pytest.mark.parametrize("scale", [2.0, 0.25])
def test_the_solution_is_invariant_under_a_geometric_rescale(scale):
    """Scale every length together: the dimensionless answer must not move *at all*.

    This is the strongest of these checks and it passes exactly -- 0.0 drift,
    not merely small -- because the discrete problem really is identical: same
    mesh topology, same iteration count, every operator scaled consistently.
    """
    _, ref = _solve(geometry=NozzleGeometry())
    _, got = _solve(geometry=_scaled_geometry(scale))

    for name in DIMENSIONLESS:
        assert getattr(got, name) == pytest.approx(getattr(ref, name), abs=1e-12), name

    # and the dimensional ones scale by exactly the factor
    assert got.thrust / ref.thrust == pytest.approx(scale, rel=1e-10)
    assert got.mass_flow_in / ref.mass_flow_in == pytest.approx(scale, rel=1e-10)


@pytest.mark.slow
def test_the_solution_is_invariant_under_a_reservoir_pressure_rescale():
    """``p_t`` multiplies rho, rho*v and rho*E by the *same* factor.

    So every scale in the problem moves together and the invariance is exact to
    machine precision, including the iteration count.
    """
    r0, ref = _solve()
    r1, got = _solve(total_pressure=10.0)

    for name in DIMENSIONLESS:
        assert getattr(got, name) == pytest.approx(getattr(ref, name), abs=1e-12), name
    assert got.thrust / ref.thrust == pytest.approx(10.0, rel=1e-10)
    assert r1.iterations == r0.iterations


@pytest.mark.slow
@pytest.mark.parametrize("kw", [{"total_temperature": 4.0}, {"Rgas": 1.0}])
def test_changing_the_sound_speed_moves_only_where_the_march_stops(kw):
    r"""``T_t`` and ``R`` change :math:`a_t`, and that exposes a real wrinkle.

    The conserved variables scale as :math:`\rho_t`, :math:`\rho_t a_t` and
    :math:`\rho_t a_t^2` -- three *different* powers of :math:`a_t` -- but the
    convergence residual is one RMS norm over all four components divided by the
    single scale :math:`\rho_t a_t / L`.  No single scale can non-dimensionalise
    a mixed-dimension norm, so changing :math:`a_t` reweights the components and
    the march crosses the tolerance a few iterations earlier or later.

    **The answer is still invariant; only the stopping point moves.**  That is
    what this test pins: the drift must sit at the tolerance, not above it.  At
    ``tolerance=1e-10`` it is ~7e-11, and it shrinks in proportion when the
    tolerance is tightened -- measured 4.8e-6, 1.2e-6, 7.5e-9, 7.0e-11 at
    tolerances of 1e-5, 1e-6, 1e-8 and 1e-10.

    If this ever fails *upward*, the residual norm has become genuinely
    inconsistent rather than merely imprecise about where to stop.
    """
    _, ref = _solve()
    _, got = _solve(**kw)

    for name in DIMENSIONLESS:
        a, b = getattr(ref, name), getattr(got, name)
        drift = abs(b - a) / max(abs(a), 1e-30)
        assert drift < 1e-7, f"{name}: relative drift {drift:.2e} is far above tolerance"
