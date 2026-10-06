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
def test_the_solution_is_invariant_under_a_geometric_rescale():
    """Scale every length together: the dimensionless answer must not move *at all*.

    This is the strongest of these checks and it passes exactly -- 0.0 drift,
    not merely small -- because the discrete problem really is identical: same
    mesh topology, same iteration count, every operator scaled consistently.
    """
    scale = 2.0
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
def test_changing_the_sound_speed_changes_nothing_at_all():
    r"""``T_t`` and ``R`` change :math:`a_t`, and nothing else may move.

    This one used to be a *tolerance* test rather than an equality test, and the
    difference is the whole point of it.

    The four conserved variables scale as :math:`\rho_t`, :math:`\rho_t a_t`
    and :math:`\rho_t a_t^2` -- three different powers of :math:`a_t` -- so an
    unweighted RMS over all four, divided by the single scale
    :math:`\rho_t a_t / L`, is a mixed-dimension norm.  Changing :math:`a_t`
    reweighted the components against each other and the march crossed the
    tolerance in a different place: the converged answer drifted by ~7e-11 at
    ``tolerance=1e-10`` and the iteration count moved by 50.

    The norm now weights each component by its own power of :math:`a_t` first
    (:func:`src.backends.base.component_weights`), so the criterion is
    reference-invariant and the march is bit-identical.  The iteration count is
    asserted as well as the answer, because that is the part that used to move
    and the part a weaker assertion would let regress silently.

    ``T_t`` is the only reference varied here.  ``Rgas`` reaches :math:`a_t`
    through the same expression and was measured to behave identically
    (3e-14 drift, same iteration count), so running it too costs two more
    converged solves for no additional coverage.
    """
    r0, ref = _solve()
    r1, got = _solve(total_temperature=4.0)

    for name in DIMENSIONLESS:
        a, b = getattr(ref, name), getattr(got, name)
        drift = abs(b - a) / max(abs(a), 1e-30)
        assert drift < 1e-12, f"{name}: relative drift {drift:.2e} is above round-off"
    assert r1.iterations == r0.iterations, "the march no longer stops in the same place"


@pytest.mark.slow
def test_every_step_path_uses_the_weighted_norm():
    """The fused steppers must not carry their own copy of the norm.

    This is the bug that hid the problem for a whole round.  ``Backend.norm``
    was the one definition, but the Numba backend's fused RK4 and SSP-RK3 called
    the unweighted reduction kernel directly, so fixing ``norm`` changed nothing
    that a solve could see -- the weighted version was correct, exactly
    invariant, and simply never called.

    Comparing a fused backend against the reference one catches that: NumPy
    steps through ``Backend.rk4_step`` and so can only use ``norm``, while Numba
    runs its fused path.  If a fused path grows its own norm again, these two
    stop agreeing.
    """
    pytest.importorskip("numba")
    common = dict(order=1, refine=0, back_pressure_ratio=0.15, max_iterations=200, verbose=False)
    ref = solve_nozzle(**common, backend="numpy")
    got = solve_nozzle(**common, backend="numba")
    assert got.residual_scaled == pytest.approx(ref.residual_scaled, rel=1e-10)
