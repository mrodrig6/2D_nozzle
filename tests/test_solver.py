"""End-to-end solves: convergence, conservation, and agreement with theory.

These run a real flow solve, so they are marked ``slow`` -- but the meshes are
the coarsest ones, and the whole file finishes in well under a minute.
"""

from __future__ import annotations

import numpy as np
import pytest

from dgnozzle import (
    FlowConditions,
    NozzleGeometry,
    SolverOptions,
    performance,
    solve_nozzle,
    solve_quasi1d,
)
from dgnozzle.api import build_case
from dgnozzle.initialize import change_order

pytestmark = pytest.mark.slow

BASE = dict(contour="smooth", verbose=False, refine=0)


@pytest.fixture(scope="module")
def solved():
    return solve_nozzle(order=1, **BASE)


def test_it_converges(solved):
    assert solved.converged, solved.message
    assert solved.residual_scaled < 1e-6
    assert solved.min_density > 0.0
    assert solved.min_pressure > 0.0


def test_mass_is_conserved(solved):
    perf = performance(solved)
    assert perf.mass_imbalance < 5e-3


def test_the_two_thrust_forms_agree(solved):
    """Momentum and wall-pressure forms are identical in exact arithmetic."""
    perf = performance(solved)
    assert perf.thrust_imbalance < 2e-2
    assert perf.thrust > 0.0


def test_it_matches_quasi_1d_theory(solved):
    perf = performance(solved)
    q1d = solve_quasi1d(solved.geometry, solved.flow)
    assert perf.exit_mach_area_averaged == pytest.approx(q1d.exit_mach, rel=0.03)
    assert perf.mass_flow_in == pytest.approx(q1d.mass_flow, rel=0.02)
    assert perf.exit_pressure_ratio == pytest.approx(q1d.exit_pressure, rel=0.05)


def test_the_nozzle_is_choked(solved):
    """Mass flow must not depend on back pressure once choked."""
    other = solve_nozzle(order=1, back_pressure_ratio=0.30, **BASE)
    assert other.converged
    a = performance(solved).mass_flow_in
    b = performance(other).mass_flow_in
    assert b == pytest.approx(a, rel=2e-3)


def test_element_kinds_and_orders_bracket_the_same_thrust():
    """Triangles and quads, p=0 and p=1, must all describe the same nozzle.

    Deliberately compared against each other rather than a hard-coded number:
    the value depends on the contour family, and a literal here silently becomes
    wrong the moment the default geometry changes.
    """
    thrusts = {}
    for element in ("tri", "quad"):
        for order in (0, 1):
            res = solve_nozzle(order=order, element=element, **BASE)
            assert res.converged, f"{element} p={order}: {res.message}"
            thrusts[(element, order)] = performance(res).thrust
    values = np.array(list(thrusts.values()))
    assert values.min() > 0.0
    # At p=1 the two element kinds must agree closely.  At p=0 they need not:
    # the triangular mesh splits each cell in two, so it carries twice the
    # degrees of freedom of the quad mesh at the same refinement level, and a
    # first-order scheme is still far from the asymptotic answer.
    assert thrusts[("tri", 1)] == pytest.approx(thrusts[("quad", 1)], rel=0.05), thrusts
    for element in ("tri", "quad"):
        assert thrusts[(element, 1)] > thrusts[(element, 0)], thrusts


def test_schemes_reach_the_same_steady_state():
    """RK4 and SSP-RK3 are different paths to the same fixed point."""
    results = {}
    for scheme in ("rk4", "ssprk3"):
        res = solve_nozzle(order=1, scheme=scheme, **BASE)
        assert res.converged, f"{scheme}: {res.message}"
        results[scheme] = performance(res).thrust
    assert results["rk4"] == pytest.approx(results["ssprk3"], rel=1e-4), results


def test_initial_condition_does_not_change_the_answer():
    """The steady state is a property of the equations, not of where you start."""
    results = {}
    for ic in ("quasi1d", "uniform"):
        res = solve_nozzle(order=1, initial_condition=ic, **BASE)
        assert res.converged, f"{ic}: {res.message}"
        results[ic] = (performance(res).thrust, res.iterations)
    assert results["quasi1d"][0] == pytest.approx(results["uniform"][0], rel=1e-4), results


def test_p_continuation_reaches_the_same_answer():
    with_cont = solve_nozzle(order=2, p_continuation=True, **BASE)
    without = solve_nozzle(order=2, p_continuation=False, **BASE)
    assert with_cont.converged and without.converged
    assert performance(with_cont).thrust == pytest.approx(
        performance(without).thrust, rel=1e-4
    )


def test_order_continuation_is_exact():
    case = build_case(NozzleGeometry(contour="smooth"), FlowConditions(), order=1)
    ops1 = case.operators_at(1)
    ops2 = case.operators_at(2)
    rng = np.random.default_rng(0)
    U1 = np.array([2.0, 0.6, 0.02, 5.0]) * (
        1.0 + 0.05 * rng.standard_normal((ops1.n_elem, ops1.ref.n_basis, 4))
    )
    back = change_order(change_order(U1, ops1, ops2), ops2, ops1)
    assert np.abs(back - U1).max() < 1e-12


def test_convergence_is_independent_of_the_initial_guess():
    """A scale-based criterion must not reward or punish the starting field.

    With a criterion relative to the *first* residual, a better initial guess
    demands a tighter absolute residual, so improving the initial condition makes
    the solver appear slower and two runs cannot be compared.
    """
    results = {
        ic: solve_nozzle(order=1, initial_condition=ic, p_continuation=False, **BASE)
        for ic in ("quasi1d", "uniform")
    }
    for ic, res in results.items():
        assert res.converged, f"{ic}: {res.message}"
        assert res.residual_scaled <= 1e-6
    a, b = (performance(r).thrust for r in results.values())
    assert a == pytest.approx(b, rel=1e-5)


def test_refinement_reduces_the_entropy_error():
    errors = [
        performance(solve_nozzle(order=1, geometry_order=2, refine=r,
                                 contour="smooth", verbose=False)).entropy_error
        for r in (0, 1)
    ]
    rate = np.log(errors[0] / errors[1]) / np.log(2.0)
    assert rate > 1.7, f"observed rate {rate:.2f}, expected about 2"


def test_backends_reach_the_same_steady_state():
    from dgnozzle.backends import available_backends

    results = {}
    for backend in available_backends():
        res = solve_nozzle(order=1, backend=backend, **BASE)
        assert res.converged, f"{backend}: {res.message}"
        results[backend] = performance(res).thrust
    values = list(results.values())
    assert np.allclose(values, values[0], rtol=1e-6), results


def test_divergence_is_reported_not_hidden():
    """A too-large cfl must stop and say so, not spin forever."""
    res = solve_nozzle(order=1, cfl=50.0, max_iterations=4000, **BASE)
    assert not res.converged
    assert res.message


def test_a_mismatched_warm_start_is_rejected():
    case = build_case(NozzleGeometry(contour="smooth"), FlowConditions(), order=1)
    from dgnozzle.solver import solve_steady

    bad = np.zeros((case.operators.n_elem, 99, 4))
    with pytest.raises(ValueError, match="expected"):
        solve_steady(case.operators, case.flow, case.geometry,
                     case.discretization, SolverOptions(print_interval=0), U0=bad)


def test_unphysical_convergence_is_not_reported_as_success():
    """Residual convergence to a negative-pressure state must not count."""
    res = solve_nozzle(order=1, contour="smooth", area_ratio=4.0, refine=0,
                       limiter="none", verbose=False, max_iterations=20000)
    if res.min_pressure <= 0.0:
        assert not res.converged
        assert "not physical" in res.message
