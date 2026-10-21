"""Prepared cases must retain their geometry, flow and solver budgets."""

import numpy as np
import pytest

from src import SolverOptions, build_case, solve_nozzle
from src.sensitivity import differentiable_case
from src.timestepping.initialize import initial_state


def test_analytic_contour_accepts_a_realised_shock_free_operating_point():
    result = solve_nozzle(
        contour="analytic",
        area_ratio=10.0,
        back_pressure_ratio=0.2,
        nx=2,
        nr=1,
        order=0,
        backend="numpy",
        verbose=False,
        options=SolverOptions(max_iterations=1),
    )
    assert result.geometry.realised_area_ratio() == pytest.approx(2.5018882108)


def test_analytic_contour_refuses_a_realised_internal_shock():
    with pytest.raises(ValueError, match="shock inside the diverging section"):
        solve_nozzle(
            contour="analytic",
            area_ratio=1.1,
            back_pressure_ratio=0.6,
            nx=2,
            nr=1,
            order=0,
            backend="numpy",
            verbose=False,
            options=SolverOptions(max_iterations=1),
        )


@pytest.mark.jax
def test_prepared_sensitivity_case_applies_flow_shortcuts(tiny_case):
    dc = differentiable_case(case=tiny_case, back_pressure_ratio=0.2, total_pressure=2.0)
    assert dc.flow.back_pressure_ratio == 0.2
    assert dc.flow.total_pressure == 2.0
    assert tiny_case.flow.total_pressure != dc.flow.total_pressure


@pytest.mark.jax
@pytest.mark.parametrize("budget", [3, 7])
def test_adjoint_respects_the_budget_with_a_partial_batch(budget):
    case = build_case(contour="smooth", nx=2, nr=1, order=0)
    dc = differentiable_case(case=case)
    state = initial_state(case.operators, case.flow, case.geometry, "uniform")
    _, _, iterations, _ = dc.adjoint_solve(
        state,
        {},
        np.ones_like(state),
        tolerance=1e-30,
        max_iterations=budget,
        check_interval=5,
    )
    assert iterations == budget
