"""Adjoint design sensitivities, verified against finite differences.

These are the slowest tests in the suite -- each finite-difference check is two
extra flow solves -- so they run on the coarsest mesh at p=1.
"""

from __future__ import annotations

import numpy as np
import pytest

pytestmark = [pytest.mark.slow, pytest.mark.jax]

pytest.importorskip("jax")

from dgnozzle.sensitivity import (  # noqa: E402
    ALL_DESIGN_PARAMETERS,
    OBJECTIVES,
    check_gradient,
    differentiable_case,
    finite_difference_gradient,
)


@pytest.fixture(scope="module")
def dcase():
    # 'smooth' and a shock-free operating point: the limiter stays inactive, so
    # the discrete objective is smooth in the design variables
    return differentiable_case(
        contour="smooth", order=1, refine=0,
        back_pressure_ratio=0.15, tolerance=1e-11, max_iterations=200_000,
    )


def test_all_design_parameters_are_exposed(dcase):
    assert set(dcase.default_params()) == set(ALL_DESIGN_PARAMETERS)


def test_unknown_parameter_is_rejected(dcase):
    with pytest.raises(ValueError, match="unknown design parameter"):
        dcase.params_for(("wing_span",))


@pytest.mark.parametrize("name", ["area_ratio", "throat_x", "theta_exit"])
def test_adjoint_matches_finite_differences(dcase, name):
    out = check_gradient(dcase, "thrust", names=(name,), step=1e-4, verbose=False)
    assert out[name]["relative_error"] < 1e-4, out


def test_all_parameters_at_once_costs_one_adjoint(dcase):
    """The whole point of the adjoint: cost is independent of the count."""
    names = ("area_ratio", "throat_x", "theta_exit", "inlet_half_height", "length")
    _, adjoint, _ = dcase.value_and_gradient("thrust", names=names)
    fd = finite_difference_gradient(dcase, "thrust", names=names, step=1e-4)
    for name in names:
        scale = max(abs(adjoint[name]), abs(fd[name]), 1e-30)
        assert abs(adjoint[name] - fd[name]) / scale < 1e-4, (name, adjoint[name], fd[name])


@pytest.mark.parametrize("objective", list(OBJECTIVES))
def test_every_objective_is_differentiable(dcase, objective):
    value, grad, _ = dcase.value_and_gradient(objective, names=("area_ratio",))
    assert np.isfinite(value)
    assert np.isfinite(grad["area_ratio"])


def test_back_pressure_gradient_vanishes_at_a_supersonic_exit(dcase):
    """No characteristic enters, so back pressure cannot influence the nozzle.

    A gradient of exactly zero here is physics, not a failure to propagate.
    """
    _, grad, result = dcase.value_and_gradient("thrust", names=("back_pressure",))
    from dgnozzle.postprocess import performance

    assert performance(result).exit_mach_area_averaged > 1.0
    assert abs(grad["back_pressure"]) < 1e-12


def test_gradient_sign_is_physical(dcase):
    """Thrust must rise with area ratio at this over-expanded operating point."""
    _, grad, _ = dcase.value_and_gradient("thrust", names=("area_ratio",))
    assert grad["area_ratio"] > 0.0


def test_forward_solve_uses_the_frozen_logical_grid(dcase):
    """Otherwise the mesh clustering moves with throat_x and the finite
    difference measures node motion as well as the design change."""
    a = dcase.case_at({"throat_x": 0.10})
    b = dcase.case_at({"throat_x": 0.25})
    assert np.allclose(a.grids["x_frac"], b.grids["x_frac"])
    assert not np.allclose(a.node_coords, b.node_coords)


def test_a_failed_forward_solve_refuses_to_produce_a_gradient():
    bad = differentiable_case(
        contour="smooth", order=1, refine=0, back_pressure_ratio=0.15,
        cfl=90.0, max_iterations=500,
    )
    with pytest.raises(RuntimeError, match="did not converge"):
        bad.value_and_gradient("thrust", names=("area_ratio",))
