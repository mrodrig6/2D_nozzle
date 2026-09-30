"""Performance metrics, line-outs and field sampling."""

from __future__ import annotations

import numpy as np
import pytest

from dgnozzle import BoundaryTag, FlowConditions, NozzleGeometry, build_case, performance
from dgnozzle import physics as ph
from dgnozzle.initialize import quasi1d_initial
from dgnozzle.postprocess import (
    boundary_trace,
    centreline_profile,
    entropy_error,
    exit_profile,
    sample_field,
    scalar_field,
    wall_profile,
)
from dgnozzle.solver import SolveHistory, SolveResult


@pytest.fixture(scope="module")
def projected():
    """The quasi-1D state projected onto the DG basis -- not a converged solve,
    but a physically sensible field, which is all these tests need."""
    case = build_case(NozzleGeometry(contour="smooth"), FlowConditions(), order=1)
    U = quasi1d_initial(case.operators, case.flow, case.geometry)
    return SolveResult(
        U=U, operators=case.operators, flow=case.flow, geometry=case.geometry,
        discretization=case.discretization, converged=True, iterations=0,
        residual=0.0, residual_initial=1.0, history=SolveHistory(), backend="numpy",
    )


def test_boundary_traces_have_the_right_shapes(projected):
    ops = projected.operators
    for tag in BoundaryTag:
        tr = boundary_trace(projected.U, ops, tag)
        n = ops.topology.tag_slice(tag).stop - ops.topology.tag_slice(tag).start
        assert tr.state.shape == (n, ops.ref.n_qface, 4)
        assert tr.weight.shape == (n, ops.ref.n_qface)
        assert np.all(tr.weight > 0.0)
        assert np.allclose(np.linalg.norm(tr.normal, axis=-1), 1.0)


def test_exit_area_matches_the_geometry(projected):
    tr = boundary_trace(projected.U, projected.operators, BoundaryTag.OUTFLOW)
    exit_height = float(projected.geometry.wall(np.asarray([projected.geometry.length]))[0])
    assert float(tr.weight.sum()) == pytest.approx(exit_height, rel=1e-6)


def test_entropy_error_is_zero_for_an_isentropic_field():
    """Build a field with exactly the reservoir entropy; the measure must vanish."""
    case = build_case(NozzleGeometry(contour="smooth"), FlowConditions(), order=1)
    flow = case.flow
    ops = case.operators
    s_t = (flow.total_pressure ** (1 - flow.gamma)
           * (flow.Rgas * flow.total_temperature) ** flow.gamma)
    rho = 1.7
    p = s_t * rho**flow.gamma
    U = np.zeros((ops.n_elem, ops.ref.n_basis, 4))
    U[:, :, 0] = rho
    U[:, :, 3] = p / (flow.gamma - 1.0)
    assert entropy_error(U, ops, flow) < 1e-12


def test_entropy_error_is_positive_off_the_isentrope(projected):
    assert entropy_error(projected.U, projected.operators, projected.flow) >= 0.0


@pytest.mark.parametrize("name", ["mach", "pressure", "density", "temperature",
                                  "u", "v", "velocity", "entropy"])
def test_scalar_fields_are_finite_and_sensible(projected, name):
    from dgnozzle.postprocess import solution_at_quadrature

    values = scalar_field(solution_at_quadrature(projected.U, projected.operators),
                          projected.flow, name)
    assert np.all(np.isfinite(values))
    if name in ("mach", "pressure", "density", "temperature", "velocity", "entropy"):
        assert np.all(values >= 0.0)


def test_unknown_scalar_is_rejected(projected):
    with pytest.raises(ValueError, match="unknown scalar"):
        scalar_field(projected.U[:, :1, :], projected.flow, "vorticity")


def test_profiles_are_sorted_and_span_the_domain(projected):
    ex = exit_profile(projected)
    assert np.all(np.diff(ex["y"]) >= 0)
    assert ex["y"].min() == pytest.approx(0.0, abs=1e-9)

    cl = centreline_profile(projected)
    assert np.all(np.diff(cl["x"]) >= 0)
    assert cl["x"].min() == pytest.approx(0.0, abs=1e-9)
    assert cl["x"].max() == pytest.approx(projected.geometry.length, abs=1e-9)

    wp = wall_profile(projected)
    assert np.all(np.diff(wp["x"]) >= 0)
    assert np.all(wp["y"] > 0.0)


def test_sampled_field_covers_the_domain(projected):
    points, triangles, values = sample_field(projected, "mach", subdivisions=2)
    assert points.shape[1] == 2
    assert triangles.shape[1] == 3
    assert values.shape[0] == points.shape[0]
    assert triangles.max() < points.shape[0]
    assert np.all(np.isfinite(values))
    assert points[:, 0].min() == pytest.approx(0.0, abs=1e-9)
    assert points[:, 0].max() == pytest.approx(projected.geometry.length, abs=1e-9)


def test_performance_reports_full_nozzle_quantities(projected):
    """The mesh is a half channel; integral quantities must be doubled."""
    perf = performance(projected)
    tr = boundary_trace(projected.U, projected.operators, BoundaryTag.INFLOW)
    rho, u, v, _, _ = ph.primitives(tr.state, projected.flow.gamma)
    un = u * tr.normal[..., 0] + v * tr.normal[..., 1]
    half = -float((rho * un * tr.weight).sum())
    assert perf.mass_flow_in == pytest.approx(2.0 * half, rel=1e-12)


def test_node_coordinates_are_recovered_exactly(projected):
    """postprocess rebuilds coordinates from the geometry; they must match."""
    from dgnozzle.postprocess import _node_coords

    case = build_case(projected.geometry, projected.flow,
                      projected.discretization)
    assert np.allclose(_node_coords(projected), case.node_coords)
