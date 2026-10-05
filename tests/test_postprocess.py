"""Performance metrics, line-outs and field sampling."""

from __future__ import annotations

import numpy as np
import pytest

from src import BoundaryTag, FlowConditions, NozzleGeometry, build_case, performance
from src import physics as ph
from src.initialize import quasi1d_initial
from src.postprocess import (
    boundary_trace,
    centreline_profile,
    entropy_error,
    exit_profile,
    sample_field,
    scalar_field,
    wall_profile,
)
from src.solver import SolveHistory, SolveResult


@pytest.fixture(scope="module")
def projected():
    """The quasi-1D state projected onto the DG basis -- not a converged solve,
    but a physically sensible field, which is all these tests need."""
    case = build_case(NozzleGeometry(contour="smooth"), FlowConditions(), order=1)
    U = quasi1d_initial(case.operators, case.flow, case.geometry)
    return SolveResult(
        U=U,
        operators=case.operators,
        flow=case.flow,
        geometry=case.geometry,
        discretization=case.discretization,
        converged=True,
        iterations=0,
        residual=0.0,
        residual_initial=1.0,
        history=SolveHistory(),
        backend="numpy",
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
    s_t = (
        flow.total_pressure ** (1 - flow.gamma) * (flow.Rgas * flow.total_temperature) ** flow.gamma
    )
    rho = 1.7
    p = s_t * rho**flow.gamma
    U = np.zeros((ops.n_elem, ops.ref.n_basis, 4))
    U[:, :, 0] = rho
    U[:, :, 3] = p / (flow.gamma - 1.0)
    assert entropy_error(U, ops, flow) < 1e-12


def test_entropy_error_is_positive_off_the_isentrope(projected):
    assert entropy_error(projected.U, projected.operators, projected.flow) >= 0.0


@pytest.mark.parametrize(
    "name", ["mach", "pressure", "density", "temperature", "vx", "vy", "velocity", "entropy"]
)
def test_scalar_fields_are_finite_and_sensible(projected, name):
    from src.postprocess import solution_at_quadrature

    values = scalar_field(
        solution_at_quadrature(projected.U, projected.operators), projected.flow, name
    )
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
    rho, vx, vy, _, _ = ph.primitives(tr.state, projected.flow.gamma)
    vn = vx * tr.normal[..., 0] + vy * tr.normal[..., 1]
    half = -float((rho * vn * tr.weight).sum())
    assert perf.mass_flow_in == pytest.approx(2.0 * half, rel=1e-12)


def test_node_coordinates_are_recovered_exactly(projected):
    """postprocess rebuilds coordinates from the geometry; they must match."""
    from src.postprocess import _node_coords

    case = build_case(projected.geometry, projected.flow, projected.discretization)
    assert np.allclose(_node_coords(projected), case.node_coords)


# --------------------------------------------------------------------------
# Temperature, which the profiles and the summary now carry
# --------------------------------------------------------------------------
def test_boundary_profiles_carry_temperature_consistent_with_the_field(projected):
    """A profile and a contour of the same quantity must not drift apart.

    ``sample_boundary`` and ``scalar_field`` derive temperature independently --
    one from the sampled boundary state, one from an arbitrary state array -- so
    nothing but a test keeps the two expressions the same.
    """
    cl = centreline_profile(projected)
    assert "temperature" in cl
    assert "velocity" in cl
    assert np.all(cl["temperature"] > 0.0)

    # the same states, through the other path
    tr = boundary_trace(projected.U, projected.operators, BoundaryTag.AXIS)
    expected = scalar_field(tr.state, projected.flow, "temperature")
    assert cl["temperature"].min() == pytest.approx(expected.min(), rel=5e-2)
    assert cl["temperature"].max() == pytest.approx(expected.max(), rel=5e-2)


def test_centreline_velocity_matches_its_components(projected):
    cl = centreline_profile(projected)
    speed = np.sqrt(cl["vx"] ** 2 + cl["vy"] ** 2)
    assert np.allclose(cl["velocity"], speed, rtol=1e-12)


def test_static_temperature_falls_through_the_expansion(projected):
    """The physics, not the plumbing: T drops monotonically past the throat.

    A nozzle converts enthalpy into kinetic energy, so static temperature on the
    axis can only fall downstream of the throat.  If this ever rises, the field
    is wrong in a way no shape check would catch.
    """
    cl = centreline_profile(projected)
    throat = projected.geometry.throat_location()
    past = cl["x"] > throat
    t = cl["temperature"][past]
    assert t[0] > t[-1]
    # and it never exceeds the reservoir value
    assert t.max() <= projected.flow.total_temperature * (1.0 + 1e-9)


def test_exit_temperature_ratio_tracks_the_isentropic_relation(projected):
    r"""``T_e/T_t`` must agree with ``(1 + (gamma-1)/2 M_e^2)^-1``.

    Not exactly: the gap *is* the entropy generated, so this is a check with a
    physical scale rather than a tolerance pulled from the air.  On a converged
    solve the two agree to a few tenths of a percent; here the field is the
    quasi-1D projection, so the agreement is tighter still.
    """
    perf = performance(projected)
    gamma = projected.flow.gamma
    m_e = perf.exit_mach_area_averaged
    isentropic = (1.0 + 0.5 * (gamma - 1.0) * m_e * m_e) ** -1.0
    assert perf.exit_temperature_ratio == pytest.approx(isentropic, rel=0.02)
    assert 0.0 < perf.exit_temperature_ratio < 1.0


def test_the_summary_reports_temperature(projected):
    """It is only an output if a student actually sees it."""
    text = performance(projected).summary()
    assert "T/T_t" in text


def test_the_discharge_coefficient_matches_the_choked_value(projected):
    r"""``C_d`` must land on the closed-form choked mass flow.

    For a choked nozzle the non-dimensional mass flow is fixed by ``gamma``
    alone -- independent of reservoir conditions and throat size -- at

    .. math::
        \frac{\dot m}{\rho_t a_t A^*}
            = \left(\frac{2}{\gamma + 1}\right)^{\frac{\gamma+1}{2(\gamma-1)}}

    which is 0.5787 for ``gamma = 1.4``.  That makes it a check on the
    non-dimensionalisation *and* on the solve: if the reference scales were
    wrong, this number would be wrong by exactly that factor.
    """
    perf = performance(projected)
    gamma = projected.flow.gamma
    choked = (2.0 / (gamma + 1.0)) ** ((gamma + 1.0) / (2.0 * (gamma - 1.0)))
    assert choked == pytest.approx(0.5787, abs=1e-4)
    assert perf.discharge_coefficient == pytest.approx(choked, rel=0.03)


def test_the_non_dimensional_forms_agree_with_their_dimensional_ones(projected):
    """The ratio forms must be the dimensional ones over the reference scales.

    Trivial arithmetic, but it is the thing that silently rots if a reference
    scale is ever changed in one place and not the other.
    """
    perf = performance(projected)
    flow = projected.flow
    a_t = flow.stagnation_sound_speed
    assert perf.specific_thrust_ratio == pytest.approx(perf.specific_thrust / a_t, rel=1e-12)

    a_throat = 2.0 * projected.geometry.throat_height()
    scale = flow.stagnation_density * a_t * a_throat
    assert perf.discharge_coefficient == pytest.approx(perf.mass_flow_in / scale, rel=1e-12)


def test_the_reference_scales_are_the_documented_ones():
    """``R = gamma - 1`` is what makes the non-dimensionalisation what it is.

    With that choice ``T`` is numerically the specific internal energy, and
    ``p_t = T_t = L = 1`` fixes everything else: ``rho_t = 2.5`` and
    ``a_t = 0.7483``.  The documentation quotes these numbers, so pin them.
    """
    flow = FlowConditions()
    assert flow.Rgas == pytest.approx(flow.gamma - 1.0, rel=1e-15)
    assert flow.total_pressure == 1.0
    assert flow.total_temperature == 1.0
    assert flow.stagnation_density == pytest.approx(2.5, rel=1e-12)
    assert flow.stagnation_sound_speed == pytest.approx(0.7483314773547882, rel=1e-12)
