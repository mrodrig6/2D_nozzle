"""The limiter must restore positivity, preserve cell averages, and stay inactive
on smooth solutions."""

from __future__ import annotations

import numpy as np
import pytest

from dgnozzle import NozzleGeometry, build_case
from dgnozzle import limiter as lim
from dgnozzle.config import FlowConditions


@pytest.fixture(scope="module")
def case():
    return build_case(NozzleGeometry(contour="smooth"), FlowConditions(), order=1)


@pytest.fixture
def smooth_state(case):
    ops = case.operators
    U = np.zeros((ops.n_elem, ops.ref.n_basis, 4))
    U[:, :, 0] = 2.0
    U[:, :, 1] = 0.6
    U[:, :, 3] = 5.0
    return U


@pytest.mark.parametrize("kind", ["positivity", "barth-jespersen"])
def test_limiter_is_inactive_on_a_smooth_state(case, smooth_state, kind):
    """Accuracy preservation: a limiter that fires on smooth flow destroys the order."""
    out = lim.apply_limiter(smooth_state, case.operators, case.flow, kind)
    assert np.abs(out - smooth_state).max() == 0.0


@pytest.mark.parametrize("kind", ["positivity", "barth-jespersen"])
def test_limiter_restores_positivity(case, smooth_state, kind):
    ops, flow = case.operators, case.flow
    U = smooth_state.copy()
    U[3, 0, 0] = -0.5   # density overshoot
    U[7, 1, 3] = -1.0   # energy overshoot -> negative pressure
    assert lim.diagnose(U, ops, flow).min_pressure < 0.0

    out = lim.apply_limiter(U, ops, flow, kind)
    diag = lim.diagnose(out, ops, flow)
    assert diag.min_density > 0.0
    assert diag.min_pressure > 0.0


@pytest.mark.parametrize("kind", ["positivity", "barth-jespersen"])
def test_limiter_is_conservative(case, smooth_state, kind):
    """Scaling the deviation from the mean must leave the mean untouched."""
    ops, flow = case.operators, case.flow
    U = smooth_state.copy()
    U[3, 0, 0] = -0.5
    U[7, 1, 3] = -1.0
    before = lim.cell_means(U, ops)
    after = lim.cell_means(lim.apply_limiter(U, ops, flow, kind), ops)
    assert np.abs(after - before).max() < 1e-15


def test_limiter_survives_a_destroyed_cell_average(case, smooth_state):
    """No scaling can fix a negative mean, so the mean is floored instead of aborting."""
    ops, flow = case.operators, case.flow
    U = smooth_state.copy()
    U[3, :, 0] = -5.0
    out = lim.apply_limiter(U, ops, flow, "positivity")
    diag = lim.diagnose(out, ops, flow)
    assert diag.min_density > 0.0
    assert diag.min_pressure > 0.0


def test_limiter_is_a_no_op_at_p0():
    """A constant cannot overshoot its own mean."""
    case = build_case(NozzleGeometry(contour="smooth"), FlowConditions(), order=0)
    U = np.zeros((case.operators.n_elem, 1, 4))
    U[:, :, 0] = 1.0
    U[:, :, 3] = 2.0
    assert lim.apply_limiter(U, case.operators, case.flow, "positivity") is U


def test_barth_jespersen_bounds_by_neighbour_averages(case, smooth_state):
    """A spike must be pulled back into the range of its neighbours' means."""
    ops, flow = case.operators, case.flow
    U = smooth_state.copy()
    U[50, :, 0] = 20.0  # a cell far outside its neighbourhood
    out = lim.barth_jespersen_limiter(U, ops, flow)
    probes = lim.probe_values(out, ops)
    means = lim.cell_means(out, ops)
    nb = ops.topology.edges.face_neighbour
    upper = np.maximum(means, means[nb].max(axis=1))[:, 0]
    assert np.all(probes[..., 0].max(axis=1) <= upper + 1e-9)


@pytest.mark.numba
def test_numba_limiter_matches_the_numpy_one(case, smooth_state):
    from dgnozzle.backends import get_backend
    from dgnozzle.config import SolverOptions

    ops, flow = case.operators, case.flow
    bk = get_backend("numba", ops, flow, SolverOptions(limiter="positivity"))
    U = smooth_state.copy()
    U[3, 0, 0] = -0.5
    U[7, 1, 3] = -1.0
    U[20, 2, 0] = -0.2
    reference = lim.apply_limiter(U, ops, flow, "positivity")
    assert np.abs(bk.limit(U) - reference).max() < 1e-14
