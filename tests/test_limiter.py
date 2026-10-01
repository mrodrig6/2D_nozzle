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


@pytest.mark.numba
def test_the_cheap_screen_never_changes_the_answer(case):
    """The limiter's O(nbf) screen is sufficient, not necessary.

    It may be pessimistic -- that only costs a probe -- but it must never
    declare an inadmissible element admissible.  A field of marginal states
    exercises both branches in the same call: most elements pass the screen, some
    fail it and pass the exact probe, and some are genuinely limited.
    """
    from dgnozzle.backends import get_backend
    from dgnozzle.config import SolverOptions

    ops, flow = case.operators, case.flow
    bk = get_backend("numba", ops, flow, SolverOptions(limiter="positivity"))
    rng = np.random.default_rng(7)

    # scale the deviation from tiny (screen passes) up to large (limiter acts)
    for amplitude in (1e-6, 0.1, 0.5, 0.9, 1.5):
        mean = np.array([2.0, 0.6, 0.0, 5.0])
        U = np.broadcast_to(mean, (ops.n_elem, ops.ref.n_basis, 4)).copy()
        U += amplitude * mean * rng.standard_normal(U.shape)
        reference = lim.apply_limiter(U, ops, flow, "positivity")
        got = bk.limit(U)
        # Relative, not absolute: both implementations bisect the pressure bound
        # twelve times, and an element whose pressure sits on the threshold can
        # take the other branch on one step from round-off alone.  At amplitude
        # 0.9 exactly one element of 140 does, differing by 7e-10 where the rest
        # agree to 2e-15.  That is the bisection's own resolution showing
        # through, not the screen admitting something it should not.
        err = np.abs(got - reference).max() / np.abs(reference).max()
        assert err < 1e-12, (amplitude, err)


@pytest.mark.numba
def test_the_lebesgue_constant_bounds_the_probe_deviation(case):
    r"""``Lambda = max_x sum_i |phi_i(x)|`` is what makes the screen valid.

    The screen bounds a probe-point value by its cell mean plus
    ``Lambda * max_i |U_i - Ubar|``.  If the constant were understated the bound
    would not hold and an inadmissible element could be skipped, so check it
    directly against every probe point the kernel ever evaluates.
    """
    from dgnozzle.backends import get_backend
    from dgnozzle.config import SolverOptions

    ops = case.operators
    bk = get_backend("numba", ops, case.flow, SolverOptions())
    rng = np.random.default_rng(11)
    U = rng.standard_normal((ops.n_elem, ops.ref.n_basis, 4))

    mean = np.einsum("ei,eis->es", np.asarray(ops.mean_weights), U)
    dev = np.abs(U - mean[:, None, :]).max(axis=1)

    phi_vol = np.asarray(ops.ref.phi_vol)
    probes = [np.einsum("iq,eis->eqs", phi_vol, U)]
    side = ops.topology.edges.face_side
    phi_face = np.asarray(ops.ref.phi_face)
    for f in range(ops.topology.n_faces):
        basis = phi_face[side[:, f], f]  # (nelem, nbf, nqf)
        probes.append(np.einsum("eiq,eis->eqs", basis, U))
    values = np.concatenate(probes, axis=1)  # (nelem, n_probe, 4)

    bound = bk._lebesgue * dev[:, None, :]
    assert np.all(np.abs(values - mean[:, None, :]) <= bound + 1e-12)
