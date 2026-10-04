"""The limiter must restore positivity, preserve cell averages, and stay inactive
on smooth solutions."""

from __future__ import annotations

import numpy as np
import pytest

from src import NozzleGeometry, build_case
from src import initialize as ini
from src import limiter as lim
from src.config import FlowConditions


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


@pytest.mark.parametrize("kind", ["positivity", "superbee"])
def test_limiter_is_inactive_on_a_smooth_state(case, smooth_state, kind):
    """Accuracy preservation: a limiter that fires on smooth flow destroys the order."""
    out = lim.apply_limiter(smooth_state, case.operators, case.flow, kind)
    assert np.abs(out - smooth_state).max() == 0.0


@pytest.mark.parametrize("kind", ["positivity", "superbee"])
def test_limiter_restores_positivity(case, smooth_state, kind):
    ops, flow = case.operators, case.flow
    U = smooth_state.copy()
    U[3, 0, 0] = -0.5  # density overshoot
    U[7, 1, 3] = -1.0  # energy overshoot -> negative pressure
    assert lim.diagnose(U, ops, flow).min_pressure < 0.0

    out = lim.apply_limiter(U, ops, flow, kind)
    diag = lim.diagnose(out, ops, flow)
    assert diag.min_density > 0.0
    assert diag.min_pressure > 0.0


@pytest.mark.parametrize("kind", ["positivity", "superbee"])
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


def test_superbee_bounds_the_slope_by_the_neighbour_jump(case, smooth_state):
    """The limited increment must be what Superbee allows, face by face.

    This is the pure TVD property, so it is tested with ``tvb_constant = 0``.
    The shipped default deliberately *exempts* increments below the TVB
    threshold from this bound -- that exemption is the subject of
    :func:`test_the_tvb_threshold_switches_the_limiter_off`, and asserting the
    bound with it switched on would be asserting the opposite of what the
    threshold is for.
    """
    ops, flow = case.operators, case.flow
    rng = np.random.default_rng(3)
    U = smooth_state + 0.4 * rng.standard_normal(smooth_state.shape)

    out = lim.superbee_limiter(U, ops, flow, tvb_constant=0.0)
    means = lim.cell_means(out, ops)
    nb = ops.topology.edges.face_neighbour
    interior = nb != np.arange(ops.n_elem)[:, None]

    jump = means[nb] - means[:, None, :]
    dev = lim.face_trace_means(out, ops) - means[:, None, :]
    allowed = lim.superbee(jump, dev)
    # on an interior face the surviving increment may not exceed what Superbee
    # admits for that jump; a slack of 1e-9 covers the single scaling factor
    assert np.all(np.abs(dev[interior]) <= np.abs(allowed[interior]) + 1e-9)


def test_superbee_scales_every_component_by_the_same_factor(case, smooth_state):
    """The limited state must stay on the segment from ``U`` to its cell mean.

    This is what keeps the slope limiter from handing the positivity limiter an
    inadmissible state.  ``{rho > 0, p > 0}`` is convex, so any point on that
    segment is admissible when both ends are; a per-component factor leaves the
    segment and can land outside.  Measured, a per-component factor drove the
    cell average non-physical 10,246 times on the shocked case where the scalar
    factor needed no repairs at all.
    """
    ops, flow = case.operators, case.flow
    rng = np.random.default_rng(11)
    U = smooth_state + 0.5 * rng.standard_normal(smooth_state.shape)

    out = lim.superbee_limiter(U, ops, flow, tvb_constant=0.0)
    ubar = lim.cell_means(U, ops)
    dev, dev_lim = U - ubar[:, None, :], out - ubar[:, None, :]

    # recover theta from every entry with a deviation worth dividing by, and
    # check they all agree -- element by element, across modes and components
    big = np.abs(dev) > 1e-8
    assert big.sum() > 100, "test is vacuous"
    ratio = np.where(big, dev_lim / np.where(big, dev, 1.0), np.nan)
    for e in range(ops.n_elem):
        vals = ratio[e][big[e]]
        if vals.size:
            assert np.ptp(vals) < 1e-10, f"element {e} scaled unevenly"

    # and the limiter must actually have been doing something
    assert np.nanmin(ratio) < 0.999


def test_superbee_is_more_compressive_than_minmod(case, smooth_state):
    """Superbee's whole character: it permits twice what minmod does.

    Both sit on the edges of Sweby's TVD region -- minmod on the lower, Superbee
    on the upper -- so Superbee must never limit *more* than minmod, and on a
    steep face it must limit strictly less.
    """
    rng = np.random.default_rng(5)
    a = rng.standard_normal(2000)
    b = rng.standard_normal(2000)
    sb = lim.superbee(a, b)
    mm = lim._minmod(a, b)
    same_sign = a * b > 0
    assert np.all(np.abs(sb[same_sign]) >= np.abs(mm[same_sign]) - 1e-12)
    assert np.all(np.sign(sb[same_sign]) == np.sign(mm[same_sign]))
    # opposite signs: both must kill the increment outright
    assert np.all(sb[~same_sign] == 0.0)
    # and the textbook values
    assert lim.superbee(np.array(1.0), np.array(3.0)) == pytest.approx(2.0)
    assert lim._minmod(np.array(1.0), np.array(3.0)) == pytest.approx(1.0)


def test_the_tvb_threshold_switches_the_limiter_off(case, smooth_state):
    """With a large enough ``M`` nothing is limited, which is the point of it.

    A pure TVD limiter stays marginally active at smooth extrema, clipping them
    on some iterations and not others; that is what parks a steady residual in a
    limit cycle.
    """
    ops, flow = case.operators, case.flow
    rng = np.random.default_rng(7)
    U = smooth_state + 0.2 * rng.standard_normal(smooth_state.shape)

    active = lim.superbee_limiter(U, ops, flow, tvb_constant=0.0)
    assert np.abs(active - U).max() > 1e-6, "nothing to limit; test is vacuous"

    off = lim.superbee_limiter(U, ops, flow, tvb_constant=1e9)
    # not bit-identical: the limiter still evaluates `ubar + 1 * (U - ubar)`,
    # which round-trips through the cell average.  Round-off, not limiting.
    assert np.abs(off - U).max() < 1e-14 * np.abs(U).max()


def test_superbee_leaves_a_linear_field_alone(case):
    """A TVD limiter must not touch a field it has no reason to touch.

    On a globally linear field every interior jump agrees in sign and magnitude
    with the element's own increment, so Superbee admits it in full.  What makes
    this the sharp test is the *boundary*: ``face_neighbour`` reports an element
    as its own neighbour there, so the jump across a boundary face is identically
    zero.  Limiting against it would drive theta to zero in precisely the
    elements that carry the wall and the exit plane -- a silent, severe loss of
    accuracy exactly where thrust is integrated -- and every boundary element
    would come back flattened.
    """
    ops, flow = case.operators, case.flow
    xy = np.asarray(ops.xy_vol)
    base = np.array([2.0, 0.6, 0.05, 5.0])
    slope = np.array([0.7, -0.3, 0.2, 1.1])
    values = base + slope * (xy[..., 0] - 0.5)[..., None]
    U = ini.project(values, ops)

    out = lim.superbee_limiter(U, ops, flow)
    assert np.abs(out - U).max() < 1e-12 * np.abs(U).max()


@pytest.mark.numba
def test_numba_limiter_matches_the_numpy_one(case, smooth_state):
    from src.backends import get_backend
    from src.config import SolverOptions

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
    from src.backends import get_backend
    from src.config import SolverOptions

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
    from src.backends import get_backend
    from src.config import SolverOptions

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
