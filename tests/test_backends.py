"""All three backends must compute the same residual."""

from __future__ import annotations

import numpy as np
import pytest

from src import NozzleGeometry, build_case
from src import assembly as asm
from src.backends import available_backends, get_backend
from src.config import FlowConditions, SolverOptions


@pytest.fixture(scope="module")
def setup():
    case = build_case(NozzleGeometry(contour="smooth"), FlowConditions(), order=1)
    rng = np.random.default_rng(0)
    base = np.array([2.0, 0.6, 0.02, 5.0])
    U = np.ascontiguousarray(
        base * (1.0 + 0.03 * rng.standard_normal((case.operators.n_elem, 3, 4)))
    )
    return case, U


@pytest.mark.parametrize("backend", ["numba", "jax"])
def test_residual_matches_the_numpy_reference(setup, backend, request):
    if backend not in available_backends():
        pytest.skip(f"{backend} not installed")
    request.node.add_marker(getattr(pytest.mark, backend))
    case, U = setup
    ops, flow = case.operators, case.flow
    opts = SolverOptions(limiter="positivity")

    reference, wave_ref = asm.residual(U, ops, flow, xp=np)
    bk = get_backend(backend, ops, flow, opts)
    R, wave = bk.residual(bk.asarray(U))

    R = np.asarray(R)
    wave = np.asarray(wave)
    assert np.abs(R - reference).max() / np.abs(reference).max() < 1e-10
    assert np.abs(wave - wave_ref).max() / np.abs(wave_ref).max() < 1e-10


@pytest.mark.parametrize("backend", ["numba", "jax"])
def test_inverse_mass_matches(setup, backend):
    if backend not in available_backends():
        pytest.skip(f"{backend} not installed")
    case, U = setup
    ops, flow = case.operators, case.flow
    R, _ = asm.residual(U, ops, flow, xp=np)
    reference = asm.apply_inverse_mass(R, ops, xp=np)
    bk = get_backend(backend, ops, flow, SolverOptions())
    out = np.asarray(bk.inverse_mass(bk.asarray(R)))
    assert np.abs(out - reference).max() < 1e-11


@pytest.mark.parametrize("backend", ["numba", "jax"])
@pytest.mark.parametrize("scheme", ["rk4", "ssprk3"])
def test_one_step_matches_the_reference(setup, backend, scheme):
    if backend not in available_backends():
        pytest.skip(f"{backend} not installed")
    case, U = setup
    ops, flow = case.operators, case.flow
    opts = SolverOptions(limiter="positivity", scheme=scheme)

    ref_bk = get_backend("numpy", ops, flow, opts)
    ref, _ = ref_bk.run(U.copy(), 3, scheme)

    bk = get_backend(backend, ops, flow, opts)
    out, _ = bk.run(bk.asarray(U.copy()), 3, scheme)

    assert np.abs(np.asarray(out) - ref).max() / max(np.abs(ref).max(), 1e-30) < 1e-9


def test_unknown_backend_is_rejected(setup):
    case, _ = setup
    with pytest.raises(ValueError, match="unknown backend"):
        get_backend("cuda", case.operators, case.flow, SolverOptions())


@pytest.mark.numba
@pytest.mark.parametrize("scheme", ["rk4", "ssprk3"])
@pytest.mark.parametrize("limiter", ["none", "positivity"])
def test_the_fused_numba_step_matches_the_shared_one(setup, scheme, limiter):
    """The Numba backend reimplements both schemes; they must not drift.

    It overrides ``run`` to keep every stage inside preallocated buffers, which
    means the arithmetic of ``rk4_step`` and ``ssprk3_step`` exists twice --
    once in :class:`~src.backends.base.Backend` for NumPy and JAX, once in
    kernels here.  Ten steps is long enough that a wrong coefficient or a stale
    buffer shows up; `fastmath` reassociation keeps it from being exact.
    """
    from src import initialize

    case, _ = setup
    ops, flow = case.operators, case.flow
    opts = SolverOptions(limiter=limiter, scheme=scheme)
    # the perturbed state of `setup` oscillates within elements, and ten
    # unlimited steps of it diverge in both backends -- which agrees, but
    # compares NaN to NaN.  Start from the real initial condition instead.
    U = np.ascontiguousarray(initialize.initial_state(ops, flow, case.geometry, "quasi1d"))

    ref, res_ref = get_backend("numpy", ops, flow, opts).run(U.copy(), 10, scheme)
    out, res = get_backend("numba", ops, flow, opts).run(U.copy(), 10, scheme)

    assert np.abs(np.asarray(out) - ref).max() / np.abs(ref).max() < 1e-9
    assert abs(res - res_ref) / abs(res_ref) < 1e-9


@pytest.mark.numba
def test_reusing_the_buffers_does_not_leak_between_runs(setup):
    """Every array is preallocated and shared, so a stale one would show here."""
    case, U = setup
    ops, flow = case.operators, case.flow
    opts = SolverOptions(limiter="positivity")
    bk = get_backend("numba", ops, flow, opts)

    once, _ = bk.run(U.copy(), 6, "rk4")
    # the same march in two chunks must land in the same place
    half, _ = bk.run(U.copy(), 3, "rk4")
    twice, _ = bk.run(half, 3, "rk4")
    assert np.abs(np.asarray(twice) - np.asarray(once)).max() < 1e-13

    # and the returned array must not alias a buffer the next run overwrites
    kept = np.asarray(once).copy()
    bk.run(U.copy(), 4, "rk4")
    assert np.abs(np.asarray(once) - kept).max() == 0.0


def test_the_numba_hllc_kernel_matches_the_shared_one():
    """Two copies of a flux is two chances to get it wrong, so pin them.

    ``_numba_kernels._hllc`` is a hand transcription of
    :func:`src.physics.hllc_flux`, including the low-Mach branch and every
    guarded division; nothing but a test keeps them in step.
    """
    import numpy as np

    from src.api import solve_nozzle

    a = solve_nozzle(order=1, refine=0, flux="hllc", backend="numpy", verbose=False)
    b = solve_nozzle(order=1, refine=0, flux="hllc", backend="numba", verbose=False)
    assert a.iterations == b.iterations
    assert np.abs(a.U - b.U).max() / np.abs(a.U).max() < 1e-12


def test_the_numba_slau2_kernel_matches_the_shared_one():
    """``_numba_kernels._slau2`` is a hand transcription of the shared flux.

    It matters more here than for the other two: the shared SLAU2 was itself
    transcribed from the literature rather than from the paper, so a divergence
    between the two copies would be a second error on top of a first.
    """
    import numpy as np

    from src.api import solve_nozzle

    a = solve_nozzle(order=1, refine=0, flux="slau2", backend="numpy", verbose=False)
    b = solve_nozzle(order=1, refine=0, flux="slau2", backend="numba", verbose=False)
    assert a.iterations == b.iterations
    assert np.abs(a.U - b.U).max() / np.abs(a.U).max() < 1e-12


def test_the_folded_rate_kernel_matches_the_unfolded_one():
    """``element_rate`` must be the exact same rate, not merely a close one.

    ``M**-1`` is folded into the operators at build time, so the two paths do the
    same arithmetic in a different order.  Nothing but a test keeps them in step,
    and the fast path is the one the march actually uses -- a drift here would be
    invisible everywhere except the answer.
    """
    import numpy as np

    from src import Discretization
    from src.api import build_case
    from src.backends import _numba_kernels as nk
    from src.backends import get_backend
    from src.config import SolverOptions
    from src.initialize import initial_state

    for order in (0, 1, 2):
        cs = build_case(discretization=Discretization(order=order, refine=0))
        ops, flow, geom = cs.operators_at(order), cs.flow, cs.geometry
        bk = get_backend("numba", ops, flow, SolverOptions())
        U = np.ascontiguousarray(initial_state(ops, flow, geom, "quasi1d"))

        folded = np.empty_like(U)
        bk._rate_into(U, folded)

        unfolded = np.empty_like(U)
        bk._edges(U)
        nk.element_pass(
            U,
            bk._fw,
            bk._smax,
            bk._phi_vol,
            bk._grad_x,
            bk._grad_y,
            bk._phi_face,
            bk._face_edge,
            bk._face_side,
            bk._face_sign,
            bk._edge_length,
            bk._gamma,
            bk._inv_mass,
            True,
            unfolded,
            bk._wave,
        )
        scale = max(np.abs(unfolded).max(), 1e-30)
        assert np.abs(folded - unfolded).max() / scale < 1e-12, f"order {order}"
