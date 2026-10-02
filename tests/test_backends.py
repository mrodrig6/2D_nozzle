"""All three backends must compute the same residual."""

from __future__ import annotations

import numpy as np
import pytest

from dgnozzle import NozzleGeometry, build_case
from dgnozzle import assembly as asm
from dgnozzle.backends import available_backends, get_backend
from dgnozzle.config import FlowConditions, SolverOptions


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
    once in :class:`~dgnozzle.backends.base.Backend` for NumPy and JAX, once in
    kernels here.  Ten steps is long enough that a wrong coefficient or a stale
    buffer shows up; `fastmath` reassociation keeps it from being exact.
    """
    from dgnozzle import initialize

    case, _ = setup
    ops, flow = case.operators, case.flow
    opts = SolverOptions(limiter=limiter, scheme=scheme)
    # the perturbed state of `setup` oscillates within elements, and ten
    # unlimited steps of it diverge in both backends -- which agrees, but
    # compares NaN to NaN.  Start from the real initial condition instead.
    U = np.ascontiguousarray(
        initialize.initial_state(ops, flow, case.geometry, "quasi1d")
    )

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
    :func:`dgnozzle.physics.hllc_flux`, including the low-Mach branch and every
    guarded division; nothing but a test keeps them in step.
    """
    import numpy as np

    from dgnozzle.api import solve_nozzle

    a = solve_nozzle(order=1, refine=0, flux="hllc", backend="numpy",
                     verbose=False)
    b = solve_nozzle(order=1, refine=0, flux="hllc", backend="numba",
                     verbose=False)
    assert a.iterations == b.iterations
    assert np.abs(a.U - b.U).max() / np.abs(a.U).max() < 1e-12
