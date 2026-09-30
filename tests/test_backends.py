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
