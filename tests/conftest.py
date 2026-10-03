"""Shared fixtures.

Everything here is deliberately small: the unit tests must run in seconds so
they are actually run.  The handful of tests that need a converged solve are
marked ``slow``.
"""

from __future__ import annotations

import numpy as np
import pytest

from src import Discretization, FlowConditions, NozzleGeometry, SolverOptions, build_case


@pytest.fixture(scope="session")
def geom():
    return NozzleGeometry(contour="smooth")


@pytest.fixture(scope="session")
def flow():
    return FlowConditions(back_pressure_ratio=0.15)


@pytest.fixture(scope="session")
def tiny_case(geom, flow):
    """A 140-element p=1 case, built once for the whole session."""
    return build_case(geom, flow, Discretization(order=1, refine=0))


@pytest.fixture(scope="session")
def fast_options():
    return SolverOptions(print_interval=0, tolerance=1e-8, max_iterations=60_000)


@pytest.fixture
def random_state():
    """A physically admissible, spatially varying state for operator tests."""

    def make(ops, seed: int = 0):
        rng = np.random.default_rng(seed)
        base = np.array([2.0, 0.6, 0.02, 5.0])
        shape = (ops.n_elem, ops.ref.n_basis, 4)
        return np.ascontiguousarray(base * (1.0 + 0.03 * rng.standard_normal(shape)))

    return make


def pytest_addoption(parser):
    """``--fast`` and ``--slow`` instead of ``-m "not slow"``.

    The quoted marker expression is easy to get wrong and easy to get wrong
    *silently*: ``-m "not slow"`` with a stray space, a smart quote, or a shell
    that eats the quotes selects a different set of tests and still exits zero.
    A flag either exists or errors out.

        pytest --fast     the unit tests; seconds
        pytest --slow     the end-to-end solves and adjoint checks
        pytest            everything

    ``-m`` still works for anything finer.
    """
    group = parser.getgroup("dgnozzle")
    group.addoption(
        "--fast",
        action="store_true",
        default=False,
        help="run only the fast tests (equivalent to -m 'not slow')",
    )
    group.addoption(
        "--slow",
        action="store_true",
        default=False,
        help="run only the slow tests (equivalent to -m slow)",
    )


def pytest_configure(config):
    """Turn the flags into a marker expression, so they *deselect* rather than skip.

    Setting ``markexpr`` hands the work to pytest's own machinery, so the flags
    behave exactly as the equivalent ``-m`` would -- same counts, same summary
    line -- instead of reporting a pile of skips.
    """
    fast = config.getoption("--fast")
    slow = config.getoption("--slow")
    if fast and slow:
        raise pytest.UsageError("--fast and --slow select disjoint sets; pass neither to run both")
    if not (fast or slow):
        return
    wanted = "not slow" if fast else "slow"
    existing = config.option.markexpr
    # respect an explicit -m too, rather than silently dropping it
    config.option.markexpr = f"({existing}) and ({wanted})" if existing else wanted


def pytest_collection_modifyitems(config, items):
    """Skip backend-specific tests when the backend is not installed."""
    try:
        import numba  # noqa: F401

        have_numba = True
    except ImportError:
        have_numba = False
    try:
        import jax  # noqa: F401

        have_jax = True
    except ImportError:
        have_jax = False

    for item in items:
        if "numba" in item.keywords and not have_numba:
            item.add_marker(pytest.mark.skip(reason="numba not installed"))
        if "jax" in item.keywords and not have_jax:
            item.add_marker(pytest.mark.skip(reason="jax not installed"))
