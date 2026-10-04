"""Backend registry.

Three interchangeable execution strategies over one set of equations:

=========  =========================================================
``numba``  Compiled nopython kernels.  The default and the fastest.
``numpy``  Vectorised reference.  No compile step; easiest to read.
``jax``    Same vectorised code under ``jax.numpy``; differentiable.
=========  =========================================================
"""

from __future__ import annotations

from ..config import BACKENDS, FlowConditions, SolverOptions
from ..operators import Operators
from .base import Backend
from .numpy_backend import NumpyBackend

__all__ = ["Backend", "NumpyBackend", "get_backend", "available_backends", "BACKENDS"]


def available_backends() -> tuple[str, ...]:
    """Backends importable in this environment, fastest first."""
    found = []
    try:
        import numba  # noqa: F401

        found.append("numba")
    except ImportError:
        pass
    found.append("numpy")
    try:
        import jax  # noqa: F401

        found.append("jax")
    except ImportError:
        pass
    return tuple(found)


def get_backend(name: str, ops: Operators, flow: FlowConditions, opts: SolverOptions) -> Backend:
    """Instantiate a backend by name."""
    if name == "numpy":
        return NumpyBackend(ops, flow, opts)
    if name == "numba":
        try:
            from .numba_backend import NumbaBackend
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "the 'numba' backend needs Numba installed: pip install numba"
            ) from exc
        return NumbaBackend(ops, flow, opts)
    if name == "jax":
        from .jax_backend import JaxBackend

        return JaxBackend(ops, flow, opts)
    raise ValueError(f"unknown backend {name!r}; choose from {BACKENDS}")
