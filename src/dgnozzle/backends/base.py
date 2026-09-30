r"""Backend protocol and the shared Runge-Kutta steppers.

A backend supplies four primitives -- ``residual``, ``inverse_mass``, ``limit``
and ``time_step`` -- and inherits the time integrators.  Keeping the schemes in
one place means the three backends cannot drift apart numerically, which is what
makes cross-backend agreement testable to round-off.

Schemes
-------
``rk4``
    The classical four-stage method, as in the legacy solver.  The local time
    step is frozen at the first stage, which is standard for a pseudo-time march
    to steady state -- the intermediate stages are not meant to be
    time-accurate.

``ssprk3``
    Three-stage strong-stability-preserving. Each stage is a convex combination
    of forward-Euler updates, which is exactly the structure the Zhang-Shu
    positivity limiter assumes, so ``ssprk3`` + ``positivity`` has a genuine
    positivity guarantee that ``rk4`` does not.  Prefer it for shocked cases.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable

import numpy as np

from ..config import FlowConditions, SolverOptions
from ..operators import Operators


class Backend(ABC):
    """Execution strategy for one fixed mesh, order and operating point."""

    name: str = "abstract"

    def __init__(self, ops: Operators, flow: FlowConditions, opts: SolverOptions):
        self.ops = ops
        self.flow = flow
        self.opts = opts
        # Limiter activity, accumulated across stages.  A run whose residual
        # stalls while these keep climbing is limit-cycling on an unresolved
        # feature, which is a different problem from "needs more iterations" and
        # deserves a different message.
        self.n_limited = 0
        self.n_mean_repaired = 0
        self.n_limit_calls = 0

    def limiter_activity(self) -> float:
        """Mean number of elements limited per limiter call; NaN if not tracked."""
        if self.n_limit_calls == 0:
            return float("nan")
        return self.n_limited / self.n_limit_calls

    # -- primitives a backend must provide --------------------------------
    @abstractmethod
    def residual(self, U):
        """Return ``(R, wave_sum)``."""

    @abstractmethod
    def inverse_mass(self, R):
        """Apply the block-diagonal inverse mass matrix."""

    @abstractmethod
    def limit(self, U):
        """Apply the configured limiter."""

    @abstractmethod
    def time_step(self, wave_sum):
        """Element-local pseudo-time step, shaped for broadcasting against ``U``."""

    @abstractmethod
    def norm(self, A) -> float:
        """Root-mean-square norm of an array."""

    def asarray(self, A):
        return np.asarray(A)

    def to_numpy(self, A) -> np.ndarray:
        return np.asarray(A)

    # -- integrators -------------------------------------------------------
    def rate(self, U):
        r"""``dU/dt = -M^{-1} R``, plus the wave-speed sum."""
        R, wave = self.residual(U)
        return -self.inverse_mass(R), wave

    def rk4_step(self, U):
        """One classical RK4 pseudo-time step.  Returns ``(U_new, residual_norm)``."""
        F0, wave = self.rate(U)
        dt = self.time_step(wave)
        res = self.norm(F0)

        F1, _ = self.rate(self.limit(U + 0.5 * dt * F0))
        F2, _ = self.rate(self.limit(U + 0.5 * dt * F1))
        F3, _ = self.rate(self.limit(U + dt * F2))

        U = U + dt / 6.0 * (F0 + 2.0 * F1 + 2.0 * F2 + F3)
        return self.limit(U), res

    def ssprk3_step(self, U):
        """One three-stage SSP-RK3 pseudo-time step."""
        F0, wave = self.rate(U)
        dt = self.time_step(wave)
        res = self.norm(F0)

        U1 = self.limit(U + dt * F0)
        F1, _ = self.rate(U1)
        U2 = self.limit(0.75 * U + 0.25 * (U1 + dt * F1))
        F2, _ = self.rate(U2)
        U3 = (1.0 / 3.0) * U + (2.0 / 3.0) * (U2 + dt * F2)
        return self.limit(U3), res

    def make_step(self, scheme: str) -> Callable:
        if scheme == "rk4":
            return self.rk4_step
        if scheme == "ssprk3":
            return self.ssprk3_step
        raise ValueError(f"unknown scheme {scheme!r}")

    def run(self, U, n_steps: int, scheme: str):
        """Advance ``n_steps`` steps.  Returns ``(U, residual_norm_of_last_step)``.

        Backends that can fuse the loop (JAX) override this; the default simply
        iterates, which is what NumPy and Numba want anyway since their kernels
        already release the GIL and have no per-call dispatch overhead worth
        amortising.
        """
        step = self.make_step(scheme)
        res = float("nan")
        for _ in range(n_steps):
            U, res = step(U)
        return U, res
