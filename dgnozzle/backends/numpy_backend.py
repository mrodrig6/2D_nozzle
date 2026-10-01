"""Reference backend: pure NumPy, fully vectorised.

This is the readable implementation and the correctness reference for the other
two.  It needs no compilation, so it is also the right choice for a quick
one-off run or when debugging: every intermediate is an ordinary NumPy array you
can inspect.
"""

from __future__ import annotations

import numpy as np

from .. import assembly as asm
from .. import limiter as lim
from .base import Backend


class NumpyBackend(Backend):
    name = "numpy"

    def residual(self, U):
        return asm.residual(U, self.ops, self.flow, xp=np)

    def inverse_mass(self, R):
        return asm.apply_inverse_mass(R, self.ops, xp=np)

    def limit(self, U):
        if self.opts.limiter == "none" or self.ops.ref.order == 0:
            return U
        out = lim.apply_limiter(
            U, self.ops, self.flow, self.opts.limiter,
            tvb_constant=self.opts.tvb_constant, xp=np,
        )
        self.n_limit_calls += 1
        changed = np.abs(out - U).max(axis=(1, 2)) > 0.0
        self.n_limited += int(changed.sum())
        return out

    def time_step(self, wave_sum):
        dt = asm.local_time_step(
            wave_sum, self.ops, self.ops.ref.order, self.opts.cfl,
            self.opts.scheme, xp=np,
        )
        return dt[:, None, None]

    def norm(self, A) -> float:
        # A diverging march overflows here by design; the caller checks for a
        # non-finite norm and reports divergence, so the warning is noise.
        with np.errstate(over="ignore", invalid="ignore"):
            A = np.asarray(A)
            return float(np.sqrt(np.mean(A * A)))
