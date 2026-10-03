r"""JAX backend: differentiable, and the one students take gradients through.

The residual is the *same* vectorised code as the NumPy backend -- only the array
module differs.  That is the payoff of writing :mod:`src.assembly` and
:mod:`src.physics` against an injected ``xp``: there is no second
implementation of the physics to keep in sync, so a gradient taken here is a
gradient of the scheme that actually ran.

Performance notes
-----------------
* ``run`` fuses ``n_steps`` iterations into a single ``lax.fori_loop`` inside one
  ``jit``.  Without that, per-step Python dispatch dominates at the mesh sizes
  this solver targets.
* The residual norm is carried through the loop as a device array and pulled back
  to the host once, at the end.  Calling ``float()`` every step would force a
  synchronisation and throw away the fusion.
* JAX defaults to 32-bit.  :func:`enable_float64` is called on import, because a
  steady-state residual cannot be driven to 1e-6 relative in single precision.

For gradients with respect to the nozzle shape, see :mod:`src.sensitivity`,
which wraps the converged solve in a ``custom_vjp`` implementing the discrete
adjoint -- far cheaper and far more accurate than differentiating through the
pseudo-time history.
"""

from __future__ import annotations

import numpy as np

from .. import assembly as asm
from .. import limiter as lim
from .base import Backend

_FLOAT64_ENABLED = False


def enable_float64() -> None:
    """Switch JAX to double precision.  Must happen before any array is made."""
    global _FLOAT64_ENABLED
    if _FLOAT64_ENABLED:
        return
    import jax

    jax.config.update("jax_enable_x64", True)
    _FLOAT64_ENABLED = True


def require_jax():
    """Import JAX, with a clear message if it is missing."""
    try:
        import jax
        import jax.numpy as jnp
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "the 'jax' backend needs JAX installed: pip install 'dgnozzle[jax]' "
            "(or pip install 'jax[cpu]')"
        ) from exc
    enable_float64()
    return jax, jnp


class JaxBackend(Backend):
    name = "jax"

    def __init__(self, ops, flow, opts):
        super().__init__(ops, flow, opts)
        self.jax, self.jnp = require_jax()
        self._step_jit = {}
        self._run_jit = {}

    # -- primitives --------------------------------------------------------
    def residual(self, U):
        return asm.residual(U, self.ops, self.flow, xp=self.jnp)

    def inverse_mass(self, R):
        return asm.apply_inverse_mass(R, self.ops, xp=self.jnp)

    def limit(self, U):
        return lim.apply_limiter(
            U, self.ops, self.flow, self.opts.limiter,
            tvb_constant=self.opts.tvb_constant, xp=self.jnp,
        )

    def time_step(self, wave_sum):
        dt = asm.local_time_step(
            wave_sum, self.ops, self.ops.ref.order, self.opts.cfl,
            self.opts.scheme, xp=self.jnp,
        )
        return self.positivity_scale() * dt[:, None, None]

    def norm(self, A):
        """Device-side RMS norm; kept as an array so it can live inside a loop."""
        return self.jnp.sqrt(self.jnp.mean(A * A))

    def asarray(self, A):
        return self.jnp.asarray(A)

    def to_numpy(self, A) -> np.ndarray:
        return np.asarray(A)

    # -- fused stepping ----------------------------------------------------
    def make_step(self, scheme: str):
        if scheme not in self._step_jit:
            raw = super().make_step(scheme)
            self._step_jit[scheme] = self.jax.jit(raw)
        return self._step_jit[scheme]

    def run(self, U, n_steps: int, scheme: str):
        """Fuse ``n_steps`` iterations into one compiled ``fori_loop``."""
        jax, jnp = self.jax, self.jnp
        key = (scheme, int(n_steps))
        if key not in self._run_jit:
            raw_step = Backend.make_step(self, scheme)

            def body(_, carry):
                Uc, _res = carry
                return raw_step(Uc)

            @jax.jit
            def runner(U0):
                init = (U0, jnp.asarray(0.0))
                return jax.lax.fori_loop(0, int(n_steps), body, init)

            self._run_jit[key] = runner
        U, res = self._run_jit[key](jnp.asarray(U))
        return U, float(res)
