r"""The state vector, the thermodynamics, and the Euler flux itself.

.. math::
    \mathbf{U} = (\rho,\ \rho v_x,\ \rho v_y,\ \rho E)^T, \qquad
    p = (\gamma - 1)\Bigl(\rho E - \tfrac{1}{2}\rho |\mathbf{v}|^2\Bigr),
    \qquad \mathbf{v} = (v_x, v_y)^T

All routines take arrays whose **last axis has length 4** and whose leading axes
are arbitrary, so the same function serves a single point, an edge's quadrature
points, or every edge in the mesh at once.

Everything is branchless -- selections use ``where`` rather than ``if`` -- so the
module runs unchanged under NumPy and under ``jax.jit``, and stays
differentiable.

Robustness
----------
A transient excursion to a negative density or a negative Roe-averaged sound
speed is normal and recoverable in a pseudo-time march, so aborting on one
throws away a run that would have converged.  Here density and pressure are
floored at :data:`FLOOR` inside the flux evaluation, which keeps the march
finite; the solver separately monitors the *unfloored* state and reports when
the floor was active, so a genuinely bad solution is never silently accepted.
"""

from __future__ import annotations

from typing import NamedTuple

import numpy as np

#: Floor applied to density and pressure inside flux evaluations.
FLOOR = 1e-10


class FluxResult(NamedTuple):
    """A numerical flux and the maximum signal speed that produced it."""

    flux: object  # (..., 4)
    max_speed: object  # (...)


# --------------------------------------------------------------------------
# Primitive variables
# --------------------------------------------------------------------------
def primitives(U, gamma: float, xp=np):
    r"""Return ``(rho, vx, vy, p, H)`` with ``rho`` and ``p`` floored positive."""
    rho = xp.maximum(U[..., 0], FLOOR)
    vx = U[..., 1] / rho
    vy = U[..., 2] / rho
    p = xp.maximum((gamma - 1.0) * (U[..., 3] - 0.5 * rho * (vx * vx + vy * vy)), FLOOR)
    H = (U[..., 3] + p) / rho
    return rho, vx, vy, p, H


def pressure(U, gamma: float, xp=np):
    """Static pressure, floored positive."""
    return primitives(U, gamma, xp=xp)[3]


def mach_number(U, gamma: float, xp=np):
    """Local Mach number."""
    rho, vx, vy, p, _ = primitives(U, gamma, xp=xp)
    return xp.sqrt((vx * vx + vy * vy) * rho / (gamma * p))


def sound_speed(U, gamma: float, xp=np):
    rho, _, _, p, _ = primitives(U, gamma, xp=xp)
    return xp.sqrt(gamma * p / rho)


def euler_flux(U, gamma: float, xp=np):
    r"""The inviscid flux pair :math:`(F, G)`, each shaped like ``U``."""
    rho, vx, vy, p, H = primitives(U, gamma, xp=xp)
    F = xp.stack([rho * vx, rho * vx * vx + p, rho * vx * vy, rho * vx * H], axis=-1)
    G = xp.stack([rho * vy, rho * vx * vy, rho * vy * vy + p, rho * vy * H], axis=-1)
    return F, G


def normal_flux(U, nx, ny, gamma: float, xp=np):
    r"""The projected flux :math:`F n_x + G n_y`."""
    rho, vx, vy, p, H = primitives(U, gamma, xp=xp)
    vn = vx * nx + vy * ny
    return xp.stack(
        [rho * vn, rho * vx * vn + p * nx, rho * vy * vn + p * ny, rho * H * vn], axis=-1
    )
