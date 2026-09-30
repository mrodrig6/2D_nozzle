r"""Compressible Euler physics: fluxes, the Roe solver, boundary conditions.

State vector, in conserved variables:

.. math::
    U = (\rho,\ \rho u,\ \rho v,\ \rho E)^T, \qquad
    p = (\gamma - 1)\Bigl(\rho E - \tfrac{1}{2}\rho (u^2 + v^2)\Bigr)

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
    r"""Return ``(rho, u, v, p, H)`` with ``rho`` and ``p`` floored positive."""
    rho = xp.maximum(U[..., 0], FLOOR)
    u = U[..., 1] / rho
    v = U[..., 2] / rho
    p = xp.maximum((gamma - 1.0) * (U[..., 3] - 0.5 * rho * (u * u + v * v)), FLOOR)
    H = (U[..., 3] + p) / rho
    return rho, u, v, p, H


def pressure(U, gamma: float, xp=np):
    """Static pressure, floored positive."""
    return primitives(U, gamma, xp=xp)[3]


def mach_number(U, gamma: float, xp=np):
    """Local Mach number."""
    rho, u, v, p, _ = primitives(U, gamma, xp=xp)
    return xp.sqrt((u * u + v * v) * rho / (gamma * p))


def sound_speed(U, gamma: float, xp=np):
    rho, _, _, p, _ = primitives(U, gamma, xp=xp)
    return xp.sqrt(gamma * p / rho)


def euler_flux(U, gamma: float, xp=np):
    r"""The inviscid flux pair :math:`(F, G)`, each shaped like ``U``."""
    rho, u, v, p, H = primitives(U, gamma, xp=xp)
    F = xp.stack([rho * u, rho * u * u + p, rho * u * v, rho * u * H], axis=-1)
    G = xp.stack([rho * v, rho * u * v, rho * v * v + p, rho * v * H], axis=-1)
    return F, G


def normal_flux(U, nx, ny, gamma: float, xp=np):
    r"""The projected flux :math:`F n_x + G n_y`."""
    rho, u, v, p, H = primitives(U, gamma, xp=xp)
    un = u * nx + v * ny
    return xp.stack(
        [rho * un, rho * u * un + p * nx, rho * v * un + p * ny, rho * H * un], axis=-1
    )


# --------------------------------------------------------------------------
# Roe flux
# --------------------------------------------------------------------------
def roe_flux(UL, UR, nx, ny, gamma: float, *, entropy_fix: float = 0.05, xp=np) -> FluxResult:
    r"""Roe's approximate Riemann solver with the Harten-Hyman entropy fix.

    .. math::
        \hat{F} = \tfrac{1}{2}\bigl(F(U_L) + F(U_R)\bigr)\cdot n
                  - \tfrac{1}{2}\, |\hat{A}|\,(U_R - U_L)

    The dissipation term is evaluated in the factored form of the standard
    AE623 formulation rather than by forming :math:`|\hat{A}|` explicitly.
    Eigenvalues smaller in magnitude than ``entropy_fix * c`` are replaced by
    :math:`(\lambda^2 + \epsilon^2)/(2\epsilon)`, which prevents the expansion
    shock that an unmodified Roe flux admits at a sonic point.
    """
    rhoL, uL, vL, pL, HL = primitives(UL, gamma, xp=xp)
    rhoR, uR, vR, pR, HR = primitives(UR, gamma, xp=xp)

    FL = normal_flux(UL, nx, ny, gamma, xp=xp)
    FR = normal_flux(UR, nx, ny, gamma, xp=xp)

    # Roe average (density-weighted)
    sL = xp.sqrt(rhoL)
    sR = xp.sqrt(rhoR)
    denom = sL + sR
    u = (sL * uL + sR * uR) / denom
    v = (sL * vL + sR * vR) / denom
    H = (sL * HL + sR * HR) / denom
    q2 = u * u + v * v
    c = xp.sqrt(xp.maximum((gamma - 1.0) * (H - 0.5 * q2), FLOOR))
    un = u * nx + v * ny

    lam1 = xp.abs(un + c)
    lam2 = xp.abs(un - c)
    lam3 = xp.abs(un)
    max_speed = xp.maximum(lam1, lam2)

    eps = entropy_fix * c
    def fix(lam):
        return xp.where(lam < eps, (lam * lam + eps * eps) / (2.0 * eps), lam)

    lam1 = fix(lam1)
    lam2 = fix(lam2)
    lam3 = fix(lam3)

    dU = UR - UL
    drho, drhou, drhov, drhoE = dU[..., 0], dU[..., 1], dU[..., 2], dU[..., 3]

    G1 = (gamma - 1.0) * (0.5 * q2 * drho - u * drhou - v * drhov + drhoE)
    G2 = -un * drho + drhou * nx + drhov * ny

    s1 = 0.5 * (lam1 + lam2)
    s2 = 0.5 * (lam1 - lam2)
    C1 = (G1 / (c * c)) * (s1 - lam3) + (G2 / c) * s2
    C2 = (G1 / c) * s2 + (s1 - lam3) * G2

    diss = xp.stack(
        [
            lam3 * drho + C1,
            lam3 * drhou + C1 * u + C2 * nx,
            lam3 * drhov + C1 * v + C2 * ny,
            lam3 * drhoE + C1 * H + C2 * un,
        ],
        axis=-1,
    )
    return FluxResult(0.5 * (FL + FR) - 0.5 * diss, max_speed)


# --------------------------------------------------------------------------
# Boundary conditions
# --------------------------------------------------------------------------
def wall_flux(U, nx, ny, gamma: float, xp=np) -> FluxResult:
    r"""Inviscid (slip) wall, also used for the symmetry axis.

    The normal velocity is removed and the wall pressure recovered from the
    remaining energy, giving a flux that transmits pressure but no mass:

    .. math::
        p_b = (\gamma - 1)\Bigl(\rho E - \tfrac{1}{2}\rho |v_\parallel|^2\Bigr),
        \qquad \hat{F} = (0,\ p_b n_x,\ p_b n_y,\ 0)^T
    """
    rho = xp.maximum(U[..., 0], FLOOR)
    u = U[..., 1] / rho
    v = U[..., 2] / rho
    un = u * nx + v * ny
    ut2 = (u - un * nx) ** 2 + (v - un * ny) ** 2
    pb = xp.maximum((gamma - 1.0) * (U[..., 3] - 0.5 * rho * ut2), FLOOR)
    zero = xp.zeros_like(pb)
    flux = xp.stack([zero, pb * nx, pb * ny, zero], axis=-1)
    return FluxResult(flux, xp.sqrt(gamma * pb / rho))


def inflow_flux(
    U, nx, ny, gamma: float, *, Tt: float, pt: float, Rgas: float, alpha: float = 0.0, xp=np
) -> FluxResult:
    r"""Subsonic stagnation inflow: total temperature, total pressure and flow angle.

    One characteristic leaves the domain, carrying the interior Riemann
    invariant :math:`J^+ = u_n + 2a/(\gamma - 1)` (with ``n`` the *outward*
    normal).  Combining it with the isentropic stagnation relations gives a
    quadratic for the inflow Mach number,

    .. math::
        \Bigl(\tfrac{\gamma-1}{2}\beta - n_d^2\Bigr) M^2
        - \tfrac{4 n_d}{\gamma-1} M
        + \Bigl(\beta - \bigl(\tfrac{2}{\gamma-1}\bigr)^2\Bigr) = 0,
        \qquad \beta = \Bigl(\tfrac{J^+}{a_t}\Bigr)^2,\
        n_d = n \cdot \hat{d}

    where :math:`\hat{d}` is the prescribed inflow direction.

    The smallest non-negative root is the physical branch.  Taking
    ``(-b + sqrt(disc)) / 2a`` unconditionally is correct only while ``a > 0``:
    when the leading coefficient changes sign -- which happens for weak inflow --
    that root is negative and the boundary state becomes meaningless.
    """
    at2 = gamma * Rgas * Tt  # stagnation speed of sound squared
    at = np.sqrt(at2)
    rho_t = gamma * pt / at2

    rho, u, v, p, _ = primitives(U, gamma, xp=xp)
    a = xp.sqrt(gamma * p / rho)
    un = u * nx + v * ny
    Jp = un + 2.0 * a / (gamma - 1.0)

    beta = (Jp / at) ** 2
    nd = nx * np.cos(alpha) + ny * np.sin(alpha)

    aa = 0.5 * (gamma - 1.0) * beta - nd * nd
    bb = -4.0 * nd / (gamma - 1.0)
    cc = beta - (2.0 / (gamma - 1.0)) ** 2

    # keep the leading coefficient away from zero without changing its sign
    tiny = 1e-12
    aa_safe = xp.where(xp.abs(aa) < tiny, xp.where(aa >= 0, tiny, -tiny), aa)
    disc = xp.maximum(bb * bb - 4.0 * aa_safe * cc, 0.0)
    sq = xp.sqrt(disc)
    r1 = (-bb + sq) / (2.0 * aa_safe)
    r2 = (-bb - sq) / (2.0 * aa_safe)

    huge = 1e30
    c1 = xp.where(r1 >= 0.0, r1, huge)
    c2 = xp.where(r2 >= 0.0, r2, huge)
    M = xp.minimum(c1, c2)
    M = xp.where(M > 0.5 * huge, 0.0, M)
    # the relation assumes a subsonic inlet; clamp transients that overshoot
    M = xp.clip(M, 0.0, 1.0)

    ab = xp.sqrt(at2 / (1.0 + 0.5 * (gamma - 1.0) * M * M))
    qb = M * ab
    ub = qb * np.cos(alpha)
    vb = qb * np.sin(alpha)
    rhob = rho_t * (1.0 + 0.5 * (gamma - 1.0) * M * M) ** (-1.0 / (gamma - 1.0))
    pb = rhob * ab * ab / gamma
    Hb = at2 / (gamma - 1.0)  # total enthalpy is conserved from the reservoir
    unb = ub * nx + vb * ny

    flux = xp.stack(
        [rhob * unb, rhob * ub * unb + pb * nx, rhob * vb * unb + pb * ny, rhob * Hb * unb],
        axis=-1,
    )
    return FluxResult(flux, xp.abs(unb) + ab)


def outflow_flux(U, nx, ny, gamma: float, *, p_back: float, xp=np) -> FluxResult:
    r"""Pressure outflow that switches automatically to supersonic extrapolation.

    *Subsonic* outflow (:math:`u_n / a < 1`) admits one incoming characteristic,
    so exactly one condition may be imposed.  Static pressure is set to
    ``p_back``; entropy and the outgoing invariant are carried from the interior,
    and the tangential velocity is unchanged:

    .. math::
        \rho_b = \rho_L\Bigl(\frac{p_b}{p_L}\Bigr)^{1/\gamma},\quad
        u_{n,b} = u_{n,L} + \frac{2}{\gamma-1}\bigl(a_L - a_b\bigr),\quad
        v_{t,b} = v_{t,L}

    *Supersonic* outflow admits none, so the interior state is extrapolated and
    ``p_back`` is ignored.  The two branches are blended by the local normal
    Mach number, which makes a shock that moves across the exit plane during the
    transient harmless.

    .. note::
       Extrapolating unconditionally -- ignoring the subsonic branch -- would
       leave the back pressure with no effect on the flow at all, and with it
       every operating point whose exit is subsonic.
    """
    rho, u, v, p, _ = primitives(U, gamma, xp=xp)
    a = xp.sqrt(gamma * p / rho)
    un = u * nx + v * ny

    # -- subsonic branch: impose static pressure.  Multiplying by ones_like
    # rather than calling float() keeps this working when p_back is a JAX tracer,
    # which is what makes d(thrust)/d(back pressure) available.
    pb = xp.asarray(p_back) * xp.ones_like(p)
    rhob = rho * (pb / p) ** (1.0 / gamma)
    ab = xp.sqrt(gamma * pb / rhob)
    unb = un + 2.0 / (gamma - 1.0) * (a - ab)
    ut_x = u - un * nx
    ut_y = v - un * ny
    ub = ut_x + unb * nx
    vb = ut_y + unb * ny
    Eb = pb / ((gamma - 1.0) * rhob) + 0.5 * (ub * ub + vb * vb)
    U_sub = xp.stack([rhob, rhob * ub, rhob * vb, rhob * Eb], axis=-1)

    # -- blend with the extrapolated (supersonic) state
    supersonic = (un / a >= 1.0)[..., None]
    U_b = xp.where(supersonic, U, U_sub)

    flux = normal_flux(U_b, nx, ny, gamma, xp=xp)
    rb, ubb, vbb, pbb, _ = primitives(U_b, gamma, xp=xp)
    ab2 = xp.sqrt(gamma * pbb / rb)
    return FluxResult(flux, xp.abs(ubb * nx + vbb * ny) + ab2)
