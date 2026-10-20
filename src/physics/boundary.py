r"""The boundary fluxes: the inviscid wall, and the two nozzle ends.

A nozzle has three kinds of boundary, and only one of them is geometric.  The
wall carries pressure and nothing else.  The inflow and the outflow are
*characteristic*: how many conditions may be imposed depends on how many
characteristics enter the domain, which depends on the flow, which is why both
of these are written as a blend rather than as a fixed prescription.
"""

from __future__ import annotations

import numpy as np

from .thermo import FLOOR, FluxResult, normal_flux, primitives


# --------------------------------------------------------------------------
# Boundary conditions
# --------------------------------------------------------------------------
def wall_flux(U, nx, ny, gamma: float, xp=np) -> FluxResult:
    r"""Inviscid (slip) wall, also used for the symmetry axis.

    The normal velocity is removed and the wall pressure recovered from the
    remaining energy, giving a flux that transmits pressure but no mass:

    .. math::
        p_b = (\gamma - 1)\Bigl(\rho E - \tfrac{1}{2}\rho |\mathbf{v}_t|^2\Bigr),
        \qquad \hat{\mathbf{F}} = (0,\ p_b n_x,\ p_b n_y,\ 0)^T
    """
    rho = xp.maximum(U[..., 0], FLOOR)
    vx = U[..., 1] / rho
    vy = U[..., 2] / rho
    vn = vx * nx + vy * ny
    vt2 = (vx - vn * nx) ** 2 + (vy - vn * ny) ** 2
    pb = xp.maximum((gamma - 1.0) * (U[..., 3] - 0.5 * rho * vt2), FLOOR)
    zero = xp.zeros_like(pb)
    flux = xp.stack([zero, pb * nx, pb * ny, zero], axis=-1)
    return FluxResult(flux, xp.sqrt(gamma * pb / rho))


def inflow_flux(
    U, nx, ny, gamma: float, *, Tt: float, pt: float, Rgas: float, alpha: float = 0.0, xp=np
) -> FluxResult:
    r"""Subsonic stagnation inflow: total temperature, total pressure and flow angle.

    One characteristic leaves the domain, carrying the interior Riemann
    invariant :math:`J^+ = v_n + 2a/(\gamma - 1)` (with :math:`\mathbf{n}` the *outward*
    normal).  Combining it with the isentropic stagnation relations gives a
    quadratic for the inflow Mach number,

    .. math::
        \Bigl(\tfrac{\gamma-1}{2}\beta - n_d^2\Bigr) M^2
        - \tfrac{4 n_d}{\gamma-1} M
        + \Bigl(\beta - \bigl(\tfrac{2}{\gamma-1}\bigr)^2\Bigr) = 0,
        \qquad \beta = \Bigl(\tfrac{J^+}{a_t}\Bigr)^2,\
        n_d = \mathbf{n} \cdot \hat{\mathbf{d}}

    where :math:`\hat{\mathbf{d}}` is the prescribed inflow direction.

    The smallest non-negative root is the physical branch.  Taking
    ``(-b + sqrt(disc)) / 2a`` unconditionally is correct only while ``a > 0``:
    when the leading coefficient changes sign -- which happens for weak inflow --
    that root is negative and the boundary state becomes meaningless.
    """
    at2 = gamma * Rgas * Tt  # stagnation speed of sound squared
    at = np.sqrt(at2)
    rho_t = gamma * pt / at2

    rho, vx, vy, p, _ = primitives(U, gamma, xp=xp)
    a = xp.sqrt(gamma * p / rho)
    vn = vx * nx + vy * ny
    Jp = vn + 2.0 * a / (gamma - 1.0)

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
    vxb = qb * np.cos(alpha)
    vyb = qb * np.sin(alpha)
    rhob = rho_t * (1.0 + 0.5 * (gamma - 1.0) * M * M) ** (-1.0 / (gamma - 1.0))
    pb = rhob * ab * ab / gamma
    Hb = at2 / (gamma - 1.0)  # total enthalpy is conserved from the reservoir
    vnb = vxb * nx + vyb * ny

    flux = xp.stack(
        [rhob * vnb, rhob * vxb * vnb + pb * nx, rhob * vyb * vnb + pb * ny, rhob * Hb * vnb],
        axis=-1,
    )
    return FluxResult(flux, xp.abs(vnb) + ab)


def outflow_flux(
    U,
    nx,
    ny,
    gamma: float,
    *,
    p_back: float,
    rho_t: float = 0.0,
    p_t: float = 1.0,
    backflow_band: float = 0.05,
    xp=np,
) -> FluxResult:
    r"""Pressure outflow that switches automatically to supersonic extrapolation.

    *Subsonic* outflow (:math:`v_n / a < 1`) admits one incoming characteristic,
    so exactly one condition may be imposed.  Static pressure is set to
    ``p_back``; entropy and the outgoing invariant are carried from the interior,
    and the tangential velocity is unchanged:

    .. math::
        \rho_b = \rho_L\Bigl(\frac{p_b}{p_L}\Bigr)^{1/\gamma},\quad
        v_{n,b} = v_{n,L} + \frac{2}{\gamma-1}\bigl(a_L - a_b\bigr),\quad
        \mathbf{v}_{t,b} = \mathbf{v}_{t,L}

    *Supersonic* outflow admits none, so the interior state is extrapolated and
    ``p_back`` is ignored.  The two branches are blended by the local normal
    Mach number, which makes a shock that moves across the exit plane during the
    transient harmless.

    .. note::
       Extrapolating unconditionally -- ignoring the subsonic branch -- would
       leave the back pressure with no effect on the flow at all, and with it
       every operating point whose exit is subsonic.

    *Reverse flow* (:math:`v_n < 0`, the flow entering through the exit plane)
    is a third case, and the subsonic-outflow formula above is **ill-posed**
    there.  Count the characteristics with :math:`\mathbf{n}` outward: subsonic
    outflow has :math:`v_n > 0` and :math:`v_n + a > 0` leaving and only
    :math:`v_n - a < 0` entering, so exactly one condition may be imposed.
    Reverse flow has :math:`v_n < 0` *and* :math:`v_n - a < 0` entering -- the
    entropy wave, the shear wave and one acoustic wave, three in all -- leaving
    only :math:`v_n + a > 0` to come from the interior.  Imposing pressure alone
    under-determines it by two conditions.

    So the backflow branch imposes three things from the reservoir that the back
    pressure belongs to -- its entropy, through
    :math:`\rho_b = \rho_t (p_b/p_t)^{1/\gamma}`; the static pressure
    :math:`p_b`; and the direction, normal to the plane -- and takes the normal
    velocity from the one outgoing invariant,
    :math:`v_{n,b} = v_{n} + \tfrac{2}{\gamma-1}(a - a_b)`.

    The two subsonic branches do not agree at :math:`v_n = 0` (one keeps the
    interior entropy and tangential velocity, the other imposes the reservoir's
    and zero), so they are blended smoothly across
    :math:`0 \le v_n/a \le` ``backflow_band`` rather than switched.  A hard
    switch at a boundary the solution is still moving across is exactly the kind
    of discontinuity that parks a residual instead of converging.

    ``rho_t = 0`` disables the branch and restores the previous behaviour, which
    is what makes the change measurable rather than merely asserted.
    """
    rho, vx, vy, p, _ = primitives(U, gamma, xp=xp)
    a = xp.sqrt(gamma * p / rho)
    vn = vx * nx + vy * ny

    # -- subsonic branch: impose static pressure.  Multiplying by ones_like
    # rather than calling float() keeps this working when p_back is a JAX tracer,
    # which is what makes d(thrust)/d(back pressure) available.
    pb = xp.asarray(p_back) * xp.ones_like(p)
    rhob = rho * (pb / p) ** (1.0 / gamma)
    ab = xp.sqrt(gamma * pb / rhob)
    vnb = vn + 2.0 / (gamma - 1.0) * (a - ab)
    vtx = vx - vn * nx
    vty = vy - vn * ny
    vxb = vtx + vnb * nx
    vyb = vty + vnb * ny
    Eb = pb / ((gamma - 1.0) * rhob) + 0.5 * (vxb * vxb + vyb * vyb)
    U_sub = xp.stack([rhob, rhob * vxb, rhob * vyb, rhob * Eb], axis=-1)

    # -- reverse flow: three conditions from the reservoir, one from inside ---
    if rho_t > 0.0:
        # reservoir entropy: rho_b = rho_t (p_b / p_t)^(1/gamma)
        rho_bf = rho_t * (pb / p_t) ** (1.0 / gamma)
        a_bf = xp.sqrt(gamma * pb / rho_bf)
        vn_bf = vn + 2.0 / (gamma - 1.0) * (a - a_bf)
        # direction normal to the plane: no tangential velocity carried in
        vx_bf = vn_bf * nx
        vy_bf = vn_bf * ny
        E_bf = pb / ((gamma - 1.0) * rho_bf) + 0.5 * (vx_bf * vx_bf + vy_bf * vy_bf)
        U_bf = xp.stack([rho_bf, rho_bf * vx_bf, rho_bf * vy_bf, rho_bf * E_bf], axis=-1)
        # smooth blend over 0 <= vn/a <= band, so nothing switches discontinuously
        t = xp.clip((vn / a) / max(backflow_band, 1e-12), 0.0, 1.0)[..., None]
        U_sub = t * U_sub + (1.0 - t) * U_bf

    # -- blend with the extrapolated (supersonic) state
    supersonic = (vn / a >= 1.0)[..., None]
    U_b = xp.where(supersonic, U, U_sub)

    flux = normal_flux(U_b, nx, ny, gamma, xp=xp)
    rho_o, vx_o, vy_o, p_o, _ = primitives(U_b, gamma, xp=xp)
    a_o = xp.sqrt(gamma * p_o / rho_o)
    return FluxResult(flux, xp.abs(vx_o * nx + vy_o * ny) + a_o)
