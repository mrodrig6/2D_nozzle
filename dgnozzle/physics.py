r"""Compressible Euler physics: fluxes, the Roe solver, boundary conditions.

State vector, in conserved variables:

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


# --------------------------------------------------------------------------
# Roe flux
# --------------------------------------------------------------------------
def roe_flux(UL, UR, nx, ny, gamma: float, *, entropy_fix: float = 0.05, xp=np) -> FluxResult:
    r"""Roe's approximate Riemann solver with the Harten-Hyman entropy fix.

    .. math::
        \hat{\mathbf{F}} = \tfrac{1}{2}\bigl(\mathbf{F}(\mathbf{U}_L)
                           + \mathbf{F}(\mathbf{U}_R)\bigr)\cdot\mathbf{n}
                  - \tfrac{1}{2}\, |\hat{\mathbf{A}}|\,(\mathbf{U}_R - \mathbf{U}_L)

    The dissipation term is evaluated in the factored form of the standard
    AE623 formulation rather than by forming :math:`|\hat{\mathbf{A}}|` explicitly.
    Eigenvalues smaller in magnitude than ``entropy_fix * c`` are replaced by
    :math:`(\lambda^2 + \epsilon^2)/(2\epsilon)`, which prevents the expansion
    shock that an unmodified Roe flux admits at a sonic point.
    """
    rhoL, vxL, vyL, pL, HL = primitives(UL, gamma, xp=xp)
    rhoR, vxR, vyR, pR, HR = primitives(UR, gamma, xp=xp)

    FL = normal_flux(UL, nx, ny, gamma, xp=xp)
    FR = normal_flux(UR, nx, ny, gamma, xp=xp)

    # Roe average (density-weighted)
    sL = xp.sqrt(rhoL)
    sR = xp.sqrt(rhoR)
    denom = sL + sR
    vx = (sL * vxL + sR * vxR) / denom
    vy = (sL * vyL + sR * vyR) / denom
    H = (sL * HL + sR * HR) / denom
    q2 = vx * vx + vy * vy
    c = xp.sqrt(xp.maximum((gamma - 1.0) * (H - 0.5 * q2), FLOOR))
    vn = vx * nx + vy * ny

    lam1 = xp.abs(vn + c)
    lam2 = xp.abs(vn - c)
    lam3 = xp.abs(vn)
    max_speed = xp.maximum(lam1, lam2)

    eps = entropy_fix * c
    def fix(lam):
        return xp.where(lam < eps, (lam * lam + eps * eps) / (2.0 * eps), lam)

    lam1 = fix(lam1)
    lam2 = fix(lam2)
    lam3 = fix(lam3)

    dU = UR - UL
    drho, drho_vx, drho_vy, drhoE = dU[..., 0], dU[..., 1], dU[..., 2], dU[..., 3]

    G1 = (gamma - 1.0) * (0.5 * q2 * drho - vx * drho_vx - vy * drho_vy + drhoE)
    G2 = -vn * drho + drho_vx * nx + drho_vy * ny

    s1 = 0.5 * (lam1 + lam2)
    s2 = 0.5 * (lam1 - lam2)
    C1 = (G1 / (c * c)) * (s1 - lam3) + (G2 / c) * s2
    C2 = (G1 / c) * s2 + (s1 - lam3) * G2

    diss = xp.stack(
        [
            lam3 * drho + C1,
            lam3 * drho_vx + C1 * vx + C2 * nx,
            lam3 * drho_vy + C1 * vy + C2 * ny,
            lam3 * drhoE + C1 * H + C2 * vn,
        ],
        axis=-1,
    )
    return FluxResult(0.5 * (FL + FR) - 0.5 * diss, max_speed)


def hllc_flux(
    UL, UR, nx, ny, gamma: float, *,
    low_mach: float = 0.0, xp=np,
) -> FluxResult:
    r"""HLLC with Batten's wave speeds, optionally with a low-Mach correction.

    Three waves instead of Roe's full eigen-decomposition: a left acoustic wave
    at :math:`S_L`, a right one at :math:`S_R`, and the contact at :math:`S_M`
    between them.  Restoring that contact is what separates HLLC from HLL, whose
    missing middle wave smears every shear layer and contact discontinuity.

    .. math::
        \hat{\mathbf{F}} = \begin{cases}
            \mathbf{F}_L, & 0 \le S_L,\\
            \mathbf{F}_L + S_L(\mathbf{U}^{*}_L - \mathbf{U}_L),
                & S_L \le 0 \le S_M,\\
            \mathbf{F}_R + S_R(\mathbf{U}^{*}_R - \mathbf{U}_R),
                & S_M \le 0 \le S_R,\\
            \mathbf{F}_R, & S_R \le 0,
        \end{cases}

    with the star states carrying the contact's normal velocity and each side's
    own tangential velocity,

    .. math::
        \rho^{*}_K = \rho_K \frac{S_K - v_{nK}}{S_K - S_M}, \qquad
        \mathbf{v}^{*}_K = \mathbf{v}_K + (S_M - v_{nK})\,\mathbf{n},

    .. math::
        E^{*}_K = E_K + (S_M - v_{nK})
            \left(S_M + \frac{p_K}{\rho_K (S_K - v_{nK})}\right).

    Why this is the one to reach for
    -------------------------------
    **It is positivity-preserving, and the limiter needs that.**  With Batten's
    wave-speed estimates,

    .. math::
        S_L = \min(v_{nL} - a_L,\ \tilde{v}_n - \tilde{a}), \qquad
        S_R = \max(v_{nR} + a_R,\ \tilde{v}_n + \tilde{a}),

    (the tilde being the Roe average) HLLC provably keeps density and pressure
    positive under a CFL condition.  The Roe flux does **not**, with or without
    the entropy fix.  That matters more here than it sounds: the Zhang-Shu
    positivity limiter this code relies on by default has a theorem, and the
    theorem *assumes* the underlying first-order flux is positivity-preserving.
    Under ``flux='roe'`` that assumption is simply unmet, which is the honest
    explanation for the cell-average repairs the shocked cases report.

    **There is nothing to tune.**  An unmodified Roe flux admits an entropy-
    violating expansion shock wherever an eigenvalue crosses zero -- at the
    throat of every choked nozzle -- so it needs the Harten-Hyman fix and that
    fix needs a constant (``entropy_fix = 0.05``).  The HLL family cannot
    produce an expansion shock at all, so there is no fix and no constant.

    **It is cheaper**, because no eigenvector matrix is ever formed.

    Parameters
    ----------
    low_mach
        Cutoff Mach number :math:`M_{\mathrm{lim}}` for the low-Mach correction
        of Fleischmann et al.  ``0`` (the default) disables it and gives
        standard HLLC.  When positive, the *acoustic* wave speeds are scaled by

        .. math::
            \phi = \min\!\left(1,\ \frac{\max(|M_L|,\ |M_R|)}
                                        {M_{\mathrm{lim}}}\right),

        which shrinks the acoustic dissipation where the flow is slow while
        leaving the contact wave untouched.  ``phi = 1`` recovers standard HLLC
        identically, which is pinned by a test.

        .. warning::
           The *form* here follows Fleischmann et al.; check the cutoff constant
           against the paper before relying on the low-Mach branch
           quantitatively.  The ``low_mach = 0`` path is the verified one.
    """
    rhoL, vxL, vyL, pL, HL = primitives(UL, gamma, xp=xp)
    rhoR, vxR, vyR, pR, HR = primitives(UR, gamma, xp=xp)

    vnL = vxL * nx + vyL * ny
    vnR = vxR * nx + vyR * ny
    aL = xp.sqrt(xp.maximum(gamma * pL / rhoL, FLOOR))
    aR = xp.sqrt(xp.maximum(gamma * pR / rhoR, FLOOR))

    # -- Roe average, for Batten's estimates --------------------------------
    sL_, sR_ = xp.sqrt(rhoL), xp.sqrt(rhoR)
    den = sL_ + sR_
    vx = (sL_ * vxL + sR_ * vxR) / den
    vy = (sL_ * vyL + sR_ * vyR) / den
    H = (sL_ * HL + sR_ * HR) / den
    vn_t = vx * nx + vy * ny
    a_t = xp.sqrt(
        xp.maximum((gamma - 1.0) * (H - 0.5 * (vx * vx + vy * vy)), FLOOR)
    )

    SL = xp.minimum(vnL - aL, vn_t - a_t)
    SR = xp.maximum(vnR + aR, vn_t + a_t)

    if low_mach > 0.0:
        # shrink only the acoustic part of each estimate, about the fluid speed
        mach = xp.maximum(xp.abs(vnL) / aL, xp.abs(vnR) / aR)
        phi = xp.minimum(1.0, mach / low_mach)
        SL = xp.minimum(vnL, vn_t) - phi * (xp.minimum(vnL, vn_t) - SL)
        SR = xp.maximum(vnR, vn_t) + phi * (SR - xp.maximum(vnR, vn_t))

    # -- contact speed -------------------------------------------------------
    mL = rhoL * (SL - vnL)
    mR = rhoR * (SR - vnR)
    SM = (rhoR * vnR * (SR - vnR) - rhoL * vnL * (SL - vnL) + pL - pR) / xp.where(
        xp.abs(mR - mL) > FLOOR, mR - mL, FLOOR
    )

    FL = normal_flux(UL, nx, ny, gamma, xp=xp)
    FR = normal_flux(UR, nx, ny, gamma, xp=xp)

    def star(U, rho, vx_, vy_, p, vn, S):
        """``U* `` for one side: contact normal velocity, own tangential part."""
        fac = (S - vn) / xp.where(xp.abs(S - SM) > FLOOR, S - SM, FLOOR)
        rho_s = rho * fac
        vxs = vx_ + (SM - vn) * nx
        vys = vy_ + (SM - vn) * ny
        E = U[..., 3] / rho
        Es = E + (SM - vn) * (
            SM + p / xp.where(xp.abs(rho * (S - vn)) > FLOOR, rho * (S - vn), FLOOR)
        )
        return xp.stack([rho_s, rho_s * vxs, rho_s * vys, rho_s * Es], axis=-1)

    UsL = star(UL, rhoL, vxL, vyL, pL, vnL, SL)
    UsR = star(UR, rhoR, vxR, vyR, pR, vnR, SR)

    FsL = FL + SL[..., None] * (UsL - UL)
    FsR = FR + SR[..., None] * (UsR - UR)

    # branchless selection, so the same code runs under jax.jit
    flux = xp.where(
        (SL >= 0.0)[..., None],
        FL,
        xp.where(
            (SR <= 0.0)[..., None],
            FR,
            xp.where((SM >= 0.0)[..., None], FsL, FsR),
        ),
    )
    max_speed = xp.maximum(xp.abs(SL), xp.abs(SR))
    return FluxResult(flux, max_speed)


def ausm_flux(
    UL, UR, nx, ny, gamma: float, *,
    cutoff_mach: float = 0.2, xp=np,
) -> FluxResult:
    r"""Liou's AUSM\ :sup:`+`-up flux: a flux-vector splitting, not a Riemann solver.

    Where the Roe flux linearises the Riemann problem and forms
    :math:`|\hat{\mathbf{A}}|\,\Delta\mathbf{U}`, AUSM splits the flux into a
    *convective* part carried by an interface mass flux and a *pressure* part,

    .. math::
        \hat{\mathbf{F}} = \dot{m}_{1/2}\,\boldsymbol{\Psi}_{\mathrm{up}}
                         + p_{1/2}\,(0,\ n_x,\ n_y,\ 0)^T ,
        \qquad \boldsymbol{\Psi} = (1,\ v_x,\ v_y,\ H)^T ,

    with :math:`\boldsymbol{\Psi}` taken from whichever side the mass flux comes
    from.  There is no eigen-decomposition anywhere in it.

    **The split polynomials.**  With :math:`M_{L,R} = v_{n\,L,R}/a_{1/2}`,

    .. math::
        \mathcal{M}^{\pm}_{(1)} = \tfrac{1}{2}(M \pm |M|), \qquad
        \mathcal{M}^{\pm}_{(2)} = \pm\tfrac{1}{4}(M \pm 1)^2 ,

    .. math::
        \mathcal{M}^{\pm}_{(4)} = \begin{cases}
            \mathcal{M}^{\pm}_{(1)} & |M| \ge 1\\
            \mathcal{M}^{\pm}_{(2)}
            \bigl(1 \mp 16\beta\,\mathcal{M}^{\mp}_{(2)}\bigr) & |M| < 1
        \end{cases}
        \qquad \beta = \tfrac{1}{8},

    and a fifth-order pressure splitting :math:`\mathcal{P}^{\pm}_{(5)}` with
    :math:`\alpha = \tfrac{3}{16}(5 f_a^2 - 4)`.  Both families are built so that
    :math:`\mathcal{M}^{+}_{(4)}(M) + \mathcal{M}^{-}_{(4)}(M) = M` and
    :math:`\mathcal{P}^{+}_{(5)}(M) + \mathcal{P}^{-}_{(5)}(M) = 1` *identically*
    -- which is what makes the scheme consistent, and is pinned by a test.

    **What "up" adds.**  Two diffusion terms, which is what makes the scheme
    work at all speeds rather than only transonic:

    .. math::
        M_{1/2} = \mathcal{M}^{+}_{(4)}(M_L) + \mathcal{M}^{-}_{(4)}(M_R)
            - \frac{K_p}{f_a}\max(1 - \sigma \bar{M}^2,\, 0)\,
              \frac{p_R - p_L}{\rho_{1/2} a_{1/2}^2}

    is the **p**\ ressure diffusion, which supplies the velocity--pressure
    coupling a plain AUSM lacks as :math:`M \to 0`, and

    .. math::
        p_{1/2} = \mathcal{P}^{+}_{(5)}(M_L)\,p_L
                + \mathcal{P}^{-}_{(5)}(M_R)\,p_R
                - K_u\,\mathcal{P}^{+}_{(5)}\mathcal{P}^{-}_{(5)}
                  (\rho_L + \rho_R)\,(f_a a_{1/2})\,(v_{n R} - v_{n L})

    is the **u** velocity diffusion.  Constants are Liou's:
    :math:`K_p = 0.25`, :math:`K_u = 0.75`, :math:`\sigma = 1`.

    **The interface sound speed** is the *numerical* one built from the critical
    speed, not an average:

    .. math::
        a_*^2 = \frac{2(\gamma-1)}{\gamma+1} H, \qquad
        \tilde{a} = \frac{a_*^2}{\max(a_*,\ |v_n|)}, \qquad
        a_{1/2} = \min(\tilde{a}_L,\ \tilde{a}_R).

    This is the part that lets the scheme capture a stationary normal shock
    without an interior point, and it is why ``a_half`` is not simply
    :math:`\tfrac{1}{2}(a_L + a_R)`.

    Why have it alongside the Roe flux
    ----------------------------------
    * **No carbuncle.**  The Roe flux admits the odd--even decoupling that shows
      up as a carbuncle ahead of a blunt body; AUSM does not.  Nothing in this
      nozzle has produced one, so this is insurance rather than a fix.
    * **Low-Mach accuracy.**  The Roe flux loses accuracy as :math:`M \to 0`;
      the ``-up`` terms are designed for exactly that, and the inlet here runs
      at :math:`M \approx 0.1`--:math:`0.3`.
    * **It is a genuinely different philosophy**, which makes "does the answer
      depend on the flux?" a question a student can actually answer.

    It is *not* expected to fix the shocked-case divergence: that is driven by
    the cell average leaving the physical state, which no choice of interface
    flux addresses.  See the README.

    Parameters
    ----------
    cutoff_mach
        Liou's :math:`M_{co}`: a reference Mach number of the order of the
        *smallest* Mach number in the flow, which sets

        .. math::
            M_0 = \min\bigl(1,\ \max(\bar{M},\ M_{co})\bigr), \qquad
            f_a = M_0 (2 - M_0) .

        It is **not** a regularisation epsilon, and this is the one place the
        scheme will bite you.  :math:`f_a` appears as :math:`K_p / f_a`, so a
        small :math:`M_{co}` makes that coefficient *large*: with
        :math:`M_{co} = 0`, a 1% pressure jump between two states **at rest**
        gives a mass flux of :math:`-9.8\times 10^{4}` where the Roe flux gives
        :math:`-3.5\times 10^{-3}`, and the march dies within 51 iterations.

        Liou's own guidance is :math:`M_{co} \sim M_\infty`.  The default
        ``0.2`` suits this nozzle, whose inlet runs at
        :math:`M \approx 0.1`--:math:`0.3`.  Raising it towards 1 recovers plain
        AUSM\ :sup:`+` by switching the low-Mach enhancement off; lowering it
        below the actual minimum Mach number of the flow is what breaks it.
    """
    rhoL, vxL, vyL, pL, HL = primitives(UL, gamma, xp=xp)
    rhoR, vxR, vyR, pR, HR = primitives(UR, gamma, xp=xp)

    vnL = vxL * nx + vyL * ny
    vnR = vxR * nx + vyR * ny

    # -- interface sound speed from the critical speed (Liou's \tilde{a}) ----
    k = 2.0 * (gamma - 1.0) / (gamma + 1.0)
    astarL = xp.sqrt(xp.maximum(k * HL, FLOOR))
    astarR = xp.sqrt(xp.maximum(k * HR, FLOOR))
    aL = astarL * astarL / xp.maximum(astarL, xp.abs(vnL))
    aR = astarR * astarR / xp.maximum(astarR, xp.abs(vnR))
    a_half = xp.minimum(aL, aR)

    ML = vnL / a_half
    MR = vnR / a_half

    # -- f_a, and the alpha it sets --------------------------------------------
    mbar2 = 0.5 * (vnL * vnL + vnR * vnR) / (a_half * a_half)
    # M_co is not an epsilon.  The pressure-diffusion term below carries
    # K_p / f_a, so driving f_a towards zero makes that coefficient *diverge*:
    # at a stagnation point with M_co = 0 and a 1% pressure jump, the mass flux
    # comes out as -9.8e4 instead of -3.5e-3.  Liou's M_co is a reference Mach
    # number of the order of the smallest Mach number in the flow, and it is
    # what keeps f_a -- and so the coefficient -- O(1).
    m0 = xp.sqrt(xp.minimum(1.0, xp.maximum(mbar2, cutoff_mach * cutoff_mach)))
    fa = m0 * (2.0 - m0)
    alpha = 0.1875 * (5.0 * fa * fa - 4.0)

    beta = 0.125
    m2p = 0.25 * (ML + 1.0) ** 2        # M^+_(2) evaluated at ML
    m2m = -0.25 * (MR - 1.0) ** 2       # M^-_(2) evaluated at MR

    m4p = xp.where(
        xp.abs(ML) >= 1.0,
        0.5 * (ML + xp.abs(ML)),
        0.25 * (ML + 1.0) ** 2 * (1.0 + 4.0 * beta * (ML - 1.0) ** 2),
    )
    m4m = xp.where(
        xp.abs(MR) >= 1.0,
        0.5 * (MR - xp.abs(MR)),
        -0.25 * (MR - 1.0) ** 2 * (1.0 + 4.0 * beta * (MR + 1.0) ** 2),
    )

    # -- interface Mach number, with the pressure-diffusion term --------------
    rho_half = 0.5 * (rhoL + rhoR)
    kp, ku, sigma = 0.25, 0.75, 1.0
    mp = -(kp / fa) * xp.maximum(1.0 - sigma * mbar2, 0.0) * (pR - pL) / (
        rho_half * a_half * a_half
    )
    m_half = m4p + m4m + mp

    # -- interface pressure, with the velocity-diffusion term -----------------
    p5p = xp.where(
        xp.abs(ML) >= 1.0,
        0.5 * (1.0 + xp.sign(ML)),
        m2p * ((2.0 - ML) + 4.0 * alpha * ML * (ML - 1.0) ** 2),
    )
    p5m = xp.where(
        xp.abs(MR) >= 1.0,
        0.5 * (1.0 - xp.sign(MR)),
        -m2m * ((2.0 + MR) - 4.0 * alpha * MR * (MR + 1.0) ** 2),
    )
    p_half = (
        p5p * pL + p5m * pR
        - ku * p5p * p5m * (rhoL + rhoR) * (fa * a_half) * (vnR - vnL)
    )

    # -- assemble: the convective part is fully upwind ------------------------
    forward = (m_half > 0.0)
    rho_up = xp.where(forward, rhoL, rhoR)
    mdot = a_half * m_half * rho_up

    f = forward[..., None]
    psi = xp.where(
        f,
        xp.stack([xp.ones_like(vxL), vxL, vyL, HL], axis=-1),
        xp.stack([xp.ones_like(vxR), vxR, vyR, HR], axis=-1),
    )
    nvec = xp.stack(
        [xp.zeros_like(nx * xp.ones_like(pL)), nx * xp.ones_like(pL),
         ny * xp.ones_like(pL), xp.zeros_like(pL)],
        axis=-1,
    )
    flux = mdot[..., None] * psi + p_half[..., None] * nvec

    # the signal speed the time step needs is the same physical quantity the Roe
    # flux reports, so it is computed from the true sound speeds rather than from
    # the numerical interface one
    cL = xp.sqrt(xp.maximum(gamma * pL / rhoL, FLOOR))
    cR = xp.sqrt(xp.maximum(gamma * pR / rhoR, FLOOR))
    max_speed = xp.maximum(xp.abs(vnL) + cL, xp.abs(vnR) + cR)
    return FluxResult(flux, max_speed)


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


def outflow_flux(U, nx, ny, gamma: float, *, p_back: float, xp=np) -> FluxResult:
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

    # -- blend with the extrapolated (supersonic) state
    supersonic = (vn / a >= 1.0)[..., None]
    U_b = xp.where(supersonic, U, U_sub)

    flux = normal_flux(U_b, nx, ny, gamma, xp=xp)
    rho_o, vx_o, vy_o, p_o, _ = primitives(U_b, gamma, xp=xp)
    a_o = xp.sqrt(gamma * p_o / rho_o)
    return FluxResult(flux, xp.abs(vx_o * nx + vy_o * ny) + a_o)
