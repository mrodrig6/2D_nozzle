"""Every interface flux this project implemented, kept for the next codebase.

This file is an **archive, not a dependency**.  Nothing in `src/` imports it and
nothing here is run by the test suite.  It exists to be handed to another
project -- specifically one solving flow over an airfoil, where a shock sits on
the body surface and the trade-offs below come out very differently.

What is here
------------
Four fluxes, lifted verbatim from this repository, with their full derivations
in the docstrings:

===============  =========================================================
`roe_flux`       Roe's approximate Riemann solver with the Harten-Hyman
                 entropy fix.  **Kept** in this project as the default.
`hllc_flux`      HLLC with Batten's wave speeds.  **Kept**: provably
                 positivity-preserving, no tunable constant.
`slau2_flux`     SLAU2, Shima & Kitamura's parameter-free low-dissipation
                 AUSM-family flux.  **Removed** from this project: it does
                 not converge in a steady DG march here.
`ausm_flux`      Liou's AUSM(+)-up.  **Removed** earlier, for the same
                 reason plus a tunable-parameter failure.
===============  =========================================================

Each takes `(UL, UR, nx, ny, gamma)` with the state ordered
`(rho, rho*vx, rho*vy, rho*E)` and the last axis of length 4, returns a
`FluxResult(flux, max_speed)`, and is branchless so it runs under NumPy or JAX.
`primitives` and `normal_flux` are included so the file stands alone.

Read this before reusing any of it
----------------------------------
The two schemes marked **Removed** are not bad schemes.  They fail *here*, in a
specific and well-characterised way, and the reason that happened is the single
most useful thing to carry forward:

* This solver refuses any operating point with a shock **inside** the nozzle, so
  the domain is shock free by construction.
* SLAU2 and AUSM(+)-up are designed for the opposite situation: robustness
  against **shock anomalies** -- the carbuncle, hypersonic heating -- at a
  captured strong shock.
* Measured here, both reach very nearly the right answer (thrust and exit Mach
  within 0.7% of Roe) and then sit in a bounded limit cycle whose residual never
  decays, four orders above Roe's.  Six mechanisms were tested; all six were
  refuted, including the interfacial-sound-speed variant the source paper itself
  recommends.  The cause is still not identified.
* Kitamura's 2016 Table 2 scores twelve fluxes on the standard 1.5D carbuncle
  test.  Against what converges *here*, the ranking is **inverted**:

      Roe (E-fix)   0/20   <- worst there, best here
      Roe           8/20
      HLLC          8/20   <- also converges here
      AUSM(+)-up   16/20
      van Leer     20/20
      SLAU2        20/20   <- best there, fails here
      AUSM(+)-up2  20/20

  The entropy fix that makes Roe usable in this DG march is exactly what takes
  Roe from 8 to 0 on their test, which Kitamura & Shima (2013) state directly:
  "too much dissipation addition to the flux yields 1D stability but in expense
  of Multi-D stability".

**For an airfoil with a shock on the surface, that inversion very likely runs
the other way.**  Property A -- robustness at a captured shock -- is exercised
there and is not exercised here, so SLAU2 and AUSM(+)-up2 become strong
candidates rather than liabilities.  Do not carry this project's conclusion
("use Roe or HLLC") across; carry the *measurement*, and re-measure.

References
----------
1. P. L. Roe, J. Comput. Phys. 43, 357-372, 1981.
2. A. Harten and J. M. Hyman, J. Comput. Phys. 50, 235-269, 1983.
3. P. Batten et al., SIAM J. Sci. Comput. 18(6), 1553-1570, 1997.
4. M.-S. Liou, J. Comput. Phys. 214, 137-170, 2006.  (AUSM(+)-up)
5. E. Shima and K. Kitamura, AIAA J. 49(8), 1693-1709, 2011.  (SLAU)
6. K. Kitamura and E. Shima, J. Comput. Phys. 245, 62-83, 2013.  (SLAU2 and
   AUSM(+)-up2; Eqs. (2.3) and (3.5) are what `slau2_flux` implements.)
7. K. Kitamura, Computers and Fluids 129, 134-145, 2016.  (The twelve-flux
   comparison quoted above.)
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


def primitives(U, gamma: float, xp=np):
    r"""Return ``(rho, vx, vy, p, H)`` with ``rho`` and ``p`` floored positive."""
    rho = xp.maximum(U[..., 0], FLOOR)
    vx = U[..., 1] / rho
    vy = U[..., 2] / rho
    p = xp.maximum((gamma - 1.0) * (U[..., 3] - 0.5 * rho * (vx * vx + vy * vy)), FLOOR)
    H = (U[..., 3] + p) / rho
    return rho, vx, vy, p, H


def normal_flux(U, nx, ny, gamma: float, xp=np):
    r"""The projected flux :math:`F n_x + G n_y`."""
    rho, vx, vy, p, H = primitives(U, gamma, xp=xp)
    vn = vx * nx + vy * ny
    return xp.stack(
        [rho * vn, rho * vx * vn + p * nx, rho * vy * vn + p * ny, rho * H * vn], axis=-1
    )


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
    UL,
    UR,
    nx,
    ny,
    gamma: float,
    *,
    low_mach: float = 0.0,
    xp=np,
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
    a_t = xp.sqrt(xp.maximum((gamma - 1.0) * (H - 0.5 * (vx * vx + vy * vy)), FLOOR))

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


def slau2_flux(UL, UR, nx, ny, gamma: float, *, xp=np) -> FluxResult:
    r"""SLAU2: the parameter-free low-dissipation AUSM-family flux.

    Shima & Kitamura's SLAU, with the Kitamura-Shima pressure flux (SLAU2).
    An AUSM-family scheme splits the interface flux into a *mass* flux carrying
    the convective field and a *pressure* flux carrying the acoustic field,

    .. math::
        \hat{\mathbf{F}} = \frac{\dot{m} + |\dot{m}|}{2}\,\boldsymbol{\Psi}_L
                         + \frac{\dot{m} - |\dot{m}|}{2}\,\boldsymbol{\Psi}_R
                         + \tilde{p}\,\mathbf{N},

    with :math:`\boldsymbol{\Psi} = (1,\ v_x,\ v_y,\ H)^T` and
    :math:`\mathbf{N} = (0,\ n_x,\ n_y,\ 0)^T`, so the mass flux is upwinded by
    its own sign and the pressure appears only in the momentum equations.

    Mass flux
    ---------
    .. math::
        \dot{m} = \tfrac{1}{2}\Bigl[
            \rho_L\bigl(v_{nL} + |\overline{v_n}|_L\bigr)
          + \rho_R\bigl(v_{nR} - |\overline{v_n}|_R\bigr)
          - \frac{\chi}{\bar{a}}\,(p_R - p_L)\Bigr]

    .. math::
        |\overline{v_n}| =
            \frac{\rho_L |v_{nL}| + \rho_R |v_{nR}|}{\rho_L + \rho_R},\qquad
        |\overline{v_n}|_{L} = (1-g)\,|\overline{v_n}| + g\,|v_{nL}|,\qquad
        |\overline{v_n}|_{R} = (1-g)\,|\overline{v_n}| + g\,|v_{nR}|

    .. math::
        g = -\max\bigl[\min(M_L, 0),\ -1\bigr]\cdot
             \min\bigl[\max(M_R, 0),\ 1\bigr] \in [0, 1],
        \qquad M_{L,R} = \frac{v_{n\,L,R}}{\bar{a}}

    .. math::
        \hat{M} = \min\!\left(1,\ \frac{1}{\bar{a}}
            \sqrt{\tfrac{1}{2}\bigl(|\mathbf{v}_L|^2 + |\mathbf{v}_R|^2\bigr)}
            \right),
        \qquad \chi = (1 - \hat{M})^2

    with :math:`\bar{a} = \tfrac{1}{2}(a_L + a_R)` and
    :math:`\bar{\rho} = \tfrac{1}{2}(\rho_L + \rho_R)`.  The switch :math:`g` is
    nonzero only where :math:`M_L < 0 < M_R`, i.e. at an expansion, and there it
    replaces the density-weighted mean normal speed by each side's own -- which
    is what keeps the mass flux from going the wrong way through a sonic point.

    Pressure flux (this is the SLAU2 part)
    --------------------------------------
    .. math::
        \tilde{p} = \frac{p_L + p_R}{2}
          + \frac{\beta_L - \beta_R}{2}\,(p_L - p_R)
          + \sqrt{\tfrac{1}{2}\bigl(|\mathbf{v}_L|^2 + |\mathbf{v}_R|^2\bigr)}\,
            (\beta_L + \beta_R - 1)\,\bar{\rho}\,\bar{a}

    .. math::
        \beta_L = \begin{cases}
            \tfrac{1}{4}(2 - M_L)(M_L + 1)^2, & |M_L| < 1\\
            \tfrac{1}{2}\bigl(1 + \operatorname{sign} M_L\bigr), & |M_L| \ge 1
        \end{cases}
        \qquad
        \beta_R = \begin{cases}
            \tfrac{1}{4}(2 + M_R)(M_R - 1)^2, & |M_R| < 1\\
            \tfrac{1}{2}\bigl(1 - \operatorname{sign} M_R\bigr), & |M_R| \ge 1
        \end{cases}

    The original SLAU wrote that third term as
    :math:`(1-\chi)(\beta_L + \beta_R - 1)\,\tfrac{1}{2}(p_L + p_R)`.  Replacing
    :math:`\tfrac{1}{2}(p_L+p_R)` by :math:`\bar{\rho}\bar{a}|\mathbf{v}|` is the
    whole of SLAU2, and it matters because the ratio of the two is
    :math:`O(1/(\gamma M))`: the SLAU form thins out exactly where a strong shock
    needs dissipation, which is where SLAU showed shock anomalies.

    Why this one, out of the AUSM family
    ------------------------------------
    **It has no tunable parameters.**  That is the reason to prefer it here, and
    it is not an aesthetic point.  AUSM\ :sup:`+`-up carries :math:`K_p`,
    :math:`K_u` and a cutoff Mach number :math:`M_{co}`; this package carried
    that scheme briefly and removed it, and one of the two failures found along
    the way was traceable to :math:`M_{co}`: read as an epsilon and floored at
    ``1e-8`` it makes :math:`K_p/f_a` diverge, which turned a 1% pressure
    difference in still air into a mass flux of :math:`-9.8\times 10^4`.  SLAU2
    has no such constant to get wrong -- :math:`g` and :math:`\chi` are built
    from the states themselves.

    **Its low-Mach dissipation scales correctly** (:math:`O(M^2)`) without a
    reference Mach number, via :math:`\chi \to 1` as :math:`M \to 0` in the mass
    flux and the :math:`|\mathbf{v}|`-weighted term vanishing in the pressure
    flux.

    .. note::
       **Checked against the paper, equation by equation.**  Kitamura & Shima,
       *J. Comput. Phys.* **245**, 62-83 (2013).  The mass flux above is their
       Eq. (2.3i); :math:`|\overline{v_n}|_{L,R}` is (2.3j);
       :math:`|\overline{v_n}|` is (2.3k); :math:`g` is (2.3l);
       :math:`\hat{M}` is (2.3e); :math:`\chi` is (2.3d);
       :math:`\bar{a} = \tfrac{1}{2}(a_L + a_R)` is (2.3h), their default;
       the split polynomials are (2.3f); and the SLAU2 pressure flux is
       Eq. (3.5), restated as (A.2) in their Appendix.  Every one matches.

       Independently, ``tests/test_physics.py`` pins five properties a
       transcription error would break: consistency
       (:math:`\hat{\mathbf{F}}(\mathbf{U},\mathbf{U}) = \mathbf{F}\cdot
       \mathbf{n}` exactly), conservation under swapping the two sides and the
       normal, exact preservation of a contact discontinuity, upwinding of every
       convected quantity in the supersonic limit, and the :math:`O(M^2)`
       pressure-dissipation scaling, measured at 4.00 per halving of :math:`M`.

    .. note::
       **Not fully upwind at a supersonic jump**, and that is the design, not a
       transcription slip.  Where :math:`g = 0` the mass flux can be rewritten

       .. math::
           \dot{m} = \tfrac{1}{2}\bigl(\rho_L v_{nL} + \rho_R v_{nR}\bigr)
             + \tfrac{1}{2}|\overline{v_n}|\,(\rho_L - \rho_R),

       a central flux plus dissipation on :math:`\Delta\rho` alone, at the fluid
       speed -- this is the "simple low-dissipation" of the name.  Pure upwinding
       would instead use each side's own speed.  So for two supersonic states
       either side of a jump, :math:`\dot{m}` differs from
       :math:`\rho_L v_{nL}` (5% for :math:`M \approx 2.5` with a 2.5:1 density
       ratio), while :math:`\boldsymbol{\Psi}` and :math:`\tilde{p}` are still
       taken entirely from the upwind side.  The difference vanishes with the
       jump, so it costs nothing in smooth supersonic flow, and a steady shock
       always has :math:`M_n < 1` behind it and so never sits in this case.

    Positivity
    ----------
    Unlike HLLC, SLAU2 comes with no positivity proof, so the Zhang-Shu
    limiter's theorem does not cover it either (see :func:`hllc_flux`).  Use
    ``flux='hllc'`` when that guarantee is what you want.
    """
    rhoL, vxL, vyL, pL, HL = primitives(UL, gamma, xp=xp)
    rhoR, vxR, vyR, pR, HR = primitives(UR, gamma, xp=xp)

    vnL = vxL * nx + vyL * ny
    vnR = vxR * nx + vyR * ny
    aL = xp.sqrt(xp.maximum(gamma * pL / rhoL, FLOOR))
    aR = xp.sqrt(xp.maximum(gamma * pR / rhoR, FLOOR))

    a_bar = 0.5 * (aL + aR)
    rho_bar = 0.5 * (rhoL + rhoR)
    ML = vnL / a_bar
    MR = vnR / a_bar

    # |v| at the interface, the scale the SLAU2 pressure dissipation rides on
    v_bar = xp.sqrt(0.5 * (vxL * vxL + vyL * vyL + vxR * vxR + vyR * vyR))
    m_hat = xp.minimum(1.0, v_bar / a_bar)
    chi = (1.0 - m_hat) ** 2

    # -- mass flux ----------------------------------------------------------
    # g is nonzero only at an expansion (M_L < 0 < M_R); elsewhere one of the
    # two factors is zero by construction, so no branch is needed.
    g = -xp.maximum(xp.minimum(ML, 0.0), -1.0) * xp.minimum(xp.maximum(MR, 0.0), 1.0)
    vn_mean = (rhoL * xp.abs(vnL) + rhoR * xp.abs(vnR)) / (rhoL + rhoR)
    vn_L = (1.0 - g) * vn_mean + g * xp.abs(vnL)
    vn_R = (1.0 - g) * vn_mean + g * xp.abs(vnR)

    mdot = 0.5 * (rhoL * (vnL + vn_L) + rhoR * (vnR - vn_R) - (chi / a_bar) * (pR - pL))

    # -- pressure flux ------------------------------------------------------
    betaL = xp.where(
        xp.abs(ML) < 1.0,
        0.25 * (2.0 - ML) * (ML + 1.0) ** 2,
        0.5 * (1.0 + xp.sign(ML)),
    )
    betaR = xp.where(
        xp.abs(MR) < 1.0,
        0.25 * (2.0 + MR) * (MR - 1.0) ** 2,
        0.5 * (1.0 - xp.sign(MR)),
    )
    p_tilde = (
        0.5 * (pL + pR)
        + 0.5 * (betaL - betaR) * (pL - pR)
        + v_bar * (betaL + betaR - 1.0) * rho_bar * a_bar
    )

    # -- assemble -----------------------------------------------------------
    psiL = xp.stack([xp.ones_like(rhoL), vxL, vyL, HL], axis=-1)
    psiR = xp.stack([xp.ones_like(rhoR), vxR, vyR, HR], axis=-1)
    zero = xp.zeros_like(rhoL)
    N = xp.stack([zero, nx * xp.ones_like(rhoL), ny * xp.ones_like(rhoL), zero], axis=-1)

    mp = (0.5 * (mdot + xp.abs(mdot)))[..., None]
    mm = (0.5 * (mdot - xp.abs(mdot)))[..., None]
    flux = mp * psiL + mm * psiR + p_tilde[..., None] * N

    max_speed = xp.maximum(xp.abs(vnL) + aL, xp.abs(vnR) + aR)
    return FluxResult(flux, max_speed)


def ausm_flux(
    UL,
    UR,
    nx,
    ny,
    gamma: float,
    *,
    cutoff_mach: float = 0.2,
    xp=np,
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
    m2p = 0.25 * (ML + 1.0) ** 2  # M^+_(2) evaluated at ML
    m2m = -0.25 * (MR - 1.0) ** 2  # M^-_(2) evaluated at MR

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
    mp = (
        -(kp / fa) * xp.maximum(1.0 - sigma * mbar2, 0.0) * (pR - pL) / (rho_half * a_half * a_half)
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
    p_half = p5p * pL + p5m * pR - ku * p5p * p5m * (rhoL + rhoR) * (fa * a_half) * (vnR - vnL)

    # -- assemble: the convective part is fully upwind ------------------------
    forward = m_half > 0.0
    rho_up = xp.where(forward, rhoL, rhoR)
    mdot = a_half * m_half * rho_up

    f = forward[..., None]
    psi = xp.where(
        f,
        xp.stack([xp.ones_like(vxL), vxL, vyL, HL], axis=-1),
        xp.stack([xp.ones_like(vxR), vxR, vyR, HR], axis=-1),
    )
    nvec = xp.stack(
        [
            xp.zeros_like(nx * xp.ones_like(pL)),
            nx * xp.ones_like(pL),
            ny * xp.ones_like(pL),
            xp.zeros_like(pL),
        ],
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
