r"""Numba nopython kernels for the DG residual.

This is the performance path.  The mathematics is identical to
:mod:`dgnozzle.assembly`; only the execution strategy differs.  Where the
vectorised backend expresses each step as an ``einsum`` over big temporaries,
these kernels fuse everything into two ``prange`` loops and keep intermediates in
registers:

1. **Edge pass** -- one iteration per global edge: evaluate both traces, call the
   Roe solver or a boundary flux, and store ``flux * |dX/dsigma| * w``.
2. **Element pass** -- one iteration per element: the volume integral, then the
   face integrals gathered through ``face_edge``.

Both loops are embarrassingly parallel *because assembly is a pure gather*.
Each iteration writes only to its own slice, so there are no atomics, no
reduction races, and no need for colouring.

``fastmath=True`` is enabled.  It permits reassociation of floating-point sums,
which changes results in the last couple of digits; ``tests/test_backends.py``
pins agreement with the NumPy backend to 1e-10.
"""

from __future__ import annotations

import numpy as np
from numba import njit, prange

FLOOR = 1e-10

# Boundary tags, mirroring dgnozzle.mesh.BoundaryTag (Numba cannot see IntEnum).
_INFLOW = 1
_OUTFLOW = 2
_WALL = 3
_AXIS = 4

_JIT = {"cache": True, "fastmath": True, "nogil": True}


@njit(inline="always", **_JIT)
def _pressure(U0, U1, U2, U3, gamma):
    rho = max(U0, FLOOR)
    p = (gamma - 1.0) * (U3 - 0.5 * (U1 * U1 + U2 * U2) / rho)
    return max(p, FLOOR)


@njit(inline="always", **_JIT)
def _normal_flux(U0, U1, U2, U3, nx, ny, gamma, out):
    rho = max(U0, FLOOR)
    vx = U1 / rho
    vy = U2 / rho
    p = _pressure(U0, U1, U2, U3, gamma)
    H = (U3 + p) / rho
    vn = vx * nx + vy * ny
    out[0] = rho * vn
    out[1] = rho * vx * vn + p * nx
    out[2] = rho * vy * vn + p * ny
    out[3] = rho * H * vn


@njit(inline="always", **_JIT)
def _roe(UL, UR, nx, ny, gamma, efix, out):
    """Roe flux with the Harten-Hyman entropy fix.  Returns the max signal speed."""
    rL = max(UL[0], FLOOR)
    vxL = UL[1] / rL
    vyL = UL[2] / rL
    pL = _pressure(UL[0], UL[1], UL[2], UL[3], gamma)
    HL = (UL[3] + pL) / rL

    rR = max(UR[0], FLOOR)
    vxR = UR[1] / rR
    vyR = UR[2] / rR
    pR = _pressure(UR[0], UR[1], UR[2], UR[3], gamma)
    HR = (UR[3] + pR) / rR

    vnL = vxL * nx + vyL * ny
    vnR = vxR * nx + vyR * ny

    fL0 = rL * vnL
    fL1 = rL * vxL * vnL + pL * nx
    fL2 = rL * vyL * vnL + pL * ny
    fL3 = rL * HL * vnL
    fR0 = rR * vnR
    fR1 = rR * vxR * vnR + pR * nx
    fR2 = rR * vyR * vnR + pR * ny
    fR3 = rR * HR * vnR

    sL = np.sqrt(rL)
    sR = np.sqrt(rR)
    den = sL + sR
    vx = (sL * vxL + sR * vxR) / den
    vy = (sL * vyL + sR * vyR) / den
    H = (sL * HL + sR * HR) / den
    q2 = vx * vx + vy * vy
    c2 = (gamma - 1.0) * (H - 0.5 * q2)
    if c2 < FLOOR:
        c2 = FLOOR
    c = np.sqrt(c2)
    vn = vx * nx + vy * ny

    l1 = abs(vn + c)
    l2 = abs(vn - c)
    l3 = abs(vn)
    smax = l1 if l1 > l2 else l2

    eps = efix * c
    if eps > 0.0:
        if l1 < eps:
            l1 = (l1 * l1 + eps * eps) / (2.0 * eps)
        if l2 < eps:
            l2 = (l2 * l2 + eps * eps) / (2.0 * eps)
        if l3 < eps:
            l3 = (l3 * l3 + eps * eps) / (2.0 * eps)

    d0 = UR[0] - UL[0]
    d1 = UR[1] - UL[1]
    d2 = UR[2] - UL[2]
    d3 = UR[3] - UL[3]

    G1 = (gamma - 1.0) * (0.5 * q2 * d0 - vx * d1 - vy * d2 + d3)
    G2 = -vn * d0 + d1 * nx + d2 * ny

    s1 = 0.5 * (l1 + l2)
    s2 = 0.5 * (l1 - l2)
    C1 = (G1 / c2) * (s1 - l3) + (G2 / c) * s2
    C2 = (G1 / c) * s2 + (s1 - l3) * G2

    out[0] = 0.5 * (fL0 + fR0) - 0.5 * (l3 * d0 + C1)
    out[1] = 0.5 * (fL1 + fR1) - 0.5 * (l3 * d1 + C1 * vx + C2 * nx)
    out[2] = 0.5 * (fL2 + fR2) - 0.5 * (l3 * d2 + C1 * vy + C2 * ny)
    out[3] = 0.5 * (fL3 + fR3) - 0.5 * (l3 * d3 + C1 * H + C2 * vn)
    return smax


@njit(inline="always", **_JIT)
def _hllc(UL, UR, nx, ny, gamma, low_mach, out):
    """HLLC with Batten's wave speeds.  Returns the max signal speed.

    Mirrors :func:`dgnozzle.physics.hllc_flux`; ``tests/test_backends.py`` pins
    the two against each other, because two copies of a flux is two chances to
    get it wrong.
    """
    rL = max(UL[0], FLOOR)
    vxL = UL[1] / rL
    vyL = UL[2] / rL
    pL = _pressure(UL[0], UL[1], UL[2], UL[3], gamma)
    HL = (UL[3] + pL) / rL

    rR = max(UR[0], FLOOR)
    vxR = UR[1] / rR
    vyR = UR[2] / rR
    pR = _pressure(UR[0], UR[1], UR[2], UR[3], gamma)
    HR = (UR[3] + pR) / rR

    vnL = vxL * nx + vyL * ny
    vnR = vxR * nx + vyR * ny
    aL = np.sqrt(max(gamma * pL / rL, FLOOR))
    aR = np.sqrt(max(gamma * pR / rR, FLOOR))

    # Roe average, for Batten's estimates
    sL_ = np.sqrt(rL)
    sR_ = np.sqrt(rR)
    den = sL_ + sR_
    vx = (sL_ * vxL + sR_ * vxR) / den
    vy = (sL_ * vyL + sR_ * vyR) / den
    H = (sL_ * HL + sR_ * HR) / den
    vn_t = vx * nx + vy * ny
    c2 = (gamma - 1.0) * (H - 0.5 * (vx * vx + vy * vy))
    if c2 < FLOOR:
        c2 = FLOOR
    a_t = np.sqrt(c2)

    SL = min(vnL - aL, vn_t - a_t)
    SR = max(vnR + aR, vn_t + a_t)

    if low_mach > 0.0:
        mach = max(abs(vnL) / aL, abs(vnR) / aR)
        phi = min(1.0, mach / low_mach)
        baseL = min(vnL, vn_t)
        baseR = max(vnR, vn_t)
        SL = baseL - phi * (baseL - SL)
        SR = baseR + phi * (SR - baseR)

    smax = max(abs(SL), abs(SR))

    fL0 = rL * vnL
    fL1 = rL * vxL * vnL + pL * nx
    fL2 = rL * vyL * vnL + pL * ny
    fL3 = rL * HL * vnL
    if SL >= 0.0:
        out[0] = fL0
        out[1] = fL1
        out[2] = fL2
        out[3] = fL3
        return smax

    fR0 = rR * vnR
    fR1 = rR * vxR * vnR + pR * nx
    fR2 = rR * vyR * vnR + pR * ny
    fR3 = rR * HR * vnR
    if SR <= 0.0:
        out[0] = fR0
        out[1] = fR1
        out[2] = fR2
        out[3] = fR3
        return smax

    mL = rL * (SL - vnL)
    mR = rR * (SR - vnR)
    dm = mR - mL
    if abs(dm) < FLOOR:
        dm = FLOOR
    SM = (rR * vnR * (SR - vnR) - rL * vnL * (SL - vnL) + pL - pR) / dm

    if SM >= 0.0:
        den2 = SL - SM
        if abs(den2) < FLOOR:
            den2 = FLOOR
        fac = (SL - vnL) / den2
        rs = rL * fac
        vxs = vxL + (SM - vnL) * nx
        vys = vyL + (SM - vnL) * ny
        den3 = rL * (SL - vnL)
        if abs(den3) < FLOOR:
            den3 = FLOOR
        Es = UL[3] / rL + (SM - vnL) * (SM + pL / den3)
        out[0] = fL0 + SL * (rs - UL[0])
        out[1] = fL1 + SL * (rs * vxs - UL[1])
        out[2] = fL2 + SL * (rs * vys - UL[2])
        out[3] = fL3 + SL * (rs * Es - UL[3])
    else:
        den2 = SR - SM
        if abs(den2) < FLOOR:
            den2 = FLOOR
        fac = (SR - vnR) / den2
        rs = rR * fac
        vxs = vxR + (SM - vnR) * nx
        vys = vyR + (SM - vnR) * ny
        den3 = rR * (SR - vnR)
        if abs(den3) < FLOOR:
            den3 = FLOOR
        Es = UR[3] / rR + (SM - vnR) * (SM + pR / den3)
        out[0] = fR0 + SR * (rs - UR[0])
        out[1] = fR1 + SR * (rs * vxs - UR[1])
        out[2] = fR2 + SR * (rs * vys - UR[2])
        out[3] = fR3 + SR * (rs * Es - UR[3])
    return smax


@njit(inline="always", **_JIT)
def _wall(Ub, nx, ny, gamma, out):
    rho = max(Ub[0], FLOOR)
    vx = Ub[1] / rho
    vy = Ub[2] / rho
    vn = vx * nx + vy * ny
    vtx = vx - vn * nx
    vty = vy - vn * ny
    pb = (gamma - 1.0) * (Ub[3] - 0.5 * rho * (vtx * vtx + vty * vty))
    if pb < FLOOR:
        pb = FLOOR
    out[0] = 0.0
    out[1] = pb * nx
    out[2] = pb * ny
    out[3] = 0.0
    return np.sqrt(gamma * pb / rho)


@njit(inline="always", **_JIT)
def _inflow(Ub, nx, ny, gamma, at2, at, rho_t, ca, sa, out):
    rho = max(Ub[0], FLOOR)
    vx = Ub[1] / rho
    vy = Ub[2] / rho
    p = _pressure(Ub[0], Ub[1], Ub[2], Ub[3], gamma)
    a = np.sqrt(gamma * p / rho)
    vn = vx * nx + vy * ny
    Jp = vn + 2.0 * a / (gamma - 1.0)

    beta = (Jp / at) * (Jp / at)
    nd = nx * ca + ny * sa
    aa = 0.5 * (gamma - 1.0) * beta - nd * nd
    bb = -4.0 * nd / (gamma - 1.0)
    cc = beta - (2.0 / (gamma - 1.0)) ** 2

    tiny = 1e-12
    if abs(aa) < tiny:
        aa = tiny if aa >= 0.0 else -tiny
    disc = bb * bb - 4.0 * aa * cc
    if disc < 0.0:
        disc = 0.0
    sq = np.sqrt(disc)
    r1 = (-bb + sq) / (2.0 * aa)
    r2 = (-bb - sq) / (2.0 * aa)

    # smallest non-negative root is the physical branch
    M = -1.0
    if r1 >= 0.0:
        M = r1
    if r2 >= 0.0 and (M < 0.0 or r2 < M):
        M = r2
    if M < 0.0:
        M = 0.0
    if M > 1.0:
        M = 1.0

    fac = 1.0 + 0.5 * (gamma - 1.0) * M * M
    ab = np.sqrt(at2 / fac)
    qb = M * ab
    vxb = qb * ca
    vyb = qb * sa
    rhob = rho_t * fac ** (-1.0 / (gamma - 1.0))
    pb = rhob * ab * ab / gamma
    Hb = at2 / (gamma - 1.0)
    vnb = vxb * nx + vyb * ny

    out[0] = rhob * vnb
    out[1] = rhob * vxb * vnb + pb * nx
    out[2] = rhob * vyb * vnb + pb * ny
    out[3] = rhob * Hb * vnb
    return abs(vnb) + ab


@njit(inline="always", **_JIT)
def _outflow(Ub, nx, ny, gamma, p_back, rho_t, p_t, band, out):
    rho = max(Ub[0], FLOOR)
    vx = Ub[1] / rho
    vy = Ub[2] / rho
    p = _pressure(Ub[0], Ub[1], Ub[2], Ub[3], gamma)
    a = np.sqrt(gamma * p / rho)
    vn = vx * nx + vy * ny

    if vn / a >= 1.0:
        # supersonic: no information enters, extrapolate
        _normal_flux(Ub[0], Ub[1], Ub[2], Ub[3], nx, ny, gamma, out)
        return abs(vn) + a

    rhob = rho * (p_back / p) ** (1.0 / gamma)
    ab = np.sqrt(gamma * p_back / rhob)
    vnb = vn + 2.0 / (gamma - 1.0) * (a - ab)
    vxb = (vx - vn * nx) + vnb * nx
    vyb = (vy - vn * ny) + vnb * ny
    Eb = p_back / ((gamma - 1.0) * rhob) + 0.5 * (vxb * vxb + vyb * vyb)

    # reverse flow: three characteristics enter, so three conditions are imposed
    # (reservoir entropy, the back pressure, a normal direction) and only the
    # normal velocity comes from the interior.  Blended over 0 <= vn/a <= band so
    # nothing switches discontinuously.
    if rho_t > 0.0 and vn / a < band:
        rho_f = rho_t * (p_back / p_t) ** (1.0 / gamma)
        a_f = np.sqrt(gamma * p_back / rho_f)
        vn_f = vn + 2.0 / (gamma - 1.0) * (a - a_f)
        vx_f = vn_f * nx
        vy_f = vn_f * ny
        E_f = p_back / ((gamma - 1.0) * rho_f) + 0.5 * (vx_f * vx_f + vy_f * vy_f)
        t = (vn / a) / band
        if t < 0.0:
            t = 0.0
        # blend the two states as conserved variables, each formed from its own
        # density before any blending
        r_out, mx_out, my_out, en_out = rhob, rhob * vxb, rhob * vyb, rhob * Eb
        r_in, mx_in, my_in, en_in = rho_f, rho_f * vx_f, rho_f * vy_f, rho_f * E_f
        _normal_flux(
            t * r_out + (1.0 - t) * r_in,
            t * mx_out + (1.0 - t) * mx_in,
            t * my_out + (1.0 - t) * my_in,
            t * en_out + (1.0 - t) * en_in,
            nx, ny, gamma, out,
        )
        return abs(vnb) + ab

    _normal_flux(rhob, rhob * vxb, rhob * vyb, rhob * Eb, nx, ny, gamma, out)
    return abs(vnb) + ab


# ==========================================================================
#  Assembly passes
# ==========================================================================
# Every kernel below writes into caller-supplied arrays.  The backend allocates
# those once and reuses them for the life of the solve, which matters more than
# it looks: a four-stage step calls the residual four times, and allocating and
# zeroing `fw`, `smax`, `R` and the mass-solve output on each call was a measured
# sixth of the step.


@njit(parallel=True, **_JIT)
def edge_pass(
    U,
    phi_face,
    iedge_elem,
    iedge_face,
    bedge_elem,
    bedge_face,
    bedge_tag,
    edge_normal,
    edge_jac,
    w_face,
    gamma,
    efix,
    flux_id,
    low_mach,
    rho_t_bf,
    p_t_bf,
    band_bf,
    at2,
    at,
    rho_t,
    ca,
    sa,
    p_back,
    fw,
    smax,
):
    """Weighted numerical flux and max signal speed for every global edge.

    Writes ``fw`` (nedge, nqf, 4) and ``smax`` (nedge); both are fully
    overwritten, so they need no zeroing by the caller.
    """
    n_int = iedge_elem.shape[0]
    n_bnd = bedge_elem.shape[0]
    n_edge = n_int + n_bnd
    nbf = U.shape[1]
    nqf = w_face.shape[0]

    for k in prange(n_edge):
        UL = np.zeros(4)
        UR = np.zeros(4)
        flux = np.zeros(4)
        best = 0.0
        if k < n_int:
            le = iedge_elem[k, 0]
            lf = iedge_face[k, 0]
            re = iedge_elem[k, 1]
            rf = iedge_face[k, 1]
            for q in range(nqf):
                for s in range(4):
                    UL[s] = 0.0
                    UR[s] = 0.0
                for i in range(nbf):
                    bl = phi_face[0, lf, i, q]
                    br = phi_face[1, rf, i, q]
                    for s in range(4):
                        UL[s] += bl * U[le, i, s]
                        UR[s] += br * U[re, i, s]
                nx = edge_normal[k, q, 0]
                ny = edge_normal[k, q, 1]
                if flux_id == 1:
                    sp = _hllc(UL, UR, nx, ny, gamma, low_mach, flux)
                else:
                    sp = _roe(UL, UR, nx, ny, gamma, efix, flux)
                if sp > best:
                    best = sp
                scale = edge_jac[k, q] * w_face[q]
                for s in range(4):
                    fw[k, q, s] = flux[s] * scale
        else:
            b = k - n_int
            be = bedge_elem[b]
            bf = bedge_face[b]
            tag = bedge_tag[b]
            for q in range(nqf):
                for s in range(4):
                    UL[s] = 0.0
                for i in range(nbf):
                    bl = phi_face[0, bf, i, q]
                    for s in range(4):
                        UL[s] += bl * U[be, i, s]
                nx = edge_normal[k, q, 0]
                ny = edge_normal[k, q, 1]
                if tag == _INFLOW:
                    sp = _inflow(UL, nx, ny, gamma, at2, at, rho_t, ca, sa, flux)
                elif tag == _OUTFLOW:
                    sp = _outflow(UL, nx, ny, gamma, p_back,
                                  rho_t_bf, p_t_bf, band_bf, flux)
                else:  # _WALL or _AXIS
                    sp = _wall(UL, nx, ny, gamma, flux)
                if sp > best:
                    best = sp
                scale = edge_jac[k, q] * w_face[q]
                for s in range(4):
                    fw[k, q, s] = flux[s] * scale
        smax[k] = best


@njit(parallel=True, **_JIT)
def element_pass(
    U,
    fw,
    smax,
    phi_vol,
    grad_x,
    grad_y,
    phi_face,
    face_edge,
    face_side,
    face_sign,
    edge_length,
    gamma,
    inv_mass,
    apply_mass,
    out,
    wave,
):
    r"""Volume integral plus the gathered face integrals, per element.

    With ``apply_mass`` false, ``out`` receives the residual :math:`R`.  With it
    true, ``out`` receives the *rate* :math:`-M^{-1}R`, which is what the time
    march actually wants.  Fusing the mass solve in here rather than running it
    as a second parallel pass saves writing :math:`R` to memory and reading it
    straight back -- about a tenth of a step, and one of four thread launches.

    The accumulation runs in a small per-element scratch rather than directly
    into ``out``, so the inner loop's read-modify-write stays in cache whatever
    the mesh size.
    """
    nelem = U.shape[0]
    nbf = U.shape[1]
    nqv = phi_vol.shape[1]
    nface = face_edge.shape[1]
    nqf = fw.shape[1]

    for e in prange(nelem):
        Re = np.zeros((nbf, 4))
        Uq = np.zeros(4)
        F = np.zeros(4)
        G = np.zeros(4)
        # ---- volume: R -= (grad_x . F + grad_y . G)
        for q in range(nqv):
            for s in range(4):
                Uq[s] = 0.0
            for i in range(nbf):
                b = phi_vol[i, q]
                for s in range(4):
                    Uq[s] += b * U[e, i, s]
            rho = max(Uq[0], FLOOR)
            vx = Uq[1] / rho
            vy = Uq[2] / rho
            p = _pressure(Uq[0], Uq[1], Uq[2], Uq[3], gamma)
            H = (Uq[3] + p) / rho
            F[0] = rho * vx
            F[1] = rho * vx * vx + p
            F[2] = rho * vx * vy
            F[3] = rho * vx * H
            G[0] = rho * vy
            G[1] = rho * vx * vy
            G[2] = rho * vy * vy + p
            G[3] = rho * vy * H
            for i in range(nbf):
                gx = grad_x[e, i, q]
                gy = grad_y[e, i, q]
                for s in range(4):
                    Re[i, s] -= gx * F[s] + gy * G[s]
        # ---- faces: R += sign * phi . fw
        acc = 0.0
        for f in range(nface):
            k = face_edge[e, f]
            sd = face_side[e, f]
            sg = face_sign[e, f]
            for q in range(nqf):
                for i in range(nbf):
                    b = sg * phi_face[sd, f, i, q]
                    for s in range(4):
                        Re[i, s] += b * fw[k, q, s]
            acc += smax[k] * edge_length[k]
        wave[e] = acc
        # ---- optionally fold in -M^{-1}
        if apply_mass:
            for i in range(nbf):
                for s in range(4):
                    tot = 0.0
                    for j in range(nbf):
                        tot += inv_mass[e, i, j] * Re[j, s]
                    out[e, i, s] = -tot
        else:
            for i in range(nbf):
                for s in range(4):
                    out[e, i, s] = Re[i, s]


@njit(parallel=True, **_JIT)
def apply_inverse_mass(inv_mass, R, out):
    """Block-diagonal mass-matrix solve, one small dense block per element.

    Still here for the unfused path that :meth:`Backend.residual` and the
    cross-backend tests use; the march goes through ``element_pass`` instead.
    """
    nelem = R.shape[0]
    nbf = R.shape[1]
    for e in prange(nelem):
        for i in range(nbf):
            for s in range(4):
                acc = 0.0
                for j in range(nbf):
                    acc += inv_mass[e, i, j] * R[e, j, s]
                out[e, i, s] = acc


# ==========================================================================
#  Time-march arithmetic
# ==========================================================================
# These replace NumPy expressions like ``U + 0.5 * dt * F0``.  Each such
# expression allocates a full state array and makes two passes over memory; done
# six times a step it was a measured tenth of the runtime, and the allocations
# churned the heap for no reason at all.


@njit(parallel=True, **_JIT)
def rms(A):
    """Root-mean-square of a state array, without materialising ``A * A``."""
    nelem, nbf, ns = A.shape
    total = 0.0
    for e in prange(nelem):
        acc = 0.0
        for i in range(nbf):
            for s in range(ns):
                v = A[e, i, s]
                acc += v * v
        total += acc
    return np.sqrt(total / (nelem * nbf * ns))


@njit(parallel=True, **_JIT)
def local_dt(wave, elem_area, factor, out):
    r"""``dt_e = factor * 2 A_e / max(sum_f s_f l_f, FLOOR)``."""
    for e in prange(wave.shape[0]):
        d = wave[e]
        if d < FLOOR:
            d = FLOOR
        out[e] = factor * 2.0 * elem_area[e] / d


@njit(parallel=True, **_JIT)
def stage(U, F, coef, dt, out):
    """``out = U + coef * dt * F``, with ``dt`` per element.

    Safe when ``out`` aliases ``U``: index ``(e, i, s)`` of the output depends
    only on index ``(e, i, s)`` of the inputs.
    """
    nelem, nbf, ns = U.shape
    for e in prange(nelem):
        c = coef * dt[e]
        for i in range(nbf):
            for s in range(ns):
                out[e, i, s] = U[e, i, s] + c * F[e, i, s]


@njit(parallel=True, **_JIT)
def combine_rk4(U, F0, F1, F2, F3, dt, out):
    """``out = U + dt / 6 * (F0 + 2 F1 + 2 F2 + F3)``."""
    nelem, nbf, ns = U.shape
    for e in prange(nelem):
        c = dt[e] / 6.0
        for i in range(nbf):
            for s in range(ns):
                out[e, i, s] = U[e, i, s] + c * (
                    F0[e, i, s] + 2.0 * F1[e, i, s] + 2.0 * F2[e, i, s] + F3[e, i, s]
                )


@njit(parallel=True, **_JIT)
def combine_ssp(U, V, F, dt, a, b, out):
    """``out = a * U + b * (V + dt * F)`` -- both non-trivial SSP-RK3 stages."""
    nelem, nbf, ns = U.shape
    for e in prange(nelem):
        d = dt[e]
        for i in range(nbf):
            for s in range(ns):
                out[e, i, s] = a * U[e, i, s] + b * (V[e, i, s] + d * F[e, i, s])


# ==========================================================================
#  Positivity limiter
# ==========================================================================


@njit(inline="always", **_JIT)
def _probe_minima(Ue, Ubar0, Ubar1, Ubar2, Ubar3, theta, phi_vol, phi_face, fside, gamma):
    """Minimum density and pressure over an element's probe points at scaling ``theta``.

    Probe points are the volume quadrature points plus the element's own trace at
    each face quadrature point -- exactly the set the scheme ever evaluates.

    Nothing is stored: the trace is formed, tested and discarded.  That is what
    makes the common case (no violation anywhere) cost a single pass with no
    allocation, instead of materialising an ``(nelem, n_probe, 4)`` array and
    sweeping it once per bisection step.
    """
    nbf = Ue.shape[0]
    nqv = phi_vol.shape[1]
    nface = fside.shape[0]
    nqf = phi_face.shape[3]

    rho_min = 1.0e300
    p_min = 1.0e300

    for q in range(nqv):
        U0 = 0.0
        U1 = 0.0
        U2 = 0.0
        U3 = 0.0
        for i in range(nbf):
            b = phi_vol[i, q]
            U0 += b * Ue[i, 0]
            U1 += b * Ue[i, 1]
            U2 += b * Ue[i, 2]
            U3 += b * Ue[i, 3]
        U0 = Ubar0 + theta * (U0 - Ubar0)
        U1 = Ubar1 + theta * (U1 - Ubar1)
        U2 = Ubar2 + theta * (U2 - Ubar2)
        U3 = Ubar3 + theta * (U3 - Ubar3)
        if U0 < rho_min:
            rho_min = U0
        r = U0 if U0 > 1e-300 else 1e-300
        pp = (gamma - 1.0) * (U3 - 0.5 * (U1 * U1 + U2 * U2) / r)
        if pp < p_min:
            p_min = pp

    for f in range(nface):
        sd = fside[f]
        for q in range(nqf):
            U0 = 0.0
            U1 = 0.0
            U2 = 0.0
            U3 = 0.0
            for i in range(nbf):
                b = phi_face[sd, f, i, q]
                U0 += b * Ue[i, 0]
                U1 += b * Ue[i, 1]
                U2 += b * Ue[i, 2]
                U3 += b * Ue[i, 3]
            U0 = Ubar0 + theta * (U0 - Ubar0)
            U1 = Ubar1 + theta * (U1 - Ubar1)
            U2 = Ubar2 + theta * (U2 - Ubar2)
            U3 = Ubar3 + theta * (U3 - Ubar3)
            if U0 < rho_min:
                rho_min = U0
            r = U0 if U0 > 1e-300 else 1e-300
            pp = (gamma - 1.0) * (U3 - 0.5 * (U1 * U1 + U2 * U2) / r)
            if pp < p_min:
                p_min = pp

    return rho_min, p_min


@njit(parallel=True, **_JIT)
def positivity_limit(
    U, phi_vol, phi_face, face_side, mean_weights, gamma, fraction, rho_floor, p_floor,
    steps, lebesgue,
):
    r"""Zhang-Shu positivity limiter, **in place**.  Returns ``(n_scaled, n_repaired)``.

    The cell average is preserved wherever it is admissible, so the limiter is
    conservative there.  Where the average itself has gone non-physical it is
    minimally repaired and counted in ``n_repaired`` -- that count being non-zero
    means the time step was too large.

    The cheap screen
    ----------------
    On a smooth solution no element violates positivity, so almost every call
    does nothing -- but proving that still cost a full sweep of the probe points,
    which made the *inactive* limiter about a seventh of a Runge-Kutta step.

    A sufficient condition avoids the sweep.  The basis is a partition of unity,
    so for any probe point :math:`x`,

    .. math::
        |\rho(x) - \bar\rho| = \Bigl|\sum_i \phi_i(x)(U_{i0} - \bar\rho)\Bigr|
        \le \Lambda \max_i |U_{i0} - \bar\rho|,
        \qquad
        \Lambda = \max_x \sum_i |\phi_i(x)| ,

    with :math:`\Lambda` (``lebesgue``) a constant of the reference element and
    the probe set, computed once.  The same bound applied to momentum and energy
    gives a lower bound on the pressure,

    .. math::
        p(x) \ge (\gamma - 1)\Bigl(
            \overline{\rho E} - \Lambda d_E
            - \tfrac{1}{2}\frac{(|\bar{\mathbf{m}}| + \Lambda d_m)^2}
                               {\bar\rho - \Lambda d_\rho}\Bigr),
        \qquad d_m = \sqrt{d_{m_x}^2 + d_{m_y}^2},

    costing one pass over the ``nbf`` coefficients instead of ``nbf`` times the
    probe count.  When the screen passes, the element is provably admissible and
    is skipped.  When it fails -- it is only sufficient, never necessary -- the
    exact probe runs as before, so the limiter's output is unchanged in every
    case.  At ``p = 0`` the deviations are identically zero and the screen always
    passes, which is why the limiter costs nothing there.
    """
    nelem, nbf, _ = U.shape
    n_scaled = 0
    n_repair = 0
    g1 = gamma - 1.0

    for e in prange(nelem):
        Ubar0 = 0.0
        Ubar1 = 0.0
        Ubar2 = 0.0
        Ubar3 = 0.0
        for i in range(nbf):
            w = mean_weights[e, i]
            Ubar0 += w * U[e, i, 0]
            Ubar1 += w * U[e, i, 1]
            Ubar2 += w * U[e, i, 2]
            Ubar3 += w * U[e, i, 3]

        # -- repair a non-physical cell average (last resort, not conservative)
        rho_new = Ubar0 if Ubar0 > rho_floor else rho_floor
        e_min = p_floor / g1 + 0.5 * (Ubar1 * Ubar1 + Ubar2 * Ubar2) / rho_new
        rhoE_new = Ubar3 if Ubar3 > e_min else e_min
        d0 = rho_new - Ubar0
        d3 = rhoE_new - Ubar3
        if d0 != 0.0 or d3 != 0.0:
            for i in range(nbf):
                U[e, i, 0] += d0
                U[e, i, 3] += d3
            Ubar0 = rho_new
            Ubar3 = rhoE_new
            n_repair += 1

        p_bar = g1 * (Ubar3 - 0.5 * (Ubar1 * Ubar1 + Ubar2 * Ubar2) / Ubar0)
        eps_rho = fraction * Ubar0
        if eps_rho < 0.5 * rho_floor:
            eps_rho = 0.5 * rho_floor
        eps_p = fraction * p_bar
        if eps_p < 0.5 * p_floor:
            eps_p = 0.5 * p_floor

        # -- cheap sufficient screen, O(nbf)
        d_rho = 0.0
        d_mx = 0.0
        d_my = 0.0
        d_en = 0.0
        for i in range(nbf):
            a = abs(U[e, i, 0] - Ubar0)
            if a > d_rho:
                d_rho = a
            a = abs(U[e, i, 1] - Ubar1)
            if a > d_mx:
                d_mx = a
            a = abs(U[e, i, 2] - Ubar2)
            if a > d_my:
                d_my = a
            a = abs(U[e, i, 3] - Ubar3)
            if a > d_en:
                d_en = a
        rho_lo = Ubar0 - lebesgue * d_rho
        if rho_lo >= eps_rho:
            m_hi = np.sqrt(Ubar1 * Ubar1 + Ubar2 * Ubar2) + lebesgue * np.sqrt(
                d_mx * d_mx + d_my * d_my
            )
            p_lo = g1 * (Ubar3 - lebesgue * d_en - 0.5 * m_hi * m_hi / rho_lo)
            if p_lo >= eps_p:
                continue  # provably admissible, no probe needed

        Ue = U[e]
        fside = face_side[e]
        rho_min, p_min = _probe_minima(
            Ue, Ubar0, Ubar1, Ubar2, Ubar3, 1.0, phi_vol, phi_face, fside, gamma
        )
        if rho_min >= eps_rho and p_min >= eps_p:
            continue  # the screen was pessimistic; still nothing to do

        # -- density bound is linear in theta, solve it directly
        theta = 1.0
        if rho_min < eps_rho:
            gap = Ubar0 - rho_min
            if gap <= 0.0:
                theta = 0.0
            else:
                theta = (Ubar0 - eps_rho) / gap
                if theta < 0.0:
                    theta = 0.0
                elif theta > 1.0:
                    theta = 1.0

        # -- pressure bound is quadratic, bisect
        _, p_hi = _probe_minima(
            Ue, Ubar0, Ubar1, Ubar2, Ubar3, theta, phi_vol, phi_face, fside, gamma
        )
        if p_hi < eps_p:
            lo = 0.0
            hi = theta
            for _ in range(steps):
                mid = 0.5 * (lo + hi)
                _, pm = _probe_minima(
                    Ue, Ubar0, Ubar1, Ubar2, Ubar3, mid, phi_vol, phi_face, fside, gamma
                )
                if pm >= eps_p:
                    lo = mid
                else:
                    hi = mid
            theta = lo

        for i in range(nbf):
            U[e, i, 0] = Ubar0 + theta * (Ue[i, 0] - Ubar0)
            U[e, i, 1] = Ubar1 + theta * (Ue[i, 1] - Ubar1)
            U[e, i, 2] = Ubar2 + theta * (Ue[i, 2] - Ubar2)
            U[e, i, 3] = Ubar3 + theta * (Ue[i, 3] - Ubar3)
        n_scaled += 1

    return n_scaled, n_repair
