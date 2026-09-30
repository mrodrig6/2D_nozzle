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
def _pressure(u0, u1, u2, u3, gamma):
    rho = max(u0, FLOOR)
    p = (gamma - 1.0) * (u3 - 0.5 * (u1 * u1 + u2 * u2) / rho)
    return max(p, FLOOR)


@njit(inline="always", **_JIT)
def _normal_flux(u0, u1, u2, u3, nx, ny, gamma, out):
    rho = max(u0, FLOOR)
    u = u1 / rho
    v = u2 / rho
    p = _pressure(u0, u1, u2, u3, gamma)
    H = (u3 + p) / rho
    un = u * nx + v * ny
    out[0] = rho * un
    out[1] = rho * u * un + p * nx
    out[2] = rho * v * un + p * ny
    out[3] = rho * H * un


@njit(inline="always", **_JIT)
def _roe(uL, uR, nx, ny, gamma, efix, out):
    """Roe flux with the Harten-Hyman entropy fix.  Returns the max signal speed."""
    rL = max(uL[0], FLOOR)
    aL = uL[1] / rL
    bL = uL[2] / rL
    pL = _pressure(uL[0], uL[1], uL[2], uL[3], gamma)
    HL = (uL[3] + pL) / rL

    rR = max(uR[0], FLOOR)
    aR = uR[1] / rR
    bR = uR[2] / rR
    pR = _pressure(uR[0], uR[1], uR[2], uR[3], gamma)
    HR = (uR[3] + pR) / rR

    unL = aL * nx + bL * ny
    unR = aR * nx + bR * ny

    fL0 = rL * unL
    fL1 = rL * aL * unL + pL * nx
    fL2 = rL * bL * unL + pL * ny
    fL3 = rL * HL * unL
    fR0 = rR * unR
    fR1 = rR * aR * unR + pR * nx
    fR2 = rR * bR * unR + pR * ny
    fR3 = rR * HR * unR

    sL = np.sqrt(rL)
    sR = np.sqrt(rR)
    den = sL + sR
    u = (sL * aL + sR * aR) / den
    v = (sL * bL + sR * bR) / den
    H = (sL * HL + sR * HR) / den
    q2 = u * u + v * v
    c2 = (gamma - 1.0) * (H - 0.5 * q2)
    if c2 < FLOOR:
        c2 = FLOOR
    c = np.sqrt(c2)
    un = u * nx + v * ny

    l1 = abs(un + c)
    l2 = abs(un - c)
    l3 = abs(un)
    smax = l1 if l1 > l2 else l2

    eps = efix * c
    if eps > 0.0:
        if l1 < eps:
            l1 = (l1 * l1 + eps * eps) / (2.0 * eps)
        if l2 < eps:
            l2 = (l2 * l2 + eps * eps) / (2.0 * eps)
        if l3 < eps:
            l3 = (l3 * l3 + eps * eps) / (2.0 * eps)

    d0 = uR[0] - uL[0]
    d1 = uR[1] - uL[1]
    d2 = uR[2] - uL[2]
    d3 = uR[3] - uL[3]

    G1 = (gamma - 1.0) * (0.5 * q2 * d0 - u * d1 - v * d2 + d3)
    G2 = -un * d0 + d1 * nx + d2 * ny

    s1 = 0.5 * (l1 + l2)
    s2 = 0.5 * (l1 - l2)
    C1 = (G1 / c2) * (s1 - l3) + (G2 / c) * s2
    C2 = (G1 / c) * s2 + (s1 - l3) * G2

    out[0] = 0.5 * (fL0 + fR0) - 0.5 * (l3 * d0 + C1)
    out[1] = 0.5 * (fL1 + fR1) - 0.5 * (l3 * d1 + C1 * u + C2 * nx)
    out[2] = 0.5 * (fL2 + fR2) - 0.5 * (l3 * d2 + C1 * v + C2 * ny)
    out[3] = 0.5 * (fL3 + fR3) - 0.5 * (l3 * d3 + C1 * H + C2 * un)
    return smax


@njit(inline="always", **_JIT)
def _wall(ub, nx, ny, gamma, out):
    rho = max(ub[0], FLOOR)
    u = ub[1] / rho
    v = ub[2] / rho
    un = u * nx + v * ny
    tx = u - un * nx
    ty = v - un * ny
    pb = (gamma - 1.0) * (ub[3] - 0.5 * rho * (tx * tx + ty * ty))
    if pb < FLOOR:
        pb = FLOOR
    out[0] = 0.0
    out[1] = pb * nx
    out[2] = pb * ny
    out[3] = 0.0
    return np.sqrt(gamma * pb / rho)


@njit(inline="always", **_JIT)
def _inflow(ub, nx, ny, gamma, at2, at, rho_t, ca, sa, out):
    rho = max(ub[0], FLOOR)
    u = ub[1] / rho
    v = ub[2] / rho
    p = _pressure(ub[0], ub[1], ub[2], ub[3], gamma)
    a = np.sqrt(gamma * p / rho)
    un = u * nx + v * ny
    Jp = un + 2.0 * a / (gamma - 1.0)

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
    ubx = qb * ca
    uby = qb * sa
    rhob = rho_t * fac ** (-1.0 / (gamma - 1.0))
    pb = rhob * ab * ab / gamma
    Hb = at2 / (gamma - 1.0)
    unb = ubx * nx + uby * ny

    out[0] = rhob * unb
    out[1] = rhob * ubx * unb + pb * nx
    out[2] = rhob * uby * unb + pb * ny
    out[3] = rhob * Hb * unb
    return abs(unb) + ab


@njit(inline="always", **_JIT)
def _outflow(ub, nx, ny, gamma, p_back, out):
    rho = max(ub[0], FLOOR)
    u = ub[1] / rho
    v = ub[2] / rho
    p = _pressure(ub[0], ub[1], ub[2], ub[3], gamma)
    a = np.sqrt(gamma * p / rho)
    un = u * nx + v * ny

    if un / a >= 1.0:
        # supersonic: no information enters, extrapolate
        _normal_flux(ub[0], ub[1], ub[2], ub[3], nx, ny, gamma, out)
        return abs(un) + a

    rhob = rho * (p_back / p) ** (1.0 / gamma)
    ab = np.sqrt(gamma * p_back / rhob)
    unb = un + 2.0 / (gamma - 1.0) * (a - ab)
    ubx = (u - un * nx) + unb * nx
    uby = (v - un * ny) + unb * ny
    Eb = p_back / ((gamma - 1.0) * rhob) + 0.5 * (ubx * ubx + uby * uby)
    _normal_flux(rhob, rhob * ubx, rhob * uby, rhob * Eb, nx, ny, gamma, out)
    return abs(unb) + ab


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
    at2,
    at,
    rho_t,
    ca,
    sa,
    p_back,
):
    """Weighted numerical flux and max signal speed for every global edge."""
    n_int = iedge_elem.shape[0]
    n_bnd = bedge_elem.shape[0]
    n_edge = n_int + n_bnd
    nbf = U.shape[1]
    nqf = w_face.shape[0]

    fw = np.zeros((n_edge, nqf, 4))
    smax = np.zeros(n_edge)

    for k in prange(n_edge):
        uL = np.zeros(4)
        uR = np.zeros(4)
        flux = np.zeros(4)
        best = 0.0
        if k < n_int:
            le = iedge_elem[k, 0]
            lf = iedge_face[k, 0]
            re = iedge_elem[k, 1]
            rf = iedge_face[k, 1]
            for q in range(nqf):
                for s in range(4):
                    uL[s] = 0.0
                    uR[s] = 0.0
                for i in range(nbf):
                    bl = phi_face[0, lf, i, q]
                    br = phi_face[1, rf, i, q]
                    for s in range(4):
                        uL[s] += bl * U[le, i, s]
                        uR[s] += br * U[re, i, s]
                nx = edge_normal[k, q, 0]
                ny = edge_normal[k, q, 1]
                sp = _roe(uL, uR, nx, ny, gamma, efix, flux)
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
                    uL[s] = 0.0
                for i in range(nbf):
                    bl = phi_face[0, bf, i, q]
                    for s in range(4):
                        uL[s] += bl * U[be, i, s]
                nx = edge_normal[k, q, 0]
                ny = edge_normal[k, q, 1]
                if tag == _INFLOW:
                    sp = _inflow(uL, nx, ny, gamma, at2, at, rho_t, ca, sa, flux)
                elif tag == _OUTFLOW:
                    sp = _outflow(uL, nx, ny, gamma, p_back, flux)
                else:  # _WALL or _AXIS
                    sp = _wall(uL, nx, ny, gamma, flux)
                if sp > best:
                    best = sp
                scale = edge_jac[k, q] * w_face[q]
                for s in range(4):
                    fw[k, q, s] = flux[s] * scale
        smax[k] = best
    return fw, smax


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
):
    """Volume integral plus the gathered face integrals, per element."""
    nelem = U.shape[0]
    nbf = U.shape[1]
    nqv = phi_vol.shape[1]
    nface = face_edge.shape[1]
    nqf = fw.shape[1]

    R = np.zeros((nelem, nbf, 4))
    wave = np.zeros(nelem)

    for e in prange(nelem):
        uq = np.zeros(4)
        F = np.zeros(4)
        G = np.zeros(4)
        # ---- volume: R -= (grad_x . F + grad_y . G)
        for q in range(nqv):
            for s in range(4):
                uq[s] = 0.0
            for i in range(nbf):
                b = phi_vol[i, q]
                for s in range(4):
                    uq[s] += b * U[e, i, s]
            rho = max(uq[0], FLOOR)
            u = uq[1] / rho
            v = uq[2] / rho
            p = _pressure(uq[0], uq[1], uq[2], uq[3], gamma)
            H = (uq[3] + p) / rho
            F[0] = rho * u
            F[1] = rho * u * u + p
            F[2] = rho * u * v
            F[3] = rho * u * H
            G[0] = rho * v
            G[1] = rho * u * v
            G[2] = rho * v * v + p
            G[3] = rho * v * H
            for i in range(nbf):
                gx = grad_x[e, i, q]
                gy = grad_y[e, i, q]
                for s in range(4):
                    R[e, i, s] -= gx * F[s] + gy * G[s]
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
                        R[e, i, s] += b * fw[k, q, s]
            acc += smax[k] * edge_length[k]
        wave[e] = acc
    return R, wave


@njit(parallel=True, **_JIT)
def apply_inverse_mass(inv_mass, R):
    """Block-diagonal mass-matrix solve, one small dense block per element."""
    nelem = R.shape[0]
    nbf = R.shape[1]
    out = np.zeros_like(R)
    for e in prange(nelem):
        for i in range(nbf):
            for s in range(4):
                acc = 0.0
                for j in range(nbf):
                    acc += inv_mass[e, i, j] * R[e, j, s]
                out[e, i, s] = acc
    return out


@njit(inline="always", **_JIT)
def _probe_minima(Ue, ub0, ub1, ub2, ub3, theta, phi_vol, phi_face, fside, gamma):
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
        u0 = 0.0
        u1 = 0.0
        u2 = 0.0
        u3 = 0.0
        for i in range(nbf):
            b = phi_vol[i, q]
            u0 += b * Ue[i, 0]
            u1 += b * Ue[i, 1]
            u2 += b * Ue[i, 2]
            u3 += b * Ue[i, 3]
        u0 = ub0 + theta * (u0 - ub0)
        u1 = ub1 + theta * (u1 - ub1)
        u2 = ub2 + theta * (u2 - ub2)
        u3 = ub3 + theta * (u3 - ub3)
        if u0 < rho_min:
            rho_min = u0
        r = u0 if u0 > 1e-300 else 1e-300
        pp = (gamma - 1.0) * (u3 - 0.5 * (u1 * u1 + u2 * u2) / r)
        if pp < p_min:
            p_min = pp

    for f in range(nface):
        sd = fside[f]
        for q in range(nqf):
            u0 = 0.0
            u1 = 0.0
            u2 = 0.0
            u3 = 0.0
            for i in range(nbf):
                b = phi_face[sd, f, i, q]
                u0 += b * Ue[i, 0]
                u1 += b * Ue[i, 1]
                u2 += b * Ue[i, 2]
                u3 += b * Ue[i, 3]
            u0 = ub0 + theta * (u0 - ub0)
            u1 = ub1 + theta * (u1 - ub1)
            u2 = ub2 + theta * (u2 - ub2)
            u3 = ub3 + theta * (u3 - ub3)
            if u0 < rho_min:
                rho_min = u0
            r = u0 if u0 > 1e-300 else 1e-300
            pp = (gamma - 1.0) * (u3 - 0.5 * (u1 * u1 + u2 * u2) / r)
            if pp < p_min:
                p_min = pp

    return rho_min, p_min


@njit(parallel=True, **_JIT)
def positivity_limit(
    U, phi_vol, phi_face, face_side, mean_weights, gamma, fraction, rho_floor, p_floor, steps
):
    """Zhang-Shu positivity limiter.  Returns ``(U_limited, n_scaled, n_mean_repaired)``.

    The cell average is preserved wherever it is admissible, so the limiter is
    conservative there.  Where the average itself has gone non-physical it is
    minimally repaired and counted in ``n_mean_repaired`` -- that count being
    non-zero means the time step was too large.
    """
    nelem, nbf, _ = U.shape
    out = U.copy()
    n_scaled = np.zeros(nelem, dtype=np.int64)
    n_repair = np.zeros(nelem, dtype=np.int64)

    for e in prange(nelem):
        ub0 = 0.0
        ub1 = 0.0
        ub2 = 0.0
        ub3 = 0.0
        for i in range(nbf):
            w = mean_weights[e, i]
            ub0 += w * U[e, i, 0]
            ub1 += w * U[e, i, 1]
            ub2 += w * U[e, i, 2]
            ub3 += w * U[e, i, 3]

        # -- repair a non-physical cell average (last resort, not conservative)
        rho_new = ub0 if ub0 > rho_floor else rho_floor
        e_min = p_floor / (gamma - 1.0) + 0.5 * (ub1 * ub1 + ub2 * ub2) / rho_new
        rhoE_new = ub3 if ub3 > e_min else e_min
        d0 = rho_new - ub0
        d3 = rhoE_new - ub3
        if d0 != 0.0 or d3 != 0.0:
            for i in range(nbf):
                out[e, i, 0] += d0
                out[e, i, 3] += d3
            ub0 = rho_new
            ub3 = rhoE_new
            n_repair[e] = 1

        p_bar = (gamma - 1.0) * (ub3 - 0.5 * (ub1 * ub1 + ub2 * ub2) / ub0)
        eps_rho = fraction * ub0
        if eps_rho < 0.5 * rho_floor:
            eps_rho = 0.5 * rho_floor
        eps_p = fraction * p_bar
        if eps_p < 0.5 * p_floor:
            eps_p = 0.5 * p_floor

        Ue = out[e]
        fside = face_side[e]
        rho_min, p_min = _probe_minima(Ue, ub0, ub1, ub2, ub3, 1.0, phi_vol, phi_face, fside, gamma)
        if rho_min >= eps_rho and p_min >= eps_p:
            continue  # the common case: nothing to do

        # -- density bound is linear in theta, solve it directly
        theta = 1.0
        if rho_min < eps_rho:
            gap = ub0 - rho_min
            if gap <= 0.0:
                theta = 0.0
            else:
                theta = (ub0 - eps_rho) / gap
                if theta < 0.0:
                    theta = 0.0
                elif theta > 1.0:
                    theta = 1.0

        # -- pressure bound is quadratic, bisect
        _, p_hi = _probe_minima(Ue, ub0, ub1, ub2, ub3, theta, phi_vol, phi_face, fside, gamma)
        if p_hi < eps_p:
            lo = 0.0
            hi = theta
            for _ in range(steps):
                mid = 0.5 * (lo + hi)
                _, pm = _probe_minima(
                    Ue, ub0, ub1, ub2, ub3, mid, phi_vol, phi_face, fside, gamma
                )
                if pm >= eps_p:
                    lo = mid
                else:
                    hi = mid
            theta = lo

        for i in range(nbf):
            out[e, i, 0] = ub0 + theta * (Ue[i, 0] - ub0)
            out[e, i, 1] = ub1 + theta * (Ue[i, 1] - ub1)
            out[e, i, 2] = ub2 + theta * (Ue[i, 2] - ub2)
            out[e, i, 3] = ub3 + theta * (Ue[i, 3] - ub3)
        n_scaled[e] = 1

    return out, n_scaled.sum(), n_repair.sum()
