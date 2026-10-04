"""Numba backend: the performance path.

Hands the hot loops to the nopython kernels in :mod:`._numba_kernels`.  The
operator arrays are made contiguous once, at construction, so the kernels never
pay for strided access.

Everything is preallocated
--------------------------
This backend owns its working set: the four stage rates, three state buffers,
the edge flux table, the wave-speed sum and the local time step are allocated
once and reused for the life of the solve.  A four-stage step evaluates the
residual four times, and each evaluation used to allocate and zero an edge table
and a residual array, while the stage arithmetic (``U + 0.5 * dt * F0`` and
friends) allocated a full state array six more times.  Together that was a
measured sixth of a step, spent entirely on the allocator.

The schemes are therefore implemented here as well as in :class:`Backend`, which
is a real risk of drift -- so ``tests/test_backends.py`` pins this backend's
``rk4_step`` and ``ssprk3_step`` against the base-class versions step for step.

The positivity limiter has its own kernel and runs in place.  It is called once
per Runge-Kutta stage, so four times per iteration; its cheap sufficient screen
(see :func:`._numba_kernels.positivity_limit`) is what keeps the inactive case,
which is every iteration of a smooth run, from costing a seventh of the step.

The Superbee slope limiter stays in vectorised NumPy: it is opt-in, wanted only
for shocked cases, and needs neighbour cell averages, which do not fit the same
in-place element-local pattern.
"""

from __future__ import annotations

import numpy as np

from .. import assembly as asm
from .. import limiter as lim
from . import _numba_kernels as nk
from .base import Backend


def _c(a) -> np.ndarray:
    return np.ascontiguousarray(np.asarray(a, dtype=np.float64))


def _ci(a) -> np.ndarray:
    return np.ascontiguousarray(np.asarray(a, dtype=np.int64))


class NumbaBackend(Backend):
    name = "numba"

    def __init__(self, ops, flow, opts):
        super().__init__(ops, flow, opts)
        # Every flux the package offers has a kernel, so this guard only fires
        # if one is added without one.  It raises rather than falling back,
        # because silently returning a Roe answer under another label is the one
        # failure mode a student could not detect.
        _KERNEL_FLUXES = {"roe": 0, "hllc": 1}
        _flux = getattr(flow, "flux", "roe")
        if _flux not in _KERNEL_FLUXES:
            raise NotImplementedError(
                f"the numba kernels implement {sorted(_KERNEL_FLUXES)}, so "
                f"flux={_flux!r} would silently give you a Roe answer here. "
                f"Add a kernel for it, or use backend='numpy'."
            )
        self._flux_id = _KERNEL_FLUXES[_flux]
        self._low_mach = float(getattr(flow, "hllc_low_mach", 0.0))
        self._rho_t_bf = float(flow.stagnation_density) if getattr(flow, "backflow", False) else 0.0
        self._p_t_bf = float(flow.total_pressure)
        self._band_bf = 0.05
        ed = ops.topology.edges
        self._phi_face = _c(ops.ref.phi_face)
        self._phi_vol = _c(ops.ref.phi_vol)
        self._w_face = _c(ops.ref.w_face)
        self._grad_x = _c(ops.grad_x)
        self._grad_y = _c(ops.grad_y)
        self._grad_x_m = _c(ops.grad_x_m)
        self._grad_y_m = _c(ops.grad_y_m)
        self._face_basis_m = _c(ops.face_basis_m)
        self._inv_mass = _c(ops.inv_mass)
        self._edge_normal = _c(ops.edge_normal)
        self._edge_jac = _c(ops.edge_jac)
        self._edge_length = _c(ops.edge_length)
        self._face_sign = _c(ops.face_sign)
        self._elem_area = _c(ops.elem_area)
        self._iedge_elem = _ci(ed.iedge_elem)
        self._iedge_face = _ci(ed.iedge_face)
        self._bedge_elem = _ci(ed.bedge_elem)
        self._bedge_face = _ci(ed.bedge_face)
        self._bedge_tag = _ci(ed.bedge_tag)
        self._face_edge = _ci(ed.face_edge)
        self._face_side = _ci(ed.face_side)
        self._mean_weights = _c(ops.mean_weights)

        rho_floor, p_floor = lim.reference_floors(flow)
        self._rho_floor = float(rho_floor)
        self._p_floor = float(p_floor)

        self._gamma = float(flow.gamma)
        self._efix = float(flow.entropy_fix)
        self._at = float(flow.stagnation_sound_speed)
        self._at2 = self._at**2
        self._rho_t = float(flow.stagnation_density)
        self._ca = float(np.cos(flow.inflow_angle))
        self._sa = float(np.sin(flow.inflow_angle))
        self._p_back = float(flow.back_pressure)

        # The Lebesgue-like constant of the basis over the probe set, which is
        # what makes the limiter's cheap screen a *valid* sufficient condition.
        # Taken over the volume points and both sides of every face, so it bounds
        # whichever trace an element actually needs.
        self._lebesgue = float(
            max(
                np.abs(self._phi_vol).sum(axis=0).max(),
                np.abs(self._phi_face).sum(axis=2).max(),
            )
        )

        # -- the working set, allocated once ------------------------------
        nelem = ops.n_elem
        nbf = ops.ref.n_basis
        nedge = ops.n_edges_total
        nqf = ops.ref.w_face.shape[0]
        self._fw = np.zeros((nedge, nqf, 4))
        self._smax = np.zeros(nedge)
        self._wave = np.zeros(nelem)
        self._dt = np.zeros(nelem)
        self._R = np.zeros((nelem, nbf, 4))
        self._F = [np.zeros((nelem, nbf, 4)) for _ in range(4)]
        self._S = [np.zeros((nelem, nbf, 4)) for _ in range(3)]

    # -- the four primitives, unfused -------------------------------------
    # These keep the base-class integrators and the cross-backend tests working.
    # The march below does not use them.
    def residual(self, U):
        self._edges(_c(U))
        nk.element_pass(
            _c(U),
            self._fw,
            self._smax,
            self._phi_vol,
            self._grad_x,
            self._grad_y,
            self._phi_face,
            self._face_edge,
            self._face_side,
            self._face_sign,
            self._edge_length,
            self._gamma,
            self._inv_mass,
            False,
            self._R,
            self._wave,
        )
        return self._R.copy(), self._wave.copy()

    def inverse_mass(self, R):
        R = _c(R)
        out = np.empty_like(R)
        nk.apply_inverse_mass(self._inv_mass, R, out)
        return out

    def limit(self, U):
        out = _c(U).copy()
        self._limit_inplace(out)
        return out

    def time_step(self, wave_sum):
        dt = asm.local_time_step(
            wave_sum, self.ops, self.ops.ref.order, self.opts.cfl, self.opts.scheme, xp=np
        )
        return dt[:, None, None]

    def norm(self, A) -> float:
        A = _c(A)
        with np.errstate(over="ignore", invalid="ignore"):
            return float(nk.rms(A))

    # -- fused internals ---------------------------------------------------
    def _edges(self, U) -> None:
        nk.edge_pass(
            U,
            self._phi_face,
            self._iedge_elem,
            self._iedge_face,
            self._bedge_elem,
            self._bedge_face,
            self._bedge_tag,
            self._edge_normal,
            self._edge_jac,
            self._w_face,
            self._gamma,
            self._efix,
            self._flux_id,
            self._low_mach,
            self._rho_t_bf,
            self._p_t_bf,
            self._band_bf,
            self._at2,
            self._at,
            self._rho_t,
            self._ca,
            self._sa,
            self._p_back,
            self._fw,
            self._smax,
        )

    def _rate_into(self, U, out) -> None:
        """``out = -M^{-1} R(U)``, filling ``self._wave`` on the way.

        Uses the operators with ``M^{-1}`` folded in, so there is no mass solve
        here at all.
        """
        self._edges(U)
        nk.element_rate(
            U,
            self._fw,
            self._smax,
            self._phi_vol,
            self._grad_x_m,
            self._grad_y_m,
            self._face_basis_m,
            self._face_edge,
            self._face_sign,
            self._edge_length,
            self._gamma,
            out,
            self._wave,
        )

    def _dt_into(self) -> np.ndarray:
        nk.local_dt(
            self._wave,
            self._elem_area,
            self.positivity_scale()
            * asm.step_coefficient(self.ops.ref.order, self.opts.cfl, self.opts.scheme),
            self._dt,
        )
        return self._dt

    def _limit_inplace(self, U) -> None:
        """Limit ``U`` in place, respecting the configured limiter."""
        kind = self.opts.limiter
        if kind == "none" or self.ops.ref.order == 0:
            return

        n_slope = 0
        if kind == "superbee":
            limited = lim.superbee_limiter(
                U, self.ops, self.flow, tvb_constant=self.opts.tvb_constant, xp=np
            )
            # Count slope-limiter activity as well as positivity activity.
            # Reporting only the latter made a run whose residual was
            # limit-cycling on the slope limiter show `limiter active on ~0.00
            # elem/call`, which pointed the diagnosis in exactly the wrong
            # direction.
            n_slope = int((np.abs(limited - U).max(axis=(1, 2)) > 0.0).sum())
            U[...] = limited

        n_scaled, n_repair = nk.positivity_limit(
            U,
            self._phi_vol,
            self._phi_face,
            self._face_side,
            self._mean_weights,
            self._gamma,
            1e-8,
            self._rho_floor,
            self._p_floor,
            12,
            self._lebesgue,
        )
        self.n_limited += max(int(n_scaled), n_slope)
        self.n_mean_repaired += int(n_repair)
        self.n_limit_calls += 1

    # -- the march ---------------------------------------------------------
    def run(self, U, n_steps: int, scheme: str):
        """Advance ``n_steps`` steps entirely inside the preallocated buffers.

        ``U`` is copied in once and out once, so a 50-step chunk pays two array
        copies rather than six per step.
        """
        step = self._rk4_fused if scheme == "rk4" else self._ssprk3_fused
        if scheme not in ("rk4", "ssprk3"):
            raise ValueError(f"unknown scheme {scheme!r}")

        cur, nxt = self._S[0], self._S[1]
        cur[...] = _c(U)
        res = float("nan")
        with np.errstate(over="ignore", invalid="ignore"):
            for _ in range(n_steps):
                res = step(cur, nxt)
                cur, nxt = nxt, cur
        return cur.copy(), res

    def _rk4_fused(self, U, out) -> float:
        """One classical RK4 step: ``U`` in, ``out`` written, residual returned."""
        F = self._F
        self._rate_into(U, F[0])
        dt = self._dt_into()
        res = float(nk.rms(F[0]))

        W = self._S[2]
        nk.stage(U, F[0], 0.5, dt, W)
        self._limit_inplace(W)
        self._rate_into(W, F[1])

        nk.stage(U, F[1], 0.5, dt, W)
        self._limit_inplace(W)
        self._rate_into(W, F[2])

        nk.stage(U, F[2], 1.0, dt, W)
        self._limit_inplace(W)
        self._rate_into(W, F[3])

        nk.combine_rk4(U, F[0], F[1], F[2], F[3], dt, out)
        self._limit_inplace(out)
        return res

    def _ssprk3_fused(self, U, out) -> float:
        """One three-stage SSP-RK3 step."""
        F = self._F
        self._rate_into(U, F[0])
        dt = self._dt_into()
        res = float(nk.rms(F[0]))

        U1 = self._S[2]
        nk.stage(U, F[0], 1.0, dt, U1)
        self._limit_inplace(U1)
        self._rate_into(U1, F[1])

        # out = 0.75 U + 0.25 (U1 + dt F1)
        nk.combine_ssp(U, U1, F[1], dt, 0.75, 0.25, out)
        self._limit_inplace(out)
        self._rate_into(out, F[2])

        # U1 = U/3 + 2/3 (out + dt F2), then hand it back as the result
        nk.combine_ssp(U, out, F[2], dt, 1.0 / 3.0, 2.0 / 3.0, U1)
        self._limit_inplace(U1)
        out[...] = U1
        return res
