"""Numba backend: the performance path.

Hands the hot loops to the nopython kernels in :mod:`._numba_kernels`.  The
operator arrays are made contiguous once, at construction, so the kernels never
pay for strided access.

The positivity limiter has its own kernel.  It runs once per Runge-Kutta stage,
so four times per iteration, and in vectorised NumPy it cost roughly three times
a residual evaluation -- more than half the whole step.  The kernel's fast path
(no element violating positivity, which is every iteration of a smooth run) is a
single pass over the probe points with no allocation at all.

The Barth-Jespersen limiter stays in vectorised NumPy: it is opt-in, wanted only
for shocked cases, and needs neighbour averages that do not fit the same
zero-allocation pattern.
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
        ed = ops.topology.edges
        self._phi_face = _c(ops.ref.phi_face)
        self._phi_vol = _c(ops.ref.phi_vol)
        self._w_face = _c(ops.ref.w_face)
        self._grad_x = _c(ops.grad_x)
        self._grad_y = _c(ops.grad_y)
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

    def residual(self, U):
        U = _c(U)
        fw, smax = nk.edge_pass(
            U, self._phi_face, self._iedge_elem, self._iedge_face,
            self._bedge_elem, self._bedge_face, self._bedge_tag,
            self._edge_normal, self._edge_jac, self._w_face,
            self._gamma, self._efix, self._at2, self._at, self._rho_t,
            self._ca, self._sa, self._p_back,
        )
        return nk.element_pass(
            U, fw, smax, self._phi_vol, self._grad_x, self._grad_y, self._phi_face,
            self._face_edge, self._face_side, self._face_sign, self._edge_length,
            self._gamma,
        )

    def inverse_mass(self, R):
        return nk.apply_inverse_mass(self._inv_mass, _c(R))

    def limit(self, U):
        kind = self.opts.limiter
        if kind == "none" or self.ops.ref.order == 0:
            return U
        if kind == "barth-jespersen":
            U = lim.barth_jespersen_limiter(U, self.ops, self.flow, xp=np)
        out, n_scaled, n_repair = nk.positivity_limit(
            _c(U), self._phi_vol, self._phi_face, self._face_side, self._mean_weights,
            self._gamma, 1e-8, self._rho_floor, self._p_floor, 12,
        )
        self.n_limited += int(n_scaled)
        self.n_mean_repaired += int(n_repair)
        self.n_limit_calls += 1
        return out

    def time_step(self, wave_sum):
        dt = asm.local_time_step(wave_sum, self.ops, self.ops.ref.order, self.opts.cfl, xp=np)
        return dt[:, None, None]

    def norm(self, A) -> float:
        A = np.asarray(A)
        return float(np.sqrt(np.mean(A * A)))
