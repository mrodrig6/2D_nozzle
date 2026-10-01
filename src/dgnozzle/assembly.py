r"""Vectorised DG residual assembly, shared by the NumPy and JAX backends.

Discrete weak form
------------------
For each element :math:`\Omega_e` and basis function :math:`\phi_i`,

.. math::
    \int_{\Omega_e} \phi_i \frac{\partial u_h}{\partial t}\,d\Omega
    = \int_{\Omega_e} \nabla \phi_i \cdot \vec{F}(u_h)\,d\Omega
      - \oint_{\partial \Omega_e} \phi_i\, \hat{F}(u_h^+, u_h^-, \vec{n})\,ds

Writing the right-hand side as :math:`-R_i` gives the semi-discrete system
:math:`M_e \,\dot{U}_e = -R_e`, which the solver marches to :math:`R = 0`.

Why this is fast
----------------
Two things:

1. **No Python-level loop over elements or edges.**  Every step is a batched
   ``einsum``.  Looping over elements, then edges, then quadrature points in
   interpreted code -- and rebuilding the per-edge basis tables on every call --
   costs orders of magnitude more than the arithmetic itself.
2. **Pure gather, never scatter.**  Because ``operators.face_edge`` maps each
   ``(element, local face)`` to exactly one global edge, the face integrals are
   collected by indexing rather than by ``np.add.at``/``scatter_add``.  Gathers
   vectorise cleanly, parallelise without atomics, and -- crucially -- let the
   identical code path run under ``jax.jit``.

The only Python loop left runs over the 3 or 4 *local* faces of the reference
element, a fixed tiny count.
"""

from __future__ import annotations

from typing import NamedTuple

import numpy as np

from . import physics as ph
from .config import FlowConditions
from .mesh import BoundaryTag
from .operators import Operators


class ResidualResult(NamedTuple):
    """Residual and the per-element wave-speed sum used for the local time step."""

    residual: object  # (nelem, nbf, 4)
    wave_sum: object  # (nelem,)  sum_f (max signal speed on f) * (length of f)


def volume_term(U, ops: Operators, flow: FlowConditions, xp=np):
    r"""The element integral :math:`-\int \nabla\phi_i \cdot \vec{F}\,d\Omega`."""
    u_q = xp.einsum("iq,eis->eqs", xp.asarray(ops.ref.phi_vol), U)
    F, G = ph.euler_flux(u_q, flow.gamma, xp=xp)
    return -(
        xp.einsum("eiq,eqs->eis", ops.grad_x, F) + xp.einsum("eiq,eqs->eis", ops.grad_y, G)
    )


def edge_fluxes(U, ops: Operators, flow: FlowConditions, xp=np):
    """Numerical flux and max signal speed at every quadrature point of every edge.

    Returns ``(flux, max_speed)`` shaped ``(n_edges_total, n_qface, 4)`` and
    ``(n_edges_total, n_qface)``, in global edge numbering: interior edges first,
    then boundary edges grouped by tag.
    """
    topo = ops.topology
    edges = topo.edges
    phi_face = xp.asarray(ops.ref.phi_face)
    nrm = ops.edge_normal
    ni = ops.n_interior

    # ---- interior edges: Roe flux between the two traces
    lelem, relem = edges.iedge_elem[:, 0], edges.iedge_elem[:, 1]
    lface, rface = edges.iedge_face[:, 0], edges.iedge_face[:, 1]
    UL = xp.einsum("kiq,kis->kqs", phi_face[0][lface], U[lelem])
    UR = xp.einsum("kiq,kis->kqs", phi_face[1][rface], U[relem])
    nx_i, ny_i = nrm[:ni, :, 0], nrm[:ni, :, 1]
    interior = ph.roe_flux(
        UL, UR, nx_i, ny_i, flow.gamma, entropy_fix=flow.entropy_fix, xp=xp
    )

    # ---- boundary edges: one call per tag, on a contiguous slice
    belem, bface = edges.bedge_elem, edges.bedge_face
    Ub = xp.einsum("kiq,kis->kqs", phi_face[0][bface], U[belem])
    nx_b, ny_b = nrm[ni:, :, 0], nrm[ni:, :, 1]

    pieces: list[object] = []
    speeds: list[object] = []
    for tag in (BoundaryTag.INFLOW, BoundaryTag.OUTFLOW, BoundaryTag.WALL, BoundaryTag.AXIS):
        sl = topo.tag_slice(tag)
        if sl.stop <= sl.start:
            continue
        U_s, nxs, nys = Ub[sl], nx_b[sl], ny_b[sl]
        if tag is BoundaryTag.INFLOW:
            res = ph.inflow_flux(
                U_s, nxs, nys, flow.gamma,
                Tt=flow.total_temperature, pt=flow.total_pressure,
                Rgas=flow.Rgas, alpha=flow.inflow_angle, xp=xp,
            )
        elif tag is BoundaryTag.OUTFLOW:
            res = ph.outflow_flux(U_s, nxs, nys, flow.gamma, p_back=flow.back_pressure, xp=xp)
        else:  # WALL and AXIS share the inviscid slip flux
            res = ph.wall_flux(U_s, nxs, nys, flow.gamma, xp=xp)
        pieces.append(res.flux)
        speeds.append(res.max_speed)

    flux = xp.concatenate([interior.flux, *pieces], axis=0)
    speed = xp.concatenate([interior.max_speed, *speeds], axis=0)
    return flux, speed


def face_term(flux, ops: Operators, xp=np):
    r"""The face integral :math:`+\oint \phi_i \hat{F}\,ds`, gathered per element.

    ``flux`` carries the owner's outward normal, so the neighbour of an interior
    edge takes the same value with a sign flip -- which is precisely why the
    scheme is conservative.
    """
    w = xp.asarray(ops.ref.w_face)
    fw = flux * (ops.edge_jac * w[None, :])[..., None]  # (nedge, nqf, 4)

    total = None
    for f in range(ops.topology.n_faces):
        gathered = fw[ops.face_edge[:, f]]  # (nelem, nqf, 4)
        contrib = xp.einsum("eiq,eqs->eis", ops.face_basis[:, f], gathered)
        contrib = contrib * ops.face_sign[:, f][:, None, None]
        total = contrib if total is None else total + contrib
    return total


def wave_speed_sum(speed, ops: Operators, xp=np):
    """``sum_f s_f l_f`` per element, for the local time step."""
    per_edge = speed.max(axis=1) * ops.edge_length  # (nedge,)
    gathered = per_edge[ops.face_edge]  # (nelem, nface)
    return gathered.sum(axis=1)


def residual(U, ops: Operators, flow: FlowConditions, xp=np) -> ResidualResult:
    """Full DG residual ``R`` with ``M dU/dt = -R``."""
    flux, speed = edge_fluxes(U, ops, flow, xp=xp)
    R = volume_term(U, ops, flow, xp=xp) + face_term(flux, ops, xp=xp)
    return ResidualResult(R, wave_speed_sum(speed, ops, xp=xp))


#: Largest ``cfl`` -- in the units of this module, so already the coefficient
#: multiplying ``2 A_e / sum_f s_f l_f`` -- at which the march still converged,
#: by polynomial order.  Measured by bisection to about 3%, over the ``bell``
#: and ``smooth`` contours at refinement levels 0 and 1; the two contours agreed
#: to within a bisection step at every order, and the tighter refinement level is
#: the one recorded.  ``tests/test_solver.py`` re-checks that a step at
#: ``cfl = 1`` stays finite, so a change that eats the margin fails the suite.
#:
#: These are *limits*, not recommendations: the shipped default keeps a 30%
#: margin on them, see :data:`dgnozzle.config.DEFAULT_CFL`.
STABILITY_LIMIT: dict[str, tuple[float, ...]] = {
    "rk4": (1.625, 0.875, 0.500),
    "ssprk3": (1.437, 0.779, 0.448),
}

#: Beyond the measured orders, fall back to ``K / (p + 1)``.  That scaling fits
#: the measurements better than the textbook ``1/(2p+1)``: the stable coefficient
#: falls as 1 : 0.54 : 0.31 across ``p = 0, 1, 2``, against 1 : 0.5 : 0.33 for
#: ``1/(p+1)`` and 1 : 0.33 : 0.2 for ``1/(2p+1)``.  ``K`` is the smallest the
#: measurements support, so the fallback is conservative where it is used.
#:
#: SSP-RK3 comes out at 0.885, 0.890 and 0.897 times RK4 at the three measured
#: orders -- near enough constant that the two schemes differ by a scheme factor
#: and not by their order dependence, which is a useful sanity check on both
#: columns.
STABILITY_FALLBACK: dict[str, float] = {"rk4": 1.5, "ssprk3": 1.32}



def stability_limit(order: int, scheme: str = "rk4") -> float:
    """The measured stable step coefficient for this order and scheme."""
    try:
        table = STABILITY_LIMIT[scheme]
    except KeyError:
        raise ValueError(
            f"unknown scheme {scheme!r}; choose from {tuple(STABILITY_LIMIT)}"
        ) from None
    if order < len(table):
        return table[order]
    return STABILITY_FALLBACK[scheme] / (order + 1)


def step_coefficient(order: int, cfl: float, scheme: str = "rk4") -> float:
    r"""The scalar in :math:`\Delta t_e = c\,2 A_e / \sum_f s_f l_f`.

    ``cfl`` is a *fraction of the stability limit* for this order and scheme, so
    ``cfl = 1`` sits at the measured edge of stability and ``cfl = 0.7`` -- the
    default -- keeps a 30% margin, whatever ``p`` is and whichever scheme runs.

    .. note::
       This is a change of units.  The step used to be
       :math:`\mathrm{cfl}/(2p+1) \cdot 2A_e/\sum_f s_f l_f`, which made the same
       ``cfl`` mean very different fractions of the stable step at different
       orders: measured, ``cfl = 1`` sat at 62% of the limit at ``p = 0`` but
       only 38% at ``p = 1`` and 40% at ``p = 2``.  The old default was therefore
       leaving most of a factor of two on the floor at every order above zero,
       and a value a user had tuned at one order was meaningless at another.
       A script that passed an explicit ``cfl`` above 1 will now be past the
       stability limit rather than comfortably inside it.
    """
    return cfl * stability_limit(order, scheme)


def local_time_step(
    wave_sum, ops: Operators, order: int, cfl: float, scheme: str = "rk4", xp=np
):
    r"""Element-local pseudo-time step.

    .. math::
        \Delta t_e = c(p, \text{scheme},\ \mathrm{cfl})\,
                     \frac{2\,A_e}{\sum_f s_f\, l_f}

    with :math:`c` from :func:`step_coefficient`.
    """
    denom = xp.maximum(wave_sum, ph.FLOOR)
    return step_coefficient(order, cfl, scheme) * 2.0 * ops.elem_area / denom


def residual_scale(flow: FlowConditions, length: float) -> float:
    r"""The natural magnitude of :math:`\dot{U}` for this problem.

    ``R`` has the units of a rate of change of a conserved variable, so the
    reservoir state and the nozzle length give the scale
    :math:`\rho_t a_t / L`.  Dividing by it turns the residual into a
    dimensionless number that means the same thing on every mesh, at every
    order, and from every initial guess.
    """
    return float(flow.stagnation_density * flow.stagnation_sound_speed / max(length, 1e-30))


def apply_inverse_mass(R, ops: Operators, xp=np):
    r"""Apply the block-diagonal :math:`M^{-1}`, element by element."""
    return xp.einsum("eij,ejs->eis", ops.inv_mass, R)
