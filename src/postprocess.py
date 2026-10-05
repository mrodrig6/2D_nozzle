r"""Performance metrics and diagnostics extracted from a converged solution.

Thrust
------
For an inviscid steady flow the axial momentum balance over the nozzle interior
closes exactly:

.. math::
    \underbrace{\int_{exit}\!\!\bigl(\rho v_x^2 + p\bigr)dA
     - \int_{inlet}\!\!\bigl(\rho v_x^2 + p\bigr)dA}_{F\ \text{(momentum form)}}
    \;=\;
    \underbrace{-\oint_{wall} p\, n_x\, ds}_{F\ \text{(wall form)}}

Both are computed.  They are mathematically identical but *discretely* distinct,
so the gap between them is a free, physically meaningful measure of
discretisation error -- it goes to zero under refinement and is reported as
:attr:`Performance.thrust_imbalance`.

Two things worth stating explicitly
-----------------------------------
1. The throat height in the thrust normalisation comes from the geometry, never
   from a constant.  Hard-coding it silently mis-normalises every coefficient
   computed at a different ``area_ratio`` or ``throat_x``.
2. The wall integral runs over the contoured wall alone.  The symmetry axis
   carries its own tag even though it shares the inviscid flux, because folding
   it into the wall integral is harmless only while the axis stays horizontal.

Reported thrust is for the **full planar nozzle per unit depth** -- twice the
half-channel that is actually meshed.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import NamedTuple

import numpy as np

from . import physics as ph
from .config import FlowConditions
from .mesh import BoundaryTag
from .operators import Operators
from .quasi1d import Quasi1DSolution, solve_quasi1d
from .solver import SolveResult


class BoundaryTrace(NamedTuple):
    """State and metrics at the quadrature points of one tagged boundary."""

    state: np.ndarray  # (nedge, nqf, 4)
    normal: np.ndarray  # (nedge, nqf, 2) unit outward
    weight: np.ndarray  # (nedge, nqf) w * |dX/dsigma|, i.e. ds
    xy: np.ndarray  # (nedge, nqf, 2)


def boundary_trace(U, ops: Operators, tag: BoundaryTag) -> BoundaryTrace:
    """Evaluate the solution on one tagged boundary."""
    topo = ops.topology
    sl = topo.tag_slice(tag)
    edges = topo.edges
    belem = edges.bedge_elem[sl]
    bface = edges.bedge_face[sl]
    phi = ops.ref.phi_face[0][bface]  # (n, nbf, nqf) owner-side basis
    state = np.einsum("kiq,kis->kqs", phi, U[belem])

    g = ops.n_interior + np.arange(sl.start, sl.stop)
    ds = ops.edge_jac[g] * ops.ref.w_face[None, :]
    return BoundaryTrace(state, ops.edge_normal[g], ds, ops.xy_face[g])


def volume_integral(values, ops: Operators) -> float:
    """Integrate values given at the volume quadrature points over the domain."""
    return float((np.asarray(values) * ops.weighted_det).sum())


def solution_at_quadrature(U, ops: Operators) -> np.ndarray:
    """The solution at the volume quadrature points, ``(nelem, nqv, 4)``."""
    return np.einsum("iq,eis->eqs", ops.ref.phi_vol, U)


# --------------------------------------------------------------------------
@dataclass(frozen=True)
class Performance:
    """Integral performance metrics of a converged solution."""

    thrust: float
    """Axial force, full planar nozzle per unit depth, momentum form."""
    thrust_wall: float
    """The same quantity from the wall-pressure integral."""
    thrust_imbalance: float
    """``|momentum - wall| / |momentum|``: a discretisation-error indicator."""
    thrust_coefficient: float
    """``thrust / (p_t A_throat)``, with ``A_throat = 2 y_throat``."""
    mass_flow_in: float
    mass_flow_out: float
    mass_imbalance: float
    """``|out - in| / in``: zero for an exactly conservative converged solution."""
    entropy_error: float
    r"""``sqrt( (1/A) \int (s/s_t - 1)^2 dA )``.  Meaningful only shock-free."""
    exit_mach_area_averaged: float
    exit_pressure_ratio: float
    """Area-averaged exit static pressure over ``p_t``."""
    exit_temperature_ratio: float
    r"""Area-averaged exit static temperature over ``T_t``.

    Reported as a ratio rather than in kelvin because the solver is
    non-dimensional: it is :math:`T_e/T_t`, which for an isentropic expansion
    would be :math:`(1 + \tfrac{\gamma-1}{2}M_e^2)^{-1}`.  Comparing the two is
    the cheapest check that the expansion is clean -- they part company exactly
    where total temperature has stopped being conserved.
    """
    specific_thrust: float
    """``thrust / mass_flow``, the effective exhaust velocity."""
    ideal_thrust: float
    """Quasi-1D shock-free thrust for the same geometry and reservoir."""
    thrust_efficiency: float
    """``thrust / ideal_thrust``."""

    def summary(self) -> str:
        return (
            f"thrust        {self.thrust:.6f}  (c_F = {self.thrust_coefficient:.6f}, "
            f"{100 * self.thrust_efficiency:.2f}% of ideal)\n"
            f"  wall form   {self.thrust_wall:.6f}  "
            f"(imbalance {self.thrust_imbalance:.2e})\n"
            f"mass flow     in {self.mass_flow_in:.6f}, out {self.mass_flow_out:.6f}  "
            f"(imbalance {self.mass_imbalance:.2e})\n"
            f"exit          M = {self.exit_mach_area_averaged:.4f}, "
            f"p/p_t = {self.exit_pressure_ratio:.5f}, "
            f"T/T_t = {self.exit_temperature_ratio:.5f}\n"
            f"entropy error {self.entropy_error:.4e}"
        )


def _axial_momentum_flux(tr: BoundaryTrace, gamma: float) -> float:
    r"""``\int (rho v_x (v.n) + p n_x) ds`` over a boundary, outward normal."""
    rho, vx, vy, p, _ = ph.primitives(tr.state, gamma)
    vn = vx * tr.normal[..., 0] + vy * tr.normal[..., 1]
    return float(((rho * vx * vn + p * tr.normal[..., 0]) * tr.weight).sum())


def _mass_flux(tr: BoundaryTrace, gamma: float) -> float:
    rho, vx, vy, _, _ = ph.primitives(tr.state, gamma)
    vn = vx * tr.normal[..., 0] + vy * tr.normal[..., 1]
    return float((rho * vn * tr.weight).sum())


def entropy_error(U, ops: Operators, flow: FlowConditions) -> float:
    r"""RMS deviation of ``s / s_t`` from 1 over the domain.

    ``s = p / rho^gamma`` and ``s_t = p_t^{1-gamma} (R T_t)^gamma``.  For
    shock-free flow the exact solution is isentropic, so this is a true error
    measure that converges at the scheme's design rate -- which makes it the
    standard quantity for an order-of-accuracy study.  Across a shock entropy
    genuinely rises, so the number then reflects physics, not error.
    """
    gamma = flow.gamma
    s_t = flow.total_pressure ** (1.0 - gamma) * (flow.Rgas * flow.total_temperature) ** gamma
    u_q = solution_at_quadrature(U, ops)
    rho, _, _, p, _ = ph.primitives(u_q, gamma)
    ratio = (p / rho**gamma) / s_t
    integral = volume_integral((ratio - 1.0) ** 2, ops)
    area = float(ops.elem_area.sum())
    return float(np.sqrt(integral / area))


def performance(result: SolveResult, *, quasi1d: Quasi1DSolution | None = None) -> Performance:
    """Compute all integral performance metrics from a solve result."""
    ops = result.operators
    flow = result.flow
    U = result.U
    gamma = flow.gamma

    inlet = boundary_trace(U, ops, BoundaryTag.INFLOW)
    outlet = boundary_trace(U, ops, BoundaryTag.OUTFLOW)
    wall = boundary_trace(U, ops, BoundaryTag.WALL)

    # factor 2: the mesh covers the upper half of a symmetric planar channel
    half_to_full = 2.0
    mom_out = _axial_momentum_flux(outlet, gamma)
    mom_in = _axial_momentum_flux(inlet, gamma)
    # the inlet's outward normal points upstream, so its flux already carries -1
    thrust = half_to_full * (mom_out + mom_in)

    _, _, _, p_wall, _ = ph.primitives(wall.state, gamma)
    thrust_wall = half_to_full * float(-(p_wall * wall.normal[..., 0] * wall.weight).sum())

    denom = abs(thrust) if abs(thrust) > 0.0 else 1.0
    imbalance = abs(thrust - thrust_wall) / denom

    mdot_in = half_to_full * (-_mass_flux(inlet, gamma))
    mdot_out = half_to_full * _mass_flux(outlet, gamma)

    a_throat = 2.0 * result.geometry.throat_height()
    cf = thrust / (flow.total_pressure * a_throat)

    # area-averaged exit quantities
    rho_e, u_e, v_e, p_e, _ = ph.primitives(outlet.state, gamma)
    area_e = float(outlet.weight.sum())
    mach_e = float((ph.mach_number(outlet.state, gamma) * outlet.weight).sum() / area_e)
    pr_e = float((p_e * outlet.weight).sum() / area_e / flow.total_pressure)
    t_e = p_e / (flow.Rgas * rho_e)
    tr_e = float((t_e * outlet.weight).sum() / area_e / flow.total_temperature)

    q1d = quasi1d if quasi1d is not None else solve_quasi1d(result.geometry, flow)
    ideal = _ideal_thrust(q1d, flow)

    return Performance(
        thrust=thrust,
        thrust_wall=thrust_wall,
        thrust_imbalance=imbalance,
        thrust_coefficient=cf,
        mass_flow_in=mdot_in,
        mass_flow_out=mdot_out,
        mass_imbalance=abs(mdot_out - mdot_in) / (abs(mdot_in) if mdot_in else 1.0),
        entropy_error=entropy_error(U, ops, flow),
        exit_mach_area_averaged=mach_e,
        exit_pressure_ratio=pr_e,
        exit_temperature_ratio=tr_e,
        specific_thrust=thrust / mdot_in if mdot_in else float("nan"),
        ideal_thrust=ideal,
        thrust_efficiency=thrust / ideal if ideal else float("nan"),
    )


def _ideal_thrust(q1d: Quasi1DSolution, flow: FlowConditions) -> float:
    """Quasi-1D axial force for the same geometry: momentum form, full nozzle."""
    rho, vx, p, a = q1d.density, q1d.velocity, q1d.pressure, q1d.area
    exit_term = (rho[-1] * vx[-1] ** 2 + p[-1]) * a[-1]
    inlet_term = (rho[0] * vx[0] ** 2 + p[0]) * a[0]
    return float(exit_term - inlet_term)


# --------------------------------------------------------------------------
# Line-outs and field sampling
# --------------------------------------------------------------------------
def sample_boundary(
    result: SolveResult, tag: BoundaryTag, n_points: int = 25
) -> dict[str, np.ndarray]:
    r"""Sample the solution uniformly along a tagged boundary.

    Returns ``x``, ``y`` and the derived scalars, ordered along the boundary.

    Sampling at *uniformly spaced* points rather than at the edge quadrature
    points matters for two reasons: Gauss points never reach a face's endpoints,
    so a quadrature-point profile stops short of the inlet and exit planes; and
    at low order there are only two or three of them per face, which is far too
    coarse to plot.
    """
    from . import elements as el

    ops = result.operators
    topo = ops.topology
    sl = topo.tag_slice(tag)
    belem = topo.edges.bedge_elem[sl]
    bface = topo.edges.bedge_face[sl]
    coords = _node_coords(result)
    gamma = result.flow.gamma

    sigma = np.linspace(0.0, 1.0, max(2, int(n_points)))
    xs, ys, states = [], [], []
    for e, f in zip(belem, bface, strict=True):
        pts = el.edge_reference_coords(topo.kind, int(f), sigma)
        phi_u, _, _ = el.shape_functions(topo.kind, ops.ref.order, pts)
        phi_g, _, _ = el.shape_functions(topo.kind, topo.geometry_order, pts)
        xy = phi_g.T @ coords[topo.elem_nodes[e]]
        xs.append(xy[:, 0])
        ys.append(xy[:, 1])
        states.append(phi_u.T @ result.U[e])

    x = np.concatenate(xs)
    y = np.concatenate(ys)
    state = np.concatenate(states, axis=0)

    # order along whichever coordinate varies across this boundary
    key = y if tag in (BoundaryTag.INFLOW, BoundaryTag.OUTFLOW) else x
    order = np.argsort(key, kind="stable")

    rho, vx, vy, pres, _ = ph.primitives(state, gamma)
    return {
        "x": x[order],
        "y": y[order],
        "mach": np.asarray(ph.mach_number(state, gamma))[order],
        "pressure": np.asarray(pres)[order],
        "density": np.asarray(rho)[order],
        # static temperature from the ideal gas law, the same expression
        # `scalar_field` uses, so a profile and a contour of the same quantity
        # cannot drift apart
        "temperature": np.asarray(pres / (result.flow.Rgas * rho))[order],
        "vx": np.asarray(vx)[order],
        "vy": np.asarray(vy)[order],
        "velocity": np.asarray(np.sqrt(vx * vx + vy * vy))[order],
    }


def exit_profile(result: SolveResult, n_points: int = 25) -> dict[str, np.ndarray]:
    """Profiles across the exit plane, sorted by ``y``.

    The spread in Mach number across the exit is a direct measure of divergence
    loss: a nozzle that leaves the flow with transverse velocity has wasted
    momentum.
    """
    return sample_boundary(result, BoundaryTag.OUTFLOW, n_points)


def centreline_profile(result: SolveResult, n_points: int = 25) -> dict[str, np.ndarray]:
    """Axial profiles along the symmetry axis, sorted by ``x``."""
    return sample_boundary(result, BoundaryTag.AXIS, n_points)


def wall_profile(result: SolveResult, n_points: int = 25) -> dict[str, np.ndarray]:
    """Profiles along the contoured wall, sorted by ``x``."""
    return sample_boundary(result, BoundaryTag.WALL, n_points)


def inlet_profile(result: SolveResult, n_points: int = 25) -> dict[str, np.ndarray]:
    """Profiles across the inlet plane, sorted by ``y``."""
    return sample_boundary(result, BoundaryTag.INFLOW, n_points)


SCALARS = ("mach", "pressure", "density", "temperature", "vx", "vy", "velocity", "entropy")


def scalar_field(state: np.ndarray, flow: FlowConditions, name: str) -> np.ndarray:
    """Derive a named scalar from conserved states shaped ``(..., 4)``."""
    gamma = flow.gamma
    rho, vx, vy, p, _ = ph.primitives(state, gamma)
    if name == "mach":
        return np.asarray(ph.mach_number(state, gamma))
    if name == "pressure":
        return np.asarray(p)
    if name == "density":
        return np.asarray(rho)
    if name == "temperature":
        return np.asarray(p / (flow.Rgas * rho))
    if name == "vx":
        return np.asarray(vx)
    if name == "vy":
        return np.asarray(vy)
    if name == "velocity":
        return np.asarray(np.sqrt(vx * vx + vy * vy))
    if name == "entropy":
        s_t = flow.total_pressure ** (1.0 - gamma) * (flow.Rgas * flow.total_temperature) ** gamma
        return np.asarray((p / rho**gamma) / s_t)
    raise ValueError(f"unknown scalar {name!r}; choose from {SCALARS}")


def sample_field(
    result: SolveResult, name: str = "mach", subdivisions: int = 3
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Sample a scalar on a subdivided mesh for plotting.

    Returns ``(points, triangles, values)`` suitable for
    ``matplotlib.pyplot.tripcolor``.  Each element is subdivided ``subdivisions``
    times per edge so that the high-order variation *inside* an element is
    visible -- plotting only the vertices would throw away most of what ``p = 2``
    buys.
    """
    from . import elements as el

    ops = result.operators
    topo = ops.topology
    n = max(1, int(subdivisions))
    kind = topo.kind

    # reference sample points and their local triangulation
    if kind == "tri":
        ij = [(i, j) for j in range(n + 1) for i in range(n + 1 - j)]
        index = {p: k for k, p in enumerate(ij)}
        ref = np.array([(i / n, j / n) for i, j in ij])
        tris = []
        for j in range(n):
            for i in range(n - j):
                tris.append([index[(i, j)], index[(i + 1, j)], index[(i, j + 1)]])
                if i + j < n - 1:
                    tris.append([index[(i + 1, j)], index[(i + 1, j + 1)], index[(i, j + 1)]])
    else:
        ij = [(i, j) for j in range(n + 1) for i in range(n + 1)]
        index = {p: k for k, p in enumerate(ij)}
        ref = np.array([(i / n, j / n) for i, j in ij])
        tris = []
        for j in range(n):
            for i in range(n):
                a, b = index[(i, j)], index[(i + 1, j)]
                c, d = index[(i + 1, j + 1)], index[(i, j + 1)]
                tris += [[a, b, c], [a, c, d]]
    tris = np.asarray(tris, dtype=np.int64)

    phi_u, _, _ = el.shape_functions(kind, ops.ref.order, ref)
    phi_g, _, _ = el.shape_functions(kind, topo.geometry_order, ref)
    coords = np.asarray(_node_coords(result))
    Xe = coords[topo.elem_nodes]  # (nelem, nbfQ, 2)

    xy = np.einsum("nk,end->ekd", phi_g, Xe)  # (nelem, npts, 2)
    state = np.einsum("ik,eis->eks", phi_u, result.U)
    values = scalar_field(state, result.flow, name)

    npts = ref.shape[0]
    nelem = topo.n_elem
    points = xy.reshape(-1, 2)
    offsets = (np.arange(nelem) * npts)[:, None, None]
    triangles = (tris[None, :, :] + offsets).reshape(-1, 3)
    return points, triangles, values.reshape(-1)


def _node_coords(result: SolveResult) -> np.ndarray:
    """Recover node coordinates from an operator set's topology.

    ``Operators`` does not keep the coordinate array, so it is rebuilt from the
    geometry and the logical grids -- which is exact, since that is how it was
    made in the first place.
    """
    from .mesh import nozzle_node_coords, nozzle_x_distribution

    topo = result.operators.topology
    disc = result.discretization
    geom = result.geometry
    n_sub = topo.geometry_order * 2 ** topo.logical["refine"]
    x_frac = nozzle_x_distribution(
        topo.logical["nx_nodes"],
        nx_base=topo.logical["nx_base"],
        n_sub=n_sub,
        spacing=disc.x_spacing,
        x_throat=geom.throat_location() / geom.length,
        cluster_strength=disc.cluster_strength,
        cluster_width=disc.cluster_width,
    )
    r_frac = np.linspace(0.0, 1.0, topo.logical["nr_nodes"])
    return np.asarray(nozzle_node_coords(x_frac, r_frac, geom.params(), geom.contour))
