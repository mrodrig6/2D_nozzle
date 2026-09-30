r"""The high-level interface.

This is the layer students use.  One call takes design variables to a converged
solution and its performance numbers::

    from dgnozzle import solve_nozzle

    result = solve_nozzle(area_ratio=3.0, back_pressure_ratio=0.12, order=1)
    print(result.summary())

Everything below it is available when needed -- :func:`build_case` hands back the
mesh and operators so a sweep can reuse them -- but nothing in the normal
workflow requires touching it.

``p``-continuation
------------------
With ``p_continuation=True`` a run solves at ``p = 0`` first, then re-projects up
one order at a time, warm-starting each stage from the last.  Re-projection is
exact -- the coarse space sits inside the fine one -- so it cannot change the
answer.

It is **off by default**, because measurement says it is not a speed-up: it cost
2-28% more wall time than a direct solve in every case that converged either
way, the quasi-1D initial condition having already removed the transient the
``p = 0`` stage exists to remove.  Its value is as a fallback for a high-order
solve that will not start at all.

When it is on, the returned :class:`~dgnozzle.solver.SolveResult` reports the
**total** cost across all stages -- iterations, wall time and a concatenated
residual history.  Reporting only the final stage, as an earlier version did,
makes a continuation run look cheaper than a direct solve, which is exactly
backwards.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field, replace
from typing import Any

import numpy as np

from . import initialize as ini
from .config import Discretization, FlowConditions, SolverOptions
from .geometry import NozzleGeometry, check_contour
from .mesh import MeshTopology, build_nozzle_mesh
from .operators import Operators, build_operators
from .quasi1d import Quasi1DSolution, solve_quasi1d
from .solver import SolveHistory, SolveResult, solve_steady

#: Keyword shortcuts accepted by :func:`solve_nozzle`, grouped by target object.
_GEOMETRY_KEYS = (
    "area_ratio", "throat_x", "throat_half_height", "inlet_half_height",
    "length", "contour", "theta_initial_deg", "theta_exit_deg",
    "bezier_w1", "bezier_w2",
)
_FLOW_KEYS = (
    "gamma", "Rgas", "total_temperature", "total_pressure",
    "back_pressure_ratio", "inflow_angle", "entropy_fix",
)
_DISC_KEYS = (
    "element", "order", "geometry_order", "refine", "nx", "nr",
    "x_spacing", "cluster_strength", "cluster_width",
)
_OPTS_KEYS = (
    "cfl", "tolerance", "max_iterations", "scheme", "limiter",
    "initial_condition", "p_continuation", "print_interval", "divergence_factor",
)


@dataclass
class Case:
    """A prepared problem: geometry, mesh and DG operators, ready to solve.

    Build one with :func:`build_case` and reuse it across a parameter sweep that
    only changes the *flow* -- the mesh and every metric term stay valid, which
    removes the dominant setup cost from each sweep point.
    """

    geometry: NozzleGeometry
    flow: FlowConditions
    discretization: Discretization
    topology: MeshTopology
    node_coords: np.ndarray
    operators: Operators
    grids: dict[str, np.ndarray] = field(default_factory=dict)
    _order_cache: dict[int, Operators] = field(default_factory=dict, repr=False)

    def operators_at(self, order: int) -> Operators:
        """Operators for the same mesh at a different polynomial order."""
        if order == self.discretization.order:
            return self.operators
        if order not in self._order_cache:
            self._order_cache[order] = build_operators(
                self.topology, self.node_coords, order
            )
        return self._order_cache[order]

    def quasi1d(self, n_points: int = 601) -> Quasi1DSolution:
        """The quasi-1D reference solution for this case."""
        return solve_quasi1d(self.geometry, self.flow, n_points=n_points)

    def summary(self) -> str:
        return (
            f"{self.geometry.describe()}\n"
            f"{self.topology.summary()}\n"
            f"p={self.discretization.order}, {self.discretization.n_dof} DOF per variable"
        )


def _split_overrides(overrides: dict[str, Any]) -> tuple[dict, dict, dict, dict]:
    geo, flo, dis, opt = {}, {}, {}, {}
    for key, value in overrides.items():
        if key in _GEOMETRY_KEYS:
            geo[key] = value
        elif key in _FLOW_KEYS:
            flo[key] = value
        elif key in _DISC_KEYS:
            dis[key] = value
        elif key in _OPTS_KEYS:
            opt[key] = value
        else:
            known = sorted(_GEOMETRY_KEYS + _FLOW_KEYS + _DISC_KEYS + _OPTS_KEYS)
            raise TypeError(
                f"unknown keyword {key!r}. Accepted shortcuts: {', '.join(known)}"
            )
    return geo, flo, dis, opt


def build_case(
    geometry: NozzleGeometry | None = None,
    flow: FlowConditions | None = None,
    discretization: Discretization | None = None,
    *,
    check: bool = True,
    **overrides: Any,
) -> Case:
    """Build the mesh and DG operators for a nozzle.

    Accepts either the three configuration objects or keyword shortcuts for any
    of their fields (``area_ratio=3.0``, ``order=2``, ...).
    """
    geo_kw, flow_kw, disc_kw, extra = _split_overrides(overrides)
    if extra:
        raise TypeError(f"build_case got solver options {sorted(extra)}; pass them to solve_nozzle")

    geom = (geometry or NozzleGeometry())
    if geo_kw:
        geom = NozzleGeometry(**{**geom.as_dict(), **geo_kw})
    flw = (flow or FlowConditions())
    if flow_kw:
        flw = flw.replace(**flow_kw)
    dsc = (discretization or Discretization())
    if disc_kw:
        dsc = dsc.replace(**disc_kw)

    if check:
        check_contour(geom)

    topo, coords, grids = build_nozzle_mesh(
        geom,
        kind=dsc.element,
        nx=dsc.nx,
        nr=dsc.nr,
        refine=dsc.refine,
        geometry_order=dsc.geometry_order,
        x_spacing=dsc.x_spacing,
        cluster_strength=dsc.cluster_strength,
        cluster_width=dsc.cluster_width,
    )
    coords = np.asarray(coords)
    ops = build_operators(topo, coords, dsc.order)
    return Case(
        geometry=geom,
        flow=flw,
        discretization=dsc,
        topology=topo,
        node_coords=coords,
        operators=ops,
        grids=grids,
    )


def case_from_grids(
    geometry: NozzleGeometry,
    flow: FlowConditions,
    discretization: Discretization,
    topology: MeshTopology,
    x_frac: np.ndarray,
    r_frac: np.ndarray,
) -> Case:
    """Rebuild a :class:`Case` on a *given* logical grid and topology.

    Used for sensitivity analysis and optimisation, where the node distribution
    must stay a fixed function of the design variables.  With the default
    ``x_spacing='throat'`` the axial clustering follows the throat, so rebuilding
    the mesh from scratch at a perturbed ``throat_x`` also *redistributes* the
    nodes -- and a finite difference then measures the node motion as well as the
    design change.  Freezing the logical grid removes that inconsistency and
    makes the adjoint gradient agree with a finite difference of the same
    discrete objective.
    """
    from .mesh import nozzle_node_coords

    coords = np.asarray(nozzle_node_coords(x_frac, r_frac, geometry.params(), geometry.contour))
    ops = build_operators(topology, coords, discretization.order)
    return Case(
        geometry=geometry,
        flow=flow,
        discretization=discretization,
        topology=topology,
        node_coords=coords,
        operators=ops,
        grids={"x_frac": np.asarray(x_frac), "r_frac": np.asarray(r_frac)},
    )


def solve_nozzle(
    geometry: NozzleGeometry | None = None,
    flow: FlowConditions | None = None,
    discretization: Discretization | None = None,
    options: SolverOptions | None = None,
    *,
    case: Case | None = None,
    backend: str = "numba",
    U0: np.ndarray | None = None,
    verbose: bool = True,
    progress: Callable[[int, float], None] | None = None,
    **overrides: Any,
) -> SolveResult:
    """Solve a nozzle from design variables to a converged flow field.

    Parameters
    ----------
    geometry, flow, discretization, options
        Configuration objects.  Any field may instead be given as a keyword
        shortcut, e.g. ``solve_nozzle(area_ratio=3.0, order=2, cfl=0.8)``.
    case
        A prepared :class:`Case` to reuse.  Skips mesh generation, which is what
        a sweep over flow conditions wants.  Geometry and discretisation
        overrides are rejected when a case is supplied, since they would
        invalidate it.
    backend
        ``'numba'`` (default, fastest), ``'numpy'`` (reference) or ``'jax'``
        (differentiable).
    U0
        Warm start.  Must match the target order's shape; use
        :func:`dgnozzle.initialize.change_order` to convert.  Warm-starting from
        a neighbouring sweep point typically saves most of the iterations.
    verbose
        Print progress and a summary line.

    Returns
    -------
    SolveResult
        Carries the solution, the operators, and the convergence record.  Check
        :attr:`~dgnozzle.solver.SolveResult.converged` before trusting the
        numbers; :attr:`~dgnozzle.solver.SolveResult.message` says what went
        wrong when it is ``False``.
    """
    geo_kw, flow_kw, disc_kw, opt_kw = _split_overrides(overrides)

    if case is not None:
        if geo_kw or disc_kw or geometry is not None or discretization is not None:
            raise TypeError(
                "geometry/discretization overrides invalidate a prepared case; "
                "either drop `case` or rebuild it with build_case()"
            )
        cs = case
        flw = flow or cs.flow
        if flow_kw:
            flw = flw.replace(**flow_kw)
    else:
        cs = build_case(geometry, flow, discretization, **{**geo_kw, **flow_kw, **disc_kw})
        flw = cs.flow

    opts = options or SolverOptions()
    if opt_kw:
        opts = opts.replace(**opt_kw)
    if not verbose:
        opts = opts.replace(print_interval=0)

    disc = cs.discretization
    target = disc.order

    if verbose:
        q1d = solve_quasi1d(cs.geometry, flw)
        print("=== dgnozzle ===")
        print(f"  {cs.geometry.describe()}")
        print(f"  {cs.topology.summary()}")
        print(f"  p={target}, Q={disc.geometry_order}, {disc.n_dof} DOF/variable, "
              f"backend={backend}")
        print(f"  {q1d.summary()}")
        print(f"  critical ratios: {q1d.critical.describe()}")
        print(f"  {'iter':>9s}   {'residual':>12s}")

    orders = list(range(target + 1)) if (opts.p_continuation and U0 is None) else [target]

    result: SolveResult | None = None
    U = U0
    # p-continuation runs several marches; the reported cost must be their sum,
    # not just the final stage's, or a continuation run looks cheaper than it is
    # and cannot be compared against a direct solve.
    total_iterations = 0
    total_time = 0.0
    combined = SolveHistory()
    for order in orders:
        ops = cs.operators_at(order)
        stage_opts = opts
        if order != target:
            # intermediate stages only need to remove the transient
            stage_opts = opts.replace(tolerance=max(opts.tolerance, 1e-4))
        if U is not None and order != orders[0]:
            U = ini.change_order(U, prev_ops, ops)  # noqa: F821
        if verbose and len(orders) > 1:
            print(f"  -- stage p={order} --")
        result = solve_steady(
            ops, flw, cs.geometry, disc.replace(order=order), stage_opts,
            backend=backend, U0=U, progress=progress,
        )
        U = result.U
        prev_ops = ops  # noqa: F841
        for it, res in zip(result.history.iterations, result.history.residual, strict=True):
            combined.iterations.append(total_iterations + it)
            combined.residual.append(res)
        total_iterations += result.iterations
        total_time += result.history.wall_time
        if not result.converged and order != target:
            if verbose:
                print(f"  (stage p={order} stopped: {result.message}; continuing)")

    assert result is not None
    if len(orders) > 1:
        combined.wall_time = total_time
        result = replace(
            result,
            iterations=total_iterations,
            history=combined,
            discretization=disc,
        )
    if verbose:
        print(f"  {result.summary()}")
        if result.message:
            print(f"  note: {result.message}")
    return result
