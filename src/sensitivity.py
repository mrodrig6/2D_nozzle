r"""Exact design sensitivities by the discrete adjoint.

What this gives you
-------------------
For a scalar performance functional :math:`J` -- thrust, exit Mach number,
whatever -- and a set of design variables :math:`\vec{a}` (area ratio, throat
position, wall angles, Bezier weights, back pressure), this module returns
:math:`dJ/d\vec{a}` exactly, at a cost independent of how many design variables
there are.  A 20-variable shape gradient costs one adjoint solve, where finite
differences would cost 20 (or 40) extra flow solves.

Why an adjoint rather than ``jax.grad`` straight through the solver
-------------------------------------------------------------------
Differentiating the pseudo-time march itself would tape thousands of iterations,
which is ruinous in memory and gives a gradient of the *iteration history* rather
than of the converged answer.  Instead the converged state is characterised
implicitly by :math:`R(U, \vec{a}) = 0`, and the implicit function theorem gives

.. math::
    \frac{dJ}{d\vec{a}}
    = \frac{\partial J}{\partial \vec{a}}
      - \frac{\partial J}{\partial U}
        \Bigl(\frac{\partial R}{\partial U}\Bigr)^{-1}
        \frac{\partial R}{\partial \vec{a}}
    = \frac{\partial J}{\partial \vec{a}}
      - \lambda^T \frac{\partial R}{\partial \vec{a}},
    \qquad
    \Bigl(\frac{\partial R}{\partial U}\Bigr)^{T}\!\lambda
    = \Bigl(\frac{\partial J}{\partial U}\Bigr)^{T}

Only the *converged* state is needed, and both Jacobian actions come from JAX's
``jvp``/``vjp`` on the residual -- so nothing is ever assembled as a matrix.

How the adjoint system is solved
--------------------------------
By a pseudo-time march, exactly mirroring the forward solver:

.. math::
    \frac{d\lambda}{d\tau} = -M^{-1}
    \Bigl(\bigl(\partial R/\partial U\bigr)^{T}\lambda
          - \bigl(\partial J/\partial U\bigr)^{T}\Bigr)

Because :math:`M` is symmetric positive definite and block diagonal,
:math:`M^{-1}(\partial R/\partial U)^T` is similar to
:math:`M^{-1}\partial R/\partial U` and has the *same* spectrum, so the forward
solver's local time step and CFL limit carry over unchanged.  That makes the
adjoint march as robust as the forward one and needs no preconditioner -- unlike
a Krylov solve on the same operator.

Validity
--------
The gradient is of the *discrete* problem, so it matches a finite difference of
the solver's own output (to the accuracy of the difference), which is what
gradient-based optimisation needs.  Two caveats worth telling students:

* The geometry must be smooth in the design variables.  It is, for every contour
  family here.
* The *flow* should be shock-free, and limiting inactive.  A limiter that
  switches on and off introduces kinks, and a shock's position is only
  piecewise-differentiable in the design variables.  Use
  :func:`check_gradient` to confirm the adjoint against finite differences for
  any new objective or operating point.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from . import assembly as asm
from . import physics as ph
from .api import Case
from .config import Discretization, FlowConditions, SolverOptions
from .geometry import DESIGN_PARAMETERS, NozzleGeometry
from .mesh import BoundaryTag, MeshTopology, nozzle_node_coords
from .operators import ReferenceData, build_operators, reference_data
from .solver import SolveResult

#: Design variables accepted in a parameter dict, in addition to the geometric
#: ones in :data:`src.geometry.DESIGN_PARAMETERS`.
FLOW_DESIGN_PARAMETERS = ("back_pressure",)

ALL_DESIGN_PARAMETERS = tuple(DESIGN_PARAMETERS) + FLOW_DESIGN_PARAMETERS

OBJECTIVES = ("thrust", "thrust_coefficient", "exit_mach", "mass_flow", "exit_pressure")


# --------------------------------------------------------------------------
# Objectives, written in the injected array module so they can be differentiated
# --------------------------------------------------------------------------
def _boundary_slice(topology: MeshTopology, tag: BoundaryTag):
    return topology.tag_slice(tag)


def _trace(U, ops, tag: BoundaryTag, xp):
    """State, unit normal and ``ds`` on a tagged boundary, differentiably."""
    topo = ops.topology
    sl = _boundary_slice(topo, tag)
    edges = topo.edges
    belem = edges.bedge_elem[sl]
    bface = edges.bedge_face[sl]
    phi = xp.asarray(ops.ref.phi_face[0])[bface]
    state = xp.einsum("kiq,kis->kqs", phi, U[belem])
    g = ops.n_interior + np.arange(sl.start, sl.stop)
    ds = ops.edge_jac[g] * xp.asarray(ops.ref.w_face)[None, :]
    return state, ops.edge_normal[g], ds


def objective_value(U, ops, flow: FlowConditions, name: str, geom: NozzleGeometry, xp):
    """Evaluate a named scalar objective.  All are for the full planar nozzle."""
    gamma = flow.gamma
    if name in ("thrust", "thrust_coefficient"):
        out_s, out_n, out_ds = _trace(U, ops, BoundaryTag.OUTFLOW, xp)
        in_s, in_n, in_ds = _trace(U, ops, BoundaryTag.INFLOW, xp)

        def axial(state, n, ds):
            rho, vx, vy, p, _ = ph.primitives(state, gamma, xp=xp)
            vn = vx * n[..., 0] + vy * n[..., 1]
            return ((rho * vx * vn + p * n[..., 0]) * ds).sum()

        thrust = 2.0 * (axial(out_s, out_n, out_ds) + axial(in_s, in_n, in_ds))
        if name == "thrust":
            return thrust
        return thrust / (flow.total_pressure * 2.0 * geom.throat_height())

    if name == "mass_flow":
        s, n, ds = _trace(U, ops, BoundaryTag.INFLOW, xp)
        rho, vx, vy, _, _ = ph.primitives(s, gamma, xp=xp)
        vn = vx * n[..., 0] + vy * n[..., 1]
        return -2.0 * (rho * vn * ds).sum()

    if name in ("exit_mach", "exit_pressure"):
        s, n, ds = _trace(U, ops, BoundaryTag.OUTFLOW, xp)
        area = ds.sum()
        if name == "exit_mach":
            return (ph.mach_number(s, gamma, xp=xp) * ds).sum() / area
        _, _, _, p, _ = ph.primitives(s, gamma, xp=xp)
        return (p * ds).sum() / area / flow.total_pressure

    raise ValueError(f"unknown objective {name!r}; choose from {OBJECTIVES}")


# --------------------------------------------------------------------------
@dataclass
class DifferentiableCase:
    """A nozzle problem set up for differentiation with respect to its design.

    Build one with :func:`differentiable_case`.  The mesh *topology* is fixed;
    the node coordinates, every metric term, the residual and the objective are
    all rebuilt from the design variables inside the traced computation, so the
    gradient accounts for mesh motion as well as the flow response.
    """

    topology: MeshTopology
    reference: ReferenceData
    x_frac: np.ndarray
    r_frac: np.ndarray
    geometry: NozzleGeometry
    flow: FlowConditions
    discretization: Discretization
    options: SolverOptions
    _jax: Any = field(default=None, repr=False)
    _jnp: Any = field(default=None, repr=False)

    def __post_init__(self) -> None:
        from .backends.jax_backend import require_jax

        self._jax, self._jnp = require_jax()

    # -- parameter handling -------------------------------------------------
    def default_params(self) -> dict[str, float]:
        """All design variables at their current values."""
        params = dict(self.geometry.params())
        params["back_pressure"] = float(self.flow.back_pressure)
        return params

    def params_for(self, names: Sequence[str]) -> dict[str, float]:
        """A parameter dict restricted to ``names`` (the ones to differentiate)."""
        full = self.default_params()
        missing = [n for n in names if n not in full]
        if missing:
            raise ValueError(
                f"unknown design parameter(s) {missing}; choose from {ALL_DESIGN_PARAMETERS}"
            )
        return {n: full[n] for n in names}

    def _merged(self, params: Mapping[str, Any]) -> dict[str, Any]:
        merged = dict(self.default_params())
        merged.update(params)
        return merged

    # -- the traced pipeline -----------------------------------------------
    def operators(self, params: Mapping[str, Any]):
        """DG operators built from design variables (differentiable)."""
        jnp = self._jnp
        full = self._merged(params)
        coords = nozzle_node_coords(self.x_frac, self.r_frac, full, self.geometry.contour, xp=jnp)
        return build_operators(
            self.topology,
            coords,
            self.discretization.order,
            xp=jnp,
            ref=self.reference,
            check_jacobian=False,
        )

    def _flow_with(self, params: Mapping[str, Any]):
        """A flow object whose back pressure may be a tracer."""
        full = self._merged(params)
        pb = full["back_pressure"]

        class _Flow:
            # a lightweight stand-in: FlowConditions validates floats, and
            # back_pressure must be allowed to be a JAX tracer here
            gamma = self.flow.gamma
            Rgas = self.flow.Rgas
            total_temperature = self.flow.total_temperature
            total_pressure = self.flow.total_pressure
            inflow_angle = self.flow.inflow_angle
            entropy_fix = self.flow.entropy_fix
            back_pressure = pb
            stagnation_sound_speed = self.flow.stagnation_sound_speed
            stagnation_density = self.flow.stagnation_density
            # Every attribute assembly.residual reads has to be here.  This
            # stand-in shadows FlowConditions rather than subclassing it, so a
            # field added there is silently missing here until something asks
            # for it -- which is exactly how `flux` went unnoticed.
            flux = self.flow.flux
            hllc_low_mach = self.flow.hllc_low_mach
            backflow = self.flow.backflow
            allow_shock_in_nozzle = self.flow.allow_shock_in_nozzle

        return _Flow()

    def residual(self, U, params: Mapping[str, Any]):
        """DG residual as a function of state *and* design variables."""
        ops = self.operators(params)
        R, _ = asm.residual(U, ops, self._flow_with(params), xp=self._jnp)
        return R

    def objective(self, U, params: Mapping[str, Any], name: str):
        """Scalar objective as a function of state *and* design variables.

        The throat height enters the thrust-coefficient normalisation, so it is
        taken from the (possibly traced) parameter dict rather than from the
        concrete geometry -- otherwise the gradient would miss that dependence.
        """
        return objective_value(
            U,
            self.operators(params),
            self._flow_with(params),
            name,
            _ThroatShim(self._merged(params)),
            self._jnp,
        )

    # -- solves ------------------------------------------------------------
    def case_at(self, params: Mapping[str, Any] | None = None) -> Case:
        """A concrete :class:`Case` at these design variables, on the frozen grid."""
        from .api import case_from_grids

        params = dict(params or {})
        geom_kw = {k: v for k, v in params.items() if k in DESIGN_PARAMETERS}
        geom = self.geometry.with_params(geom_kw) if geom_kw else self.geometry
        flow = self.flow
        if "back_pressure" in params:
            flow = flow.replace(
                back_pressure_ratio=float(params["back_pressure"]) / flow.total_pressure
            )
        return case_from_grids(
            geom, flow, self.discretization, self.topology, self.x_frac, self.r_frac
        )

    def solve_forward(
        self, params: Mapping[str, Any] | None = None, *, backend="numba", U0=None, verbose=False
    ) -> SolveResult:
        """Converge the flow for these design variables, on the frozen logical grid.

        The grid is deliberately *not* regenerated: see
        :func:`src.api.case_from_grids` for why that matters for gradients.
        """
        from .api import solve_nozzle

        cs = self.case_at(params)
        return solve_nozzle(
            case=cs,
            flow=cs.flow,
            options=self.options,
            backend=backend,
            U0=U0,
            verbose=verbose,
        )

    def adjoint_solve(
        self,
        U,
        params: Mapping[str, Any],
        seed,
        *,
        tolerance: float = 1e-8,
        max_iterations: int = 100_000,
        check_interval: int = 100,
        verbose: bool = False,
    ):
        r"""Solve :math:`(\partial R/\partial U)^T \lambda = \text{seed}`.

        Marched in pseudo-time with the same local step and CFL as the forward
        solver, which is stable because the transposed operator has the same
        spectrum (see the module docstring).  Returns ``(lambda, residual_norm,
        iterations, converged)``.
        """
        jax, jnp = self._jax, self._jnp
        ops = self.operators(params)
        flow = self._flow_with(params)

        def R_of_U(U):
            R, _ = asm.residual(U, ops, flow, xp=jnp)
            return R

        _, vjp_U = jax.vjp(R_of_U, U)
        seed = jnp.asarray(seed)

        # frozen local time step from the converged state
        _, wave = asm.residual(U, ops, flow, xp=jnp)
        dt = asm.local_time_step(
            wave,
            ops,
            self.discretization.order,
            self.options.cfl,
            self.options.scheme,
            xp=jnp,
        )[:, None, None]
        inv_mass = ops.inv_mass

        def defect(lam):
            return vjp_U(lam)[0] - seed

        def rate(lam):
            return -jnp.einsum("eij,ejs->eis", inv_mass, defect(lam))

        @jax.jit
        def rk4(lam):
            f0 = rate(lam)
            f1 = rate(lam + 0.5 * dt * f0)
            f2 = rate(lam + 0.5 * dt * f1)
            f3 = rate(lam + dt * f2)
            return lam + dt / 6.0 * (f0 + 2.0 * f1 + 2.0 * f2 + f3)

        @jax.jit
        def chunk(lam):
            return jax.lax.fori_loop(0, check_interval, lambda _, l: rk4(l), lam)

        lam = jnp.zeros_like(seed)
        norm0 = float(jnp.sqrt(jnp.mean(seed * seed)))
        if norm0 == 0.0:
            return lam, 0.0, 0, True

        res = norm0
        it = 0
        converged = False
        while it < max_iterations:
            lam = chunk(lam)
            it += check_interval
            res = float(jnp.sqrt(jnp.mean(defect(lam) ** 2)))
            if verbose:
                print(f"    adjoint {it:>7d}   {res:.4e}  ({res / norm0:.2e} relative)")
            if not np.isfinite(res):
                break
            if res <= tolerance * norm0:
                converged = True
                break
        return lam, res / norm0, it, converged

    # -- the gradient -------------------------------------------------------
    def value_and_gradient(
        self,
        objective: str = "thrust",
        params: Mapping[str, Any] | None = None,
        *,
        names: Sequence[str] | None = None,
        backend: str = "numba",
        U0=None,
        adjoint_tolerance: float = 1e-8,
        verbose: bool = False,
    ) -> tuple[float, dict[str, float], SolveResult]:
        """Objective value and its exact gradient with respect to design variables.

        Parameters
        ----------
        objective
            One of :data:`OBJECTIVES`.
        params
            Design variables to evaluate at; defaults to the case's own values.
        names
            Which variables to differentiate with respect to.  Defaults to the
            keys of ``params``, or to ``('area_ratio',)`` if none given.
        backend
            Backend for the *forward* solve.  ``'numba'`` is fastest; the adjoint
            always runs under JAX.

        Returns
        -------
        (value, gradient, forward_result)
        """
        jax, jnp = self._jax, self._jnp
        params = dict(params or {})
        if names is None:
            names = tuple(params.keys()) or ("area_ratio",)
        base = self.params_for(names)
        base.update({k: v for k, v in params.items() if k in names})

        if verbose:
            print(f"  forward solve at {base}")
        fwd = self.solve_forward(base, backend=backend, U0=U0, verbose=False)
        if not fwd.converged:
            raise RuntimeError(
                f"the forward solve did not converge, so no gradient can be taken: {fwd.message}"
            )
        U = jnp.asarray(fwd.U)

        # dJ/dU and the explicit dJ/da at fixed U
        def J_of_U(u, pr):
            return self.objective(u, pr, objective)

        value, J_U = jax.value_and_grad(J_of_U, argnums=0)(U, base)
        _, J_a = jax.value_and_grad(J_of_U, argnums=1)(U, base)

        if verbose:
            print(f"  J = {float(value):.8f}; solving adjoint")
        lam, rel, its, ok = self.adjoint_solve(
            U, base, J_U, tolerance=adjoint_tolerance, verbose=verbose
        )
        if not ok:
            raise RuntimeError(
                f"the adjoint march stalled at relative residual {rel:.2e} after "
                f"{its} iterations; the forward solution may not be tight enough "
                "or a limiter may be active (try tolerance=1e-10, limiter='none' "
                "on a shock-free point)"
            )

        # dR/da contracted with lambda
        def R_of_params(pr):
            return self.residual(U, pr)

        _, vjp_a = jax.vjp(R_of_params, base)
        (R_a,) = vjp_a(lam)

        grad = {k: float(J_a[k]) - float(R_a[k]) for k in base}
        return float(value), grad, fwd


class _ThroatShim:
    """Exposes ``throat_height()`` from a raw parameter dict (possibly traced)."""

    def __init__(self, params: Mapping[str, Any]):
        self._p = params

    def throat_height(self):
        return self._p["throat_half_height"]


def differentiable_case(
    geometry: NozzleGeometry | None = None,
    flow: FlowConditions | None = None,
    discretization: Discretization | None = None,
    options: SolverOptions | None = None,
    *,
    case: Case | None = None,
    **overrides: Any,
) -> DifferentiableCase:
    """Build a :class:`DifferentiableCase`.

    Accepts the same keyword shortcuts as :func:`src.api.solve_nozzle`.
    """
    from .api import _split_overrides, build_case

    geo_kw, flow_kw, disc_kw, opt_kw = _split_overrides(overrides)
    if case is None:
        case = build_case(geometry, flow, discretization, **{**geo_kw, **flow_kw, **disc_kw})
    elif geo_kw or disc_kw:
        raise TypeError("geometry/discretization overrides invalidate a prepared case")
    opts = options or SolverOptions()
    if opt_kw:
        opts = opts.replace(**opt_kw)
    disc = case.discretization
    return DifferentiableCase(
        topology=case.topology,
        reference=reference_data(disc.element, disc.order, disc.geometry_order),
        x_frac=np.asarray(case.grids["x_frac"]),
        r_frac=np.asarray(case.grids["r_frac"]),
        geometry=case.geometry,
        flow=flow or case.flow,
        discretization=disc,
        options=opts,
    )


# --------------------------------------------------------------------------
# Verification
# --------------------------------------------------------------------------
def finite_difference_gradient(
    dcase: DifferentiableCase,
    objective: str = "thrust",
    params: Mapping[str, float] | None = None,
    *,
    names: Sequence[str] | None = None,
    step: float | Mapping[str, float] = 1e-4,
    backend: str = "numba",
    relative: bool = True,
) -> dict[str, float]:
    """Central-difference gradient, for verifying the adjoint.

    ``step`` is relative to each parameter's magnitude by default, which is the
    right scaling when the variables have very different units (an area ratio of
    2.5 next to a throat height of 0.14).
    """
    names = tuple(names) if names is not None else tuple((params or {}).keys()) or ("area_ratio",)
    base = dcase.params_for(names)
    if params:
        base.update({k: v for k, v in params.items() if k in names})

    def evaluate(p: Mapping[str, float]) -> float:
        res = dcase.solve_forward(p, backend=backend, verbose=False)
        if not res.converged:
            raise RuntimeError(f"forward solve failed inside finite differences: {res.message}")
        ops = res.operators
        import numpy as _np

        return float(objective_value(res.U, ops, res.flow, objective, res.geometry, _np))

    grad: dict[str, float] = {}
    for name in names:
        h = step[name] if isinstance(step, Mapping) else step
        if relative:
            h = h * max(abs(base[name]), 1e-3)
        plus = dict(base)
        plus[name] = base[name] + h
        minus = dict(base)
        minus[name] = base[name] - h
        grad[name] = (evaluate(plus) - evaluate(minus)) / (2.0 * h)
    return grad


def check_gradient(
    dcase: DifferentiableCase,
    objective: str = "thrust",
    names: Sequence[str] = ("area_ratio",),
    *,
    step: float = 1e-4,
    backend: str = "numba",
    verbose: bool = True,
) -> dict[str, dict[str, float]]:
    """Compare the adjoint gradient against central differences.

    Returns, per parameter, the adjoint value, the finite-difference value and
    their relative discrepancy.  Agreement to a few times ``step`` is the
    expected outcome; a large discrepancy means either the forward solve is not
    tight enough or the objective is not smooth at this operating point.
    """
    value, adj, _ = dcase.value_and_gradient(objective, names=names, backend=backend)
    fd = finite_difference_gradient(dcase, objective, names=names, step=step, backend=backend)
    out: dict[str, dict[str, float]] = {}
    for name in names:
        scale = max(abs(adj[name]), abs(fd[name]), 1e-30)
        out[name] = {
            "adjoint": adj[name],
            "finite_difference": fd[name],
            "relative_error": abs(adj[name] - fd[name]) / scale,
        }
    if verbose:
        print(f"objective {objective} = {value:.10f}")
        print(f"{'parameter':22s} {'adjoint':>14s} {'finite diff':>14s} {'rel err':>10s}")
        for name, row in out.items():
            print(
                f"{name:22s} {row['adjoint']:14.6e} {row['finite_difference']:14.6e} "
                f"{row['relative_error']:10.2e}"
            )
    return out
