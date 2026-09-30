r"""The pseudo-time march to steady state.

The semi-discrete system :math:`M \dot{U} = -R(U)` is integrated in pseudo-time
with an element-local step until :math:`R \to 0`.  The march is explicit, which
is the right trade for this problem size: each step is cheap, the memory
footprint is tiny, and there is no linear solver to precondition.

Convergence is measured on the **rate of change**,

.. math::
    \|\dot{U}\|_{\mathrm{rms}} = \sqrt{\frac{1}{4 N} \sum \bigl(M^{-1} R\bigr)^2},

scaled by the problem's own physical magnitude :math:`\rho_t a_t / L`.

That scaling is deliberate, and the third choice tried here.  Testing the
*unnormalised* residual against a fixed threshold makes "converged" mean
different things on different meshes and at different operating points.  But
testing it *relative to the first iteration* is no better: it demands a tighter
absolute residual the better the initial guess is, so improving the initial
condition makes the solver look slower and two runs started differently cannot
be compared at all.  A fixed physical scale is independent of both the mesh and
the starting field.

Two related traps are worth naming.  The residual must be measured at a state
the solver actually holds -- not at an intermediate Runge-Kutta stage, whose
rate is not the residual anywhere.  And the march needs an iteration cap, so
that a diverging run reports the problem instead of running forever.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np

from . import assembly as asm
from . import initialize as ini
from . import limiter as lim
from .backends import Backend, get_backend
from .config import Discretization, FlowConditions, SolverOptions
from .geometry import NozzleGeometry
from .operators import Operators


@dataclass
class SolveHistory:
    """Residual history and diagnostics of one march."""

    iterations: list[int] = field(default_factory=list)
    residual: list[float] = field(default_factory=list)
    wall_time: float = 0.0

    def as_arrays(self) -> tuple[np.ndarray, np.ndarray]:
        return np.asarray(self.iterations), np.asarray(self.residual)


@dataclass
class SolveResult:
    """Outcome of a steady-state solve."""

    U: np.ndarray
    operators: Operators
    flow: FlowConditions
    geometry: NozzleGeometry
    discretization: Discretization
    converged: bool
    iterations: int
    residual: float
    residual_initial: float
    history: SolveHistory
    backend: str
    residual_scale: float = 1.0
    """The physical magnitude the residual is measured against, rho_t a_t / L."""
    message: str = ""
    min_density: float = float("nan")
    min_pressure: float = float("nan")
    limiter_activity: float = float("nan")
    """Mean elements limited per limiter call; non-zero at convergence means
    the solution is being held physical by the limiter rather than resolved."""
    mean_repairs: int = 0
    """Times a cell average had to be floored. Non-zero means cfl was too large."""

    @property
    def residual_relative(self) -> float:
        """Residual as a fraction of its value at the first iteration (reporting only)."""
        if self.residual_initial == 0.0:
            return 0.0
        return self.residual / self.residual_initial

    @property
    def residual_scaled(self) -> float:
        """Residual divided by ``rho_t a_t / L``.  This is what convergence tests."""
        return self.residual / self.residual_scale

    def summary(self) -> str:
        state = "converged" if self.converged else "NOT CONVERGED"
        base = (
            f"{state} in {self.iterations} iterations "
            f"({self.history.wall_time:.2f} s, {self.backend}): "
            f"residual {self.residual_scaled:.3e} scaled "
            f"({self.residual_relative:.2e} of initial); "
            f"min rho {self.min_density:.4e}, min p {self.min_pressure:.4e}"
        )
        if self.limiter_activity == self.limiter_activity and self.limiter_activity > 0:
            base += f"; limiter active on ~{self.limiter_activity:.2f} elem/call"
        if self.mean_repairs:
            base += f"; {self.mean_repairs} cell-average repairs (cfl too large)"
        return base


def march(
    U,
    backend: Backend,
    opts: SolverOptions,
    scale: float,
    *,
    label: str = "",
    progress: Callable[[int, float], None] | None = None,
) -> tuple[object, SolveHistory, bool, int, float, float, str]:
    """Advance to steady state.  Returns ``(U, history, converged, iters, res, res0, message)``.

    Iterations are run in chunks so that the JAX backend can fuse them and so
    that progress reporting costs nothing per step.
    """
    history = SolveHistory()
    # Convergence is tested every `check_interval` steps, independently of how
    # often progress is printed.  Tying the two together (as an earlier version
    # did) either spams the log or runs long past the tolerance.
    chunk = min(max(1, opts.check_interval), opts.max_iterations)
    next_print = opts.print_interval
    window: list[float] = []

    # one step alone first, to capture the initial residual
    U, res0 = backend.run(U, 1, opts.scheme)
    res = res0
    done = 1
    history.iterations.append(done)
    history.residual.append(float(res0))
    converged = False
    message = ""
    t0 = time.perf_counter()

    if not np.isfinite(res0):
        history.wall_time = time.perf_counter() - t0
        return U, history, False, done, float(res0), float(res0), "initial residual is not finite"
    if res0 <= opts.tolerance * scale:
        history.wall_time = time.perf_counter() - t0
        return U, history, True, done, float(res0), float(res0), ""

    while done < opts.max_iterations:
        n = min(chunk, opts.max_iterations - done)
        U, res = backend.run(U, n, opts.scheme)
        done += n
        history.iterations.append(done)
        history.residual.append(float(res))

        if progress is not None:
            progress(done, float(res))
        elif opts.print_interval > 0 and done >= next_print:
            print(f"  {label}{done:>7d}   {res / scale:.6e}   "
                  f"({res / res0:.3e} of initial)")
            next_print = done + opts.print_interval

        if not np.isfinite(res):
            message = (
                f"residual became non-finite after {done} iterations; reduce cfl "
                "(try 0.5) or enable limiter='barth-jespersen'"
            )
            break
        if res <= opts.tolerance * scale:
            converged = True
            break
        if res > opts.divergence_factor * res0:
            message = (
                f"residual grew by {res / res0:.1e}x after {done} iterations, which is "
                f"past divergence_factor={opts.divergence_factor:g}; reduce cfl or "
                "start from initial_condition='quasi1d'"
            )
            break

        # -- stall detection -------------------------------------------------
        # A limiter that switches on and off between iterations parks the
        # residual at a fixed level instead of converging.  Detect that and stop,
        # rather than spending the whole iteration budget on a limit cycle.
        window.append(res)
        if len(window) > opts.stall_window:
            window.pop(0)
        if len(window) == opts.stall_window:
            best, worst = min(window), max(window)
            if worst > 0.0 and best / worst > opts.stall_ratio:
                active = backend.limiter_activity()
                span = opts.stall_window * chunk
                message = (
                    f"the residual has stalled at {res / scale:.3e} (scaled; target "
                    f"{opts.tolerance:g}) -- it changed by less than "
                    f"{100 * (1 - opts.stall_ratio):.0f}% over the last {span} iterations"
                )
                if np.isfinite(active) and active > 0.0:
                    message += (
                        f", with the limiter active on ~{active:.1f} element(s) per call. "
                        "That is a limit cycle, not slow convergence: the mesh cannot "
                        "resolve a shock or a strong expansion. Refine (refine=+1), "
                        "reduce the polynomial order, or move the operating point"
                    )
                else:
                    message += (
                        ". Try a smaller cfl, scheme='ssprk3', or a finer mesh"
                    )
                break

    if not converged and not message:
        message = (
            f"hit max_iterations={opts.max_iterations} at scaled residual "
            f"{res / scale:.2e} (target {opts.tolerance:g}); raise max_iterations or "
            "relax tolerance"
        )

    history.wall_time = time.perf_counter() - t0
    return U, history, converged, done, float(res), float(res0), message


def solve_steady(
    ops: Operators,
    flow: FlowConditions,
    geom: NozzleGeometry,
    disc: Discretization,
    opts: SolverOptions | None = None,
    *,
    backend: str = "numba",
    U0=None,
    progress: Callable[[int, float], None] | None = None,
) -> SolveResult:
    """Solve for the steady state on a prepared operator set.

    Used directly when you already hold the operators (a sweep reusing one mesh,
    say).  Most callers want :func:`dgnozzle.api.solve_nozzle` instead.
    """
    opts = opts or SolverOptions()
    bk = get_backend(backend, ops, flow, opts)

    if U0 is None:
        U = ini.initial_state(ops, flow, geom, opts.initial_condition, xp=np)
    else:
        U = np.asarray(U0)
        if U.shape != (ops.n_elem, ops.ref.n_basis, 4):
            raise ValueError(
                f"U0 has shape {U.shape}, expected "
                f"{(ops.n_elem, ops.ref.n_basis, 4)}; use dgnozzle.initialize."
                "change_order or rebuild the mesh to match"
            )
    U = bk.asarray(U)

    scale = asm.residual_scale(flow, geom.length)
    U, hist, conv, iters, res, res0, msg = march(U, bk, opts, scale, progress=progress)

    U_np = bk.to_numpy(U)
    diag = lim.diagnose(U_np, ops, flow)
    if conv and (diag.min_density <= 0.0 or diag.min_pressure <= 0.0):
        conv = False
        msg = (
            f"residual converged but the solution is not physical (min rho "
            f"{diag.min_density:.3e}, min p {diag.min_pressure:.3e}); "
            "enable a limiter or refine the mesh"
        )

    return SolveResult(
        U=U_np,
        operators=ops,
        flow=flow,
        geometry=geom,
        discretization=disc,
        converged=conv,
        iterations=iters,
        residual=res,
        residual_initial=res0,
        residual_scale=scale,
        history=hist,
        backend=backend,
        message=msg,
        min_density=diag.min_density,
        min_pressure=diag.min_pressure,
        limiter_activity=bk.limiter_activity(),
        mean_repairs=int(bk.n_mean_repaired),
    )
