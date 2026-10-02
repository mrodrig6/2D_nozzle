"""Command-line interface: ``python -m dgnozzle ...``.

Four subcommands, covering the workflows the lab needs without writing a script:

``solve``    one operating point, optionally saving figures
``sweep``    a parameter sweep to a CSV table
``geometry`` inspect and plot a contour without running the flow solver
``bench``    time the backends against each other
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence

import numpy as np


def _geometry_args(p: argparse.ArgumentParser) -> None:
    g = p.add_argument_group("geometry")
    g.add_argument("--contour", default="bell", help="contour family (default: bell)")
    g.add_argument("--area-ratio", type=float, default=2.5019, help="exit/throat area ratio")
    g.add_argument("--throat-x", type=float, default=0.1388, help="throat location, 0-1")
    g.add_argument("--theta-initial-deg", type=float, default=None, help="wall angle at throat")
    g.add_argument("--theta-exit-deg", type=float, default=0.0, help="wall angle at exit")
    g.add_argument("--bezier-w1", type=float, default=0.55)
    g.add_argument("--bezier-w2", type=float, default=0.90)


def _flow_args(p: argparse.ArgumentParser) -> None:
    g = p.add_argument_group("flow")
    g.add_argument("--back-pressure-ratio", type=float, default=0.15, help="p_back / p_total")
    g.add_argument("--gamma", type=float, default=1.4)


def _disc_args(p: argparse.ArgumentParser) -> None:
    g = p.add_argument_group("discretisation")
    g.add_argument("--element", default="tri", choices=("tri", "quad"))
    g.add_argument("--flux", default="roe", choices=("roe", "hllc"),
                   help="interface flux: Roe (most accurate) or "
                        "HLLC (positivity-preserving)")
    g.add_argument("-p", "--order", type=int, default=1, help="polynomial order p")
    g.add_argument("-Q", "--geometry-order", type=int, default=1, help="geometry order Q")
    g.add_argument("-r", "--refine", type=int, default=0, help="refinement level")
    g.add_argument("--x-spacing", default="throat", choices=("throat", "inlet", "uniform"))


def _solver_args(p: argparse.ArgumentParser) -> None:
    g = p.add_argument_group("solver")
    g.add_argument("--backend", default="numba", choices=("numba", "numpy", "jax"))
    g.add_argument(
        "--cfl", type=float, default=None,
        help="Courant number: cfl/(2p+1) times the geometric step. The default "
             "is 70%% of the largest value measured to converge at this order",
    )
    g.add_argument("--tolerance", type=float, default=1e-6)
    g.add_argument("--max-iterations", type=int, default=200_000)
    g.add_argument("--scheme", default="rk4", choices=("rk4", "ssprk3"))
    g.add_argument("--limiter", default="positivity",
                   choices=("none", "positivity", "superbee"))
    g.add_argument("--tvb-constant", type=float, default=50.0,
                   help="Cockburn-Shu TVB threshold M for limiter=superbee")
    g.add_argument(
        "--p-continuation",
        action="store_true",
        help="solve at p=0 and re-project upward; a fallback for a high-order "
             "solve that will not start, not a speed-up",
    )
    g.add_argument("-q", "--quiet", action="store_true")


def _collect(args) -> dict:
    """Gather the solver keywords a subcommand's parser actually defined.

    Read with ``getattr`` and a default rather than by attribute access: not
    every subcommand registers every argument group.  ``geometry`` runs no flow
    solve, so it declares no discretisation or solver options, and reaching for
    ``args.element`` there raises ``AttributeError`` before the command does any
    work.  Only keys the parser supplied are returned, so a subcommand never
    passes along a setting the user had no way to give it.
    """
    defined = vars(args)
    names = (
        "contour", "area_ratio", "throat_x", "theta_exit_deg",
        "bezier_w1", "bezier_w2", "back_pressure_ratio", "gamma",
        "element", "order", "geometry_order", "refine", "x_spacing",
        "flux",
        "cfl", "tolerance", "max_iterations", "scheme", "limiter",
        "p_continuation", "tvb_constant",
    )
    out = {name: defined[name] for name in names if name in defined}
    if defined.get("theta_initial_deg") is not None:
        out["theta_initial_deg"] = defined["theta_initial_deg"]
    return out


def _cmd_solve(args) -> int:
    from .api import solve_nozzle
    from .postprocess import performance

    result = solve_nozzle(verbose=not args.quiet, backend=args.backend, **_collect(args))
    print()
    print(performance(result).summary())
    if not result.converged:
        print(f"\nWARNING: not converged — {result.message}", file=sys.stderr)

    if args.figure:
        from .plotting import overview

        fig = overview(result)
        fig.savefig(args.figure, dpi=150)
        print(f"\nfigure written to {args.figure}")
    if args.save:
        np.savez_compressed(
            args.save,
            U=result.U,
            converged=result.converged,
            residual=result.residual,
            **{f"geom_{k}": v for k, v in result.geometry.as_dict().items()
               if v is not None},
        )
        print(f"solution written to {args.save}")
    return 0 if result.converged else 1


def _cmd_sweep(args) -> int:
    from .sweep import sweep

    values = np.linspace(args.start, args.stop, args.count)
    fixed = _collect(args)
    fixed.pop(args.parameter, None)
    result = sweep(progress=not args.quiet, backend=args.backend,
                   **{args.parameter: values}, **fixed)
    print()
    print(result.table(tuple(args.metrics)))
    if args.csv:
        result.to_csv(args.csv)
        print(f"\ntable written to {args.csv}")
    if args.figure:
        from .plotting import plot_sweep

        ax = plot_sweep(result, args.metrics[0])
        ax.figure.savefig(args.figure, dpi=150, bbox_inches="tight")
        print(f"figure written to {args.figure}")
    for point, message in result.failures():
        print(f"FAILED at {point}: {message}", file=sys.stderr)
    return 0 if result.all_converged else 1


def _cmd_geometry(args) -> int:
    from .geometry import NozzleGeometry, check_contour
    from .quasi1d import critical_ratios

    kw = _collect(args)
    geom = NozzleGeometry(
        **{k: v for k, v in kw.items()
           if k in ("contour", "area_ratio", "throat_x", "theta_initial_deg",
                    "theta_exit_deg", "bezier_w1", "bezier_w2")}
    )
    print(geom.describe())
    diag = check_contour(geom)
    for key, value in diag.items():
        print(f"  {key:26s} {value:.6g}")
    print()
    print("  " + critical_ratios(diag["area_ratio"], args.gamma).describe())
    if args.figure:
        from .plotting import plot_contour

        ax = plot_contour(geom)
        ax.figure.savefig(args.figure, dpi=150, bbox_inches="tight")
        print(f"\nfigure written to {args.figure}")
    return 0


def _cmd_bench(args) -> int:
    import time

    from .api import build_case
    from .backends import available_backends, get_backend
    from .config import SolverOptions
    from .initialize import initial_state

    kw = _collect(args)
    case = build_case(**{k: v for k, v in kw.items()
                         if k not in ("cfl", "tolerance", "max_iterations", "scheme",
                                      "limiter", "p_continuation")})
    opts = SolverOptions(cfl=args.cfl, limiter=args.limiter, scheme=args.scheme)
    ops = case.operators
    U0 = initial_state(ops, case.flow, case.geometry, "quasi1d")

    print(f"{case.topology.summary()}")
    print(f"p={case.discretization.order}, {case.discretization.n_dof} DOF/variable")
    print(f"timing {args.steps} steps of {args.scheme}\n")
    print(f"{'backend':10s} {'total [s]':>11s} {'per step [ms]':>15s} {'speedup':>9s}")

    baseline = None
    for name in available_backends():
        bk = get_backend(name, ops, case.flow, opts)
        U = bk.asarray(U0)
        bk.run(U, 2, args.scheme)  # warm up / compile
        t0 = time.perf_counter()
        bk.run(U, args.steps, args.scheme)
        dt = time.perf_counter() - t0
        baseline = baseline or dt
        print(f"{name:10s} {dt:11.3f} {dt / args.steps * 1e3:15.4f} {baseline / dt:8.2f}x")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m dgnozzle",
        description="2D Euler DG nozzle solver",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    s = sub.add_parser("solve", help="solve one operating point")
    _geometry_args(s)
    _flow_args(s)
    _disc_args(s)
    _solver_args(s)
    s.add_argument("--figure", help="write an overview figure to this path")
    s.add_argument("--save", help="write the solution to this .npz path")
    s.set_defaults(func=_cmd_solve)

    w = sub.add_parser("sweep", help="sweep one parameter")
    w.add_argument("parameter", help="parameter to sweep, e.g. area_ratio")
    w.add_argument("start", type=float)
    w.add_argument("stop", type=float)
    w.add_argument("count", type=int)
    _geometry_args(w)
    _flow_args(w)
    _disc_args(w)
    _solver_args(w)
    w.add_argument("--metrics", nargs="+",
                   default=["thrust_coefficient", "exit_mach", "entropy_error"])
    w.add_argument("--csv", help="write the full table here")
    w.add_argument("--figure", help="write a plot here")
    w.set_defaults(func=_cmd_sweep)

    g = sub.add_parser("geometry", help="inspect a contour, no flow solve")
    _geometry_args(g)
    _flow_args(g)
    g.add_argument("--figure", help="write a contour plot here")
    g.set_defaults(func=_cmd_geometry)

    b = sub.add_parser("bench", help="time the backends against each other")
    _geometry_args(b)
    _flow_args(b)
    _disc_args(b)
    _solver_args(b)
    b.add_argument("--steps", type=int, default=200)
    b.set_defaults(func=_cmd_bench)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
