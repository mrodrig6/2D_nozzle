r"""Parameter sweeps.

A sweep is the main thing students do with this solver: vary a design variable,
watch a performance metric respond, and explain the trend.  Two features make
that cheap:

**Warm starting.**  Consecutive sweep points are nearby designs, so the previous
converged field is an excellent initial guess for the next.  Because the mesh
*topology* never changes when a design variable moves -- only the node
coordinates do -- the previous solution is always shape-compatible and can be
handed straight to the next solve.  For a smooth sweep this typically cuts total
cost by 2-5x.  Multi-dimensional grids are walked in serpentine order so that
successive points stay adjacent.

**Process parallelism.**  Optional, and a genuine trade-off rather than a free
win: the Numba kernels are already thread-parallel across elements, so running
several solves at once oversubscribes the machine unless each worker is pinned to
one thread.  ``parallel=n`` does exactly that, and disables warm starting (which
cannot cross a process boundary).  It pays off for many small solves; a sweep of
a few large ones is better left serial.
"""

from __future__ import annotations

import itertools
import os
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from .api import solve_nozzle
from .config import Discretization, FlowConditions, SolverOptions
from .geometry import NozzleGeometry
from .postprocess import Performance, performance
from .quasi1d import operating_regime, solve_quasi1d

#: Metrics recorded at every sweep point.
METRICS = (
    "thrust",
    "thrust_wall",
    "thrust_coefficient",
    "thrust_imbalance",
    "thrust_efficiency",
    "specific_thrust",
    "mass_flow_in",
    "mass_imbalance",
    "entropy_error",
    "exit_mach",
    "exit_pressure_ratio",
    "ideal_thrust",
    "quasi1d_exit_mach",
    "quasi1d_shock_x",
    "iterations",
    "wall_time",
    "residual",
    "min_density",
    "min_pressure",
)


@dataclass
class SweepResult:
    """Tabulated results of a parameter sweep.

    Index a metric by name (``result['thrust']``) or a swept parameter
    (``result['area_ratio']``); both return 1-D arrays aligned with
    :attr:`points`.
    """

    parameters: tuple[str, ...]
    points: list[dict[str, float]]
    values: dict[str, np.ndarray]
    converged: np.ndarray
    regimes: list[str]
    messages: list[str]
    shape: tuple[int, ...] = ()

    def __getitem__(self, key: str) -> np.ndarray:
        if key in self.values:
            return self.values[key]
        if key in self.parameters:
            return np.asarray([p[key] for p in self.points])
        raise KeyError(
            f"{key!r} is neither a swept parameter {self.parameters} nor a metric; "
            f"metrics are {tuple(self.values)}"
        )

    @property
    def n_points(self) -> int:
        return len(self.points)

    @property
    def all_converged(self) -> bool:
        return bool(np.all(self.converged))

    def reshape(self, key: str) -> np.ndarray:
        """A metric reshaped to the sweep's grid shape (for a 2-D contour plot)."""
        return self[key].reshape(self.shape)

    def failures(self) -> list[tuple[dict[str, float], str]]:
        """The points that did not converge, with the solver's explanation."""
        return [
            (p, m)
            for p, m, ok in zip(self.points, self.messages, self.converged, strict=True)
            if not ok
        ]

    def table(
        self,
        metrics: Sequence[str] = ("thrust_coefficient", "exit_mach", "entropy_error"),
    ) -> str:
        """A fixed-width table, for printing."""
        cols = list(self.parameters) + list(metrics) + ["ok"]
        head = "  ".join(f"{c:>18s}" for c in cols)
        lines = [head, "-" * len(head)]
        for i, pt in enumerate(self.points):
            row = [f"{pt[p]:18.6g}" for p in self.parameters]
            row += [f"{self.values[m][i]:18.6g}" for m in metrics]
            row.append(f"{'yes' if self.converged[i] else 'NO':>18s}")
            lines.append("  ".join(row))
        return "\n".join(lines)

    def to_csv(self, path: str) -> None:
        """Write every parameter and metric to a CSV file."""
        cols = list(self.parameters) + list(self.values) + ["converged", "regime"]
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(",".join(cols) + "\n")
            for i, pt in enumerate(self.points):
                row = [repr(pt[p]) for p in self.parameters]
                row += [repr(float(self.values[m][i])) for m in self.values]
                row += [str(bool(self.converged[i])), self.regimes[i]]
                fh.write(",".join(row) + "\n")

    def to_dict(self) -> dict[str, np.ndarray]:
        out = {p: self[p] for p in self.parameters}
        out.update(self.values)
        out["converged"] = self.converged
        return out


def _serpentine(grids: list[np.ndarray]) -> list[tuple[int, ...]]:
    """Grid indices walked so that consecutive entries are adjacent.

    Reverses every other row of each axis in turn, so a warm start always comes
    from a neighbouring design rather than from the far end of the previous row.
    """
    ranges = [range(len(g)) for g in grids]
    out: list[tuple[int, ...]] = []
    for idx in itertools.product(*ranges):
        out.append(idx)
    if len(grids) <= 1:
        return out
    # reorder: for each block of the last axis, reverse when the preceding index sums odd
    n_last = len(grids[-1])
    reordered: list[tuple[int, ...]] = []
    for block_start in range(0, len(out), n_last):
        block = out[block_start : block_start + n_last]
        if (sum(block[0][:-1])) % 2 == 1:
            block = block[::-1]
        reordered.extend(block)
    return reordered


def _extract(
    result, perf: Performance, geom: NozzleGeometry, flow: FlowConditions
) -> dict[str, float]:
    q1d = solve_quasi1d(geom, flow)
    return {
        "thrust": perf.thrust,
        "thrust_wall": perf.thrust_wall,
        "thrust_coefficient": perf.thrust_coefficient,
        "thrust_imbalance": perf.thrust_imbalance,
        "thrust_efficiency": perf.thrust_efficiency,
        "specific_thrust": perf.specific_thrust,
        "mass_flow_in": perf.mass_flow_in,
        "mass_imbalance": perf.mass_imbalance,
        "entropy_error": perf.entropy_error,
        "exit_mach": perf.exit_mach_area_averaged,
        "exit_pressure_ratio": perf.exit_pressure_ratio,
        "ideal_thrust": perf.ideal_thrust,
        "quasi1d_exit_mach": q1d.exit_mach,
        "quasi1d_shock_x": float("nan") if q1d.shock_x is None else q1d.shock_x,
        "iterations": float(result.iterations),
        "wall_time": result.history.wall_time,
        "residual": result.residual,
        "min_density": result.min_density,
        "min_pressure": result.min_pressure,
    }


def _is_axis(value: Any) -> bool:
    """A sweep axis is an iterable of values; anything scalar is a fixed setting.

    This is what lets ``sweep(area_ratio=np.linspace(2, 4, 5), order=1)`` mean
    "sweep the area ratio, at p = 1" rather than treating ``order`` as a
    one-point axis.
    """
    if isinstance(value, (str, bytes, bool)):
        return False
    if isinstance(value, (int, float, np.integer, np.floating)):
        return False
    try:
        iter(value)
    except TypeError:
        return False
    return True


def _split_grids(kwargs: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    """Separate sweep axes (iterables) from fixed overrides (scalars)."""
    axes = {k: v for k, v in kwargs.items() if _is_axis(v)}
    fixed = {k: v for k, v in kwargs.items() if not _is_axis(v)}
    return axes, fixed


def _solve_point(
    point: Mapping[str, float],
    fixed: Mapping[str, Any],
    geometry: NozzleGeometry | None,
    flow: FlowConditions | None,
    discretization: Discretization | None,
    options: SolverOptions | None,
    backend: str,
    U0,
):
    res = solve_nozzle(
        geometry, flow, discretization, options,
        backend=backend, U0=U0, verbose=False, **dict(fixed), **dict(point),
    )
    perf = performance(res) if res.converged else None
    return res, perf


def _worker(args):  # pragma: no cover - runs in a subprocess
    os.environ.setdefault("NUMBA_NUM_THREADS", "1")
    point, fixed, geometry, flow, discretization, options, backend = args
    res, perf = _solve_point(
        point, fixed, geometry, flow, discretization, options, backend, None
    )
    if perf is None:
        return point, None, res.converged, res.message, res.geometry, res.flow, res
    metrics = _extract(res, perf, res.geometry, res.flow)
    return point, metrics, res.converged, res.message, None, None, None


def sweep(
    geometry: NozzleGeometry | None = None,
    flow: FlowConditions | None = None,
    discretization: Discretization | None = None,
    options: SolverOptions | None = None,
    *,
    backend: str = "numba",
    warm_start: bool = True,
    parallel: int | None = None,
    progress: bool | Callable[[int, int, dict], None] = True,
    **grids: Iterable[float],
) -> SweepResult:
    """Sweep one or more parameters and tabulate performance.

    Parameters
    ----------
    geometry, flow, discretization, options
        Baseline configuration; every sweep point starts from these.
    backend
        Solver backend for each point.
    warm_start
        Reuse the previous point's converged field as the next one's initial
        guess.  Requires ``parallel=None``.
    parallel
        Number of worker processes.  Each is pinned to one Numba thread to avoid
        oversubscription; warm starting is disabled.
    progress
        ``True`` prints a line per point, ``False`` is silent, or pass a callable
        ``(index, total, point) -> None``.
    **grids
        Keywords accepted by :func:`dgnozzle.api.solve_nozzle`.  A keyword whose
        value is an **iterable** becomes a sweep axis
        (``area_ratio=np.linspace(2.0, 4.0, 9)``); a **scalar** is a fixed setting
        applied at every point (``order=1``, ``contour='smooth'``).  Several axes
        form a full Cartesian grid.

    Returns
    -------
    SweepResult

    Examples
    --------
    >>> import numpy as np
    >>> from dgnozzle import sweep                          # doctest: +SKIP
    >>> r = sweep(area_ratio=np.linspace(2.0, 4.0, 5), order=1)   # doctest: +SKIP
    >>> print(r.table())                                    # doctest: +SKIP
    """
    grid_kw, fixed = _split_grids(grids)
    if not grid_kw:
        raise ValueError(
            "sweep needs at least one iterable parameter grid, e.g. "
            "sweep(area_ratio=np.linspace(2, 4, 5)); scalar keywords like "
            f"{sorted(fixed) or ['order=1']} are fixed settings, not axes"
        )
    names = tuple(grid_kw)
    axes = [np.atleast_1d(np.asarray(list(grid_kw[n]), dtype=float)) for n in names]
    shape = tuple(len(a) for a in axes)
    order = _serpentine(axes)
    total = len(order)

    if parallel is not None and warm_start:
        warm_start = False

    values: dict[str, list[float]] = {m: [float("nan")] * total for m in METRICS}
    flat_points: list[dict[str, float]] = [None] * total  # type: ignore[list-item]
    converged = np.zeros(total, dtype=bool)
    regimes: list[str] = [""] * total
    messages: list[str] = [""] * total

    def flat_index(idx: tuple[int, ...]) -> int:
        return int(np.ravel_multi_index(idx, shape))

    def report(k: int, idx: tuple[int, ...], point: dict[str, float], ok: bool) -> None:
        if progress is True:
            desc = ", ".join(f"{n}={point[n]:g}" for n in names)
            print(f"  [{k + 1}/{total}] {desc}  {'ok' if ok else 'FAILED'}")
        elif callable(progress):
            progress(k, total, point)

    if parallel is None:
        U_prev = None
        for k, idx in enumerate(order):
            point = {n: float(axes[i][j]) for i, (n, j) in enumerate(zip(names, idx, strict=True))}
            fi = flat_index(idx)
            flat_points[fi] = point
            res, perf = _solve_point(
                point, fixed, geometry, flow, discretization, options, backend,
                U_prev if warm_start else None,
            )
            converged[fi] = res.converged
            messages[fi] = res.message
            regimes[fi] = operating_regime(
                res.geometry.realised_area_ratio(),
                res.flow.back_pressure_ratio,
                res.flow.gamma,
            )[0].value
            if perf is not None:
                for m, v in _extract(res, perf, res.geometry, res.flow).items():
                    values[m][fi] = v
                if warm_start:
                    U_prev = res.U
            elif warm_start:
                U_prev = None  # do not propagate a failed field
            report(k, idx, point, bool(res.converged))
    else:  # pragma: no cover - exercised only with parallel>1
        from concurrent.futures import ProcessPoolExecutor

        jobs = []
        for idx in order:
            point = {n: float(axes[i][j]) for i, (n, j) in enumerate(zip(names, idx, strict=True))}
            flat_points[flat_index(idx)] = point
            jobs.append((point, fixed, geometry, flow, discretization, options, backend))
        with ProcessPoolExecutor(max_workers=int(parallel)) as pool:
            for k, (idx, out) in enumerate(zip(order, pool.map(_worker, jobs), strict=True)):
                point, metrics, ok, msg, geom_f, flow_f, _ = out
                fi = flat_index(idx)
                converged[fi] = ok
                messages[fi] = msg
                if metrics is not None:
                    for m, v in metrics.items():
                        values[m][fi] = v
                    regimes[fi] = ""
                report(k, idx, point, bool(ok))

    return SweepResult(
        parameters=names,
        points=flat_points,
        values={m: np.asarray(v) for m, v in values.items()},
        converged=converged,
        regimes=regimes,
        messages=messages,
        shape=shape,
    )
