r"""Matplotlib figures: mesh, fields, line-outs, convergence, sweeps.

Every function takes an optional ``ax`` and returns the axes it drew on, so
figures compose.  Nothing calls ``plt.show()`` -- the caller decides.

Field plots sample *inside* each element (see
:func:`src.postprocess.sample_field`).  Plotting element vertices alone
would show a piecewise-linear picture no matter the polynomial order, hiding
exactly what ``p = 2`` is for.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np

from .geometry import NozzleGeometry
from .mesh import TAG_NAMES, BoundaryTag, MeshTopology
from .postprocess import (
    centreline_profile,
    exit_profile,
    sample_field,
    wall_profile,
)
from .quasi1d import solve_quasi1d
from .solver import SolveResult
from .sweep import SweepResult


def _require_matplotlib():
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "plotting needs matplotlib: pip install 'dgnozzle[plot]'"
        ) from exc
    return plt


def _axes(ax=None, figsize=(9.0, 3.2)):
    plt = _require_matplotlib()
    if ax is None:
        _, ax = plt.subplots(figsize=figsize)
    return ax


# --------------------------------------------------------------------------
def plot_contour(
    geom: NozzleGeometry, ax=None, *, n: int = 400, mirror: bool = True, **kwargs: Any
):
    """Draw the nozzle wall (and its mirror image), with the throat marked."""
    ax = _axes(ax)
    x = np.linspace(0.0, geom.length, n)
    y = np.asarray(geom.wall(x))
    kwargs.setdefault("color", "k")
    kwargs.setdefault("lw", 1.6)
    ax.plot(x, y, **kwargs)
    if mirror:
        ax.plot(x, -y, **{**kwargs, "label": None})
        ax.axhline(0.0, color="0.7", lw=0.6, ls=":")
    xt = geom.throat_location()
    ax.axvline(xt, color="0.6", lw=0.7, ls="--")
    ax.annotate(
        "throat", (xt, 0.0), textcoords="offset points", xytext=(4, 6),
        fontsize=8, color="0.4",
    )
    ax.set_xlabel("$x$ [m]")
    ax.set_ylabel("$y$ [m]")
    ax.set_aspect("equal")
    return ax


def plot_mesh(
    result_or_topology, node_coords=None, ax=None, *, lw: float = 0.35, color_boundaries=True
):
    """Draw element edges, optionally colouring the tagged boundaries.

    Accepts a :class:`~src.solver.SolveResult` or a
    ``(topology, node_coords)`` pair.
    """
    from . import elements as el
    from .postprocess import _node_coords

    ax = _axes(ax)
    if isinstance(result_or_topology, MeshTopology):
        topo = result_or_topology
        if node_coords is None:
            raise ValueError("pass node_coords alongside a bare topology")
        coords = np.asarray(node_coords)
    else:
        topo = result_or_topology.operators.topology
        coords = _node_coords(result_or_topology)

    # element outlines, sampled along each edge so curved (Q>1) sides show
    sigma = np.linspace(0.0, 1.0, 2 * topo.geometry_order + 1)
    segs = []
    for f in range(topo.n_faces):
        pts = el.edge_reference_coords(topo.kind, f, sigma)
        phi, _, _ = el.shape_functions(topo.kind, topo.geometry_order, pts)
        xy = np.einsum("nk,end->ekd", phi, coords[topo.elem_nodes])
        segs.append(xy)
    from matplotlib.collections import LineCollection

    ax.add_collection(
        LineCollection(np.concatenate(segs, axis=0), colors="0.45", linewidths=lw)
    )

    if color_boundaries:
        palette = {
            BoundaryTag.INFLOW: "tab:blue",
            BoundaryTag.OUTFLOW: "tab:red",
            BoundaryTag.WALL: "k",
            BoundaryTag.AXIS: "tab:green",
        }
        for tag, colour in palette.items():
            sl = topo.tag_slice(tag)
            if sl.stop <= sl.start:
                continue
            e = topo.edges.bedge_elem[sl]
            f = topo.edges.bedge_face[sl]
            pieces = []
            for ei, fi in zip(e, f, strict=True):
                pts = el.edge_reference_coords(topo.kind, int(fi), sigma)
                phi, _, _ = el.shape_functions(topo.kind, topo.geometry_order, pts)
                pieces.append(phi.T @ coords[topo.elem_nodes[ei]])
            ax.add_collection(
                LineCollection(pieces, colors=colour, linewidths=1.8, label=TAG_NAMES[tag])
            )
        ax.legend(fontsize=8, loc="upper left", frameon=False, ncol=4)

    ax.autoscale_view()
    ax.set_xlabel("$x$ [m]")
    ax.set_ylabel("$y$ [m]")
    ax.set_aspect("equal")
    ax.set_title(f"{topo.n_elem} {topo.kind} elements, $Q={topo.geometry_order}$", fontsize=10)
    return ax


def plot_field(
    result: SolveResult,
    name: str = "mach",
    ax=None,
    *,
    subdivisions: int = 3,
    levels: int | Sequence[float] = 24,
    cmap: str = "viridis",
    mirror: bool = True,
    colorbar: bool = True,
    shading: str = "gouraud",
):
    """Filled contour plot of a derived scalar over the nozzle.

    ``mirror`` also draws the reflected lower half, so the picture shows the
    physical channel rather than the half that was meshed.
    """
    plt = _require_matplotlib()
    ax = _axes(ax, figsize=(9.5, 3.6))
    pts, tris, vals = sample_field(result, name, subdivisions)

    import matplotlib.tri as mtri

    def draw(sign: float):
        faces = tris if sign > 0 else tris[:, ::-1]
        tri = mtri.Triangulation(pts[:, 0], sign * pts[:, 1], faces)
        return ax.tricontourf(tri, vals, levels=levels, cmap=cmap)

    art = draw(1.0)
    if mirror:
        draw(-1.0)

    plot_contour(result.geometry, ax=ax, mirror=mirror, color="k", lw=1.2)
    if colorbar:
        plt.colorbar(art, ax=ax, label=name.replace("_", " "), pad=0.02)
    ax.set_title(
        f"{name} — {result.geometry.contour} contour, "
        f"AR={result.geometry.realised_area_ratio():.3f}, "
        f"$p_b/p_t$={result.flow.back_pressure_ratio:.3f}, "
        f"$p$={result.discretization.order}",
        fontsize=10,
    )
    return ax


def plot_centreline(
    result: SolveResult, ax=None, *, quantity: str = "mach", compare_quasi1d: bool = True
):
    """Axial profile along the symmetry axis, against quasi-1D theory.

    The gap between the two curves *is* the two-dimensionality of the flow: it is
    small in the converging section and grows through the expansion, where the
    streamlines are no longer parallel.
    """
    ax = _axes(ax, figsize=(7.5, 3.4))
    cl = centreline_profile(result)
    ax.plot(cl["x"], cl[quantity], "-", color="tab:blue", lw=1.6, label="DG (axis)")

    if compare_quasi1d:
        q = solve_quasi1d(result.geometry, result.flow)
        ref = {"mach": q.mach, "pressure": q.pressure,
               "density": q.density, "vx": q.velocity}
        ax.plot(q.x, ref[quantity], "--", color="0.4", lw=1.2, label="quasi-1D theory")
        if q.shock_x is not None:
            ax.axvline(q.shock_x, color="tab:red", lw=0.8, ls=":",
                       label=f"quasi-1D shock ($M_1$={q.shock_mach:.2f})")

    ax.axvline(result.geometry.throat_location(), color="0.7", lw=0.7, ls="--")
    ax.set_xlabel("$x$ [m]")
    ax.set_ylabel({"mach": "Mach number", "pressure": "$p$", "density": r"$\rho$",
                   "vx": "$v_x$"}[quantity])
    ax.legend(fontsize=8, frameon=False)
    ax.grid(alpha=0.25)
    return ax


def plot_wall(result: SolveResult, ax=None):
    """Wall pressure and wall Mach number along the contour."""
    ax = _axes(ax, figsize=(7.5, 3.4))
    wp = wall_profile(result)
    ax.plot(wp["x"], wp["pressure"], "-", color="tab:red", lw=1.5)
    ax.set_xlabel("$x$ [m]")
    ax.set_ylabel("wall $p$", color="tab:red")
    ax.tick_params(axis="y", colors="tab:red")
    twin = ax.twinx()
    twin.plot(wp["x"], wp["mach"], "-", color="tab:blue", lw=1.5)
    twin.set_ylabel("wall $M$", color="tab:blue")
    twin.tick_params(axis="y", colors="tab:blue")
    ax.axvline(result.geometry.throat_location(), color="0.7", lw=0.7, ls="--")
    ax.grid(alpha=0.25)
    return ax


def plot_exit_profile(result: SolveResult, ax=None, *, n_points: int = 40):
    """Mach number across the exit plane, with the quasi-1D value for reference.

    A uniform exit profile means the nozzle turns the flow fully axial; the
    spread is a direct measure of divergence loss.
    """
    ax = _axes(ax, figsize=(4.2, 3.8))
    ep = exit_profile(result, n_points)
    ax.plot(ep["mach"], ep["y"], "-", color="tab:blue", lw=1.6, label="DG")
    q = solve_quasi1d(result.geometry, result.flow)
    ax.axvline(q.exit_mach, color="0.4", ls="--", lw=1.2, label="quasi-1D")
    ax.set_xlabel("Mach number")
    ax.set_ylabel("$y$ [m]")
    ax.set_title("exit plane", fontsize=10)
    ax.legend(fontsize=8, frameon=False)
    ax.grid(alpha=0.25)
    return ax


def plot_convergence(results, ax=None, labels: Sequence[str] | None = None):
    """Residual history on a log scale.  Accepts one result or several."""
    ax = _axes(ax, figsize=(6.0, 3.6))
    if isinstance(results, SolveResult):
        results = [results]
    for k, res in enumerate(results):
        it, r = res.history.as_arrays()
        lab = (labels[k] if labels else
               f"p={res.discretization.order}, ref={res.discretization.refine}")
        ax.semilogy(it, r / r[0], lw=1.4, label=lab)
    ax.set_xlabel("iteration")
    ax.set_ylabel(r"$\|\dot{U}\|_{\rm rms}$ / initial")
    ax.legend(fontsize=8, frameon=False)
    ax.grid(alpha=0.25, which="both")
    return ax


def plot_sweep(
    result: SweepResult,
    metric: str = "thrust_coefficient",
    ax=None,
    *,
    reference: str | None = None,
):
    """Plot a 1-D sweep, or a 2-D sweep as a filled contour."""
    plt = _require_matplotlib()
    ax = _axes(ax, figsize=(6.2, 3.8))
    if len(result.parameters) == 1:
        (name,) = result.parameters
        x = result[name]
        ax.plot(x, result[metric], "o-", lw=1.5, ms=4, label=metric)
        if reference:
            ax.plot(x, result[reference], "s--", lw=1.1, ms=3, color="0.4", label=reference)
            ax.legend(fontsize=8, frameon=False)
        bad = ~result.converged
        if bad.any():
            ax.plot(x[bad], result[metric][bad], "x", color="tab:red", ms=9,
                    label="not converged")
            ax.legend(fontsize=8, frameon=False)
        ax.set_xlabel(name.replace("_", " "))
        ax.set_ylabel(metric.replace("_", " "))
    elif len(result.parameters) == 2:
        a, b = result.parameters
        xa = np.unique(result[a])
        xb = np.unique(result[b])
        Z = result.reshape(metric)
        art = ax.contourf(xb, xa, Z, levels=24, cmap="viridis")
        plt.colorbar(art, ax=ax, label=metric.replace("_", " "), pad=0.02)
        ax.set_xlabel(b.replace("_", " "))
        ax.set_ylabel(a.replace("_", " "))
    else:
        raise ValueError(
            f"cannot plot a {len(result.parameters)}-parameter sweep directly; "
            "slice it first, e.g. result.reshape(metric)[i]"
        )
    ax.grid(alpha=0.25)
    return ax


def plot_convergence_study(
    dofs: Sequence[float], errors: Sequence[float], ax=None, *, label: str = "",
    expected_rate: float | None = None,
):
    """Log-log error against ``1/sqrt(DOF)`` with a fitted slope.

    ``1/sqrt(DOF)`` is the effective mesh size in 2D, so the fitted slope is the
    observed order of accuracy.
    """
    ax = _axes(ax, figsize=(5.4, 4.0))
    h = 1.0 / np.sqrt(np.asarray(dofs, dtype=float))
    e = np.asarray(errors, dtype=float)
    slope = float(np.polyfit(np.log(h), np.log(e), 1)[0])
    ax.loglog(h, e, "o-", lw=1.5, ms=5, label=f"{label} (slope {slope:.2f})")
    if expected_rate is not None:
        ref = e[0] * (h / h[0]) ** expected_rate
        ax.loglog(h, ref, "--", color="0.5", lw=1.0, label=f"rate {expected_rate:g}")
    ax.set_xlabel(r"$1/\sqrt{\rm DOF}$")
    ax.set_ylabel("error")
    ax.legend(fontsize=8, frameon=False)
    ax.grid(alpha=0.25, which="both")
    return ax


def overview(result: SolveResult, figsize=(12.0, 8.0)):
    """A four-panel summary figure: field, mesh, centreline and exit profile."""
    plt = _require_matplotlib()
    fig = plt.figure(figsize=figsize, constrained_layout=True)
    gs = fig.add_gridspec(3, 2, height_ratios=[1.1, 1.1, 1.0])
    plot_field(result, "mach", ax=fig.add_subplot(gs[0, :]))
    plot_mesh(result, ax=fig.add_subplot(gs[1, 0]))
    plot_wall(result, ax=fig.add_subplot(gs[1, 1]))
    plot_centreline(result, ax=fig.add_subplot(gs[2, 0]))
    plot_exit_profile(result, ax=fig.add_subplot(gs[2, 1]))
    return fig
