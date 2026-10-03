r"""Draw the external wave structure on top of the nozzle geometry.

Kept with the rest of :mod:`src.external` rather than in :mod:`src.plotting`,
so that the external post-processing is one self-contained thing to read, add to
or remove.
"""

from __future__ import annotations

import numpy as np

from ..plotting import plot_contour
from .jet import JetCells, jet_wave_cells
from .plume import ExitWaves, exit_wave_structure

__all__ = [
    "FIELD_RAMPS",
    "REGIME_COLOURS",
    "plot_exit_waves",
    "plot_jet_cells",
    "plot_jet_field",
]

#: One hue per regime, assigned by regime and never by position, so a figure
#: showing two of the three keeps the same colours as one showing all three.
#: Validated for colour-vision deficiency: worst adjacent pair dE 10.4 (deutan),
#: 24.8 normal vision, all three above 3:1 against a white surface.
REGIME_COLOURS = {
    "under-expanded": "#3b5bdb",
    "design": "#087f5b",
    "over-expanded": "#c2410c",
}


def plot_exit_waves(
    result,
    waves: ExitWaves | None = None,
    *,
    ax=None,
    mirror: bool = True,
    extent: float = 0.45,
    label: bool = True,
):
    """Plot the nozzle wall and the first wave leaving the lip.

    ``extent`` is how far downstream to draw the wave, as a fraction of the
    nozzle length.  The wave is a *straight line at the computed angle*: that is
    what the closed-form relation gives, and drawing it curved would imply
    detail the relation does not contain.
    """
    import matplotlib.pyplot as plt

    if waves is None:
        waves = exit_wave_structure(result)
    if ax is None:
        _, ax = plt.subplots(figsize=(7.2, 3.4), constrained_layout=True)

    geom = result.geometry
    plot_contour(geom, ax=ax, mirror=mirror, color="0.25", lw=1.4)

    length = geom.length
    y_lip = float(np.asarray(geom.wall(np.array([length])))[0])
    reach = extent * length
    colour = REGIME_COLOURS[waves.regime]

    def ray(angle, style, width, alpha=1.0):
        """A line from the lip at ``angle`` below the axis direction."""
        dx = reach
        dy = -reach * np.tan(angle)
        for s in (1.0,) if not mirror else (1.0, -1.0):
            ax.plot(
                [length, length + dx],
                [s * y_lip, s * (y_lip + dy)],
                style,
                color=colour,
                lw=width,
                alpha=alpha,
                solid_capstyle="round",
                zorder=3,
            )

    if waves.regime == "over-expanded":
        # a single oblique shock, drawn solid: it is a discontinuity
        ray(waves.wave_angle, "-", 2.0)
    elif waves.regime == "under-expanded":
        # the fan, drawn as a few characteristics between the two bounding angles
        lead, trail = waves.fan_angles
        for frac in np.linspace(0.0, 1.0, 7):
            a = lead + frac * (trail - lead)
            ray(a, "-", 1.0, alpha=0.75)
    # the design point gets no wave, which is the point of it

    ax.axhline(0.0, color="0.75", lw=0.6, ls=":", zorder=1)
    ax.axvline(length, color="0.75", lw=0.7, ls="--", zorder=1)
    ax.set_xlabel("$x$")
    ax.set_ylabel("$y$")
    ax.set_xlim(-0.02 * length, length + reach * 1.04)
    ax.set_aspect("equal", adjustable="box")
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)

    if label:
        ax.set_title(
            f"{waves.regime}   $p_e/p_{{amb}}$ = {waves.pressure_mismatch:.3f}"
            f"   $M_e$ = {waves.exit_mach:.2f}",
            fontsize=10,
            loc="left",
            color="0.2",
        )
        if waves.regime != "design":
            # The waves run downstream and *toward* the axis in both regimes,
            # even though the under-expanded jet turns outward.  These are Mach
            # waves and a shock, not streamlines, and conflating the two is the
            # standard way to misread this picture.
            kind = (
                "expansion fan (Mach waves)"
                if waves.regime == "under-expanded"
                else "oblique shock"
            )
            way = "outward" if waves.regime == "under-expanded" else "inward"
            ax.text(
                0.99,
                0.06,
                f"{kind} \u2014 flow turns {abs(np.degrees(waves.turn_angle)):.1f}\u00b0 {way}",
                transform=ax.transAxes,
                fontsize=8,
                color=colour,
                ha="right",
                va="bottom",
            )
    return ax


def plot_jet_cells(
    result,
    cells: JetCells | None = None,
    *,
    n_cells: int = 3,
    ax=None,
    ax_pressure=None,
    mirror: bool = True,
):
    """Draw the extended downstream domain: the wave cells and the axis pressure.

    Two panels sharing ``x``, because the periodicity is the point and it reads
    far better as a trace than as a pattern:

    * **top** -- the nozzle, the jet boundary, and each wave as a straight
      characteristic.  Compressions are solid and expansions dashed, so wave
      *type* survives a greyscale print and does not rely on colour.
    * **bottom** -- :math:`p/p_{\\rm amb}` along the symmetry axis, a staircase
      that crosses 1 once per half cell.  The horizontal line at 1 is the
      ambient the jet is trying to match and keeps overshooting.

    ``n_cells`` is how many shock cells to march, and therefore how far
    downstream the domain extends.

    Pass ``ax`` alone for the wave pattern by itself, or ``ax`` and
    ``ax_pressure`` to place both panels into axes you already own; pass neither
    and a two-panel figure is made here.
    """
    import matplotlib.pyplot as plt

    if cells is None:
        cells = jet_wave_cells(result, cells=n_cells)

    if ax is None:
        _, axes = plt.subplots(
            2,
            1,
            figsize=(9.0, 5.0),
            height_ratios=(2.0, 1.0),
            sharex=True,
            constrained_layout=True,
        )
        ax, ax_p = axes
    else:
        ax_p = ax_pressure

    geom = result.geometry
    plot_contour(geom, ax=ax, mirror=mirror, color="0.25", lw=1.4)
    colour = REGIME_COLOURS[cells.regime]
    signs = (1.0, -1.0) if mirror else (1.0,)

    # the jet boundary: a streamline, so it is the thing that visibly bulges
    bnd = cells.boundary
    for s in signs:
        ax.plot(
            bnd[:, 0],
            s * bnd[:, 1],
            color=colour,
            lw=1.8,
            alpha=0.9,
            zorder=3,
            label="jet boundary" if s > 0 else None,
        )

    # waves, labelled once each so the legend has one entry per type
    seen = set()
    for k, seg in enumerate(cells.wave_segments):
        # the region reached by wave k is regions[k + 1]
        compressive = cells.regions[k + 1].nu < cells.regions[k].nu
        style = "-" if compressive else "--"
        name = "compression" if compressive else "expansion"
        for s in signs:
            lab = None
            if s > 0 and name not in seen:
                lab = name
                seen.add(name)
            ax.plot(
                seg[:, 0],
                s * seg[:, 1],
                style,
                color=colour,
                lw=1.3 if compressive else 1.0,
                alpha=0.85,
                zorder=2,
                label=lab,
            )

    ax.axhline(0.0, color="0.75", lw=0.6, ls=":", zorder=1)
    ax.axvline(cells.lip_x, color="0.75", lw=0.7, ls="--", zorder=1)
    ax.set_ylabel("$y$")
    ax.set_aspect("equal", adjustable="box")
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.legend(loc="upper left", fontsize=8, frameon=False, ncol=3)
    ew = cells.exit_waves
    ax.set_title(
        f"{cells.regime} jet   $p_e/p_{{amb}}$ = {ew.pressure_mismatch:.3f}"
        f"   $M_e$ = {ew.exit_mach:.2f}"
        f"   cell length {cells.cell_length:.3f}",
        fontsize=10,
        loc="left",
    )

    if ax_p is not None:
        amb = ew.ambient_pressure_ratio
        # the axis crossings bound each on-axis region
        xs = [cells.lip_x]
        for seg in cells.wave_segments:
            if abs(seg[1, 1]) < 1e-12:
                xs.append(float(seg[1, 0]))
        axis_regions = [r for r in cells.regions if r.on_axis]
        n = min(len(xs) - 1, len(axis_regions))
        for i in range(n):
            ax_p.plot(
                [xs[i], xs[i + 1]],
                [axis_regions[i].pressure_ratio / amb] * 2,
                color=colour,
                lw=2.0,
                solid_capstyle="butt",
            )
            if i + 1 < n:
                ax_p.plot(
                    [xs[i + 1]] * 2,
                    [
                        axis_regions[i].pressure_ratio / amb,
                        axis_regions[i + 1].pressure_ratio / amb,
                    ],
                    color=colour,
                    lw=0.8,
                    alpha=0.5,
                )
        ax_p.axhline(1.0, color="0.45", lw=0.8, ls="--")
        ax_p.annotate(
            "ambient",
            xy=(0.995, 1.0),
            xycoords=("axes fraction", "data"),
            ha="right",
            va="bottom",
            fontsize=8,
            color="0.35",
        )
        ax_p.set_xlabel("$x$")
        ax_p.set_ylabel("$p / p_{amb}$ on the axis")
        for side in ("top", "right"):
            ax_p.spines[side].set_visible(False)
    else:
        ax.set_xlabel("$x$")

    if cells.stopped_because:
        ax.annotate(
            "march stopped: " + cells.stopped_because.split(":")[0],
            xy=(0.99, 0.04),
            xycoords="axes fraction",
            ha="right",
            fontsize=8,
            color="0.35",
        )
    return ax


#: How each field is encoded.  The jobs differ, so the ramps differ:
#:
#: * **pressure** is a *polarity* -- above or below the ambient the jet is trying
#:   to match -- so it gets a diverging ramp pinned with its neutral midpoint
#:   exactly at ``p_amb``.  Blue reads as under-ambient (over-expanded locally),
#:   red as over-ambient.  The midpoint carries meaning, which is the whole
#:   reason not to use a sequential ramp here.
#: * **velocity** and **Mach** are *magnitudes* with no special middle value, so
#:   they get a single-hue sequential ramp, light to dark.
#:
#: Neither is a rainbow, and neither hue is reused from REGIME_COLOURS, so a
#: field plot can never be misread as a regime plot.
FIELD_RAMPS = {
    "pressure": ("coolwarm", "diverging", r"$p / p_{amb}$"),
    "velocity": ("Purples", "sequential", r"$v / a_t$"),
    "mach": ("Purples", "sequential", r"$M$"),
}


def plot_jet_field(
    result,
    cells: JetCells | None = None,
    *,
    quantity: str = "pressure",
    n_cells: int = 3,
    ax=None,
    mirror: bool = True,
    colorbar: bool = True,
):
    """Fill the cell pattern, colouring each region by one flow quantity.

    ``plot_jet_cells`` draws the *waves*; this draws the *regions between* them,
    which is what makes the periodicity read as alternating states rather than
    as a line drawing.

    Parameters
    ----------
    quantity
        ``'pressure'``, ``'velocity'`` or ``'mach'``.  See :data:`FIELD_RAMPS`
        for why pressure is encoded differently from the other two.
    n_cells
        Shock cells to march, if ``cells`` is not supplied.

    Notes
    -----
    Each region is drawn with a thin surface-coloured edge rather than a stroke
    in the ramp, so neighbouring fills are separated by a gap instead of running
    together -- the wave *is* the gap, which keeps the structure legible without
    drawing a second set of lines over the top of the colour.

    The last region is not drawn: it has no downstream wave, so its extent is
    not something the march computed.
    """
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize

    if quantity not in FIELD_RAMPS:
        raise ValueError(f"quantity must be one of {sorted(FIELD_RAMPS)}, got {quantity!r}")
    if cells is None:
        cells = jet_wave_cells(result, cells=n_cells)
    if ax is None:
        _, ax = plt.subplots(figsize=(9.5, 3.4), constrained_layout=True)

    cmap_name, kind, label = FIELD_RAMPS[quantity]
    amb = cells.exit_waves.ambient_pressure_ratio
    polys = cells.region_polygons()
    if not polys:
        raise ValueError("the march produced no closed region to fill")

    def value(region):
        if quantity == "pressure":
            return region.pressure_ratio / amb
        if quantity == "velocity":
            return region.velocity_ratio
        return region.mach

    vals = [value(r) for r, _ in polys]
    span = 0.0
    if kind == "diverging":
        # p/p_amb is a *ratio*, so it is symmetric in the log, not in the value:
        # twice ambient and half ambient are equal and opposite departures.  A
        # linear ramp centred on 1 would both exaggerate the high side and run
        # the scale off the bottom into negative pressure ratios, which mean
        # nothing.  So colour on log(p/p_amb), symmetric about 0.
        logs = np.log(np.asarray(vals))
        span = float(max(np.abs(logs).max(), 1e-9))
        norm = Normalize(vmin=-span, vmax=span)

        def colour_of(v):
            return norm(np.log(v))
    else:
        norm = Normalize(vmin=min(vals), vmax=max(vals))
        colour_of = norm
    cmap = plt.get_cmap(cmap_name)

    signs = (1.0, -1.0) if mirror else (1.0,)
    for (_region, poly), v in zip(polys, vals, strict=True):
        for s in signs:
            xy = np.column_stack([poly[:, 0], s * poly[:, 1]])
            ax.fill(
                xy[:, 0],
                xy[:, 1],
                facecolor=cmap(colour_of(v)),
                edgecolor="white",
                linewidth=1.2,
                zorder=2,
            )

    plot_contour(result.geometry, ax=ax, mirror=mirror, color="0.25", lw=1.4)
    ax.axhline(0.0, color="0.35", lw=0.6, ls=":", zorder=4)
    ax.set_xlabel("$x$")
    ax.set_ylabel("$y$")
    ax.set_aspect("equal", adjustable="box")
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ew = cells.exit_waves
    ax.set_title(
        f"{cells.regime} jet, coloured by {quantity}"
        f"   $p_e/p_{{amb}}$ = {ew.pressure_mismatch:.3f}   $M_e$ = {ew.exit_mach:.2f}",
        fontsize=10,
        loc="left",
    )

    if colorbar:
        sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
        cb = ax.figure.colorbar(sm, ax=ax, pad=0.015, fraction=0.045)
        cb.set_label(label, fontsize=9)
        if kind == "diverging":
            # the bar is in log space; label it with the ratios people read
            ticks = sorted({-span, -span / 2, 0.0, span / 2, span})
            cb.set_ticks(ticks)
            cb.set_ticklabels([f"{np.exp(t):.2f}" for t in ticks])
            cb.ax.axhline(0.0, color="0.2", lw=1.0)  # ambient
    return ax
