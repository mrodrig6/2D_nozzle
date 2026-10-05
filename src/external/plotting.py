r"""Draw the external wave structure on top of the nozzle geometry.

Kept with the rest of :mod:`src.external` rather than in :mod:`src.plotting`,
so that the external post-processing is one self-contained thing to read, add to
or remove.
"""

from __future__ import annotations

import functools

import numpy as np

from ..plotting import plot_contour
from ..quasi1d import solve_quasi1d
from .jet import JetCells, jet_wave_cells
from .plume import ExitWaves, exit_wave_structure

__all__ = [
    "FIELD_RAMPS",
    "PLUME_QUANTITIES",
    "REGIME_COLOURS",
    "latex_rc",
    "plot_exit_waves",
    "plot_jet_cells",
    "plot_jet_field",
    "plot_nozzle_and_plume",
]


def latex_rc(use_latex: str | bool = "auto") -> dict:
    r"""Matplotlib settings that typeset the figures the way LaTeX would.

    Parameters
    ----------
    use_latex
        ``True`` drives a real LaTeX installation through
        ``text.usetex``.  ``False`` uses Matplotlib's own mathtext with the
        Computer Modern fonts, which needs nothing installed and is visually
        very close.  ``'auto'`` (the default) picks the first when it will
        actually work and the second otherwise.

    Why ``'auto'`` rather than just switching ``text.usetex`` on
    ---------------------------------------------------------
    ``text.usetex`` needs more than LaTeX: on a raster backend Matplotlib
    shells out to ``latex`` **and** ``dvipng``, and a machine with a perfectly
    good ``pdflatex`` but no ``dvipng`` fails at ``savefig`` with a LaTeX log
    dump rather than anything that reads like a missing dependency.  That is a
    bad failure for a teaching code, where the figure is often the first thing
    a student runs.  So the real thing is used when it is genuinely available,
    and the Computer Modern fallback -- same fonts, no install -- otherwise.

    Returns
    -------
    dict
        rcParams, for ``matplotlib.pyplot.rc_context``.
    """
    import shutil

    if use_latex == "auto":
        use_latex = bool(shutil.which("latex")) and bool(shutil.which("dvipng"))

    if use_latex:
        return {
            "text.usetex": True,
            "font.family": "serif",
            "text.latex.preamble": r"\usepackage{amsmath}",
        }
    return {
        "text.usetex": False,
        # Computer Modern for the *maths*, which is what carries the LaTeX look
        # in these figures -- the labels are nearly all $x$, $y$, $p_b/p_t$,
        # $M_e$.
        "mathtext.fontset": "cm",
        "font.family": "serif",
        # ...but DejaVu Serif for the prose.  Matplotlib ships `cmr10`, the real
        # Computer Modern roman, and it is the obvious choice here -- except that
        # it covers little beyond ASCII, so an em dash or a Greek letter in a
        # title silently renders as a tofu box rather than failing.  A teaching
        # code should not have that trap in it, and the difference between the
        # two serifs in a title is far smaller than the cost of a broken glyph.
        "font.serif": ["DejaVu Serif"],
        "axes.formatter.use_mathtext": True,
    }


def _styled(fn):
    """Draw inside :func:`latex_rc`'s settings, and take ``use_latex`` for it.

    A decorator rather than a context manager inside each function, so the
    styling decision lives in exactly one place and the plotting code stays
    about the physics.  Text objects capture the font and ``usetex`` settings
    when they are created, so wrapping the call is enough for everything these
    functions draw.  Axes supplied by the caller were created outside, though,
    so a caller who wants their *own* figure's tick labels styled too should
    wrap that figure in ``plt.rc_context(latex_rc())`` themselves.
    """

    @functools.wraps(fn)
    def wrapper(*args, use_latex: str | bool = "auto", **kwargs):
        import matplotlib.pyplot as plt

        with plt.rc_context(latex_rc(use_latex)):
            return fn(*args, **kwargs)

    return wrapper


#: One hue per regime, assigned by regime and never by position, so a figure
#: showing two of the three keeps the same colours as one showing all three.
#:
#: Sampled from **viridis** at 0.225, 0.525 and 0.725.  Those three stops are
#: not arbitrary: the obvious choice of spreading across the whole ramp fails a
#: categorical palette check, because viridis runs from near-black purple to
#: near-white yellow and categorical hues have to share a lightness band.  These
#: three were searched for and validated -- lightness band, chroma floor, CVD
#: separation (worst adjacent dE 16.0 protan, 11.6 tritan) and normal-vision
#: separation (16.2) all pass.  The lightest sits at 2.17:1 against white, below
#: the 3:1 bar, which is why every figure using these also carries a legend or
#: names the regime in its title: colour is never the only cue.
REGIME_COLOURS = {
    "under-expanded": "#3e4a89",
    "design": "#1f968b",
    "over-expanded": "#50c46a",
}


@_styled
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


@_styled
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
#: * **velocity** and **Mach** are *magnitudes* with no special middle value, so
#:   viridis is exactly the right tool: perceptually uniform, monotone in
#:   lightness, and readable under every common colour-vision deficiency.
#: * **pressure** is a *polarity* -- above or below the ambient the jet is trying
#:   to match -- so it keeps a diverging ramp, whose neutral midpoint sits
#:   exactly on :math:`p_{amb}`.  That puts the sign in the hue, where the eye
#:   reads it without consulting the colourbar, which a sequential ramp cannot
#:   do however well it is labelled.  Pass ``cmap='viridis'`` to
#:   :func:`plot_jet_field` for one ramp across every field.
#:
#: Pressure is scaled on :math:`\log(p/p_{amb})` whichever ramp is used, because
#: a pressure ratio is symmetric in the log and not in the value.
FIELD_RAMPS = {
    "pressure": ("coolwarm", "diverging", r"$p / p_{amb}$"),
    "velocity": ("viridis", "sequential", r"$v / a_t$"),
    "mach": ("viridis", "sequential", r"$M$"),
}


@_styled
def plot_jet_field(
    result,
    cells: JetCells | None = None,
    *,
    quantity: str = "pressure",
    n_cells: int = 3,
    ax=None,
    mirror: bool = True,
    colorbar: bool = True,
    cmap: str | None = None,
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
    if cmap is not None:
        cmap_name = cmap
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


#: What each stitched quantity is, on both sides of the exit plane.  Every one
#: is a *ratio*, which is what makes the two sides commensurate: the interior
#: comes from the quasi-1D solution in solver units and the exterior from the
#: wave march in its own, so stitching raw values would join two different
#: scales and look continuous while being wrong.
PLUME_QUANTITIES = {
    "mach": r"$M$",
    "pressure": r"$p / p_t$",
    "temperature": r"$T / T_t$",
    "velocity": r"$v / a_t$",
}


def _interior_ratios(q1d, flow, quantity: str) -> np.ndarray:
    """The quasi-1D interior field, as the same ratio the exterior reports."""
    if quantity == "mach":
        return np.asarray(q1d.mach)
    if quantity == "pressure":
        return np.asarray(q1d.pressure) / flow.total_pressure
    if quantity == "temperature":
        return np.asarray(q1d.temperature) / flow.total_temperature
    return np.asarray(q1d.velocity) / flow.stagnation_sound_speed


def _exterior_ratio(region, gamma: float, quantity: str) -> float:
    """The same ratio for one external wave-cell region."""
    if quantity == "mach":
        return region.mach
    if quantity == "pressure":
        return region.pressure_ratio
    if quantity == "temperature":
        # isentropic from the region's Mach number, which is what the march
        # tracks; p and T are not independent along an isentrope
        return 1.0 / (1.0 + 0.5 * (gamma - 1.0) * region.mach**2)
    return region.velocity_ratio


@_styled
def plot_nozzle_and_plume(
    result,
    cells: JetCells | None = None,
    *,
    quantity: str = "mach",
    n_cells: int = 3,
    ambient_pressure_ratio: float | None = None,
    ax=None,
    colorbar: bool = True,
    cmap: str = "viridis",
):
    r"""The quasi-1D nozzle and the external plume, stitched into one field.

    Left of the exit plane is the **quasi-1D solution inside the nozzle**; right
    of it is the **wave-cell march outside**, both on one colour scale and one
    axis. The two are joined at the lip, which is where they genuinely meet: the
    march is started from the exit state the quasi-1D solution ends at, so the
    picture is continuous because the physics is, not because it was drawn that
    way.

    This is the whole flow a student is reasoning about -- reservoir, throat,
    expansion, and then the shock cells that carry on downstream -- in a single
    frame, rather than an interior plot and an exterior plot that have to be
    mentally joined.

    **The small step at the lip is real, and it is the point.**  Inside is
    quasi-1D theory; outside is a march started from the *computed* exit state,
    which is a 2D solve.  Those two exit states are not the same -- quasi-1D
    assumes parallel streamlines at the exit and the real flow is still
    diverging -- so the colour jumps by roughly half a percent to one percent
    across the dashed exit line.  That jump is the two-dimensionality of the
    exit flow, the same effect that puts this nozzle's true design point at
    :math:`p_b/p_t \approx 0.0669` rather than the quasi-1D 0.0640.  Drawing the
    two sides from one model would hide it.

    Parameters
    ----------
    quantity
        One of :data:`PLUME_QUANTITIES`.  Everything is a ratio, because the two
        regions report in different units and stitching raw values would join
        two scales and look seamless while being wrong.
    n_cells
        Shock cells to march downstream, if ``cells`` is not supplied.

    Notes
    -----
    Inside the nozzle the field is constant across ``y`` at each station, which
    is the quasi-1D assumption drawn rather than described.  Outside, each wave
    cell is uniform, which is the wave march's own assumption.  Neither is a
    2D solution -- for that, inside the nozzle, see
    :func:`src.plotting.plot_dg_vs_quasi1d`.
    """
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize

    if quantity not in PLUME_QUANTITIES:
        raise ValueError(f"quantity must be one of {sorted(PLUME_QUANTITIES)}, got {quantity!r}")
    if cells is None:
        cells = jet_wave_cells(result, ambient_pressure_ratio=ambient_pressure_ratio, cells=n_cells)
    if ax is None:
        _, ax = plt.subplots(figsize=(11.0, 3.6), constrained_layout=True)

    geom, flow = result.geometry, result.flow
    q1d = solve_quasi1d(geom, flow)
    inner = _interior_ratios(q1d, flow, quantity)

    polys = cells.region_polygons()
    outer = [_exterior_ratio(r, flow.gamma, quantity) for r, _ in polys]

    lo = float(min(inner.min(), min(outer))) if outer else float(inner.min())
    hi = float(max(inner.max(), max(outer))) if outer else float(inner.max())
    norm = Normalize(vmin=lo, vmax=hi)
    ramp = plt.get_cmap(cmap)
    levels = np.linspace(lo, hi, 24)

    # ---- inside: quasi-1D, constant across the channel at each station -----
    wall = np.asarray(geom.wall(q1d.x))
    xx = np.repeat(np.asarray(q1d.x)[:, None], 2, axis=1)
    yy = np.stack([-wall, wall], axis=1)
    zz = np.repeat(inner[:, None], 2, axis=1)
    art = ax.contourf(xx, yy, zz, levels=levels, cmap=ramp, extend="both")

    # ---- outside: one flat colour per wave cell ----------------------------
    for (region, poly), value in zip(polys, outer, strict=True):
        del region
        for sign in (1.0, -1.0):
            ax.fill(
                poly[:, 0],
                sign * poly[:, 1],
                facecolor=ramp(norm(value)),
                edgecolor="white",
                linewidth=0.9,
                zorder=2,
            )

    plot_contour(geom, ax=ax, mirror=True, color="k", lw=1.4)
    ax.axvline(cells.lip_x, color="k", lw=1.0, ls="--", zorder=4)
    ax.axhline(0.0, color="0.4", lw=0.5, ls=":", zorder=4)
    ax.set_xlabel("$x$")
    ax.set_ylabel("$y$")
    ax.set_aspect("equal", adjustable="box")
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    if colorbar:
        plt.colorbar(art, ax=ax, label=PLUME_QUANTITIES[quantity], pad=0.015, fraction=0.04)

    ew = cells.exit_waves
    ax.set_title(
        f"quasi-1D nozzle stitched to the {cells.regime} plume   "
        f"$p_e/p_{{amb}}$ = {ew.pressure_mismatch:.3f}   $M_e$ = {ew.exit_mach:.2f}",
        fontsize=10,
        loc="left",
    )
    return ax
