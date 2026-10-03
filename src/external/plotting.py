r"""Draw the external wave structure on top of the nozzle geometry.

Kept with the rest of :mod:`src.external` rather than in :mod:`src.plotting`,
so that the external post-processing is one self-contained thing to read, add to
or remove.
"""

from __future__ import annotations

import numpy as np

from ..plotting import plot_contour
from .plume import ExitWaves, exit_wave_structure

__all__ = ["REGIME_COLOURS", "plot_exit_waves"]

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
