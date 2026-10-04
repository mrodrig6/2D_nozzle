r"""The repeating wave pattern downstream of the lip: the jet's "shock diamonds".

:mod:`src.external.plume` stops at the first wave leaving the lip.  This module
extends the domain horizontally, marching that wave through its reflections to
give the periodic cell structure a student actually sees in a photograph of a
jet.

The construction
----------------
Two boundaries close the problem, and each reflects a wave in a different way:

* the **symmetry axis** :math:`y = 0`, where the flow must run axial, so a wave
  reflects as the *same* family -- an expansion reflects as an expansion;
* the **jet boundary**, a free surface at constant :math:`p = p_{\rm amb}`,
  where a wave reflects as the *opposite* family -- a compression reflects as an
  expansion.

Writing states as :math:`(\nu, \theta)` with :math:`\nu` the Prandtl-Meyer
function and :math:`\theta` the flow angle (positive *away* from the axis), and
:math:`\theta_1 = \nu_1 - \nu_0` the turn through the lip wave, the march closes
after four states and then repeats:

.. math::
    (\nu_0, 0)
    \;\xrightarrow{\text{lip}}\; (\nu_1, \theta_1)
    \;\xrightarrow{\text{axis}}\; (\nu_1 + \theta_1, 0)
    \;\xrightarrow{\text{boundary}}\; (\nu_1, -\theta_1)
    \;\xrightarrow{\text{axis}}\; (\nu_0, 0)
    \;\xrightarrow{\text{boundary}}\; (\nu_1, \theta_1) \;\cdots

That period-4 cycle *is* the cell structure: the jet alternately over- and
under-expands about the ambient pressure, and the pattern is strictly periodic
because every wave is taken as isentropic.  Which way the cycle runs depends on
the regime -- :math:`\theta_1 > 0` under-expanded, :math:`\theta_1 < 0`
over-expanded -- so one march covers both.

What this model is not
----------------------
**Every wave is treated as isentropic**, including the compressions.  That is
the assumption that makes the pattern exactly periodic, and it is the first
thing to go in a real jet:

* a compressive turn steepens into an oblique shock, which loses total pressure,
  so the real cells decay downstream instead of repeating forever;
* at the axis a compression has to turn the flow by :math:`2\theta_1`, and once
  that exceeds the maximum attached deflection the regular reflection assumed
  here becomes a **Mach reflection** -- the Mach disc of a strongly
  under-expanded jet.  :func:`jet_wave_cells` checks this and stops the march
  rather than drawing a pattern that does not exist;
* each wave is collapsed to a single straight characteristic at the mean Mach
  angle of the regions it separates, so a fan appears as one line.

So read the cell *spacing* and the pressure *swing* as quantitative, and the
number of cells as "this is the pattern", not as a prediction of how far the jet
stays organised.  Nothing here is a substitute for putting the external region
in the mesh.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import waves
from .plume import ExitWaves, exit_wave_structure

__all__ = ["JetCells", "JetRegion", "jet_wave_cells"]


@dataclass(frozen=True)
class JetRegion:
    """One uniform region of the cell pattern, between two waves."""

    index: int
    mach: float
    pressure_ratio: float
    """``p / p_t``, against the same reservoir the solve used."""
    flow_angle: float
    """Radians, positive away from the axis."""
    nu: float
    """Prandtl-Meyer angle, radians."""
    velocity_ratio: float
    r"""``v / a_t``: speed over the stagnation sound speed.

    The natural non-dimensional speed for this flow, and a function of the Mach
    number alone, :math:`v/a_t = M\,/\sqrt{1 + \tfrac{\gamma-1}{2}M^2}`.  It is
    bounded (unlike :math:`M`) by the vacuum value
    :math:`\sqrt{2/(\gamma-1)}`, which is what makes it the readable thing to
    colour by: a jet expanding hard runs the Mach number up without the speed
    changing much, and the colour should say so.
    """
    on_axis: bool
    """True where this region touches the symmetry axis with ``theta = 0``."""


@dataclass(frozen=True)
class JetCells:
    """The marched pattern: regions, wave segments, and the jet boundary."""

    regions: list[JetRegion]
    wave_segments: np.ndarray
    """``(n_waves, 2, 2)``: each wave as a straight segment ``[[x0,y0],[x1,y1]]``."""
    boundary: np.ndarray
    """``(n_points, 2)``: the jet boundary as a polyline from the lip."""
    cell_length: float
    """Axial length of one shock cell: boundary to axis and back out."""
    exit_half_height: float
    lip_x: float
    regime: str
    exit_waves: ExitWaves
    stopped_because: str = ""
    """Empty if the requested cells were all marched; otherwise why it stopped."""

    def region_polygons(self) -> list[tuple[JetRegion, np.ndarray]]:
        r"""Each closed region paired with the region it is.

        The wave nodes alternate between the jet boundary and the axis, so every
        region after the first is the triangle on three consecutive nodes.  The
        first is the one exception: it is closed on the left by the exit plane
        rather than by a wave, so it is the triangle on the lip, the axis point
        below the lip, and the first axis crossing.

        The final region is left out: it has no downstream wave yet, so it is
        open, and filling it would draw a boundary the march has not computed.
        """
        segs = self.wave_segments
        if len(segs) == 0:
            return []
        nodes = [segs[0][0]] + [seg[1] for seg in segs]
        out = []
        first = np.array([[self.lip_x, self.exit_half_height], [self.lip_x, 0.0], nodes[1]])
        out.append((self.regions[0], first))
        for k in range(1, len(segs)):
            out.append((self.regions[k], np.array([nodes[k - 1], nodes[k], nodes[k + 1]])))
        return out

    @property
    def x_extent(self) -> float:
        """How far downstream of the lip the pattern reaches."""
        if len(self.wave_segments) == 0:
            return 0.0
        return float(self.wave_segments[:, :, 0].max() - self.lip_x)

    def describe(self) -> str:
        n = len(self.wave_segments)
        p = [r.pressure_ratio for r in self.regions]
        out = [
            f"{self.regime} jet: {n} waves, cell length {self.cell_length:.4f} "
            f"({self.cell_length / self.exit_half_height:.2f} exit half-heights), "
            f"reaching x = {self.lip_x + self.x_extent:.4f}",
            f"  Mach swings over {min(r.mach for r in self.regions):.3f} "
            f"to {max(r.mach for r in self.regions):.3f}, "
            f"p/p_t over {min(p):.5f} to {max(p):.5f}",
        ]
        if self.stopped_because:
            out.append(f"  march stopped early: {self.stopped_because}")
        return "\n".join(out)


def _mach_from_pressure(p_over_pt: float, gamma: float) -> float:
    """Invert the isentropic pressure relation for the Mach number."""
    t = p_over_pt ** (-(gamma - 1.0) / gamma)
    return float(np.sqrt(max(2.0 * (t - 1.0) / (gamma - 1.0), 0.0)))


def _pressure_from_mach(mach: float, gamma: float) -> float:
    return float((1.0 + 0.5 * (gamma - 1.0) * mach * mach) ** (-gamma / (gamma - 1.0)))


def _intersect(p0, d0, p1, d1):
    """Intersection of two rays, as ``p0 + t*d0``.  ``None`` if parallel."""
    a = np.array([[d0[0], -d1[0]], [d0[1], -d1[1]]], dtype=float)
    det = a[0, 0] * a[1, 1] - a[0, 1] * a[1, 0]
    if abs(det) < 1e-14:
        return None
    b = np.array([p1[0] - p0[0], p1[1] - p0[1]], dtype=float)
    t = (b[0] * a[1, 1] - b[1] * a[0, 1]) / det
    return np.array([p0[0] + t * d0[0], p0[1] + t * d0[1]])


def jet_wave_cells(
    result,
    ambient_pressure_ratio: float | None = None,
    *,
    cells: int = 3,
    exit_waves: ExitWaves | None = None,
) -> JetCells:
    r"""March the lip wave through ``cells`` periods of reflection.

    Parameters
    ----------
    result
        A converged :class:`~src.solver.SolveResult` with a supersonic exit.
    ambient_pressure_ratio
        ``p_amb / p_t``; defaults to the solve's own back pressure.
    cells
        How many shock cells to march.  One cell is boundary to axis and back
        out, so two waves; the horizontal extent therefore grows roughly
        linearly in ``cells``.  Note the *pressure* pattern repeats only every
        two cells -- the period-4 state cycle above spans two of them.
    exit_waves
        A previously computed :class:`ExitWaves`, to avoid recomputing it.

    Raises
    ------
    ValueError
        Via :func:`~src.external.plume.exit_wave_structure`, if the solve did not
        converge, the exit is subsonic, or the lip turn is impossible.  Also if
        the point is the design point, where there is no wave to march.
    """
    if cells < 1:
        raise ValueError(f"cells must be at least 1, got {cells}")
    ew = exit_wave_structure(result, ambient_pressure_ratio) if exit_waves is None else exit_waves
    if ew.regime == "design":
        raise ValueError(
            "at the design point the lip wave vanishes, so there is no cell "
            "pattern to march; perturb the back pressure to see one"
        )

    gamma = result.flow.gamma
    geom = result.geometry
    lip_x = float(geom.length)
    half_height = float(np.asarray(geom.wall(np.array([lip_x])))[0])

    # the jet's own reservoir pressure, from the measured exit state -- not the
    # solver's p_t, so the march stays consistent with the exit it starts from
    m0 = ew.exit_mach
    pt_jet = ew.exit_pressure_ratio / _pressure_from_mach(m0, gamma)
    m1 = _mach_from_pressure(ew.ambient_pressure_ratio / pt_jet, gamma)
    if m1 <= 1.0:
        raise ValueError(
            f"matching the ambient pressure would take the jet subsonic "
            f"(M = {m1:.3f}); the characteristic march does not apply"
        )

    nu0 = waves.prandtl_meyer(m0, gamma)
    nu1 = waves.prandtl_meyer(m1, gamma)
    theta1 = nu1 - nu0  # > 0 under-expanded (turns outward), < 0 over-expanded

    def region(index, nu, theta, on_axis):
        mach = waves.prandtl_meyer_inverse(nu, gamma)
        return JetRegion(
            index=index,
            mach=mach,
            pressure_ratio=pt_jet * _pressure_from_mach(mach, gamma),
            flow_angle=theta,
            nu=nu,
            velocity_ratio=float(mach / np.sqrt(1.0 + 0.5 * (gamma - 1.0) * mach * mach)),
            on_axis=on_axis,
        )

    # the period-4 cycle of (nu, theta), as the module docstring derives
    cycle = [
        (nu1, theta1, False),
        (nu1 + theta1, 0.0, True),
        (nu1, -theta1, False),
        (nu0, 0.0, True),
    ]

    regions = [region(0, nu0, 0.0, True)]
    segments: list[np.ndarray] = []
    boundary = [np.array([lip_x, half_height])]
    stopped = ""

    # march state: where the next wave starts, and whether that is the boundary
    point = np.array([lip_x, half_height])
    at_boundary = True
    prev = regions[0]

    for step in range(2 * cells):
        nu, theta, on_axis = cycle[step % 4]

        # Both ways the march can fail are the same physical event seen twice:
        # the compression the pattern needs is one the flow cannot deliver while
        # staying supersonic and attached.  That is a Mach reflection -- the Mach
        # disc of a strongly mismatched jet -- and everything below it would be
        # fiction, so stop and say which wave stopped it.
        compressive = nu < prev.nu
        if compressive and nu < 0.0:
            stopped = (
                f"the compression at wave {step + 1} would take the jet subsonic "
                f"(nu = {np.degrees(nu):.2f} deg < 0): this is the Mach disc, "
                "which the model does not carry"
            )
            break
        if on_axis and compressive and abs(theta - prev.flow_angle) > 0.0:
            turn = abs(theta - prev.flow_angle)
            theta_max, _ = waves.max_deflection_angle(prev.mach, gamma)
            if turn > theta_max:
                stopped = (
                    f"the turn at the axis, {np.degrees(turn):.2f} deg, exceeds the "
                    f"maximum attached deflection {np.degrees(theta_max):.2f} deg at "
                    f"M = {prev.mach:.3f}: the reflection is a Mach reflection "
                    "(a Mach disc), which this model does not carry"
                )
                break

        try:
            nxt = region(step + 1, nu, theta, on_axis)
        except ValueError as exc:
            # the only remaining way out is an expansion past the vacuum limit
            stopped = f"the expansion reached the vacuum limit ({exc})"
            break

        mu = 0.5 * (waves.mach_angle(prev.mach) + waves.mach_angle(nxt.mach))
        th = 0.5 * (prev.flow_angle + nxt.flow_angle)

        if at_boundary:
            # heading inward to the axis
            ang = th - mu
            if ang >= -1e-12:
                stopped = "the inbound characteristic no longer reaches the axis"
                break
            end = np.array([point[0] + (0.0 - point[1]) / np.tan(ang), 0.0])
            # the boundary kinks here, following the new region's flow angle;
            # the lip is already boundary[0], so do not record it twice
            if not np.allclose(boundary[-1], point):
                boundary.append(point)
            at_boundary = False
        else:
            # heading outward; find where it meets the boundary, which runs at
            # the flow angle of the region that touches it
            ang = th + mu
            b0 = boundary[-1]
            bdir = np.array([1.0, np.tan(prev.flow_angle)])
            hit = _intersect(point, np.array([1.0, np.tan(ang)]), b0, bdir)
            if hit is None or hit[0] <= point[0] or hit[1] <= 0.0:
                stopped = "the outbound characteristic no longer meets the jet boundary"
                break
            end = hit
            at_boundary = True

        segments.append(np.array([point, end]))
        regions.append(nxt)
        point = end
        prev = nxt

    if at_boundary and len(segments) > 0:
        boundary.append(point)

    # one cell is two waves; measure the first complete one
    if len(segments) >= 2:
        cell_length = float(segments[1][1, 0] - segments[0][0, 0])
    elif len(segments) == 1:
        cell_length = 2.0 * float(segments[0][1, 0] - segments[0][0, 0])
    else:
        cell_length = 0.0

    return JetCells(
        regions=regions,
        wave_segments=np.array(segments) if segments else np.empty((0, 2, 2)),
        boundary=np.array(boundary),
        cell_length=cell_length,
        exit_half_height=half_height,
        lip_x=lip_x,
        regime=ew.regime,
        exit_waves=ew,
        stopped_because=stopped,
    )
