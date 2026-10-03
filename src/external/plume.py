r"""The wave structure just outside the exit plane, from the converged exit state.

A nozzle-design exercise avoids a shock *inside* the diverging section, so the
DG domain stays shock free and converges.  The interesting waves then live
immediately downstream of the lip, outside the computed domain:

==================  ===========================  ==========================
regime              condition                    external wave
==================  ===========================  ==========================
over-expanded       :math:`p_e < p_{\rm amb}`     oblique shock turning in
design              :math:`p_e = p_{\rm amb}`     none
under-expanded      :math:`p_e > p_{\rm amb}`     Prandtl-Meyer fan turning out
==================  ===========================  ==========================

This module evaluates that structure in closed form from the exit Mach number
and pressure the solver already computes.  It is **not** a plume solver: it gives
the first wave leaving the lip, not the barrel shock or the Mach disc, both of
which need the external domain in the mesh and shock capturing that converges in
steady state.

It is deliberately explicit -- nothing in the solver calls it.  Post-processing
that can legitimately have no answer (a required turn past maximum deflection)
should fail where the user asked for it, not inside a flow solve.

.. note::
   **The quasi-1D design back pressure is not this nozzle's design point.**  Run
   at the quasi-1D value :math:`p_b/p_t = 0.0640` for ``area_ratio = 2.5`` and the
   computed exit pressure comes out at :math:`0.0669` -- a 4.5% mismatch, so the
   jet is slightly *under-expanded* there and this module says so.

   That offset is **not** discretisation error: it is 4.57%, 4.54%, 4.56% and
   4.54% at ``(p, refine)`` of ``(1,0)``, ``(1,1)``, ``(2,0)`` and ``(2,1)``,
   so it does not shrink with resolution.  It is the two-dimensionality of the
   flow.  Quasi-1D theory assumes parallel streamlines at the exit plane; the
   real exit flow is still diverging, so its area-averaged static pressure
   differs from the one-dimensional prediction.  The nozzle's own design point
   -- where the external wave actually vanishes -- is near
   :math:`p_b/p_t = 0.0669`, and finding it is a worthwhile exercise in itself.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..postprocess import performance
from . import waves

__all__ = ["ExitWaves", "exit_wave_structure"]


@dataclass(frozen=True)
class ExitWaves:
    """The first wave leaving the nozzle lip."""

    regime: str
    """``'over-expanded'``, ``'design'`` or ``'under-expanded'``."""
    exit_mach: float
    exit_pressure_ratio: float
    """``p_e / p_t``, the exit static pressure over the reservoir pressure."""
    ambient_pressure_ratio: float
    """``p_amb / p_t``."""
    pressure_mismatch: float
    """``p_e / p_amb``.  Below 1 over-expanded, above 1 under-expanded."""
    turn_angle: float
    """Flow deflection through the wave, radians.  Positive turns toward the axis."""
    wave_angle: float | None = None
    """Oblique-shock angle beta from the flow direction, radians.  ``None`` if no shock."""
    fan_angles: tuple[float, float] | None = None
    """Leading and trailing characteristic angles of the expansion fan, radians."""
    downstream_mach: float | None = None
    """Mach number after the wave.  ``None`` for the design point."""

    def describe(self) -> str:
        d = np.degrees
        if self.regime == "design":
            return (
                f"design: p_e/p_amb = {self.pressure_mismatch:.4f}, "
                f"M_e = {self.exit_mach:.3f}, no external wave"
            )
        if self.regime == "over-expanded":
            return (
                f"over-expanded: p_e/p_amb = {self.pressure_mismatch:.4f}, "
                f"M_e = {self.exit_mach:.3f} -> {self.downstream_mach:.3f}, "
                f"oblique shock at beta = {d(self.wave_angle):.2f} deg, "
                f"turning the flow {d(self.turn_angle):.2f} deg inward"
            )
        lead, trail = self.fan_angles
        return (
            f"under-expanded: p_e/p_amb = {self.pressure_mismatch:.4f}, "
            f"M_e = {self.exit_mach:.3f} -> {self.downstream_mach:.3f}, "
            f"Prandtl-Meyer fan from {d(lead):.2f} to {d(trail):.2f} deg, "
            f"turning the flow {d(self.turn_angle):.2f} deg outward"
        )


def exit_wave_structure(
    result, ambient_pressure_ratio: float | None = None, *, tol: float = 1e-3
) -> ExitWaves:
    r"""The external wave implied by a converged solution's exit state.

    Parameters
    ----------
    result
        A converged :class:`~src.solver.SolveResult`.
    ambient_pressure_ratio
        ``p_amb / p_t``.  Defaults to the solve's own ``back_pressure_ratio``,
        which is the ambient the operating point was chosen against.
    tol
        Relative pressure mismatch inside which the point counts as design.
        ``1e-3`` rather than machine zero because the regime boundary is a
        measured pressure, not an exact one.

    Raises
    ------
    ValueError
        If the solve did not converge, if the exit is subsonic (there is no
        external wave system to speak of), or if an over-expanded turn exceeds
        the maximum attached-shock deflection.
    """
    if not result.converged:
        raise ValueError(
            "the external wave structure is read off a converged exit state; "
            f"this solve reports converged=False ({result.message})"
        )
    perf = performance(result)
    gamma = result.flow.gamma
    m_e = float(perf.exit_mach_area_averaged)
    pe = float(perf.exit_pressure_ratio)
    amb = (
        float(result.flow.back_pressure_ratio)
        if ambient_pressure_ratio is None
        else float(ambient_pressure_ratio)
    )
    if m_e <= 1.0:
        raise ValueError(
            f"the exit is subsonic (M_e = {m_e:.3f}); the external relations here "
            "assume a supersonic jet leaving the lip"
        )

    mismatch = pe / amb
    if abs(mismatch - 1.0) <= tol:
        return ExitWaves("design", m_e, pe, amb, mismatch, 0.0)

    if mismatch < 1.0:
        # over-expanded: the jet must be compressed, so an oblique shock turns it in
        ratio = amb / pe
        theta_max, _ = waves.max_deflection_angle(m_e, gamma)
        beta = waves.oblique_shock_angle(m_e, ratio, gamma)
        theta = waves.deflection_from_wave_angle(m_e, beta, gamma)
        if theta > theta_max:
            raise ValueError(
                f"the required turn {np.degrees(theta):.2f} deg exceeds the maximum "
                f"attached deflection {np.degrees(theta_max):.2f} deg at M={m_e:.3f}; "
                "the shock detaches and this relation does not apply"
            )
        mn1 = m_e * np.sin(beta)
        mn2 = np.sqrt(
            (1.0 + 0.5 * (gamma - 1.0) * mn1**2)
            / (gamma * mn1**2 - 0.5 * (gamma - 1.0))
        )
        m_down = float(mn2 / np.sin(beta - theta))
        return ExitWaves(
            "over-expanded", m_e, pe, amb, mismatch, theta,
            wave_angle=beta, downstream_mach=m_down,
        )

    # under-expanded: the jet expands to ambient through a Prandtl-Meyer fan
    m_down = float(
        np.sqrt(
            (
                (1.0 + 0.5 * (gamma - 1.0) * m_e**2)
                * (amb / pe) ** (-(gamma - 1.0) / gamma)
                - 1.0
            )
            / (0.5 * (gamma - 1.0))
        )
    )
    theta = waves.prandtl_meyer(m_down, gamma) - waves.prandtl_meyer(m_e, gamma)
    lead = waves.mach_angle(m_e)
    trail = waves.mach_angle(m_down) - theta
    return ExitWaves(
        "under-expanded", m_e, pe, amb, mismatch, theta,
        fan_angles=(lead, trail), downstream_mach=m_down,
    )
