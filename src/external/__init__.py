"""External flow just downstream of the exit plane, in closed form.

Separate from the solver on purpose: this is post-processing that reads a
converged exit state, never part of a solve.  See :mod:`src.external.plume`.
"""

from __future__ import annotations

from .plotting import REGIME_COLOURS, plot_exit_waves
from .plume import ExitWaves, exit_wave_structure
from .waves import (
    deflection_from_wave_angle,
    mach_angle,
    max_deflection_angle,
    oblique_shock_angle,
    prandtl_meyer,
    prandtl_meyer_inverse,
    pressure_ratio_across_oblique_shock,
)

__all__ = [
    "ExitWaves",
    "REGIME_COLOURS",
    "deflection_from_wave_angle",
    "exit_wave_structure",
    "mach_angle",
    "max_deflection_angle",
    "oblique_shock_angle",
    "plot_exit_waves",
    "prandtl_meyer",
    "prandtl_meyer_inverse",
    "pressure_ratio_across_oblique_shock",
]
