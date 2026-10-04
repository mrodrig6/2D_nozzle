"""External flow just downstream of the exit plane, in closed form.

Separate from the solver on purpose: this is post-processing that reads a
converged exit state, never part of a solve.

:mod:`src.external.plume` gives the first wave at the lip;
:mod:`src.external.jet` extends the domain downstream, marching that wave
through its reflections to the periodic cell pattern.
"""

from __future__ import annotations

from .jet import JetCells, JetRegion, jet_wave_cells
from .plotting import (
    FIELD_RAMPS,
    REGIME_COLOURS,
    latex_rc,
    plot_exit_waves,
    plot_jet_cells,
    plot_jet_field,
)
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
    "JetCells",
    "JetRegion",
    "FIELD_RAMPS",
    "REGIME_COLOURS",
    "deflection_from_wave_angle",
    "exit_wave_structure",
    "jet_wave_cells",
    "latex_rc",
    "mach_angle",
    "max_deflection_angle",
    "oblique_shock_angle",
    "plot_exit_waves",
    "plot_jet_cells",
    "plot_jet_field",
    "prandtl_meyer",
    "prandtl_meyer_inverse",
    "pressure_ratio_across_oblique_shock",
]
