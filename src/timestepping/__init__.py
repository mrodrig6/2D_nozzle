"""Marching the residual to steady state.

``step`` sizes the time step and measures convergence; ``initialize`` says
where the march starts; ``march`` runs the loop.  The stage combinations
themselves live in :mod:`src.backends.base`, so that all three backends share
one definition of each scheme.
"""

from .initialize import (
    change_order,
    initial_state,
    project,
    quasi1d_initial,
    uniform_initial,
    uniform_state,
)
from .march import SolveHistory, SolveResult, march, solve_steady
from .step import (
    CFL_MARGIN,
    local_time_step,
    recommended_cfl,
    residual_scale,
    stability_limit,
    step_coefficient,
)

__all__ = [
    "CFL_MARGIN",
    "SolveHistory",
    "SolveResult",
    "change_order",
    "initial_state",
    "local_time_step",
    "march",
    "project",
    "quasi1d_initial",
    "recommended_cfl",
    "residual_scale",
    "solve_steady",
    "stability_limit",
    "step_coefficient",
    "uniform_initial",
    "uniform_state",
]
