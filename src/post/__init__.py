"""Reading the answer: performance integrals, derived fields, and figures.

``plotting`` is not re-exported, because matplotlib is an optional dependency
and importing this package should not require it.  ``src.post.plotting`` is a lazy
attribute on the top-level package for the same reason.
"""

from .postprocess import (
    SCALARS,
    BoundaryTrace,
    Performance,
    boundary_trace,
    centreline_profile,
    entropy_error,
    exit_profile,
    inlet_profile,
    performance,
    sample_boundary,
    sample_field,
    scalar_field,
    solution_at_quadrature,
    volume_integral,
    wall_profile,
)

__all__ = [
    "SCALARS",
    "BoundaryTrace",
    "Performance",
    "boundary_trace",
    "centreline_profile",
    "entropy_error",
    "exit_profile",
    "inlet_profile",
    "performance",
    "sample_boundary",
    "sample_field",
    "scalar_field",
    "solution_at_quadrature",
    "volume_integral",
    "wall_profile",
]
