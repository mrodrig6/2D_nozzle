"""The right-hand side: the reference element mapped onto a mesh, and the residual.

``operators`` precomputes everything a residual evaluation needs -- metrics,
mass matrices, normals -- once, into element-batched arrays; ``assembly`` sums
the volume and face terms over them; ``limiter`` keeps the result admissible.
"""

from .assembly import (
    ResidualResult,
    apply_inverse_mass,
    edge_fluxes,
    face_term,
    residual,
    volume_term,
    wave_speed_sum,
)
from .limiter import (
    LimiterDiagnostics,
    apply_limiter,
    cell_means,
    diagnose,
    floor_cell_means,
    positivity_limiter,
    superbee_limiter,
)
from .operators import Operators, ReferenceData, build_operators, reference_data

__all__ = [
    "LimiterDiagnostics",
    "Operators",
    "ReferenceData",
    "ResidualResult",
    "apply_inverse_mass",
    "apply_limiter",
    "build_operators",
    "cell_means",
    "diagnose",
    "edge_fluxes",
    "face_term",
    "floor_cell_means",
    "positivity_limiter",
    "reference_data",
    "residual",
    "superbee_limiter",
    "volume_term",
    "wave_speed_sum",
]
