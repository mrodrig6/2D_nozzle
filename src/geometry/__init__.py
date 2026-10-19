"""What shape, and what mesh inside it.

``contour`` describes the nozzle wall; ``mesh`` fills the region between the
walls with elements.  Nothing here knows about the flow.
"""

from .contour import (
    ANALYTIC_AREA_RATIO,
    ANALYTIC_EXIT_Y,
    ANALYTIC_INLET_Y,
    ANALYTIC_THROAT_X,
    ANALYTIC_THROAT_Y,
    CONTOURS,
    DESIGN_PARAMETERS,
    NozzleGeometry,
    analytic_contour,
    check_contour,
    wall_half_height,
)
from .mesh import (
    BoundaryTag,
    EdgeConnectivity,
    MeshTopology,
    build_edges,
    build_nozzle_mesh,
    nozzle_boundary_tagger,
    nozzle_node_coords,
    nozzle_x_distribution,
    structured_nozzle_topology,
)

__all__ = [
    "ANALYTIC_AREA_RATIO",
    "ANALYTIC_EXIT_Y",
    "ANALYTIC_INLET_Y",
    "ANALYTIC_THROAT_X",
    "ANALYTIC_THROAT_Y",
    "BoundaryTag",
    "CONTOURS",
    "DESIGN_PARAMETERS",
    "EdgeConnectivity",
    "MeshTopology",
    "NozzleGeometry",
    "analytic_contour",
    "build_edges",
    "build_nozzle_mesh",
    "check_contour",
    "nozzle_boundary_tagger",
    "nozzle_node_coords",
    "nozzle_x_distribution",
    "structured_nozzle_topology",
    "wall_half_height",
]
