"""The reference element: what the approximation space is, before any mesh.

This is the layer a discontinuous Galerkin code has and a finite-volume code
does not.  Everything here is defined on the unit triangle or the unit square
and knows nothing about the nozzle, the mesh or the flow: ``elements`` gives
the Lagrange bases and where their nodes sit, ``quadrature`` gives the rules
that integrate them exactly.

Both are pure tables.  :mod:`src.spatial.operators` is what maps them onto a
real mesh.
"""

from .elements import (
    ELEMENT_KINDS,
    edge_dreference_dsigma,
    edge_reference_coords,
    edge_vertex_local_indices,
    n_basis,
    n_edges,
    quadrature_degree,
    reference_nodes,
    shape_functions,
    vertex_indices,
)
from .quadrature import (
    QuadratureRule,
    collapsed_gauss_jacobi_triangle,
    gauss_legendre_1d,
    gauss_legendre_square,
    triangle_rule,
    volume_rule,
)

__all__ = [
    "ELEMENT_KINDS",
    "QuadratureRule",
    "collapsed_gauss_jacobi_triangle",
    "edge_dreference_dsigma",
    "edge_reference_coords",
    "edge_vertex_local_indices",
    "gauss_legendre_1d",
    "gauss_legendre_square",
    "n_basis",
    "n_edges",
    "quadrature_degree",
    "reference_nodes",
    "shape_functions",
    "triangle_rule",
    "vertex_indices",
    "volume_rule",
]
