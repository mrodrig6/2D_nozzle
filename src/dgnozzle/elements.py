"""Reference elements: Lagrange bases on the unit triangle and unit square.

The solver core is element-agnostic.  Everything an element must expose is
collected here so that adding a new element kind requires no changes elsewhere.

Reference geometry
------------------
``tri``   vertices ``(0,0), (1,0), (0,1)``; local edge ``f`` is the edge
          *opposite* vertex ``f``, traversed from vertex ``(f+1) % 3`` to vertex
          ``(f+2) % 3``.  This reproduces the edge numbering of the legacy
          MATLAB code.
``quad``  vertices ``(0,0), (1,0), (1,1), (0,1)``; local edge ``f`` runs from
          vertex ``f`` to vertex ``(f+1) % 4``.

In both cases the vertices are ordered counter-clockwise, so an edge traversed
in its local direction keeps the element interior on its left, and the outward
normal is the tangent rotated by -90 degrees.

Node ordering for a Lagrange basis of order ``p``
-------------------------------------------------
``tri``   ``for iy in 0..p: for ix in 0..p-iy`` at ``(ix/p, iy/p)``
``quad``  ``for iy in 0..p: for ix in 0..p``   at ``(ix/p, iy/p)``

For ``p == 0`` the single node sits at the element centroid and the basis is the
constant 1.
"""

from __future__ import annotations

from functools import lru_cache

import numpy as np

ELEMENT_KINDS = ("tri", "quad")


def n_edges(kind: str) -> int:
    """Number of edges of a reference element."""
    if kind == "tri":
        return 3
    if kind == "quad":
        return 4
    raise ValueError(f"unknown element kind {kind!r}; use one of {ELEMENT_KINDS}")


def n_basis(kind: str, order: int) -> int:
    """Number of Lagrange basis functions of the given order."""
    if order < 0:
        raise ValueError(f"order must be non-negative, got {order}")
    if kind == "tri":
        return (order + 1) * (order + 2) // 2
    if kind == "quad":
        return (order + 1) ** 2
    raise ValueError(f"unknown element kind {kind!r}; use one of {ELEMENT_KINDS}")


@lru_cache(maxsize=32)
def reference_nodes(kind: str, order: int) -> np.ndarray:
    """Lagrange node coordinates, shape ``(n_basis, 2)``."""
    if order == 0:
        centroid = (1.0 / 3.0, 1.0 / 3.0) if kind == "tri" else (0.5, 0.5)
        return np.asarray([centroid], dtype=float)

    h = 1.0 / order
    pts: list[tuple[float, float]] = []
    if kind == "tri":
        for iy in range(order + 1):
            for ix in range(order + 1 - iy):
                pts.append((ix * h, iy * h))
    elif kind == "quad":
        for iy in range(order + 1):
            for ix in range(order + 1):
                pts.append((ix * h, iy * h))
    else:
        raise ValueError(f"unknown element kind {kind!r}; use one of {ELEMENT_KINDS}")
    return np.asarray(pts, dtype=float)


@lru_cache(maxsize=32)
def vertex_indices(kind: str, order: int) -> tuple[int, ...]:
    """Indices, within the order-``order`` node list, of the element vertices.

    Returned counter-clockwise, matching the local edge numbering.
    """
    if order == 0:
        raise ValueError("an order-0 node list has no vertices; geometry order must be >= 1")
    if kind == "tri":
        # (0,0) is first; (1,0) is the last node of the iy = 0 row; (0,1) is last.
        return (0, order, n_basis("tri", order) - 1)
    if kind == "quad":
        q = order
        return (0, q, (q + 1) * q + q, (q + 1) * q)
    raise ValueError(f"unknown element kind {kind!r}; use one of {ELEMENT_KINDS}")


@lru_cache(maxsize=32)
def _monomial_exponents(kind: str, order: int) -> tuple[tuple[int, int], ...]:
    """Monomial exponent pairs ``(r, s)`` spanning the Lagrange space."""
    if kind == "tri":
        return tuple((r, s) for s in range(order + 1) for r in range(order + 1 - s))
    if kind == "quad":
        return tuple((r, s) for s in range(order + 1) for r in range(order + 1))
    raise ValueError(f"unknown element kind {kind!r}; use one of {ELEMENT_KINDS}")


@lru_cache(maxsize=32)
def _basis_coefficients(kind: str, order: int) -> np.ndarray:
    """Coefficient matrix ``C`` with ``phi_j = sum_k C[k, j] * monomial_k``."""
    nodes = reference_nodes(kind, order)
    expo = _monomial_exponents(kind, order)
    nbf = n_basis(kind, order)
    vander = np.empty((nbf, nbf), dtype=float)
    for i, (xi, eta) in enumerate(nodes):
        for k, (r, s) in enumerate(expo):
            vander[i, k] = xi**r * eta**s
    # phi_j(node_i) = delta_ij  =>  vander @ C = I
    return np.linalg.solve(vander, np.eye(nbf))


def shape_functions(
    kind: str, order: int, points: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Evaluate the Lagrange basis and its reference gradients.

    Parameters
    ----------
    kind
        ``'tri'`` or ``'quad'``.
    order
        Polynomial order of the basis.
    points
        ``(npts, 2)`` array of reference coordinates.

    Returns
    -------
    phi, dphi_dxi, dphi_deta
        Each of shape ``(n_basis, npts)``.
    """
    pts = np.atleast_2d(np.asarray(points, dtype=float))
    if pts.shape[1] != 2:
        raise ValueError(f"points must have shape (npts, 2), got {pts.shape}")
    npts = pts.shape[0]
    nbf = n_basis(kind, order)

    if order == 0:
        return (
            np.ones((1, npts)),
            np.zeros((1, npts)),
            np.zeros((1, npts)),
        )

    coeff = _basis_coefficients(kind, order)
    expo = _monomial_exponents(kind, order)
    xi = pts[:, 0]
    eta = pts[:, 1]

    phi = np.zeros((nbf, npts))
    dxi = np.zeros((nbf, npts))
    deta = np.zeros((nbf, npts))
    for k, (r, s) in enumerate(expo):
        ck = coeff[k, :]  # (nbf,) contribution of monomial k to each basis fn
        mono = xi**r * eta**s
        phi += ck[:, None] * mono[None, :]
        if r > 0:
            dxi += (ck * r)[:, None] * (xi ** (r - 1) * eta**s)[None, :]
        if s > 0:
            deta += (ck * s)[:, None] * (xi**r * eta ** (s - 1))[None, :]
    return phi, dxi, deta


def edge_reference_coords(kind: str, edge: int, sigma: np.ndarray) -> np.ndarray:
    """Map edge parameter ``sigma`` in ``[0, 1]`` to reference coordinates.

    ``sigma = 0`` sits at the edge's start vertex and ``sigma = 1`` at its end
    vertex, following the counter-clockwise local edge numbering.

    Returns an ``(len(sigma), 2)`` array.
    """
    s = np.asarray(sigma, dtype=float).ravel()
    zero = np.zeros_like(s)
    one = np.ones_like(s)
    if kind == "tri":
        table = {
            0: (one - s, s),  # (1,0) -> (0,1)
            1: (zero, one - s),  # (0,1) -> (0,0)
            2: (s, zero),  # (0,0) -> (1,0)
        }
    elif kind == "quad":
        table = {
            0: (s, zero),  # (0,0) -> (1,0)
            1: (one, s),  # (1,0) -> (1,1)
            2: (one - s, one),  # (1,1) -> (0,1)
            3: (zero, one - s),  # (0,1) -> (0,0)
        }
    else:
        raise ValueError(f"unknown element kind {kind!r}; use one of {ELEMENT_KINDS}")
    if edge not in table:
        raise ValueError(f"{kind} element has no local edge {edge}")
    xi, eta = table[edge]
    return np.column_stack([xi, eta])


def edge_dreference_dsigma(kind: str, edge: int) -> tuple[float, float]:
    """Constant ``(d xi / d sigma, d eta / d sigma)`` along a local edge."""
    if kind == "tri":
        table = {0: (-1.0, 1.0), 1: (0.0, -1.0), 2: (1.0, 0.0)}
    elif kind == "quad":
        table = {0: (1.0, 0.0), 1: (0.0, 1.0), 2: (-1.0, 0.0), 3: (0.0, -1.0)}
    else:
        raise ValueError(f"unknown element kind {kind!r}; use one of {ELEMENT_KINDS}")
    if edge not in table:
        raise ValueError(f"{kind} element has no local edge {edge}")
    return table[edge]


def edge_vertex_local_indices(kind: str, edge: int) -> tuple[int, int]:
    """Local *vertex* indices (0-based, CCW) bounding a local edge."""
    if kind == "tri":
        return ((edge + 1) % 3, (edge + 2) % 3)
    if kind == "quad":
        return (edge, (edge + 1) % 4)
    raise ValueError(f"unknown element kind {kind!r}; use one of {ELEMENT_KINDS}")


def quadrature_degree(kind: str, solution_order: int, geometry_order: int) -> tuple[int, int]:
    """Volume and edge quadrature degrees needed for exact mass/flux integration.

    The volume integrand ``grad(phi) . F`` is degree ``2p`` in the solution for
    an affine element; a curved element of geometry order ``Q`` adds
    ``2 (Q - 1)`` from the metric terms, and the mass matrix needs ``2p``.  One
    extra degree is carried for safety, matching the legacy ``2p + 1 + 2(Q-1)``.
    """
    p = solution_order
    q = geometry_order
    vol = 2 * p + 1 + 2 * (q - 1)
    edge = 2 * p + q
    if kind == "quad":
        # Tensor-product bases are degree p in each variable, so the product of
        # two of them reaches total degree 2p in a single direction.
        vol = max(vol, 2 * p + 1)
        edge = max(edge, 2 * p + 1)
    return vol, edge
