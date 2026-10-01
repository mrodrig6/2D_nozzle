"""Mesh topology and node coordinates.

The design separates two things that are usually tangled together:

``MeshTopology``
    Pure integer connectivity -- element-to-node, edge lists, boundary tags,
    the assembly gather map.  It depends only on the *counts* (``nx``, ``nr``,
    refinement, element kind, geometry order), never on the design variables.
``node_coords``
    A float array of shape ``(n_nodes, 2)``.  This is where the design
    variables enter, so it is the only part that has to be differentiable.

Keeping them apart is what makes ``jax.grad`` with respect to the nozzle shape
practical: the topology is built once with NumPy and treated as static, while
the coordinates (and everything derived from them) flow through JAX.

``build_edges`` is fully generic -- hand it any element-to-node array for a
supported element kind and it returns the connectivity.  Nothing in this module
below :func:`structured_nozzle_mesh` knows what a nozzle is.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from enum import IntEnum

import numpy as np

from . import elements as el
from .geometry import NozzleGeometry, wall_half_height


class BoundaryTag(IntEnum):
    """Boundary-condition tags.  The solver dispatches on these."""

    INFLOW = 1
    OUTFLOW = 2
    WALL = 3
    AXIS = 4


TAG_NAMES: dict[BoundaryTag, str] = {
    BoundaryTag.INFLOW: "inflow",
    BoundaryTag.OUTFLOW: "outflow",
    BoundaryTag.WALL: "wall",
    BoundaryTag.AXIS: "axis",
}


# --------------------------------------------------------------------------
# Generic edge connectivity
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class EdgeConnectivity:
    """Interior/boundary edge lists plus the element-face gather map."""

    iedge_elem: np.ndarray  # (niedge, 2) left, right element
    iedge_face: np.ndarray  # (niedge, 2) left, right local face
    bedge_elem: np.ndarray  # (nbedge,)
    bedge_face: np.ndarray  # (nbedge,)
    bedge_tag: np.ndarray  # (nbedge,)
    face_edge: np.ndarray  # (nelem, nface) global edge index
    face_side: np.ndarray  # (nelem, nface) 0 = left/owner, 1 = right
    face_neighbour: np.ndarray  # (nelem, nface) element across the face; self on a boundary
    tag_slices: dict[int, slice] = field(default_factory=dict)

    @property
    def n_interior(self) -> int:
        return int(self.iedge_elem.shape[0])

    @property
    def n_boundary(self) -> int:
        return int(self.bedge_elem.shape[0])


def build_edges(
    kind: str,
    elem_nodes: np.ndarray,
    geometry_order: int,
    tagger: Callable[[np.ndarray], np.ndarray],
    node_coords: np.ndarray,
) -> EdgeConnectivity:
    """Build edge connectivity for an arbitrary conforming mesh of one element kind.

    Faces are matched on their two corner vertices, so meshes of any geometry
    order work.  Of the two elements sharing an interior face, the one with the
    lower index is taken as the *left* element; the outward normal is defined
    from its local face orientation.

    Parameters
    ----------
    kind
        ``'tri'`` or ``'quad'``.
    elem_nodes
        ``(nelem, n_nodes_per_elem)`` element-to-node array in the reference
        node ordering of :mod:`dgnozzle.elements`.
    geometry_order
        ``Q``; used to locate the corner vertices within each element's nodes.
    tagger
        Callable mapping an ``(nbedge, 2)`` array of boundary-face midpoints to
        an ``(nbedge,)`` array of :class:`BoundaryTag` values.
    node_coords
        ``(n_nodes, 2)`` coordinates, used only to evaluate ``tagger``.
    """
    nface = el.n_edges(kind)
    vidx = np.asarray(el.vertex_indices(kind, geometry_order))
    corners = elem_nodes[:, vidx]  # (nelem, nface) corner node ids, CCW
    nelem = corners.shape[0]

    # face f of element e runs from corner a to corner b
    face_a = np.empty((nelem, nface), dtype=np.int64)
    face_b = np.empty((nelem, nface), dtype=np.int64)
    for f in range(nface):
        ia, ib = el.edge_vertex_local_indices(kind, f)
        face_a[:, f] = corners[:, ia]
        face_b[:, f] = corners[:, ib]

    flat_a = face_a.ravel()
    flat_b = face_b.ravel()
    flat_elem = np.repeat(np.arange(nelem, dtype=np.int64), nface)
    flat_face = np.tile(np.arange(nface, dtype=np.int64), nelem)

    key_lo = np.minimum(flat_a, flat_b)
    key_hi = np.maximum(flat_a, flat_b)
    order = np.lexsort((key_hi, key_lo))
    s_lo, s_hi = key_lo[order], key_hi[order]

    # A face is shared when it is identical to its successor in sorted order.
    same = np.empty(s_lo.shape, dtype=bool)
    same[:-1] = (s_lo[:-1] == s_lo[1:]) & (s_hi[:-1] == s_hi[1:])
    same[-1] = False
    first_of_pair = np.flatnonzero(same)
    if np.any(same[:-1] & same[1:]):
        raise ValueError("non-conforming mesh: a face is shared by more than two elements")

    paired = np.zeros(s_lo.shape, dtype=bool)
    paired[first_of_pair] = True
    paired[first_of_pair + 1] = True
    lone = np.flatnonzero(~paired)

    # -- interior edges; lower element index becomes the left element
    a_slot = order[first_of_pair]
    b_slot = order[first_of_pair + 1]
    ea, eb = flat_elem[a_slot], flat_elem[b_slot]
    left_is_a = ea <= eb
    left_slot = np.where(left_is_a, a_slot, b_slot)
    right_slot = np.where(left_is_a, b_slot, a_slot)

    iedge_elem = np.column_stack([flat_elem[left_slot], flat_elem[right_slot]])
    iedge_face = np.column_stack([flat_face[left_slot], flat_face[right_slot]])

    # -- boundary edges
    b_slot_all = order[lone]
    bedge_elem = flat_elem[b_slot_all]
    bedge_face = flat_face[b_slot_all]

    mid = 0.5 * (node_coords[flat_a[b_slot_all]] + node_coords[flat_b[b_slot_all]])
    bedge_tag = np.asarray(tagger(np.asarray(mid, dtype=float)), dtype=np.int64)
    if bedge_tag.shape != bedge_elem.shape:
        raise ValueError("tagger must return one tag per boundary face")

    # -- sort boundary edges by tag so each BC acts on a contiguous slice
    b_order = np.argsort(bedge_tag, kind="stable")
    bedge_elem = bedge_elem[b_order]
    bedge_face = bedge_face[b_order]
    bedge_tag = bedge_tag[b_order]
    b_slot_all = b_slot_all[b_order]

    tag_slices: dict[int, slice] = {}
    if bedge_tag.size:
        uniq, starts = np.unique(bedge_tag, return_index=True)
        stops = np.append(starts[1:], bedge_tag.size)
        tag_slices = {
            int(t): slice(int(a), int(b))
            for t, a, b in zip(uniq, starts, stops, strict=True)
        }

    # -- gather map: every (element, local face) points at one global edge
    niedge = iedge_elem.shape[0]
    face_edge = np.full((nelem, nface), -1, dtype=np.int64)
    face_side = np.zeros((nelem, nface), dtype=np.int64)
    face_edge[flat_elem[left_slot], flat_face[left_slot]] = np.arange(niedge)
    face_side[flat_elem[left_slot], flat_face[left_slot]] = 0
    face_edge[flat_elem[right_slot], flat_face[right_slot]] = np.arange(niedge)
    face_side[flat_elem[right_slot], flat_face[right_slot]] = 1
    face_edge[bedge_elem, bedge_face] = niedge + np.arange(bedge_elem.size)
    face_side[bedge_elem, bedge_face] = 0

    if np.any(face_edge < 0):
        raise ValueError("internal error: some element face was left unassigned")

    # Neighbour across each face; a boundary face points back at its own element
    # so that a min/max over neighbours degenerates harmlessly there.
    face_neighbour = np.repeat(np.arange(nelem, dtype=np.int64)[:, None], nface, axis=1)
    face_neighbour[iedge_elem[:, 0], iedge_face[:, 0]] = iedge_elem[:, 1]
    face_neighbour[iedge_elem[:, 1], iedge_face[:, 1]] = iedge_elem[:, 0]

    return EdgeConnectivity(
        iedge_elem=iedge_elem,
        iedge_face=iedge_face,
        bedge_elem=bedge_elem,
        bedge_face=bedge_face,
        bedge_tag=bedge_tag,
        face_edge=face_edge,
        face_side=face_side,
        face_neighbour=face_neighbour,
        tag_slices=tag_slices,
    )


# --------------------------------------------------------------------------
# Mesh container
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class MeshTopology:
    """Integer connectivity of a single-element-kind mesh."""

    kind: str
    geometry_order: int
    elem_nodes: np.ndarray
    n_nodes: int
    edges: EdgeConnectivity
    logical: dict[str, int] = field(default_factory=dict)

    @property
    def n_elem(self) -> int:
        return int(self.elem_nodes.shape[0])

    @property
    def n_faces(self) -> int:
        return el.n_edges(self.kind)

    def tag_slice(self, tag: BoundaryTag) -> slice:
        """Slice of the boundary-edge arrays carrying ``tag`` (may be empty)."""
        return self.edges.tag_slices.get(int(tag), slice(0, 0))

    def summary(self) -> str:
        counts = {
            TAG_NAMES[BoundaryTag(t)]: s.stop - s.start
            for t, s in sorted(self.edges.tag_slices.items())
        }
        return (
            f"{self.n_elem} {self.kind} elements (Q={self.geometry_order}), "
            f"{self.n_nodes} nodes, {self.edges.n_interior} interior edges, "
            f"boundary: {counts}"
        )


# --------------------------------------------------------------------------
# Node distributions
# --------------------------------------------------------------------------
def _inlet_clustered_distribution(n_intervals: int) -> np.ndarray:
    """Inlet-clustered points on ``[0, 1]``, geometrically graded.

    Resolves a throat that sits near the inlet, but puts the fine spacing in the
    wrong place for a throat further downstream -- see ``'throat'``.
    """
    a, b = 0.7, 1.2
    v = np.logspace(a, b, n_intervals + 1)
    return (v - 10.0**a) / (10.0**b - 10.0**a)


def _clustered_x_distribution(
    n_intervals: int, x_throat: float, strength: float, width: float
) -> np.ndarray:
    """Points on ``[0, 1]`` clustered around ``x_throat``.

    A target spacing ``h(x) = 1 + (strength - 1) (1 - exp(-((x - x_t)/w)^2))`` is
    integrated and inverted, so the spacing is ``1`` at the throat and grows to
    ``strength`` far away.  The node count is exact by construction.
    """
    if strength < 1.0:
        raise ValueError(f"cluster strength must be >= 1, got {strength}")
    fine = np.linspace(0.0, 1.0, 20001)
    h = 1.0 + (strength - 1.0) * (1.0 - np.exp(-(((fine - x_throat) / width) ** 2)))
    density = 1.0 / h
    cdf = np.concatenate([[0.0], np.cumsum(0.5 * (density[1:] + density[:-1]) * np.diff(fine))])
    cdf /= cdf[-1]
    targets = np.linspace(0.0, 1.0, n_intervals + 1)
    x = np.interp(targets, cdf, fine)
    x[0], x[-1] = 0.0, 1.0
    return x


def _subdivide(base: np.ndarray, n_sub: int) -> np.ndarray:
    """Linearly split each interval of ``base`` into ``n_sub`` pieces.

    Because refinement only subdivides, the node set at one refinement level is
    a subset of the next, so the coarse nodes survive exactly into the fine
    mesh.
    """
    if n_sub == 1:
        return base.copy()
    n_base = base.size - 1
    out = np.empty(n_base * n_sub + 1)
    frac = np.arange(n_sub) / n_sub
    for i in range(n_base):
        out[i * n_sub : (i + 1) * n_sub] = base[i] * (1.0 - frac) + base[i + 1] * frac
    out[-1] = base[-1]
    return out


# --------------------------------------------------------------------------
# Structured nozzle mesh
# --------------------------------------------------------------------------
def _cell_element_nodes(kind: str, geometry_order: int) -> list[np.ndarray]:
    """Node offsets, in cell-local ``(i, j)`` index units, for each element in a cell.

    Every element's node list is generated by mapping the reference Lagrange
    nodes through the affine (triangle) or bilinear (quad) map onto the cell's
    integer sub-grid, so the ordering automatically matches
    :func:`dgnozzle.elements.reference_nodes` for any ``Q``.
    """
    q = geometry_order
    ref = el.reference_nodes(kind, q)
    out: list[np.ndarray] = []

    if kind == "tri":
        # lower-left triangle and upper-right triangle of the cell
        corner_sets = [
            ((0, 0), (q, 0), (0, q)),
            ((q, 0), (q, q), (0, q)),
        ]
        for a, b, c in corner_sets:
            a_v, b_v, c_v = (np.asarray(v, float) for v in (a, b, c))
            ij = a_v + ref[:, :1] * (b_v - a_v) + ref[:, 1:2] * (c_v - a_v)
            out.append(np.rint(ij).astype(np.int64))
    elif kind == "quad":
        ij = ref * q
        out.append(np.rint(ij).astype(np.int64))
    else:
        raise ValueError(f"unknown element kind {kind!r}")

    for arr in out:
        if not np.all(np.abs(arr - (ref * 0 + arr)) < 1e-9):  # pragma: no cover
            raise AssertionError("non-integral node offsets")
    return out


def nozzle_boundary_tagger(length: float, tol: float) -> Callable[[np.ndarray], np.ndarray]:
    """Tag boundary faces of the nozzle channel from their midpoints.

    ``x = 0`` is the inflow plane, ``x = length`` the outflow plane, ``y = 0``
    the symmetry axis and everything else the contoured wall.  The axis and the
    wall get different tags even though they share the same inviscid flux, so
    that thrust can be integrated over the wall alone.
    """

    def tagger(midpoints: np.ndarray) -> np.ndarray:
        x = midpoints[:, 0]
        y = midpoints[:, 1]
        tags = np.full(x.shape, int(BoundaryTag.WALL), dtype=np.int64)
        tags[np.abs(y) < tol] = int(BoundaryTag.AXIS)
        tags[x < tol] = int(BoundaryTag.INFLOW)
        tags[x > length - tol] = int(BoundaryTag.OUTFLOW)
        return tags

    return tagger


def structured_nozzle_topology(
    kind: str = "tri",
    *,
    nx: int = 14,
    nr: int = 5,
    refine: int = 0,
    geometry_order: int = 1,
) -> tuple[MeshTopology, np.ndarray, np.ndarray]:
    """Build the topology and the logical node grid of the nozzle channel.

    Returns
    -------
    topology
        Connectivity, with boundary faces tagged.  Independent of the design
        variables.
    x_frac
        ``(nx_nodes,)`` normalised axial node positions in ``[0, 1]``.
    r_frac
        ``(nr_nodes,)`` normalised transverse node positions in ``[0, 1]``.

    The node coordinates themselves come from :func:`nozzle_node_coords`, which
    is where the design variables enter.
    """
    if kind not in el.ELEMENT_KINDS:
        raise ValueError(f"unknown element kind {kind!r}; use one of {el.ELEMENT_KINDS}")
    if refine < 0:
        raise ValueError(f"refine must be non-negative, got {refine}")
    if geometry_order < 1:
        raise ValueError(f"geometry_order (Q) must be at least 1, got {geometry_order}")
    if nx < 1 or nr < 1:
        raise ValueError("nx and nr must be positive")

    nref = 2**refine
    ncx, ncy = nx * nref, nr * nref  # cells
    q = geometry_order
    n_ix, n_iy = q * ncx + 1, q * ncy + 1  # node-grid size

    def node_id(i: np.ndarray | int, j: np.ndarray | int) -> np.ndarray:
        return np.asarray(j) * n_ix + np.asarray(i)

    offsets = _cell_element_nodes(kind, q)
    ci, cj = np.meshgrid(np.arange(ncx), np.arange(ncy), indexing="ij")
    base_i = (ci * q).ravel()
    base_j = (cj * q).ravel()

    elem_nodes = np.concatenate(
        [node_id(base_i[:, None] + off[None, :, 0], base_j[:, None] + off[None, :, 1])
         for off in offsets],
        axis=0,
    )

    # Provisional straight-wall coordinates, used only to tag the boundary.
    x_frac = np.linspace(0.0, 1.0, n_ix)
    r_frac = np.linspace(0.0, 1.0, n_iy)
    gx, gy = np.meshgrid(x_frac, r_frac, indexing="ij")
    probe = np.column_stack([gx.T.ravel(), gy.T.ravel()])

    tol = 0.25 / max(n_ix, n_iy)
    conn = build_edges(kind, elem_nodes, q, nozzle_boundary_tagger(1.0, tol), probe)

    topo = MeshTopology(
        kind=kind,
        geometry_order=q,
        elem_nodes=elem_nodes,
        n_nodes=n_ix * n_iy,
        edges=conn,
        logical={"nx_cells": ncx, "nr_cells": ncy, "nx_nodes": n_ix, "nr_nodes": n_iy,
                 "refine": refine, "nx_base": nx, "nr_base": nr},
    )
    return topo, x_frac, r_frac


def nozzle_x_distribution(
    n_nodes_x: int,
    *,
    nx_base: int,
    n_sub: int,
    spacing: str,
    x_throat: float,
    cluster_strength: float,
    cluster_width: float,
) -> np.ndarray:
    """Normalised axial node positions matching the topology's node count."""
    if spacing == "uniform":
        base = np.linspace(0.0, 1.0, nx_base + 1)
    elif spacing == "inlet":
        base = _inlet_clustered_distribution(nx_base)
    elif spacing == "throat":
        base = _clustered_x_distribution(nx_base, x_throat, cluster_strength, cluster_width)
    else:
        raise ValueError(
            f"unknown x_spacing {spacing!r}; use 'throat', 'inlet' or 'uniform'"
        )
    x = _subdivide(base, n_sub)
    if x.size != n_nodes_x:  # pragma: no cover - guarded by construction
        raise AssertionError(f"expected {n_nodes_x} axial nodes, built {x.size}")
    return x


def nozzle_node_coords(
    x_frac: np.ndarray,
    r_frac: np.ndarray,
    params: Mapping[str, object],
    contour: str,
    xp=np,
):
    """Node coordinates of the structured nozzle mesh.

    ``x_frac`` and ``r_frac`` are the static logical grids; ``params`` carries
    the design variables.  Written through ``xp`` so the result is
    differentiable when ``xp`` is ``jax.numpy``.

    The transverse coordinate is ``y = r * y_wall(x)``, so every node moves when
    the contour moves -- including the interior ones.  That is what lets a shape
    derivative propagate into the metric terms.
    """
    length = params.get("length", 1.0)
    x = xp.asarray(x_frac) * length
    y_wall = wall_half_height(x, params, contour, xp=xp)
    # node ordering is j-major to match node_id(i, j) = j * n_ix + i
    xx = xp.broadcast_to(x[None, :], (len(r_frac), len(x_frac)))
    yy = xp.asarray(r_frac)[:, None] * y_wall[None, :]
    return xp.stack([xx.reshape(-1), yy.reshape(-1)], axis=-1)


def build_nozzle_mesh(
    geom: NozzleGeometry,
    *,
    kind: str = "tri",
    nx: int = 14,
    nr: int = 5,
    refine: int = 0,
    geometry_order: int = 1,
    x_spacing: str = "throat",
    cluster_strength: float = 3.0,
    cluster_width: float = 0.12,
    xp=np,
) -> tuple[MeshTopology, object, dict[str, np.ndarray]]:
    """Convenience wrapper returning ``(topology, node_coords, logical_grids)``."""
    topo, _, r_frac = structured_nozzle_topology(
        kind, nx=nx, nr=nr, refine=refine, geometry_order=geometry_order
    )
    n_sub = geometry_order * 2**refine
    x_frac = nozzle_x_distribution(
        topo.logical["nx_nodes"],
        nx_base=nx,
        n_sub=n_sub,
        spacing=x_spacing,
        x_throat=geom.throat_location() / geom.length,
        cluster_strength=cluster_strength,
        cluster_width=cluster_width,
    )
    coords = nozzle_node_coords(x_frac, r_frac, geom.params(), geom.contour, xp=xp)
    return topo, coords, {"x_frac": x_frac, "r_frac": r_frac}
