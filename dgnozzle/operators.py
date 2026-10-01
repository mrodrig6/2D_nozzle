r"""Precomputed DG operators: metrics, mass matrices, normals.

Everything a residual evaluation needs is assembled once, here, into a flat
container of arrays.  Two properties are deliberate:

*Element-batched layout.*  Arrays are shaped ``(n_elem, ...)`` or
``(n_edge, ...)`` so the whole residual is a handful of ``einsum`` calls rather
than a Python loop over elements.  Interpreted per-element and per-edge loops
dominate the cost of a scheme like this one, so there are none.

*Written through an injected array module.*  Passing ``xp=jax.numpy`` makes the
metric terms -- and therefore the residual, the steady solution and the thrust
-- differentiable with respect to the node coordinates, hence with respect to
the nozzle shape.

Global edge numbering
---------------------
Interior edges occupy ``0 .. n_interior-1`` and boundary edges follow at
``n_interior + k``.  Every ``(element, local face)`` pair maps to exactly one
global edge via ``topology.edges.face_edge``, with ``face_side`` recording
whether the element is that edge's owner.  Assembly is therefore a pure
*gather* -- there is no scatter-add anywhere in the hot loop, which is what lets
the same code run unchanged under NumPy, Numba and JAX.

Sign convention
---------------
The outward normal of an edge is the tangent ``dX/dsigma`` of its owner's local
face rotated by -90 degrees.  Because element vertices are ordered
counter-clockwise, that normal points out of the owner.  The neighbour therefore
sees ``-n``, which the assembly applies as a sign flip rather than by storing a
second set of normals.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from . import elements as el
from . import quadrature as qd
from .mesh import MeshTopology


@dataclass(frozen=True)
class ReferenceData:
    """Static, mesh-independent reference-element tables."""

    kind: str
    order: int
    geometry_order: int
    n_basis: int
    n_basis_geom: int
    # volume
    points_vol: np.ndarray  # (nqv, 2) reference coordinates
    w_vol: np.ndarray  # (nqv,)
    phi_vol: np.ndarray  # (nbf, nqv)
    dphi_dxi_vol: np.ndarray  # (nbf, nqv)
    dphi_deta_vol: np.ndarray
    dgeom_dxi_vol: np.ndarray  # (nbfQ, nqv)
    dgeom_deta_vol: np.ndarray
    geom_vol: np.ndarray  # (nbfQ, nqv)
    # faces: index 0 = owner side (sigma), 1 = neighbour side (1 - sigma)
    w_face: np.ndarray  # (nqf,)
    phi_face: np.ndarray  # (2, nface, nbf, nqf)
    geom_face: np.ndarray  # (nface, nbfQ, nqf)   owner side only
    dgeom_dsigma_face: np.ndarray  # (nface, nbfQ, nqf) owner side only

    @property
    def n_qvol(self) -> int:
        return int(self.w_vol.size)

    @property
    def n_qface(self) -> int:
        return int(self.w_face.size)

    @property
    def n_faces(self) -> int:
        return el.n_edges(self.kind)


def reference_data(kind: str, order: int, geometry_order: int) -> ReferenceData:
    """Build the reference tables for one element kind / order pair."""
    deg_vol, deg_face = el.quadrature_degree(kind, order, geometry_order)
    vol = qd.volume_rule(kind, deg_vol)
    face = qd.gauss_legendre_1d(deg_face)
    sigma = face.points[:, 0]
    nface = el.n_edges(kind)

    phi_v, dxi_v, deta_v = el.shape_functions(kind, order, vol.points)
    g_v, gdxi_v, gdeta_v = el.shape_functions(kind, geometry_order, vol.points)

    nbf = el.n_basis(kind, order)
    nbfq = el.n_basis(kind, geometry_order)
    nqf = sigma.size

    phi_face = np.zeros((2, nface, nbf, nqf))
    geom_face = np.zeros((nface, nbfq, nqf))
    dgeom_dsigma = np.zeros((nface, nbfq, nqf))

    for f in range(nface):
        dxi_ds, deta_ds = el.edge_dreference_dsigma(kind, f)
        # owner side walks the edge in its local direction
        pts_own = el.edge_reference_coords(kind, f, sigma)
        # the neighbour walks the same physical edge backwards, so quadrature
        # point q of both sides lands on the same physical location
        pts_nbr = el.edge_reference_coords(kind, f, 1.0 - sigma)

        phi_face[0, f], _, _ = el.shape_functions(kind, order, pts_own)
        phi_face[1, f], _, _ = el.shape_functions(kind, order, pts_nbr)

        g_f, gdxi_f, gdeta_f = el.shape_functions(kind, geometry_order, pts_own)
        geom_face[f] = g_f
        dgeom_dsigma[f] = gdxi_f * dxi_ds + gdeta_f * deta_ds

    return ReferenceData(
        kind=kind,
        order=order,
        geometry_order=geometry_order,
        n_basis=nbf,
        n_basis_geom=nbfq,
        points_vol=vol.points,
        w_vol=vol.weights,
        phi_vol=phi_v,
        dphi_dxi_vol=dxi_v,
        dphi_deta_vol=deta_v,
        dgeom_dxi_vol=gdxi_v,
        dgeom_deta_vol=gdeta_v,
        geom_vol=g_v,
        w_face=face.weights,
        phi_face=phi_face,
        geom_face=geom_face,
        dgeom_dsigma_face=dgeom_dsigma,
    )


@dataclass(frozen=True)
class Operators:
    """Mesh-dependent DG operators.

    All float arrays here come from ``node_coords`` through the injected array
    module, so under JAX the whole container is differentiable.
    """

    ref: ReferenceData
    topology: MeshTopology
    node_coords: Any
    """The geometry nodes these operators were built from.

    Kept so that a second operator set can be built on the *same* mesh at a
    different polynomial order -- which is what ``p``-continuation needs --
    without the caller having to carry the coordinates alongside.
    """

    # volume
    det_jac: Any  # (nelem, nqv)
    weighted_det: Any  # (nelem, nqv)        w_vol * det_jac
    grad_x: Any  # (nelem, nbf, nqv)   w_vol * det_jac * dphi/dx
    grad_y: Any  # (nelem, nbf, nqv)
    inv_mass: Any  # (nelem, nbf, nbf)
    elem_area: Any  # (nelem,)
    mean_weights: Any  # (nelem, nbf) cell-average operator: ubar_s = sum_i mw_i U_is
    xy_vol: Any  # (nelem, nqv, 2) physical coords of volume quad points

    # edges, in global edge numbering
    edge_normal: Any  # (nedge, nqf, 2) unit outward normal of the owner
    edge_jac: Any  # (nedge, nqf)    |dX/dsigma|
    edge_length: Any  # (nedge,)
    xy_face: Any  # (nedge, nqf, 2)

    # gather map, broadcast to element-face layout
    face_edge: np.ndarray  # (nelem, nface)
    face_sign: Any  # (nelem, nface) +1 owner, -1 neighbour
    face_basis: Any  # (nelem, nface, nbf, nqf) basis for this element's side

    @property
    def n_elem(self) -> int:
        return int(self.topology.n_elem)

    @property
    def n_interior(self) -> int:
        return int(self.topology.edges.n_interior)

    @property
    def n_boundary(self) -> int:
        return int(self.topology.edges.n_boundary)

    @property
    def n_edges_total(self) -> int:
        return self.n_interior + self.n_boundary


def build_operators(
    topology: MeshTopology,
    node_coords,
    order: int,
    *,
    xp=np,
    ref: ReferenceData | None = None,
    check_jacobian: bool = True,
) -> Operators:
    """Assemble all DG operators for a mesh.

    Parameters
    ----------
    topology
        Integer connectivity from :mod:`dgnozzle.mesh`.
    node_coords
        ``(n_nodes, 2)`` coordinates; may be a traced JAX array.
    order
        Solution polynomial order ``p``.
    xp
        ``numpy`` or ``jax.numpy``.
    check_jacobian
        Verify that every Jacobian determinant is positive.  Skipped
        automatically when ``node_coords`` is a JAX tracer, since that cannot be
        inspected inside ``jit``.
    """
    q = topology.geometry_order
    rd = ref if ref is not None else reference_data(topology.kind, order, q)
    if rd.kind != topology.kind or rd.geometry_order != q or rd.order != order:
        raise ValueError("reference data does not match the requested mesh/order")

    X = xp.asarray(node_coords)
    elem_nodes = topology.elem_nodes
    Xe = X[elem_nodes]  # (nelem, nbfQ, 2)

    # ---- volume metrics -------------------------------------------------
    dx_dxi = Xe[:, :, 0] @ rd.dgeom_dxi_vol  # (nelem, nqv)
    dx_deta = Xe[:, :, 0] @ rd.dgeom_deta_vol
    dy_dxi = Xe[:, :, 1] @ rd.dgeom_dxi_vol
    dy_deta = Xe[:, :, 1] @ rd.dgeom_deta_vol

    det = dx_dxi * dy_deta - dx_deta * dy_dxi

    if check_jacobian and isinstance(det, np.ndarray):
        bad = int(np.sum(det <= 0.0))
        if bad:
            worst = int(np.argmin(det.min(axis=1)))
            raise ValueError(
                f"{bad} quadrature point(s) have a non-positive Jacobian determinant "
                f"(worst element {worst}, det={float(det.min()):.3e}). The mesh is "
                "inverted or tangled -- usually a contour that goes negative or "
                "doubles back. Run dgnozzle.geometry.check_contour on the geometry."
            )

    # inverse Jacobian: d(xi, eta) / d(x, y)
    inv_det = 1.0 / det
    dxi_dx = dy_deta * inv_det
    dxi_dy = -dx_deta * inv_det
    deta_dx = -dy_dxi * inv_det
    deta_dy = dx_dxi * inv_det

    # dphi/dx[e, i, q] = dphi_dxi[i, q] * dxi_dx[e, q] + dphi_deta[i, q] * deta_dx[e, q]
    wdet = rd.w_vol[None, :] * det  # (nelem, nqv)
    grad_x = wdet[:, None, :] * (
        rd.dphi_dxi_vol[None, :, :] * dxi_dx[:, None, :]
        + rd.dphi_deta_vol[None, :, :] * deta_dx[:, None, :]
    )
    grad_y = wdet[:, None, :] * (
        rd.dphi_dxi_vol[None, :, :] * dxi_dy[:, None, :]
        + rd.dphi_deta_vol[None, :, :] * deta_dy[:, None, :]
    )

    # ---- mass matrix: M[e,i,j] = sum_q w det phi_i phi_j -----------------
    mass = xp.einsum("iq,jq,eq->eij", rd.phi_vol, rd.phi_vol, wdet)
    inv_mass = xp.linalg.inv(mass)

    elem_area = wdet.sum(axis=1)
    # Cell average as a linear functional of the coefficients.  Computed by
    # quadrature rather than read off the first mode, because a Lagrange basis is
    # not orthogonal; the volume rule integrates it exactly.
    mean_weights = xp.einsum("iq,eq->ei", xp.asarray(rd.phi_vol), wdet) / elem_area[:, None]
    xy_vol = xp.einsum("nq,end->eqd", rd.geom_vol, Xe)

    # ---- edge metrics, in global edge numbering -------------------------
    edges = topology.edges
    owner_elem = np.concatenate([edges.iedge_elem[:, 0], edges.bedge_elem])
    owner_face = np.concatenate([edges.iedge_face[:, 0], edges.bedge_face])

    Xo = Xe[owner_elem]  # (nedge, nbfQ, 2)
    dgeom = xp.asarray(rd.dgeom_dsigma_face)[owner_face]  # (nedge, nbfQ, nqf)
    tangent = xp.einsum("enq,end->eqd", dgeom, Xo)  # (nedge, nqf, 2)
    jac = xp.sqrt(tangent[..., 0] ** 2 + tangent[..., 1] ** 2)
    # rotate the tangent by -90 deg -> outward normal of a CCW element
    normal = xp.stack([tangent[..., 1], -tangent[..., 0]], axis=-1) / jac[..., None]
    edge_length = (jac * rd.w_face[None, :]).sum(axis=1)
    geom_f = xp.asarray(rd.geom_face)[owner_face]  # (nedge, nbfQ, nqf)
    xy_face = xp.einsum("enq,end->eqd", geom_f, Xo)

    # ---- gather map ------------------------------------------------------
    face_sign = xp.asarray(np.where(edges.face_side == 0, 1.0, -1.0))
    nface = topology.n_faces
    # basis this element uses on each of its faces, chosen by side
    face_basis = xp.asarray(
        rd.phi_face[edges.face_side, np.arange(nface)[None, :].repeat(topology.n_elem, axis=0)]
    )

    return Operators(
        ref=rd,
        topology=topology,
        node_coords=X,
        det_jac=det,
        weighted_det=wdet,
        grad_x=grad_x,
        grad_y=grad_y,
        inv_mass=inv_mass,
        elem_area=elem_area,
        mean_weights=mean_weights,
        xy_vol=xy_vol,
        edge_normal=normal,
        edge_jac=jac,
        edge_length=edge_length,
        xy_face=xy_face,
        face_edge=edges.face_edge,
        face_sign=face_sign,
        face_basis=face_basis,
    )
