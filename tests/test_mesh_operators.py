"""Mesh topology and the metric terms.

The freestream-preservation test is the important one here: it is the strongest
available check that the volume gradient operator and the face operator are
mutually consistent, and it catches nearly every metric bug.
"""

from __future__ import annotations

import numpy as np
import pytest

from dgnozzle import BoundaryTag, NozzleGeometry, build_nozzle_mesh
from dgnozzle import physics as ph
from dgnozzle.assembly import face_term, volume_term
from dgnozzle.config import FlowConditions
from dgnozzle.operators import build_operators

KINDS = ["tri", "quad"]
ORDERS = [0, 1, 2]
GEOM_ORDERS = [1, 2]


@pytest.fixture(scope="module")
def geom():
    return NozzleGeometry(contour="smooth")


@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("refine", [0, 1])
def test_face_count_is_consistent(geom, kind, refine):
    topo, _, _ = build_nozzle_mesh(geom, kind=kind, refine=refine)
    ed = topo.edges
    assert topo.n_elem * topo.n_faces == 2 * ed.n_interior + ed.n_boundary


@pytest.mark.parametrize("kind", KINDS)
def test_element_counts_quadruple_per_refinement(geom, kind):
    counts = [build_nozzle_mesh(geom, kind=kind, refine=r)[0].n_elem for r in (0, 1, 2)]
    expected = [140, 560, 2240] if kind == "tri" else [70, 280, 1120]
    assert counts == expected


@pytest.mark.parametrize("kind", KINDS)
def test_every_boundary_is_tagged_and_wall_and_axis_are_separate(geom, kind):
    topo, coords, _ = build_nozzle_mesh(geom, kind=kind)
    coords = np.asarray(coords)
    tags = {t: topo.tag_slice(t) for t in BoundaryTag}
    assert all(s.stop > s.start for s in tags.values())
    total = sum(s.stop - s.start for s in tags.values())
    assert total == topo.edges.n_boundary

    # Check the tagged *faces*, not their owning elements: an axis element also
    # has nodes away from the axis.
    ops = build_operators(topo, coords, 1)
    ni = ops.n_interior
    for tag, check in (
        (BoundaryTag.AXIS, lambda xy: np.allclose(xy[..., 1], 0.0)),
        (BoundaryTag.WALL, lambda xy: xy[..., 1].min() > 0.05),
        (BoundaryTag.INFLOW, lambda xy: np.allclose(xy[..., 0], 0.0)),
        (BoundaryTag.OUTFLOW, lambda xy: np.allclose(xy[..., 0], 1.0)),
    ):
        sl = topo.tag_slice(tag)
        xy = ops.xy_face[ni + np.arange(sl.start, sl.stop)]
        assert check(xy), f"{tag.name} faces are in the wrong place"


@pytest.mark.parametrize("kind", KINDS)
def test_face_neighbours_are_reciprocal(geom, kind):
    topo, _, _ = build_nozzle_mesh(geom, kind=kind)
    nb = topo.edges.face_neighbour
    self_ref = (nb == np.arange(topo.n_elem)[:, None]).sum()
    assert self_ref == topo.edges.n_boundary
    for k in range(topo.edges.n_interior):
        le, re = topo.edges.iedge_elem[k]
        lf, rf = topo.edges.iedge_face[k]
        assert nb[le, lf] == re
        assert nb[re, rf] == le


@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("q", GEOM_ORDERS)
def test_jacobians_are_positive(geom, kind, q):
    topo, coords, _ = build_nozzle_mesh(geom, kind=kind, geometry_order=q)
    ops = build_operators(topo, coords, 1)
    assert np.all(ops.det_jac > 0.0)
    assert np.all(ops.elem_area > 0.0)


@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("q", GEOM_ORDERS)
def test_normals_satisfy_the_divergence_theorem(geom, kind, q):
    """sum over faces of n*ds must vanish for every element."""
    topo, coords, _ = build_nozzle_mesh(geom, kind=kind, geometry_order=q)
    ops = build_operators(topo, coords, 1)
    ns = ops.edge_normal[ops.face_edge] * ops.edge_jac[ops.face_edge][..., None]
    closure = np.einsum("efqd,ef,q->ed", ns, ops.face_sign, ops.ref.w_face)
    assert np.abs(closure).max() < 1e-12


@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("order", ORDERS)
def test_mass_matrix_inverse_is_correct(geom, kind, order):
    topo, coords, _ = build_nozzle_mesh(geom, kind=kind)
    ops = build_operators(topo, coords, order)
    mass = np.einsum("iq,jq,eq->eij", ops.ref.phi_vol, ops.ref.phi_vol, ops.weighted_det)
    identity = np.einsum("eij,ejk->eik", ops.inv_mass, mass)
    assert np.allclose(identity, np.eye(ops.ref.n_basis), atol=1e-9)


@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("order", ORDERS)
def test_cell_average_operator(geom, kind, order):
    topo, coords, _ = build_nozzle_mesh(geom, kind=kind)
    ops = build_operators(topo, coords, order)
    assert np.allclose(ops.mean_weights.sum(axis=1), 1.0)
    U = np.zeros((ops.n_elem, ops.ref.n_basis, 4))
    U[:, :, 0] = 3.7
    assert np.allclose(np.einsum("ei,eis->es", ops.mean_weights, U)[:, 0], 3.7)


@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("order", ORDERS)
@pytest.mark.parametrize("q", GEOM_ORDERS)
def test_freestream_preservation(geom, kind, order, q):
    """A uniform state with the exact edge flux must give exactly zero residual.

    This is the discrete geometric conservation law.  If the metric terms are
    inconsistent in any way -- wrong Jacobian, wrong normal, a mismatched
    quadrature between the volume and the face -- it fails.
    """
    topo, coords, _ = build_nozzle_mesh(geom, kind=kind, geometry_order=q)
    ops = build_operators(topo, coords, order)
    flow = FlowConditions()
    state = np.array([2.0, 0.6, -0.15, 5.0])

    U = np.zeros((ops.n_elem, ops.ref.n_basis, 4))
    U[:, :, :] = state

    exact = ph.normal_flux(
        np.broadcast_to(state, (ops.n_edges_total, ops.ref.n_qface, 4)),
        ops.edge_normal[..., 0],
        ops.edge_normal[..., 1],
        flow.gamma,
    )
    residual = volume_term(U, ops, flow) + face_term(exact, ops)
    assert np.abs(residual).max() < 1e-13


@pytest.mark.parametrize("q", GEOM_ORDERS)
def test_curved_elements_improve_the_geometry(geom, q):
    """Q=2 must represent the curved wall far better than Q=1 at equal element count."""
    exact = float(np.trapezoid(np.asarray(geom.wall(np.linspace(0, 1, 400_001))),
                               np.linspace(0, 1, 400_001)))
    errors = {}
    for qq in (1, 2):
        topo, coords, _ = build_nozzle_mesh(geom, geometry_order=qq)
        ops = build_operators(topo, coords, 1)
        errors[qq] = abs(float(ops.elem_area.sum()) - exact) / exact
    assert errors[2] < errors[1] / 20.0


def test_inverted_mesh_is_rejected_with_a_useful_message(geom):
    topo, coords, _ = build_nozzle_mesh(geom)
    bad = np.asarray(coords).copy()
    bad[:, 1] *= -1.0  # flip the channel, inverting every element
    with pytest.raises(ValueError, match="non-positive Jacobian"):
        build_operators(topo, bad, 1)


def test_non_conforming_mesh_is_detected(geom):
    from dgnozzle.mesh import build_edges, nozzle_boundary_tagger

    topo, coords, _ = build_nozzle_mesh(geom)
    coords = np.asarray(coords)
    triple = np.vstack([topo.elem_nodes, topo.elem_nodes[:1]])
    with pytest.raises(ValueError, match="more than two elements"):
        build_edges("tri", triple, 1, nozzle_boundary_tagger(1.0, 1e-3), coords)
