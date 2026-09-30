"""Lagrange bases must interpolate, reproduce constants, and order edges consistently."""

from __future__ import annotations

import numpy as np
import pytest

from dgnozzle import elements as el


@pytest.mark.parametrize("order", [0, 1, 2, 3])
@pytest.mark.parametrize("kind", ["tri", "quad"])
def test_basis_is_interpolatory(kind, order):
    nodes = el.reference_nodes(kind, order)
    phi, _, _ = el.shape_functions(kind, order, nodes)
    assert np.allclose(phi, np.eye(el.n_basis(kind, order)), atol=1e-11)


@pytest.mark.parametrize("order", [0, 1, 2, 3])
@pytest.mark.parametrize("kind", ["tri", "quad"])
def test_partition_of_unity(kind, order):
    rng = np.random.default_rng(1)
    pts = rng.random((40, 2))
    if kind == "tri":
        pts = pts[pts.sum(axis=1) <= 1.0]
    phi, dxi, deta = el.shape_functions(kind, order, pts)
    assert np.allclose(phi.sum(axis=0), 1.0)
    assert np.allclose(dxi.sum(axis=0), 0.0, atol=1e-10)
    assert np.allclose(deta.sum(axis=0), 0.0, atol=1e-10)


@pytest.mark.parametrize("kind", ["tri", "quad"])
def test_gradients_match_finite_differences(kind):
    order = 2
    rng = np.random.default_rng(2)
    pts = rng.random((12, 2)) * 0.4 + 0.2
    h = 1e-6
    _, dxi, deta = el.shape_functions(kind, order, pts)
    fx = (
        el.shape_functions(kind, order, pts + [h, 0])[0]
        - el.shape_functions(kind, order, pts - [h, 0])[0]
    ) / (2 * h)
    fy = (
        el.shape_functions(kind, order, pts + [0, h])[0]
        - el.shape_functions(kind, order, pts - [0, h])[0]
    ) / (2 * h)
    assert np.allclose(dxi, fx, atol=1e-6)
    assert np.allclose(deta, fy, atol=1e-6)


@pytest.mark.parametrize("kind", ["tri", "quad"])
def test_edges_join_the_right_vertices(kind):
    """Local edge f must run between the vertices edge_vertex_local_indices names."""
    order = 2
    verts = el.reference_nodes(kind, order)[list(el.vertex_indices(kind, order))]
    for f in range(el.n_edges(kind)):
        a, b = el.edge_vertex_local_indices(kind, f)
        ends = el.edge_reference_coords(kind, f, np.array([0.0, 1.0]))
        assert np.allclose(ends[0], verts[a])
        assert np.allclose(ends[1], verts[b])


@pytest.mark.parametrize("kind", ["tri", "quad"])
def test_edge_parameterisation_matches_its_derivative(kind):
    for f in range(el.n_edges(kind)):
        sigma = np.array([0.2, 0.8])
        pts = el.edge_reference_coords(kind, f, sigma)
        d = np.asarray(el.edge_dreference_dsigma(kind, f))
        assert np.allclose((pts[1] - pts[0]) / (sigma[1] - sigma[0]), d)


@pytest.mark.parametrize("kind", ["tri", "quad"])
def test_vertices_are_counter_clockwise(kind):
    """The outward-normal convention depends on this."""
    verts = el.reference_nodes(kind, 1)[list(el.vertex_indices(kind, 1))]
    x, y = verts[:, 0], verts[:, 1]
    area2 = np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y)
    assert area2 > 0.0


def test_unknown_kind_is_rejected():
    with pytest.raises(ValueError, match="unknown element kind"):
        el.n_basis("hexagon", 1)
