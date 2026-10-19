"""Quadrature rules for the reference triangle, square and unit interval.

All rules returned here have **strictly positive weights**, which is a
requirement rather than a nicety: several standard symmetric triangle rules
carry a negative weight (the Dunavant degree-3 rule's first weight is
``-0.28125``), and a negative weight can destroy the positive definiteness of
the elemental mass matrix and let the entropy-error integral -- a sum of squares
-- come out negative.  Every rule below is verified against exact monomial
integrals in ``tests/test_quadrature.py``.

Conventions
-----------
Reference triangle   ``T = {(xi, eta) : xi >= 0, eta >= 0, xi + eta <= 1}``,
                     area 1/2, so the weights sum to 1/2.
Reference square     ``S = [0, 1]^2``, weights sum to 1.
Reference interval   ``[0, 1]``, weights sum to 1.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import numpy as np
from scipy.special import roots_jacobi


@dataclass(frozen=True)
class QuadratureRule:
    """A quadrature rule: ``points`` is ``(npts, ndim)``, ``weights`` is ``(npts,)``."""

    points: np.ndarray
    weights: np.ndarray

    @property
    def n_points(self) -> int:
        return int(self.weights.shape[0])

    def __post_init__(self) -> None:
        if np.any(self.weights <= 0.0):
            raise ValueError("quadrature rule has a non-positive weight")


# --------------------------------------------------------------------------
# 1D: Gauss-Legendre on [0, 1]
# --------------------------------------------------------------------------
@lru_cache(maxsize=64)
def gauss_legendre_1d(degree: int) -> QuadratureRule:
    """Gauss-Legendre rule on ``[0, 1]`` exact for polynomials up to ``degree``."""
    if degree < 0:
        raise ValueError(f"degree must be non-negative, got {degree}")
    n = max(1, -(-(degree + 1) // 2))  # ceil((degree+1)/2)
    x, w = np.polynomial.legendre.leggauss(n)
    # map [-1, 1] -> [0, 1]
    return QuadratureRule(
        points=(0.5 * (x + 1.0)).reshape(-1, 1).copy(),
        weights=(0.5 * w).copy(),
    )


# --------------------------------------------------------------------------
# Square: tensor product Gauss-Legendre on [0, 1]^2
# --------------------------------------------------------------------------
@lru_cache(maxsize=64)
def gauss_legendre_square(degree: int) -> QuadratureRule:
    """Tensor-product Gauss-Legendre rule on ``[0, 1]^2``."""
    rule = gauss_legendre_1d(degree)
    x = rule.points[:, 0]
    w = rule.weights
    xi, eta = np.meshgrid(x, x, indexing="ij")
    wt = np.outer(w, w)
    return QuadratureRule(
        points=np.column_stack([xi.ravel(), eta.ravel()]),
        weights=wt.ravel().copy(),
    )


# --------------------------------------------------------------------------
# Triangle
# --------------------------------------------------------------------------
# Symmetric positive-weight rules for the two cheapest cases.  Weights are
# given on the unit triangle (they sum to 1/2).
_TRI_SYMMETRIC: dict[int, tuple[list[tuple[float, float]], list[float]]] = {
    0: ([(1.0 / 3.0, 1.0 / 3.0)], [0.5]),
    1: ([(1.0 / 3.0, 1.0 / 3.0)], [0.5]),
    2: (
        [(1.0 / 6.0, 1.0 / 6.0), (2.0 / 3.0, 1.0 / 6.0), (1.0 / 6.0, 2.0 / 3.0)],
        [1.0 / 6.0, 1.0 / 6.0, 1.0 / 6.0],
    ),
}


@lru_cache(maxsize=64)
def collapsed_gauss_jacobi_triangle(degree: int) -> QuadratureRule:
    """Duffy-collapsed Gauss-Jacobi rule on the reference triangle.

    The Duffy transform ``(xi, eta) = (x, (1 - x) y)`` maps the unit square onto
    the triangle with Jacobian ``(1 - x)``.  Absorbing that Jacobian into a
    Gauss-Jacobi(alpha=1, beta=0) rule in ``x`` and using Gauss-Legendre in
    ``y`` gives an ``n x n`` rule, exact to degree ``2n - 1``, whose weights are
    all positive for every ``n``.
    """
    if degree < 0:
        raise ValueError(f"degree must be non-negative, got {degree}")
    n = max(1, -(-(degree + 1) // 2))

    # Gauss-Jacobi(1, 0) on [-1, 1] with weight (1 - t); map t -> x = (t + 1)/2.
    # On [-1, 1] the weight is (1 - t) = 2 (1 - x), and dt = 2 dx, so a weight
    # w_GJ on [-1, 1] corresponds to w_GJ / 4 against (1 - x) dx on [0, 1].
    t, w_gj = roots_jacobi(n, 1.0, 0.0)
    x = 0.5 * (t + 1.0)
    wx = w_gj / 4.0

    leg = gauss_legendre_1d(degree)
    y = leg.points[:, 0]
    wy = leg.weights

    X, Y = np.meshgrid(x, y, indexing="ij")
    W = np.outer(wx, wy)

    xi = X
    eta = (1.0 - X) * Y
    return QuadratureRule(
        points=np.column_stack([xi.ravel(), eta.ravel()]),
        weights=W.ravel().copy(),
    )


@lru_cache(maxsize=64)
def triangle_rule(degree: int) -> QuadratureRule:
    """Positive-weight rule on the reference triangle, exact to ``degree``."""
    if degree in _TRI_SYMMETRIC:
        pts, wts = _TRI_SYMMETRIC[degree]
        return QuadratureRule(points=np.asarray(pts, float), weights=np.asarray(wts, float))
    return collapsed_gauss_jacobi_triangle(degree)


def volume_rule(element_kind: str, degree: int) -> QuadratureRule:
    """Volume quadrature for ``'tri'`` or ``'quad'`` reference elements."""
    if element_kind == "tri":
        return triangle_rule(degree)
    if element_kind == "quad":
        return gauss_legendre_square(degree)
    raise ValueError(f"unknown element kind {element_kind!r}; use 'tri' or 'quad'")
