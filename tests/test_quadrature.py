"""Quadrature rules must integrate polynomials exactly and have positive weights."""

from __future__ import annotations

import numpy as np
import pytest

from src import quadrature as qd


def _exact_triangle_monomial(r: int, s: int) -> float:
    """int_T xi^r eta^s dA over the unit triangle = r! s! / (r+s+2)!."""
    from math import factorial

    return factorial(r) * factorial(s) / factorial(r + s + 2)


@pytest.mark.parametrize("degree", range(0, 13))
def test_triangle_rule_is_exact(degree):
    rule = qd.triangle_rule(degree)
    for r in range(degree + 1):
        for s in range(degree + 1 - r):
            approx = float((rule.points[:, 0] ** r * rule.points[:, 1] ** s * rule.weights).sum())
            assert approx == pytest.approx(_exact_triangle_monomial(r, s), abs=1e-13)


@pytest.mark.parametrize("degree", range(0, 13))
def test_square_rule_is_exact(degree):
    rule = qd.gauss_legendre_square(degree)
    for r in range(degree + 1):
        for s in range(degree + 1):
            approx = float((rule.points[:, 0] ** r * rule.points[:, 1] ** s * rule.weights).sum())
            assert approx == pytest.approx(1.0 / ((r + 1) * (s + 1)), abs=1e-13)


@pytest.mark.parametrize("degree", range(0, 20))
def test_line_rule_is_exact(degree):
    rule = qd.gauss_legendre_1d(degree)
    for r in range(degree + 1):
        approx = float((rule.points[:, 0] ** r * rule.weights).sum())
        assert approx == pytest.approx(1.0 / (r + 1), abs=1e-13)


@pytest.mark.parametrize("degree", range(0, 16))
@pytest.mark.parametrize("kind", ["tri", "quad"])
def test_all_weights_are_positive(kind, degree):
    """Several standard triangle rules carry a negative weight.

    The Dunavant degree-3 rule's first weight is -0.28125.  A negative weight can
    make the elemental mass matrix indefinite and lets the entropy-error
    integral, a sum of squares, come out negative.
    """
    rule = qd.volume_rule(kind, degree)
    assert np.all(rule.weights > 0.0)


def test_weights_sum_to_the_reference_measure():
    assert qd.triangle_rule(5).weights.sum() == pytest.approx(0.5)
    assert qd.gauss_legendre_square(5).weights.sum() == pytest.approx(1.0)
    assert qd.gauss_legendre_1d(5).weights.sum() == pytest.approx(1.0)


def test_negative_degree_is_rejected():
    with pytest.raises(ValueError, match="non-negative"):
        qd.triangle_rule(-1)
