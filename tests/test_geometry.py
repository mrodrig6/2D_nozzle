"""Contour families must hit their design targets and stay differentiable."""

from __future__ import annotations

import warnings

import numpy as np
import pytest

from src import CONTOURS, NozzleGeometry, check_contour
from src.geometry import (
    ANALYTIC_AREA_RATIO,
    ANALYTIC_THROAT_X,
    ANALYTIC_THROAT_Y,
    analytic_contour,
    wall_half_height,
)


@pytest.mark.parametrize("contour", CONTOURS)
def test_contour_hits_its_area_ratio(contour):
    geom = NozzleGeometry(contour=contour, area_ratio=2.5019, throat_x=0.1388)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        diag = check_contour(geom)
    assert diag["area_ratio"] == pytest.approx(2.5019, rel=2e-4)
    assert diag["y_throat"] == pytest.approx(geom.throat_height(), rel=2e-4)
    assert diag["y_inlet"] == pytest.approx(0.15, rel=1e-6)


@pytest.mark.parametrize("area_ratio", [1.5, 2.0, 3.0, 4.0, 6.0])
@pytest.mark.parametrize("contour", ["bell", "smooth", "conical", "bezier"])
def test_area_ratio_is_honoured_across_the_range(contour, area_ratio):
    geom = NozzleGeometry(contour=contour, area_ratio=area_ratio)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        diag = check_contour(geom)
    assert diag["area_ratio"] == pytest.approx(area_ratio, rel=1e-3)


def test_analytic_contour_matches_its_closed_form():
    """The fixed verification contour: inlet 0.15, throat 0.13989434, exit 0.35."""
    x = np.linspace(0.0, 1.0, 200_001)
    y = analytic_contour(x)
    i = int(np.argmin(y))
    assert y[0] == pytest.approx(0.15, abs=1e-12)
    assert y[-1] == pytest.approx(0.35, abs=1e-12)
    assert y[i] == pytest.approx(ANALYTIC_THROAT_Y, abs=1e-7)
    assert x[i] == pytest.approx(ANALYTIC_THROAT_X, abs=1e-5)
    assert y[-1] / y[i] == pytest.approx(ANALYTIC_AREA_RATIO, rel=1e-6)


def test_contour_never_goes_negative():
    """A wall that crosses zero produces inverted elements, so no family may."""
    for contour in CONTOURS:
        for ar in (1.5, 2.5, 4.0, 6.0):
            geom = NozzleGeometry(contour=contour, area_ratio=ar)
            y = np.asarray(geom.wall(np.linspace(0.0, geom.length, 2001)))
            assert np.all(y > 0.0), f"{contour} at AR={ar} produced a non-positive wall"


def test_contour_never_overshoots_its_exit_height():
    """A mis-scaled diverging section shows up here as a wall above the exit."""
    for contour in ("smooth", "conical", "bezier", "analytic"):
        geom = NozzleGeometry(contour=contour, area_ratio=2.5019)
        y = np.asarray(geom.wall(np.linspace(0.0, 1.0, 2001)))
        assert y.max() <= geom.throat_height() * 2.5019 * 1.001


def test_smooth_contour_is_c1_at_the_throat():
    geom = NozzleGeometry(contour="smooth")
    diag = check_contour(geom)
    assert abs(diag["throat_slope_jump"]) < 1e-3
    assert abs(diag["throat_wall_angle_deg"]) < 0.05


def test_bell_contour_has_a_deliberate_throat_corner():
    """Not a defect: it is the sharp-corner expansion of minimum-length design."""
    geom = NozzleGeometry(contour="bell", theta_initial_deg=21.0)
    diag = check_contour(geom)
    assert diag["throat_wall_angle_deg"] == pytest.approx(21.0, abs=0.1)
    assert diag["throat_slope_jump"] > 0.3


def test_explicit_wall_angles_are_honoured_at_any_length():
    for length in (0.5, 1.0, 2.0):
        geom = NozzleGeometry(contour="bell", length=length, theta_initial_deg=18.0)
        assert check_contour(geom)["throat_wall_angle_deg"] == pytest.approx(18.0, abs=0.02)


def test_bezier_weights_reproduce_the_conical_wall():
    common = dict(area_ratio=3.0, throat_x=0.2)
    bez = NozzleGeometry(contour="bezier", bezier_w1=1 / 3, bezier_w2=2 / 3, **common)
    con = NozzleGeometry(contour="conical", **common)
    x = np.linspace(0.0, 1.0, 400)
    assert np.allclose(np.asarray(bez.wall(x)), np.asarray(con.wall(x)), atol=1e-12)


def test_parameter_round_trip():
    geom = NozzleGeometry(contour="bezier", area_ratio=3.1, theta_exit_deg=4.0)
    back = geom.with_params(geom.params())
    x = np.linspace(0.0, 1.0, 101)
    assert np.allclose(np.asarray(geom.wall(x)), np.asarray(back.wall(x)))


def test_invalid_geometry_is_rejected():
    with pytest.raises(ValueError, match="area_ratio must exceed 1"):
        NozzleGeometry(area_ratio=0.8)
    with pytest.raises(ValueError, match="throat_x must lie"):
        NozzleGeometry(throat_x=1.5)
    with pytest.raises(ValueError, match="inlet_half_height must exceed"):
        NozzleGeometry(inlet_half_height=0.05)
    with pytest.raises(ValueError, match="unknown contour"):
        NozzleGeometry(contour="banana")


def test_non_monotone_contour_warns_rather_than_failing():
    """A bulging wall is an unusual design, not an error."""
    geom = NozzleGeometry(contour="bell", area_ratio=1.2, theta_initial_deg=45.0)
    with pytest.warns(UserWarning, match="not monotone"):
        check_contour(geom)


@pytest.mark.jax
@pytest.mark.parametrize("contour", ["bell", "smooth", "conical", "bezier"])
def test_contour_is_differentiable_under_jax(contour):
    import jax
    import jax.numpy as jnp

    jax.config.update("jax_enable_x64", True)
    base = NozzleGeometry(contour=contour).params()

    def exit_height(ar):
        p = dict(base)
        p["area_ratio"] = ar
        return wall_half_height(jnp.asarray([1.0]), p, contour, xp=jnp)[0]

    grad = float(jax.grad(exit_height)(2.5))
    # y_exit = AR * y_throat, so the derivative is exactly y_throat
    assert grad == pytest.approx(base["throat_half_height"], rel=1e-6)
