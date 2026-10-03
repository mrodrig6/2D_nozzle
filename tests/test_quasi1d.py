"""Quasi-1D theory: the reference solution the DG result is checked against."""

from __future__ import annotations

import numpy as np
import pytest

from src import FlowConditions, NozzleGeometry, Regime, critical_ratios, solve_quasi1d
from src.quasi1d import (
    area_over_throat,
    mach_from_area,
    mach_from_pressure_ratio,
    normal_shock_mach,
    normal_shock_static_pressure_ratio,
    normal_shock_total_pressure_ratio,
    operating_regime,
    pressure_ratio,
)

GAMMA = 1.4


@pytest.mark.parametrize("mach", [0.05, 0.2, 0.5, 0.9, 1.4, 2.0, 3.0, 5.0])
def test_area_mach_inversion_round_trips(mach):
    ar = float(area_over_throat(mach, GAMMA))
    back = float(mach_from_area(ar, GAMMA, supersonic=mach > 1.0))
    assert back == pytest.approx(mach, abs=1e-9)


def test_area_relation_is_unity_at_sonic():
    assert float(area_over_throat(1.0, GAMMA)) == pytest.approx(1.0, abs=1e-14)


def test_pressure_ratio_inversion_round_trips():
    for mach in (0.1, 0.5, 1.0, 2.5):
        r = float(pressure_ratio(mach, GAMMA))
        assert float(mach_from_pressure_ratio(r, GAMMA)) == pytest.approx(mach, abs=1e-10)


def test_area_below_unity_is_rejected():
    with pytest.raises(ValueError, match=r"A/A\* must be >= 1"):
        mach_from_area(0.5, GAMMA)


@pytest.mark.parametrize("m1", [1.2, 1.5, 2.0, 3.0, 5.0])
def test_normal_shock_relations(m1):
    m2 = float(normal_shock_mach(m1, GAMMA))
    assert m2 < 1.0  # a shock always decelerates to subsonic
    assert float(normal_shock_static_pressure_ratio(m1, GAMMA)) > 1.0
    pt_ratio = float(normal_shock_total_pressure_ratio(m1, GAMMA))
    assert 0.0 < pt_ratio < 1.0  # total pressure can only be lost


def test_weak_shock_limit():
    """As M1 -> 1 the shock vanishes."""
    assert float(normal_shock_mach(1.0, GAMMA)) == pytest.approx(1.0, abs=1e-9)
    assert float(normal_shock_total_pressure_ratio(1.0, GAMMA)) == pytest.approx(1.0, abs=1e-9)


def test_critical_ratios_are_ordered():
    crit = critical_ratios(2.5019, GAMMA)
    assert crit.third < crit.second < crit.first < 1.0
    assert crit.exit_mach_design > 1.0
    assert crit.exit_mach_subsonic < 1.0


def test_critical_ratios_for_the_reference_geometry():
    crit = critical_ratios(2.50189, GAMMA)
    assert crit.first == pytest.approx(0.9609, abs=1e-3)
    assert crit.second == pytest.approx(0.4345, abs=1e-3)
    assert crit.third == pytest.approx(0.06390, abs=1e-4)
    assert crit.exit_mach_design == pytest.approx(2.4436, abs=1e-3)


@pytest.mark.parametrize(
    "pb, expected",
    [
        (0.99, Regime.SUBSONIC),
        (0.90, Regime.SHOCK_IN_NOZZLE),
        (0.50, Regime.SHOCK_IN_NOZZLE),
        (0.35, Regime.OVEREXPANDED),
        (0.15, Regime.OVEREXPANDED),
        (0.03, Regime.UNDEREXPANDED),
    ],
)
def test_operating_regimes(pb, expected):
    assert operating_regime(2.50189, pb, GAMMA)[0] is expected


@pytest.fixture(scope="module")
def geom():
    return NozzleGeometry(contour="smooth")


def test_mass_flow_is_constant_along_a_choked_nozzle(geom):
    sol = solve_quasi1d(geom, FlowConditions(back_pressure_ratio=0.15))
    mdot = sol.density * sol.velocity * sol.area
    assert (mdot.max() - mdot.min()) / mdot.mean() < 1e-10


def test_mass_flow_is_constant_across_a_shock(geom):
    sol = solve_quasi1d(geom, FlowConditions(back_pressure_ratio=0.70))
    assert sol.shock_x is not None
    mdot = sol.density * sol.velocity * sol.area
    assert (mdot.max() - mdot.min()) / mdot.mean() < 1e-8


def test_shock_position_matches_the_requested_back_pressure(geom):
    for pb in (0.5, 0.7, 0.9):
        sol = solve_quasi1d(geom, FlowConditions(back_pressure_ratio=pb))
        assert sol.regime is Regime.SHOCK_IN_NOZZLE
        assert sol.exit_pressure == pytest.approx(pb, rel=1e-6)
        assert geom.throat_location() < sol.shock_x < geom.length


def test_shock_obeys_rankine_hugoniot(geom):
    sol = solve_quasi1d(geom, FlowConditions(back_pressure_ratio=0.70))
    i = int(np.searchsorted(sol.x, sol.shock_x))
    jump = sol.pressure[i + 2] / sol.pressure[i - 2]
    expected = float(normal_shock_static_pressure_ratio(sol.shock_mach, GAMMA))
    assert jump == pytest.approx(expected, rel=0.02)


def test_choked_mass_flow_is_independent_of_back_pressure(geom):
    """The defining property of choking."""
    flows = [
        solve_quasi1d(geom, FlowConditions(back_pressure_ratio=pb)).mass_flow
        for pb in (0.9, 0.5, 0.15)
    ]
    assert np.allclose(flows, flows[0], rtol=1e-9)


def test_unchoked_mass_flow_rises_as_back_pressure_falls(geom):
    a = solve_quasi1d(geom, FlowConditions(back_pressure_ratio=0.995)).mass_flow
    b = solve_quasi1d(geom, FlowConditions(back_pressure_ratio=0.975)).mass_flow
    assert b > a


def test_throat_area_comes_from_the_analytic_contour(geom):
    """Reading the throat off the sampled grid straddles the minimum and
    misclassifies operating points near a critical ratio."""
    coarse = solve_quasi1d(geom, FlowConditions(back_pressure_ratio=0.15), n_points=51)
    fine = solve_quasi1d(geom, FlowConditions(back_pressure_ratio=0.15), n_points=2001)
    assert coarse.exit_mach == pytest.approx(fine.exit_mach, rel=1e-9)
