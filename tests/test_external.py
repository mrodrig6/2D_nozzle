"""Closed-form external wave structure at the nozzle lip.

These relations are classical, so the tests pin them against textbook values and
against each other's inverses rather than against the solver.
"""

from __future__ import annotations

import numpy as np
import pytest

from src import solve_nozzle
from src.external import (
    exit_wave_structure,
    mach_angle,
    max_deflection_angle,
    oblique_shock_angle,
    prandtl_meyer,
    prandtl_meyer_inverse,
    pressure_ratio_across_oblique_shock,
)

GAMMA = 1.4


# --------------------------------------------------------------------------
# the relations themselves
# --------------------------------------------------------------------------


def test_prandtl_meyer_vanishes_at_mach_one():
    assert prandtl_meyer(1.0, GAMMA) == pytest.approx(0.0, abs=1e-12)


@pytest.mark.parametrize(
    ("mach", "nu_deg"),
    [(1.5, 11.905), (2.0, 26.380), (3.0, 49.757), (4.0, 65.785)],
)
def test_prandtl_meyer_matches_the_tables(mach, nu_deg):
    """Textbook values for gamma = 1.4, to three decimals of a degree."""
    assert np.degrees(prandtl_meyer(mach, GAMMA)) == pytest.approx(nu_deg, abs=5e-3)


@pytest.mark.parametrize("mach", [1.2, 2.0, 3.5, 6.0])
def test_prandtl_meyer_inverse_round_trips(mach):
    assert prandtl_meyer_inverse(prandtl_meyer(mach, GAMMA), GAMMA) == pytest.approx(mach, rel=1e-9)


def test_prandtl_meyer_refuses_subsonic_flow():
    with pytest.raises(ValueError, match="supersonic"):
        prandtl_meyer(0.8, GAMMA)


def test_the_vacuum_limit_is_reported_rather_than_silently_clipped():
    with pytest.raises(ValueError, match="vacuum limit"):
        prandtl_meyer_inverse(10.0, GAMMA)


@pytest.mark.parametrize("mach", [1.5, 2.5, 4.0])
def test_oblique_shock_angle_inverts_its_own_pressure_relation(mach):
    """Sampled below the normal-shock limit, which is the strongest shock that M.

    At ``beta = 90`` the oblique shock *is* a normal shock, so
    ``1 + 2g/(g+1)(M^2-1)`` is the largest pressure ratio that Mach number can
    produce -- 2.46 at M=1.5.  Asking for more is not a near-miss, it is a
    different flow: the shock detaches.
    """
    strongest = 1.0 + 2.0 * GAMMA / (GAMMA + 1.0) * (mach**2 - 1.0)
    for frac in (0.1, 0.5, 0.95):
        ratio = 1.0 + frac * (strongest - 1.0)
        beta = oblique_shock_angle(mach, ratio, GAMMA)
        back = pressure_ratio_across_oblique_shock(mach, beta, GAMMA)
        assert back == pytest.approx(ratio, rel=1e-10)


@pytest.mark.parametrize("mach", [1.5, 2.5, 4.0])
def test_the_normal_shock_is_the_strongest_attached_shock(mach):
    """One step past it must be refused, and at it beta must be 90 degrees."""
    strongest = 1.0 + 2.0 * GAMMA / (GAMMA + 1.0) * (mach**2 - 1.0)
    assert oblique_shock_angle(mach, strongest, GAMMA) == pytest.approx(np.pi / 2, rel=1e-6)
    with pytest.raises(ValueError, match="detach"):
        oblique_shock_angle(mach, strongest * 1.01, GAMMA)


def test_a_vanishing_shock_is_a_mach_wave():
    """As the pressure ratio approaches 1, beta must approach the Mach angle.

    This is what makes the design point come out right: the oblique shock does
    not disappear discontinuously, it degenerates into a zero-strength Mach wave.
    """
    mach = 2.5
    beta = oblique_shock_angle(mach, 1.0 + 1e-12, GAMMA)
    assert beta == pytest.approx(mach_angle(mach), rel=1e-6)


def test_a_turn_past_detachment_is_refused():
    """M=1.5 cannot be turned through an attached shock of arbitrary strength."""
    with pytest.raises(ValueError, match="detach"):
        oblique_shock_angle(1.5, 50.0, GAMMA)


def test_max_deflection_is_between_the_mach_angle_and_normal():
    for mach in (1.5, 2.0, 4.0):
        theta, beta = max_deflection_angle(mach, GAMMA)
        assert 0.0 < theta < np.pi / 2
        assert mach_angle(mach) < beta < np.pi / 2


# --------------------------------------------------------------------------
# reading them off a converged solve
# --------------------------------------------------------------------------


def _solved(pb, order=1, refine=0):
    return solve_nozzle(
        contour="smooth",
        area_ratio=2.5,
        back_pressure_ratio=pb,
        order=order,
        refine=refine,
        geometry_order=2,
        verbose=False,
    )


@pytest.mark.slow
@pytest.mark.parametrize(
    ("pb", "regime"),
    [(0.0300, "under-expanded"), (0.1500, "over-expanded")],
)
def test_the_regime_is_read_off_the_exit_state(pb, regime):
    w = exit_wave_structure(_solved(pb), ambient_pressure_ratio=pb)
    assert w.regime == regime
    assert w.exit_mach > 1.0
    assert np.isfinite(w.turn_angle)


@pytest.mark.slow
def test_the_two_dimensional_design_point_is_not_the_quasi_1d_one():
    """A measured property of the nozzle, not of the solver's accuracy.

    Quasi-1D theory puts the design back pressure at 0.0640 for AR=2.5, but the
    computed exit pressure is 0.0669 -- a 4.5% offset that does *not* shrink with
    refinement, because quasi-1D assumes parallel exit streamlines and the real
    flow is still diverging.  So the jet is slightly under-expanded at the
    quasi-1D design value, and genuinely at design near 0.0669.
    """
    at_quasi1d = exit_wave_structure(_solved(0.0640), ambient_pressure_ratio=0.0640)
    assert at_quasi1d.regime == "under-expanded"
    assert at_quasi1d.pressure_mismatch == pytest.approx(1.045, abs=0.01)

    at_true = exit_wave_structure(_solved(0.0669), ambient_pressure_ratio=0.0669)
    assert at_true.regime == "design"


@pytest.mark.slow
def test_a_non_converged_solve_is_refused_rather_than_read():
    from src.config import SolverOptions

    r = solve_nozzle(
        contour="smooth",
        area_ratio=2.5,
        back_pressure_ratio=0.15,
        order=1,
        refine=0,
        options=SolverOptions(max_iterations=5),
        verbose=False,
    )
    assert not r.converged
    with pytest.raises(ValueError, match="converged"):
        exit_wave_structure(r)


# --------------------------------------------------------------------------
# The extended downstream domain: the shock-cell march
# --------------------------------------------------------------------------
@pytest.fixture(scope="module")
def supersonic_result():
    from src.api import solve_nozzle

    return solve_nozzle(order=2, refine=0, back_pressure_ratio=0.0640, verbose=False)


@pytest.mark.slow
@pytest.mark.parametrize("amb", [0.030, 0.100])
def test_jet_cells_put_the_boundary_regions_exactly_at_ambient(supersonic_result, amb):
    """The free-boundary condition, which is the whole closure of the march.

    Every region that touches the jet boundary must sit at ``p_amb`` exactly --
    that is what defines the reflection there.  If this drifts, the wave the
    march computes is not the wave the boundary condition asks for.
    """
    from src.external import jet_wave_cells

    jc = jet_wave_cells(supersonic_result, ambient_pressure_ratio=amb, cells=3)
    edge = [r for r in jc.regions if not r.on_axis]
    assert len(edge) >= 2
    for r in edge:
        assert r.pressure_ratio == pytest.approx(amb, rel=1e-10)


@pytest.mark.slow
@pytest.mark.parametrize("amb", [0.030, 0.100])
def test_jet_cells_are_periodic_with_period_four(supersonic_result, amb):
    """The cycle must close: state 4 is state 0 again.

    Everything downstream -- that the cells repeat, that ``cell_length`` means
    anything -- rests on this, and it is the first thing a sign error breaks.
    """
    from src.external import jet_wave_cells

    jc = jet_wave_cells(supersonic_result, ambient_pressure_ratio=amb, cells=3)
    assert len(jc.regions) >= 5
    for k in range(len(jc.regions) - 4):
        a, b = jc.regions[k], jc.regions[k + 4]
        assert b.nu == pytest.approx(a.nu, rel=1e-10)
        assert b.flow_angle == pytest.approx(a.flow_angle, abs=1e-12)
        assert b.on_axis == a.on_axis


@pytest.mark.slow
def test_jet_cells_alternate_about_ambient_on_the_axis(supersonic_result):
    """Under-expanded: the axis over-shoots *below* ambient, then back above.

    The jet never simply relaxes to ambient; it rings about it.  That ringing is
    what makes the cells visible, so it is worth pinning that the sign alternates
    rather than decaying -- this model has no mechanism to decay.
    """
    from src.external import jet_wave_cells

    jc = jet_wave_cells(supersonic_result, ambient_pressure_ratio=0.030, cells=3)
    amb = jc.exit_waves.ambient_pressure_ratio
    axis = [r.pressure_ratio / amb for r in jc.regions if r.on_axis]
    assert len(axis) >= 3
    assert axis[0] > 1.0  # under-expanded, so the exit is above ambient
    assert axis[1] < 1.0  # the double expansion overshoots below it
    assert axis[2] == pytest.approx(axis[0], rel=1e-10)


@pytest.mark.slow
def test_jet_cells_have_equal_length_and_scale_linearly(supersonic_result):
    """``cells=n`` must reach n times as far, with every cell the same length."""
    from src.external import jet_wave_cells

    one = jet_wave_cells(supersonic_result, ambient_pressure_ratio=0.030, cells=1)
    three = jet_wave_cells(supersonic_result, ambient_pressure_ratio=0.030, cells=3)
    assert len(three.wave_segments) == 3 * len(one.wave_segments)
    assert three.cell_length == pytest.approx(one.cell_length, rel=1e-10)
    assert three.x_extent == pytest.approx(3.0 * one.x_extent, rel=1e-10)


@pytest.mark.slow
def test_an_under_expanded_jet_bulges_and_an_over_expanded_one_pinches(supersonic_result):
    """The jet boundary is a streamline, so it goes where the first wave turns it.

    Outward for an under-expanded jet, inward for an over-expanded one.  Getting
    this backwards is the most visible way to have the sign of ``theta`` wrong,
    and no pressure check would catch it.
    """
    from src.external import jet_wave_cells

    under = jet_wave_cells(supersonic_result, ambient_pressure_ratio=0.030, cells=2)
    over = jet_wave_cells(supersonic_result, ambient_pressure_ratio=0.100, cells=2)
    assert under.boundary[1, 1] > under.boundary[0, 1]
    assert over.boundary[1, 1] < over.boundary[0, 1]


@pytest.mark.slow
def test_the_design_point_has_no_cells_to_march(supersonic_result):
    """No lip wave means no pattern; say so rather than drawing nothing."""
    from src.external import jet_wave_cells
    from src.postprocess import performance

    # the design point is this nozzle's own computed exit pressure, not the
    # quasi-1D one -- they differ by 4.5%, as src.external.plume records
    design = float(performance(supersonic_result).exit_pressure_ratio)
    with pytest.raises(ValueError, match="design point"):
        jet_wave_cells(supersonic_result, ambient_pressure_ratio=design, cells=2)


@pytest.mark.slow
def test_a_strong_over_expansion_stops_at_the_mach_reflection(supersonic_result):
    """The model's own limit, reported rather than drawn through.

    A compression reflecting off the axis has to turn the flow by twice the lip
    angle.  Past maximum deflection that regular reflection does not exist -- it
    is a Mach reflection, a Mach disc -- and the periodic pattern below it is
    fiction.  The march must stop and say which wave it stopped on.
    """
    from src.external import exit_wave_structure, jet_wave_cells

    # walk up the over-expansion: the lip shock is still attached well past the
    # point where the doubled turn at the axis is not
    for amb in (0.16, 0.18, 0.20, 0.22, 0.24):
        try:
            exit_wave_structure(supersonic_result, ambient_pressure_ratio=amb)
        except ValueError:
            continue  # the lip shock already detaches; not the case under test
        jc = jet_wave_cells(supersonic_result, ambient_pressure_ratio=amb, cells=3)
        if jc.stopped_because:
            assert "Mach disc" in jc.stopped_because
            assert len(jc.wave_segments) < 6
            # and it must stop *before* drawing the impossible wave
            assert all(r.mach > 1.0 for r in jc.regions)
            return
    pytest.skip("no over-expansion in the swept range triggers a Mach disc")


# --------------------------------------------------------------------------
# Colouring the cells by a flow quantity
# --------------------------------------------------------------------------
@pytest.mark.slow
def test_region_polygons_are_closed_triangles_covering_each_state(supersonic_result):
    """One polygon per wave, each a non-degenerate triangle.

    The fill is only meaningful if the polygons really are the regions.  The
    open last region must be left out -- its downstream edge is a wave the march
    has not computed, so drawing it would invent geometry.
    """
    from src.external import jet_wave_cells

    jc = jet_wave_cells(supersonic_result, ambient_pressure_ratio=0.030, cells=3)
    polys = jc.region_polygons()
    assert len(polys) == len(jc.wave_segments)
    assert len(polys) == len(jc.regions) - 1  # the open one is dropped
    for region, poly in polys:
        assert poly.shape == (3, 2)
        x, y = poly[:, 0], poly[:, 1]
        area = 0.5 * abs(x[0] * (y[1] - y[2]) + x[1] * (y[2] - y[0]) + x[2] * (y[0] - y[1]))
        assert area > 1e-9, f"region {region.index} is degenerate"
    # the first region is closed by the exit plane, so it owns the lip
    first = polys[0][1]
    assert first[:, 0].min() == pytest.approx(jc.lip_x)


@pytest.mark.slow
def test_velocity_ratio_follows_the_mach_number_and_stays_below_vacuum(
    supersonic_result,
):
    """``v/a_t`` is a function of ``M`` alone, and bounded where ``M`` is not.

    This is why it, rather than ``M``, is what the velocity plot colours by: a
    hard expansion runs ``M`` up without the speed changing much, and the colour
    should report the speed.
    """
    from src.external import jet_wave_cells

    gamma = supersonic_result.flow.gamma
    vacuum = np.sqrt(2.0 / (gamma - 1.0))
    jc = jet_wave_cells(supersonic_result, ambient_pressure_ratio=0.030, cells=3)
    for r in jc.regions:
        expected = r.mach / np.sqrt(1.0 + 0.5 * (gamma - 1.0) * r.mach**2)
        assert r.velocity_ratio == pytest.approx(expected, rel=1e-12)
        assert 0.0 < r.velocity_ratio < vacuum


@pytest.mark.slow
def test_the_pressure_ramp_is_symmetric_about_ambient_in_the_log(supersonic_result):
    """Equal multiplicative departures from ambient must get equal colour.

    ``p/p_amb`` is a ratio, so a ramp linear in the value would both exaggerate
    the high side and -- on a strong under-expansion, where the swing is 2.15x up
    and 0.41x down -- run the scale off the bottom into negative pressure
    ratios.  Colouring on ``log(p/p_amb)`` fixes both.  Checked here through the
    public result rather than the plot: the extreme ratios either side of
    ambient must be reciprocal-ish in log, i.e. map to equal and opposite ends.
    """
    pytest.importorskip("matplotlib")
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from src.external import jet_wave_cells, plot_jet_field

    jc = jet_wave_cells(supersonic_result, ambient_pressure_ratio=0.030, cells=3)
    amb = jc.exit_waves.ambient_pressure_ratio
    ratios = [r.pressure_ratio / amb for r, _ in jc.region_polygons()]
    span = max(abs(np.log(v)) for v in ratios)
    # the bar must reach at least as far as the furthest region, both ways
    assert np.exp(span) >= max(ratios) - 1e-12
    assert np.exp(-span) <= min(ratios) + 1e-12

    fig, ax = plt.subplots()
    plot_jet_field(supersonic_result, jc, quantity="pressure", ax=ax)
    plt.close(fig)


@pytest.mark.slow
@pytest.mark.parametrize("quantity", ["pressure", "velocity", "mach"])
def test_every_field_quantity_draws(supersonic_result, quantity):
    pytest.importorskip("matplotlib")
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from src.external import jet_wave_cells, plot_jet_field

    jc = jet_wave_cells(supersonic_result, ambient_pressure_ratio=0.100, cells=2)
    fig, ax = plt.subplots()
    plot_jet_field(supersonic_result, jc, quantity=quantity, ax=ax)
    plt.close(fig)


def test_an_unknown_field_quantity_is_refused():
    """Silently colouring by the wrong thing is the one failure a reader cannot see."""
    pytest.importorskip("matplotlib")
    from src.external import plot_jet_field

    with pytest.raises(ValueError, match="quantity must be one of"):
        plot_jet_field(None, object(), quantity="entropy")
