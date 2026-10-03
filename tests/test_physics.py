"""Fluxes and boundary conditions.

Consistency and conservation of the numerical flux are the two properties the
whole scheme rests on; each boundary condition is checked against a state it
should reproduce exactly.
"""

from __future__ import annotations

import numpy as np
import pytest

from src import physics as ph

GAMMA = 1.4


@pytest.fixture
def states():
    return np.array([[1.2, 0.30, -0.10, 2.6], [0.8, 0.50, 0.20, 2.0], [2.0, 1.40, 0.05, 6.0]])


@pytest.fixture
def normals():
    n = np.array([[0.6, -0.8], [1.0, 0.0], [-0.28, 0.96]])
    return n[:, 0], n[:, 1]


def test_primitives_round_trip(states):
    rho, vx, vy, p, H = ph.primitives(states, GAMMA)
    rebuilt = np.stack(
        [rho, rho * vx, rho * vy, p / (GAMMA - 1) + 0.5 * rho * (vx * vx + vy * vy)], -1
    )
    assert np.allclose(rebuilt, states)
    assert np.allclose(H, (states[:, 3] + p) / rho)


def test_normal_flux_matches_the_projected_pair(states, normals):
    nx, ny = normals
    F, G = ph.euler_flux(states, GAMMA)
    assert np.allclose(ph.normal_flux(states, nx, ny, GAMMA), F * nx[:, None] + G * ny[:, None])


def test_roe_flux_is_consistent(states, normals):
    """F_hat(U, U, n) must equal the exact flux; otherwise the scheme is inconsistent."""
    nx, ny = normals
    out = ph.roe_flux(states, states, nx, ny, GAMMA)
    assert np.allclose(out.flux, ph.normal_flux(states, nx, ny, GAMMA))


def test_roe_flux_is_conservative(states, normals):
    """F_hat(L, R, n) = -F_hat(R, L, -n): what leaves one element enters the other."""
    nx, ny = normals
    L, R = states[:2], states[1:]
    a = ph.roe_flux(L, R, nx[:2], ny[:2], GAMMA).flux
    b = ph.roe_flux(R, L, -nx[:2], -ny[:2], GAMMA).flux
    assert np.allclose(a, -b)


def test_roe_survives_a_vacuum_like_state():
    """A near-vacuum state must give a finite flux, not abort the run."""
    bad = np.array([[1e-14, 0.0, 0.0, 1e-14]])
    good = np.array([[1.0, 0.3, 0.0, 2.5]])
    out = ph.roe_flux(bad, good, np.array([1.0]), np.array([0.0]), GAMMA)
    assert np.all(np.isfinite(out.flux))
    assert np.all(np.isfinite(out.max_speed))


def test_entropy_fix_bounds_eigenvalues_away_from_zero():
    """Without it, an unmodified Roe flux admits an expansion shock at a sonic point."""
    sonic = np.array([[1.0, 0.0, 0.0, 2.5]])
    nudged = np.array([[1.0000001, 1e-9, 0.0, 2.5]])
    fixed = ph.roe_flux(sonic, nudged, np.array([1.0]), np.array([0.0]), GAMMA, entropy_fix=0.05)
    assert np.all(np.isfinite(fixed.flux))


def test_wall_flux_transmits_pressure_only(states, normals):
    nx, ny = normals
    out = ph.wall_flux(states, nx, ny, GAMMA)
    assert np.allclose(out.flux[:, 0], 0.0)
    assert np.allclose(out.flux[:, 3], 0.0)
    # the momentum flux is p_b times the normal, so it is parallel to n
    cross = out.flux[:, 1] * ny - out.flux[:, 2] * nx
    assert np.allclose(cross, 0.0)


@pytest.mark.parametrize("mach", [0.02, 0.05, 0.2, 0.5, 0.9, 0.99])
def test_inflow_reproduces_a_uniform_isentropic_state(mach):
    """The state consistent with the reservoir must be returned unchanged."""
    Tt, pt, R = 1.0, 1.0, 0.4
    at2 = GAMMA * R * Tt
    rho_t = GAMMA * pt / at2
    fac = 1.0 + 0.5 * (GAMMA - 1) * mach**2
    a = np.sqrt(at2 / fac)
    q = mach * a
    rho = rho_t * fac ** (-1 / (GAMMA - 1))
    p = rho * a * a / GAMMA
    U = np.array([[rho, rho * q, 0.0, p / (GAMMA - 1) + 0.5 * rho * q * q]])
    nx, ny = np.array([-1.0]), np.array([0.0])  # outward at the inlet
    out = ph.inflow_flux(U, nx, ny, GAMMA, Tt=Tt, pt=pt, Rgas=R)
    assert np.allclose(out.flux, ph.normal_flux(U, nx, ny, GAMMA), atol=1e-12)


def test_inflow_picks_a_non_negative_mach_number():
    """Taking the larger root unconditionally returns a negative Mach number
    wherever the quadratic's leading coefficient changes sign."""
    rng = np.random.default_rng(0)
    for _ in range(200):
        rho = rng.uniform(0.2, 3.0)
        vx = rng.uniform(-0.9, 0.9)
        p = rng.uniform(0.05, 1.5)
        U = np.array([[rho, rho * vx, 0.0, p / (GAMMA - 1) + 0.5 * rho * vx * vx]])
        out = ph.inflow_flux(U, np.array([-1.0]), np.array([0.0]), GAMMA, Tt=1.0, pt=1.0, Rgas=0.4)
        assert np.all(np.isfinite(out.flux))
        # a physical inflow carries mass INTO the domain, i.e. flux[0] <= 0
        assert out.flux[0, 0] <= 1e-12


def test_outflow_extrapolates_when_supersonic():
    U = np.array([[1.0, 2.5, 0.0, 5.0]])
    nx, ny = np.array([1.0]), np.array([0.0])
    assert ph.mach_number(U, GAMMA)[0] > 1.0
    for pb in (0.02, 0.4, 0.9):
        out = ph.outflow_flux(U, nx, ny, GAMMA, p_back=pb)
        assert np.allclose(out.flux, ph.normal_flux(U, nx, ny, GAMMA))


def test_outflow_imposes_the_back_pressure_when_subsonic():
    """Extrapolating unconditionally would leave p_back with no effect at all."""
    U = np.array([[1.0, 0.3, 0.0, 2.0]])
    nx, ny = np.array([1.0]), np.array([0.0])
    assert ph.mach_number(U, GAMMA)[0] < 1.0
    fluxes = [ph.outflow_flux(U, nx, ny, GAMMA, p_back=pb).flux[0, 1] for pb in (0.3, 0.7)]
    assert fluxes[1] > fluxes[0]  # higher back pressure -> larger momentum flux


def test_outflow_is_continuous_across_the_sonic_point():
    """A shock crossing the exit during the transient must not jump the flux."""
    nx, ny = np.array([1.0]), np.array([0.0])
    rho, p = 1.0, 0.5
    a = np.sqrt(GAMMA * p / rho)
    prev = None
    for mach in np.linspace(0.90, 1.10, 41):
        vx = mach * a
        U = np.array([[rho, rho * vx, 0.0, p / (GAMMA - 1) + 0.5 * rho * vx * vx]])
        f = ph.outflow_flux(U, nx, ny, GAMMA, p_back=p).flux[0]
        if prev is not None:
            assert np.abs(f - prev).max() < 0.2
        prev = f


# --------------------------------------------------------------------------
# HLLC
# --------------------------------------------------------------------------


def test_hllc_flux_is_consistent(states, normals):
    nx, ny = normals
    out = ph.hllc_flux(states, states, nx, ny, GAMMA)
    assert np.allclose(out.flux, ph.normal_flux(states, nx, ny, GAMMA))


def test_hllc_flux_is_conservative(states, normals):
    nx, ny = normals
    L, R = states[:2], states[1:]
    a = ph.hllc_flux(L, R, nx[:2], ny[:2], GAMMA).flux
    b = ph.hllc_flux(R, L, -nx[:2], -ny[:2], GAMMA).flux
    assert np.allclose(a, -b)


@pytest.mark.parametrize("mach", [1.2, 3.0])
def test_hllc_is_fully_upwind_when_supersonic(mach):
    """Supersonic means one wave family, so the branch must be the exact flux."""
    rho, a = 1.0, 1.0
    p = rho * a * a / GAMMA
    UL = np.array([[rho, rho * mach * a, 0.0, p / (GAMMA - 1) + 0.5 * rho * (mach * a) ** 2]])
    UR = np.array([[0.6, 0.48 * mach * a, 0.1, 0.7 * p / (GAMMA - 1) + 0.2]])
    nx, ny = np.array([1.0]), np.array([0.0])
    got = ph.hllc_flux(UL, UR, nx, ny, GAMMA).flux
    assert np.allclose(got, ph.normal_flux(UL, nx, ny, GAMMA), rtol=0, atol=1e-14)


def test_hllc_keeps_the_star_density_positive():
    """The property the whole flux is chosen for.

    Batten's wave speeds make ``rho* = rho (S - v_n)/(S - S_M)`` positive by
    construction, which is what the Zhang-Shu limiter's theorem assumes of the
    underlying flux and what the Roe flux does not provide.  A strong expansion
    is where a flux that lacks it gives up.
    """
    rng = np.random.default_rng(17)
    for _ in range(500):
        # deliberately violent: large opposing velocities, big pressure ratio
        rho_l, rho_r = rng.uniform(1e-3, 2.0, 2)
        p_l, p_r = rng.uniform(1e-4, 2.0, 2)
        u_l, u_r = rng.uniform(-6.0, 6.0), rng.uniform(-6.0, 6.0)
        UL = np.array([[rho_l, rho_l * u_l, 0.0, p_l / (GAMMA - 1) + 0.5 * rho_l * u_l**2]])
        UR = np.array([[rho_r, rho_r * u_r, 0.0, p_r / (GAMMA - 1) + 0.5 * rho_r * u_r**2]])
        nx, ny = np.array([1.0]), np.array([0.0])
        out = ph.hllc_flux(UL, UR, nx, ny, GAMMA)
        assert np.all(np.isfinite(out.flux)), (rho_l, rho_r, p_l, p_r, u_l, u_r)
        assert np.all(out.max_speed > 0.0)


def test_hllc_low_mach_switch_recovers_standard_hllc(states, normals):
    """``phi = 1`` must be the identity, or the switch is not a switch."""
    nx, ny = normals
    L, R = states[:2], states[1:]
    plain = ph.hllc_flux(L, R, nx[:2], ny[:2], GAMMA)
    # a cutoff far below every local Mach number here gives phi = 1 everywhere
    tiny = ph.hllc_flux(L, R, nx[:2], ny[:2], GAMMA, low_mach=1e-12)
    assert np.allclose(plain.flux, tiny.flux, rtol=0, atol=1e-14)


def test_hllc_needs_no_entropy_fix():
    """A sonic point is where Roe needs its fix; HLLC must simply be smooth.

    The Roe flux admits a stationary expansion shock as an exact solution when
    an eigenvalue crosses zero, which is why ``entropy_fix`` exists.  HLLC
    cannot, so the flux has to vary smoothly straight through the sonic point --
    no constant, nothing to tune.
    """
    rho, a = 1.0, 1.0
    p = rho * a * a / GAMMA
    nx, ny = np.array([1.0]), np.array([0.0])

    def flux_at(mach):
        u = mach * a
        U = np.array([[rho, rho * u, 0.0, p / (GAMMA - 1) + 0.5 * rho * u * u]])
        nudge = np.array(
            [[rho * 1.01, rho * 1.01 * u, 0.0, p * 1.01 / (GAMMA - 1) + 0.5 * rho * 1.01 * u * u]]
        )
        return ph.hllc_flux(U, nudge, nx, ny, GAMMA).flux[0, 0]

    # sweep through M = 1 and check the mass flux has no kink
    machs = np.linspace(0.9, 1.1, 41)
    vals = np.array([flux_at(m) for m in machs])
    second = np.diff(vals, 2)
    assert np.all(np.abs(second) < 1e-3), "flux should pass smoothly through M=1"


# --------------------------------------------------------------------------
# Reverse flow at the exit plane
# --------------------------------------------------------------------------


def test_backflow_is_inert_where_the_flow_leaves():
    """With ``v_n > 0`` the branch must not touch the subsonic-outflow state."""
    rho, p = 1.0, 1.0 / GAMMA
    for mach in (0.3, 0.6, 0.9):
        a = np.sqrt(GAMMA * p / rho)
        u = mach * a
        U = np.array([[rho, rho * u, 0.0, p / (GAMMA - 1) + 0.5 * rho * u * u]])
        nx, ny = np.array([1.0]), np.array([0.0])
        off = ph.outflow_flux(U, nx, ny, GAMMA, p_back=0.6 * p)
        on = ph.outflow_flux(U, nx, ny, GAMMA, p_back=0.6 * p, rho_t=1.0, p_t=1.0)
        assert np.allclose(off.flux, on.flux), f"branch leaked at M={mach}"


def test_backflow_changes_the_flux_when_the_flow_enters():
    """And it must actually do something when ``v_n < 0``, or it is dead code."""
    rho, p = 1.0, 1.0 / GAMMA
    a = np.sqrt(GAMMA * p / rho)
    u = -0.25 * a  # reverse flow through the exit plane
    U = np.array([[rho, rho * u, 0.0, p / (GAMMA - 1) + 0.5 * rho * u * u]])
    nx, ny = np.array([1.0]), np.array([0.0])
    off = ph.outflow_flux(U, nx, ny, GAMMA, p_back=0.6 * p)
    on = ph.outflow_flux(U, nx, ny, GAMMA, p_back=0.6 * p, rho_t=1.0, p_t=1.0)
    assert not np.allclose(off.flux, on.flux)
    assert np.all(np.isfinite(on.flux))


def test_backflow_blends_continuously_through_zero_normal_velocity():
    """A hard switch here would park the residual; the blend must be smooth.

    The two subsonic branches genuinely disagree at ``v_n = 0`` -- one keeps the
    interior entropy and tangential velocity, the other imposes the reservoir's
    and zero -- so the blend is what keeps the boundary condition continuous as
    the solution moves across it.
    """
    rho, p = 1.0, 1.0 / GAMMA
    a = np.sqrt(GAMMA * p / rho)
    nx, ny = np.array([1.0]), np.array([0.0])

    def flux_at(mach):
        u = mach * a
        U = np.array(
            [[rho, rho * u, 0.3 * rho * a, p / (GAMMA - 1) + 0.5 * rho * (u * u + (0.3 * a) ** 2)]]
        )
        return ph.outflow_flux(U, nx, ny, GAMMA, p_back=0.6 * p, rho_t=1.0, p_t=1.0).flux[0]

    # continuity is tested by refinement, not by a threshold: halving the
    # sample spacing must halve the largest step.  A real switch would leave it
    # constant.  The flux is nonlinear in the blended state, so the variation is
    # steep near v_n = 0 without being discontinuous -- an absolute bound would
    # only measure that steepness.
    steps = []
    for n in (121, 241, 481):
        machs = np.linspace(-0.08, 0.13, n)
        vals = np.array([flux_at(m) for m in machs])
        steps.append(np.abs(np.diff(vals, axis=0)).max())
    for coarse, fine in zip(steps[:-1], steps[1:], strict=True):
        assert 1.8 < coarse / fine < 2.2, (
            f"step ratio {coarse / fine:.2f} is not first order; "
            "the branch is switching rather than blending"
        )


# --------------------------------------------------------------------------
# SLAU2
#
# The formulation was transcribed from the literature rather than from a copy
# of the paper (see the provenance note on ph.slau2_flux), so these tests carry
# more weight than usual: they are what stands between a transcription slip and
# a wrong answer.  Each one is a property the published scheme is claimed to
# have, chosen so that a mistyped coefficient would break at least one.
# --------------------------------------------------------------------------
def test_slau2_flux_is_consistent(states, normals):
    nx, ny = normals
    out = ph.slau2_flux(states, states, nx, ny, GAMMA)
    assert np.allclose(out.flux, ph.normal_flux(states, nx, ny, GAMMA))


def test_slau2_flux_is_conservative(states, normals):
    nx, ny = normals
    L, R = states[:2], states[1:]
    a = ph.slau2_flux(L, R, nx[:2], ny[:2], GAMMA).flux
    b = ph.slau2_flux(R, L, -nx[:2], -ny[:2], GAMMA).flux
    assert np.allclose(a, -b)


@pytest.mark.parametrize("vn", [0.0, 0.3, -0.3, 2.0])
def test_slau2_preserves_a_contact_discontinuity_exactly(vn):
    """Uniform pressure and velocity, jumping density: the upwind flux, exactly.

    This is the AUSM family's signature property and the reason the mass flux
    is weighted by density rather than averaged.  A sign error in the ``g``
    switch or in ``vn_mean`` shows up here and almost nowhere else.
    """
    p = 1.0

    def cons(rho):
        return np.array([[rho, rho * vn, 0.0, p / (GAMMA - 1) + 0.5 * rho * vn * vn]])

    UL, UR = cons(1.0), cons(5.0)
    nx, ny = np.array([1.0]), np.array([0.0])
    got = ph.slau2_flux(UL, UR, nx, ny, GAMMA).flux
    upwind = UL if vn > 0.0 else UR
    assert np.allclose(got, ph.normal_flux(upwind, nx, ny, GAMMA), rtol=0, atol=1e-14)


@pytest.mark.parametrize("mach", [1.2, 3.0])
def test_slau2_convects_from_the_upwind_side_when_supersonic(mach):
    """Supersonic: ``Psi`` and ``p_tilde`` come wholly from upwind.

    Note what is *not* asserted: that the flux equals the upwind flux.  SLAU2's
    mass flux dissipates on ``rho_R - rho_L`` at the mean fluid speed rather
    than upwinding each side separately, so at a supersonic *jump* ``mdot``
    differs from ``rho_L v_nL`` by a few percent -- by design, as the docstring
    derives.  Asserting the stronger property would be asserting a bug.
    """
    rho, a = 1.0, 1.0
    p = rho * a * a / GAMMA
    vL = mach * a
    vR = 0.95 * vL
    UL = np.array([[rho, rho * vL, 0.0, p / (GAMMA - 1) + 0.5 * rho * vL * vL]])
    UR = np.array([[0.6, 0.6 * vR, 0.06, 0.7 * p / (GAMMA - 1) + 0.5 * 0.6 * (vR * vR + 0.01)]])
    nx, ny = np.array([1.0]), np.array([0.0])

    # the premise, asserted rather than assumed: BOTH normal Mach numbers must
    # exceed 1 against the interface sound speed, or beta_L/beta_R do not
    # saturate and the flux is legitimately two-sided
    aL = ph.sound_speed(UL, GAMMA)
    aR = ph.sound_speed(UR, GAMMA)
    a_bar = 0.5 * (aL + aR)
    assert (vL / a_bar > 1.0).all() and (vR / a_bar > 1.0).all()

    got = ph.slau2_flux(UL, UR, nx, ny, GAMMA).flux

    _, vxL, vyL, pL, HL = ph.primitives(UL, GAMMA)
    mdot = got[:, 0]
    psi = np.stack([np.ones_like(vxL), vxL, vyL, HL], axis=-1)
    N = np.array([[0.0, 1.0, 0.0, 0.0]])
    assert np.allclose(got, mdot[:, None] * psi + pL[:, None] * N, rtol=0, atol=1e-13)


def test_slau2_pressure_dissipation_scales_as_mach_squared():
    """The low-Mach property, and the reason SLAU2 needs no cutoff Mach number.

    AUSM+-up buys this with ``K_p``, ``K_u`` and ``M_co``; SLAU2's ``chi`` and
    its ``|v|``-weighted pressure term deliver it from the states alone.  The
    measured ratio per halving of ``M`` is 4.00 by ``M = 0.025``.
    """
    nx, ny = np.array([1.0]), np.array([0.0])
    a = np.sqrt(GAMMA)
    ratios = []
    prev = None
    for mach in (0.2, 0.1, 0.05, 0.025):
        v = mach * a

        def cons(vel):
            return np.array([[1.0, vel, 0.0, 1.0 / (GAMMA - 1) + 0.5 * vel * vel]])

        UL, UR = cons(v), cons(v * 1.02)
        got = ph.slau2_flux(UL, UR, nx, ny, GAMMA).flux
        central = 0.5 * (ph.normal_flux(UL, nx, ny, GAMMA) + ph.normal_flux(UR, nx, ny, GAMMA))
        diss = abs(float(got[0, 1] - central[0, 1]))
        if prev is not None:
            ratios.append(prev / diss)
        prev = diss
    # O(M^2) means 4x per halving; the first interval is still feeling the
    # higher-order terms, the last is asymptotic
    assert ratios[-1] == pytest.approx(4.0, abs=0.05)
    assert all(r > 3.5 for r in ratios)


def test_slau2_is_sane_in_still_air_with_a_pressure_jump():
    """The configuration that exposed AUSM+-up, kept as a regression.

    A 1% pressure difference across a motionless interface gave AUSM+-up a mass
    flux of -9.8e4, because ``M_co`` had been read as an epsilon and floored at
    1e-8, making ``K_p / f_a`` diverge.  SLAU2 has no such constant.  The bar
    here is simply that it agrees with the two solvers that were never in doubt.
    """
    nx, ny = np.array([1.0]), np.array([0.0])

    def cons(p):
        return np.array([[1.0, 0.0, 0.0, p / (GAMMA - 1)]])

    for dp in (0.01, 0.1, 1.0):
        UL, UR = cons(1.0), cons(1.0 + dp)
        m_slau = float(ph.slau2_flux(UL, UR, nx, ny, GAMMA).flux[0, 0])
        m_roe = float(ph.roe_flux(UL, UR, nx, ny, GAMMA).flux[0, 0])
        m_hllc = float(ph.hllc_flux(UL, UR, nx, ny, GAMMA).flux[0, 0])
        # driven from high to low pressure, so the mass flux is negative
        assert m_slau < 0.0
        assert m_slau == pytest.approx(m_roe, rel=0.1)
        assert m_slau == pytest.approx(m_hllc, rel=0.35)
