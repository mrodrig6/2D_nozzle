"""Fluxes and boundary conditions.

Consistency and conservation of the numerical flux are the two properties the
whole scheme rests on; each boundary condition is checked against a state it
should reproduce exactly.
"""

from __future__ import annotations

import numpy as np
import pytest

from dgnozzle import physics as ph

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
        out = ph.inflow_flux(U, np.array([-1.0]), np.array([0.0]), GAMMA,
                             Tt=1.0, pt=1.0, Rgas=0.4)
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
