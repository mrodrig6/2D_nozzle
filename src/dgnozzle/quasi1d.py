r"""Quasi-one-dimensional nozzle theory.

Two jobs:

1. **A reference solution.**  Students can compare the DG result against exact
   isentropic theory, including the location of a normal shock in the diverging
   section.  Differences are then attributable to two-dimensionality and
   discretisation error rather than to the unknown.
2. **An initial condition.**  Starting the pseudo-time march from the quasi-1D
   state instead of a uniform freestream saves a modest number of iterations
   (about 15% at ``p = 0``, 3% at ``p = 1`` on the reference case) and, more
   importantly, converges in cases where a uniform start diverges outright --
   ``p = 2`` without ``p``-continuation, for one.

Because the nozzle is planar, the one-dimensional area is the local channel
height, ``A(x) = 2 y_wall(x)``, and the area ratio is the height ratio.

Core relation
-------------
.. math::
    \frac{A}{A^*} = \frac{1}{M}
    \left[\frac{2}{\gamma+1}\left(1 + \frac{\gamma-1}{2}M^2\right)
    \right]^{\frac{\gamma+1}{2(\gamma-1)}}

which is monotone on each of the subsonic and supersonic branches, so it is
inverted here by bracketed bisection followed by Newton polish -- robust for
every area ratio, unlike a bare Newton iteration.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np

from .config import FlowConditions
from .geometry import NozzleGeometry


class Regime(str, Enum):
    """Nozzle operating regime, set by the back-pressure ratio."""

    SUBSONIC = "subsonic"
    """Not choked: subsonic throughout, mass flow still rising with pressure drop."""

    SHOCK_IN_NOZZLE = "shock-in-nozzle"
    """Choked, with a normal shock standing in the diverging section."""

    OVEREXPANDED = "overexpanded"
    """Shock-free in the nozzle; exit pressure below back pressure (oblique shocks outside)."""

    DESIGN = "design"
    """Shock-free and perfectly expanded: exit static pressure equals back pressure."""

    UNDEREXPANDED = "underexpanded"
    """Shock-free; exit pressure above back pressure (expansion fan outside)."""


# --------------------------------------------------------------------------
# Isentropic relations
# --------------------------------------------------------------------------
def area_over_throat(mach, gamma: float):
    r"""``A / A*`` from the Mach number."""
    M = np.asarray(mach, dtype=float)
    M = np.maximum(M, 1e-12)
    exponent = (gamma + 1.0) / (2.0 * (gamma - 1.0))
    return (1.0 / M) * (2.0 / (gamma + 1.0) * (1.0 + 0.5 * (gamma - 1.0) * M * M)) ** exponent


def pressure_ratio(mach, gamma: float):
    r"""``p / p_t`` from the Mach number (isentropic)."""
    M = np.asarray(mach, dtype=float)
    return (1.0 + 0.5 * (gamma - 1.0) * M * M) ** (-gamma / (gamma - 1.0))


def density_ratio(mach, gamma: float):
    M = np.asarray(mach, dtype=float)
    return (1.0 + 0.5 * (gamma - 1.0) * M * M) ** (-1.0 / (gamma - 1.0))


def temperature_ratio(mach, gamma: float):
    M = np.asarray(mach, dtype=float)
    return 1.0 / (1.0 + 0.5 * (gamma - 1.0) * M * M)


def mach_from_pressure_ratio(p_over_pt, gamma: float):
    r"""Invert ``p / p_t`` for the Mach number."""
    r = np.clip(np.asarray(p_over_pt, dtype=float), 1e-300, 1.0)
    return np.sqrt(2.0 / (gamma - 1.0) * (r ** (-(gamma - 1.0) / gamma) - 1.0))


def mach_from_area(area_ratio, gamma: float, supersonic: bool = False, tol: float = 1e-13):
    r"""Invert the area relation for the Mach number on the requested branch.

    ``area_ratio`` is ``A / A*`` and must be at least 1.  Values marginally below
    1 (which arise from round-off when a station sits exactly at the throat) are
    clipped to 1 rather than raising.
    """
    ar = np.atleast_1d(np.asarray(area_ratio, dtype=float))
    if np.any(ar < 1.0 - 1e-9):
        raise ValueError(f"A/A* must be >= 1; got a minimum of {float(ar.min())}")
    ar = np.maximum(ar, 1.0)

    if supersonic:
        lo = np.ones_like(ar)
        hi = np.full_like(ar, 1.2)
        # grow the bracket until it spans the target
        for _ in range(200):
            need = area_over_throat(hi, gamma) < ar
            if not np.any(need):
                break
            hi = np.where(need, hi * 1.5, hi)
    else:
        lo = np.full_like(ar, 1e-10)
        hi = np.ones_like(ar)

    # A/A* decreases with M below 1 and increases above 1
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        f = area_over_throat(mid, gamma)
        if supersonic:
            too_small = f < ar
            lo = np.where(too_small, mid, lo)
            hi = np.where(too_small, hi, mid)
        else:
            too_small = f < ar
            hi = np.where(too_small, mid, hi)
            lo = np.where(too_small, lo, mid)
        if np.all(hi - lo < tol):
            break
    out = 0.5 * (lo + hi)
    return out if np.ndim(area_ratio) else float(out[0])


# --------------------------------------------------------------------------
# Normal shock relations
# --------------------------------------------------------------------------
def normal_shock_mach(M1, gamma: float):
    """Downstream Mach number across a normal shock."""
    M1 = np.asarray(M1, dtype=float)
    return np.sqrt(
        (1.0 + 0.5 * (gamma - 1.0) * M1 * M1) / (gamma * M1 * M1 - 0.5 * (gamma - 1.0))
    )


def normal_shock_static_pressure_ratio(M1, gamma: float):
    """``p2 / p1`` across a normal shock."""
    M1 = np.asarray(M1, dtype=float)
    return (2.0 * gamma * M1 * M1 - (gamma - 1.0)) / (gamma + 1.0)


def normal_shock_total_pressure_ratio(M1, gamma: float):
    r"""``p_{t2} / p_{t1}`` across a normal shock (always below 1)."""
    M1 = np.asarray(M1, dtype=float)
    g = gamma
    term1 = ((g + 1.0) * 0.5 * M1 * M1 / (1.0 + 0.5 * (g - 1.0) * M1 * M1)) ** (g / (g - 1.0))
    term2 = (2.0 * g / (g + 1.0) * M1 * M1 - (g - 1.0) / (g + 1.0)) ** (1.0 / (g - 1.0))
    return term1 / term2


# --------------------------------------------------------------------------
# Critical pressure ratios and regime
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class CriticalRatios:
    """The three classical critical back-pressure ratios of a nozzle."""

    first: float
    """``p_b / p_t`` at which the throat just reaches ``M = 1`` (subsonic exit)."""
    second: float
    """``p_b / p_t`` with a normal shock standing exactly at the exit plane."""
    third: float
    """``p_b / p_t`` for shock-free supersonic flow (the design point)."""
    exit_mach_subsonic: float
    exit_mach_design: float

    def describe(self) -> str:
        return (
            f"first={self.first:.4f} (choking), second={self.second:.4f} "
            f"(shock at exit), third={self.third:.4f} (design, M_exit="
            f"{self.exit_mach_design:.3f})"
        )


def critical_ratios(area_ratio: float, gamma: float = 1.4) -> CriticalRatios:
    """Critical back-pressure ratios for a given exit-to-throat area ratio."""
    if area_ratio <= 1.0:
        raise ValueError(f"area_ratio must exceed 1, got {area_ratio}")
    m_sub = float(mach_from_area(area_ratio, gamma, supersonic=False))
    m_sup = float(mach_from_area(area_ratio, gamma, supersonic=True))
    first = float(pressure_ratio(m_sub, gamma))
    third = float(pressure_ratio(m_sup, gamma))
    second = third * float(normal_shock_static_pressure_ratio(m_sup, gamma))
    return CriticalRatios(
        first=first,
        second=second,
        third=third,
        exit_mach_subsonic=m_sub,
        exit_mach_design=m_sup,
    )


def operating_regime(area_ratio: float, back_pressure_ratio: float, gamma: float = 1.4):
    """Classify an operating point.  Returns ``(regime, critical_ratios)``."""
    crit = critical_ratios(area_ratio, gamma)
    pb = back_pressure_ratio
    if pb >= crit.first:
        regime = Regime.SUBSONIC
    elif pb > crit.second:
        regime = Regime.SHOCK_IN_NOZZLE
    elif pb > crit.third * (1.0 + 1e-9):
        regime = Regime.OVEREXPANDED
    elif pb < crit.third * (1.0 - 1e-9):
        regime = Regime.UNDEREXPANDED
    else:
        regime = Regime.DESIGN
    return regime, crit


# --------------------------------------------------------------------------
# The full quasi-1D solution
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class Quasi1DSolution:
    """Quasi-1D state along the nozzle axis."""

    x: np.ndarray
    area: np.ndarray
    mach: np.ndarray
    pressure: np.ndarray
    density: np.ndarray
    temperature: np.ndarray
    velocity: np.ndarray
    regime: Regime
    critical: CriticalRatios
    shock_x: float | None
    shock_mach: float | None
    mass_flow: float
    exit_mach: float
    exit_pressure: float

    def as_conserved(self, gamma: float) -> np.ndarray:
        """The state as ``(n, 4)`` conserved variables with ``v = 0``."""
        rho = self.density
        u = self.velocity
        rhoE = self.pressure / (gamma - 1.0) + 0.5 * rho * u * u
        return np.column_stack([rho, rho * u, np.zeros_like(rho), rhoE])

    def summary(self) -> str:
        shock = (
            "none"
            if self.shock_x is None
            else f"x = {self.shock_x:.4f} m (M1 = {self.shock_mach:.3f})"
        )
        return (
            f"quasi-1D: {self.regime.value}, shock {shock}, "
            f"M_exit = {self.exit_mach:.4f}, p_exit/p_t = {self.exit_pressure:.4f}, "
            f"mdot = {self.mass_flow:.6f}"
        )


def solve_quasi1d(
    geom: NozzleGeometry, flow: FlowConditions, n_points: int = 601
) -> Quasi1DSolution:
    """Solve the quasi-1D nozzle problem for this geometry and back pressure.

    Handles all three regimes, locating a normal shock in the diverging section
    by bisection on its position when one is present.
    """
    gamma = flow.gamma
    x = np.linspace(0.0, geom.length, n_points)
    y = np.asarray(geom.wall(x), dtype=float)
    area = 2.0 * y  # planar nozzle, unit depth

    # Evaluate the throat and exit areas from the *analytic* contour, never by
    # interpolating the sampled grid: linear interpolation across a minimum
    # overestimates the throat height, and the resulting error in A_exit/A_throat
    # is large enough to misclassify operating points near a critical ratio.
    x_throat = geom.throat_location()
    a_throat = 2.0 * geom.throat_height()
    a_exit = 2.0 * float(geom.wall(np.asarray([geom.length]))[0])
    ar_exit = a_exit / a_throat

    regime, crit = operating_regime(ar_exit, flow.back_pressure_ratio, gamma)
    diverging = x > x_throat

    pt = flow.total_pressure
    Tt = flow.total_temperature
    shock_x: float | None = None
    shock_mach: float | None = None

    if regime is Regime.SUBSONIC:
        # Not choked: A* follows from the exit pressure, subsonic everywhere.
        m_exit = float(mach_from_pressure_ratio(flow.back_pressure_ratio, gamma))
        a_star = a_exit / float(area_over_throat(m_exit, gamma))
        mach = mach_from_area(np.maximum(area / a_star, 1.0), gamma, supersonic=False)
        pt_local = np.full_like(x, pt)
    else:
        # Choked: A* = A_throat, subsonic ahead of the throat, supersonic behind.
        a_star = a_throat
        ratio = np.maximum(area / a_star, 1.0)
        mach = np.where(
            diverging,
            mach_from_area(ratio, gamma, supersonic=True),
            mach_from_area(ratio, gamma, supersonic=False),
        )
        pt_local = np.full_like(x, pt)

        if regime is Regime.SHOCK_IN_NOZZLE:
            def exit_pressure_for_shock(xs: float) -> float:
                a1 = 2.0 * float(geom.wall(np.asarray([xs]))[0])
                m1 = float(mach_from_area(a1 / a_star, gamma, supersonic=True))
                pt_ratio = float(normal_shock_total_pressure_ratio(m1, gamma))
                a_star2 = a_star / pt_ratio
                m_e = float(mach_from_area(a_exit / a_star2, gamma, supersonic=False))
                return float(pressure_ratio(m_e, gamma)) * pt_ratio

            lo, hi = x_throat + 1e-9 * geom.length, geom.length
            # p_exit falls monotonically as the shock moves downstream
            for _ in range(200):
                mid = 0.5 * (lo + hi)
                if exit_pressure_for_shock(mid) > flow.back_pressure_ratio:
                    lo = mid
                else:
                    hi = mid
                if hi - lo < 1e-12 * geom.length:
                    break
            shock_x = 0.5 * (lo + hi)

            a1 = 2.0 * float(geom.wall(np.asarray([shock_x]))[0])
            m1 = float(mach_from_area(a1 / a_star, gamma, supersonic=True))
            pt_ratio = float(normal_shock_total_pressure_ratio(m1, gamma))
            a_star2 = a_star / pt_ratio

            behind = x > shock_x
            mach = np.where(
                behind,
                mach_from_area(np.maximum(area / a_star2, 1.0), gamma, supersonic=False),
                mach,
            )
            pt_local = np.where(behind, pt * pt_ratio, pt)
            shock_mach = m1

    p = pt_local * pressure_ratio(mach, gamma)
    T = Tt * temperature_ratio(mach, gamma)
    rho = p / (flow.Rgas * T)
    a = np.sqrt(gamma * flow.Rgas * T)
    u = mach * a
    mdot = float(rho[0] * u[0] * area[0])

    return Quasi1DSolution(
        x=x,
        area=area,
        mach=mach,
        pressure=p,
        density=rho,
        temperature=T,
        velocity=u,
        regime=regime,
        critical=crit,
        shock_x=shock_x,
        shock_mach=shock_mach,
        mass_flow=mdot,
        exit_mach=float(mach[-1]),
        exit_pressure=float(p[-1] / pt),
    )
