r"""The Euler equations: thermodynamics, the Riemann solvers, the boundaries.

Everything in ``thermo``, ``fluxes`` and ``boundary`` is a pure function of
states and normals.  Nothing in them knows about the mesh, the march, or which
backend is running, and every function takes an ``xp`` array module so the same
source runs under NumPy and under ``jax.numpy``.  That is what makes the JAX
backend the *same code* rather than a second implementation to keep in step.

================================  ===========================================
:mod:`~src.physics.thermo`        the state vector: primitive variables, the
                                  ideal-gas relations, the Euler flux itself
:mod:`~src.physics.fluxes`        the ``roe``, ``hllc`` and ``ausmup``
                                  approximate Riemann solvers
:mod:`~src.physics.boundary`      the wall, inflow and outflow fluxes
:mod:`~src.physics.quasi1d`       the reduced one-dimensional theory
================================  ===========================================

``quasi1d`` is the one module here that reads a geometry, because the area-Mach
relation is a statement about a particular contour.  It is physics all the same
-- the Euler equations under the assumption that nothing varies across the
duct -- and it serves three purposes: an exact reference to compare the DG
solution against, a starting field for the march, and the design relation that
says which area ratio gives which exit Mach number.
"""

from .boundary import inflow_flux, outflow_flux, wall_flux
from .fluxes import (
    AUSM_ALPHA,
    AUSM_BETA,
    AUSM_KP,
    AUSM_KU,
    ausm_up_flux,
    hllc_flux,
    roe_flux,
)
from .quasi1d import (
    CriticalRatios,
    Quasi1DSolution,
    Regime,
    area_over_throat,
    critical_ratios,
    density_ratio,
    is_shock_free,
    mach_from_area,
    mach_from_pressure_ratio,
    normal_shock_mach,
    normal_shock_static_pressure_ratio,
    normal_shock_total_pressure_ratio,
    operating_regime,
    pressure_ratio,
    shock_free_range,
    solve_quasi1d,
    temperature_ratio,
)
from .thermo import (
    FLOOR,
    FluxResult,
    euler_flux,
    mach_number,
    normal_flux,
    pressure,
    primitives,
    sound_speed,
)

__all__ = [
    "AUSM_ALPHA",
    "AUSM_BETA",
    "AUSM_KP",
    "AUSM_KU",
    "CriticalRatios",
    "FLOOR",
    "FluxResult",
    "Quasi1DSolution",
    "Regime",
    "area_over_throat",
    "ausm_up_flux",
    "critical_ratios",
    "density_ratio",
    "euler_flux",
    "hllc_flux",
    "inflow_flux",
    "is_shock_free",
    "mach_from_area",
    "mach_from_pressure_ratio",
    "mach_number",
    "normal_flux",
    "normal_shock_mach",
    "normal_shock_static_pressure_ratio",
    "normal_shock_total_pressure_ratio",
    "operating_regime",
    "outflow_flux",
    "pressure",
    "pressure_ratio",
    "primitives",
    "roe_flux",
    "shock_free_range",
    "solve_quasi1d",
    "sound_speed",
    "temperature_ratio",
    "wall_flux",
]
