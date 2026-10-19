r"""**dgnozzle** — a 2D compressible Euler solver for nozzle design studies.

A discontinuous Galerkin solver for the two-dimensional Euler equations on a
planar nozzle, built so that students *use* it rather than write it: change the
geometry, sweep the design space, take sensitivities, explain the trends.

Quick start
-----------
>>> from src import solve_nozzle, performance          # doctest: +SKIP
>>> result = solve_nozzle(area_ratio=3.0, back_pressure_ratio=0.12, order=1)
>>> print(performance(result).summary())

Sweep a design variable:

>>> import numpy as np                                      # doctest: +SKIP
>>> from src import sweep
>>> table = sweep(area_ratio=np.linspace(2.0, 4.0, 9), order=1)
>>> print(table.table())

Take an exact shape derivative:

>>> from src import differentiable_case                 # doctest: +SKIP
>>> dc = differentiable_case(order=1, contour='bezier')
>>> value, grad, _ = dc.value_and_gradient('thrust',
...     names=('area_ratio', 'bezier_w1', 'bezier_w2'))

Layout
------
``src/hierarchy.md`` draws the dependency tree; the subpackages are

=============================  ==============================================
:mod:`~src.config`             the stagnation state, the gas, the mesh
                               resolution; the non-dimensionalisation is
                               documented here
:mod:`~src.reference`          the reference element: Lagrange bases on the
                               unit triangle and square, and the quadrature
                               rules that integrate them exactly
:mod:`~src.geometry`           the nozzle contour families, and the mesh
                               built inside them -- triangles or quads
:mod:`~src.physics`            the equations: thermodynamics, the Roe, HLLC
                               and AUSM+-up approximate Riemann solvers, the
                               boundary fluxes, and the exact quasi-1D theory
:mod:`~src.spatial`            the right-hand side: DG operators, the
                               vectorised residual, and the limiter
:mod:`~src.timestepping`       how big a step to take, where the march
                               starts, and the march itself
:mod:`~src.backends`           how the residual gets evaluated: vectorised
                               NumPy, compiled Numba, or traceable JAX
:mod:`~src.post`               thrust, entropy error, line-outs, and
                               matplotlib figures
:mod:`~src.external`           the plume and jet just downstream of the exit
                               plane, in closed form
:mod:`~src.api`                :func:`~src.api.solve_nozzle`, the one call
:mod:`~src.sweep`              parameter sweeps with warm starting
:mod:`~src.sensitivity`        discrete-adjoint design gradients
=============================  ==============================================

The solver core is geometry-agnostic: it takes any conforming mesh of triangles
or quadrilaterals with tagged boundaries.  Only :mod:`~src.geometry` and
:mod:`~src.physics.quasi1d` know that the domain is a nozzle.

See ``docs/theory.md`` for the formulation and the geometry definition, and
``README.md`` for how to run everything.
"""

from __future__ import annotations

__version__ = "1.0.0"

from .api import Case, build_case, case_from_grids, solve_nozzle
from .config import (
    BACKENDS,
    Discretization,
    FlowConditions,
    SolverOptions,
)
from .geometry.contour import CONTOURS, DESIGN_PARAMETERS, NozzleGeometry, check_contour
from .geometry.mesh import BoundaryTag, build_nozzle_mesh
from .physics.quasi1d import (
    Quasi1DSolution,
    Regime,
    critical_ratios,
    is_shock_free,
    operating_regime,
    shock_free_range,
    solve_quasi1d,
)
from .post.postprocess import (
    Performance,
    centreline_profile,
    exit_profile,
    inlet_profile,
    performance,
    sample_boundary,
    sample_field,
    wall_profile,
)
from .sweep import METRICS, SweepResult, sweep
from .timestepping.march import SolveResult, solve_steady

__all__ = [
    "__version__",
    # configuration
    "NozzleGeometry",
    "FlowConditions",
    "Discretization",
    "SolverOptions",
    "CONTOURS",
    "DESIGN_PARAMETERS",
    "BACKENDS",
    "BoundaryTag",
    "check_contour",
    # running
    "solve_nozzle",
    "solve_steady",
    "build_case",
    "case_from_grids",
    "build_nozzle_mesh",
    "Case",
    "SolveResult",
    # results
    "performance",
    "Performance",
    "exit_profile",
    "centreline_profile",
    "wall_profile",
    "inlet_profile",
    "sample_boundary",
    "sample_field",
    # theory
    "solve_quasi1d",
    "Quasi1DSolution",
    "critical_ratios",
    "is_shock_free",
    "operating_regime",
    "shock_free_range",
    "Regime",
    # studies
    "sweep",
    "SweepResult",
    "METRICS",
    "differentiable_case",
    "check_gradient",
]


def __getattr__(name: str):
    """Import the optional-dependency modules lazily.

    ``sensitivity`` needs JAX and ``plotting`` needs matplotlib.  Deferring them
    keeps ``import src`` working -- and fast -- when neither is installed.
    """
    if name in (
        "differentiable_case",
        "check_gradient",
        "finite_difference_gradient",
        "DifferentiableCase",
        "OBJECTIVES",
    ):
        from . import sensitivity

        return getattr(sensitivity, name)
    if name == "plotting":
        from .post import plotting

        return plotting
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
