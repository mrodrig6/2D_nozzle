r"""**dgnozzle** — a 2D compressible Euler solver for nozzle design studies.

A discontinuous Galerkin solver for the two-dimensional Euler equations on a
planar nozzle, built so that students *use* it rather than write it: change the
geometry, sweep the design space, take sensitivities, explain the trends.

Quick start
-----------
>>> from dgnozzle import solve_nozzle, performance          # doctest: +SKIP
>>> result = solve_nozzle(area_ratio=3.0, back_pressure_ratio=0.12, order=1)
>>> print(performance(result).summary())

Sweep a design variable:

>>> import numpy as np                                      # doctest: +SKIP
>>> from dgnozzle import sweep
>>> table = sweep(area_ratio=np.linspace(2.0, 4.0, 9), order=1)
>>> print(table.table())

Take an exact shape derivative:

>>> from dgnozzle import differentiable_case                 # doctest: +SKIP
>>> dc = differentiable_case(order=1, contour='bezier')
>>> value, grad, _ = dc.value_and_gradient('thrust',
...     names=('area_ratio', 'bezier_w1', 'bezier_w2'))

Layout
------
=========================  ==================================================
:mod:`~dgnozzle.geometry`  nozzle contour families (the only nozzle-aware part
                           of the discretisation)
:mod:`~dgnozzle.mesh`      topology and node coordinates, triangles or quads
:mod:`~dgnozzle.operators` metrics, mass matrices, normals
:mod:`~dgnozzle.physics`   Euler fluxes, Roe solver, boundary conditions
:mod:`~dgnozzle.assembly`  the vectorised DG residual
:mod:`~dgnozzle.backends`  ``numba`` / ``numpy`` / ``jax`` execution
:mod:`~dgnozzle.limiter`   positivity and slope limiting
:mod:`~dgnozzle.solver`    the pseudo-time march
:mod:`~dgnozzle.multigrid` FAS multigrid acceleration of that march (off by
                           default -- see the module docstring for what it is
                           and is not worth)
:mod:`~dgnozzle.quasi1d`   exact quasi-1D theory, for reference and for the
                           initial condition
:mod:`~dgnozzle.postprocess` thrust, entropy error, line-outs
:mod:`~dgnozzle.sweep`     parameter sweeps with warm starting
:mod:`~dgnozzle.sensitivity` discrete-adjoint design gradients
:mod:`~dgnozzle.plotting`  matplotlib figures
=========================  ==================================================

The solver core is geometry-agnostic: it takes any conforming mesh of triangles
or quadrilaterals with tagged boundaries.  Only :mod:`~dgnozzle.geometry` and the
mesh generator in :mod:`~dgnozzle.mesh` know that the domain is a nozzle.

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
from .geometry import CONTOURS, DESIGN_PARAMETERS, NozzleGeometry, check_contour
from .mesh import BoundaryTag, build_nozzle_mesh
from .postprocess import (
    Performance,
    centreline_profile,
    exit_profile,
    inlet_profile,
    performance,
    sample_boundary,
    sample_field,
    wall_profile,
)
from .quasi1d import (
    Quasi1DSolution,
    Regime,
    critical_ratios,
    operating_regime,
    solve_quasi1d,
)
from .solver import SolveResult, solve_steady
from .sweep import METRICS, SweepResult, sweep

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
    "operating_regime",
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
    keeps ``import dgnozzle`` working -- and fast -- when neither is installed.
    """
    if name in ("differentiable_case", "check_gradient", "finite_difference_gradient",
                "DifferentiableCase", "OBJECTIVES"):
        from . import sensitivity

        return getattr(sensitivity, name)
    if name == "plotting":
        from . import plotting

        return plotting
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
