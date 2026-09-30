"""User-facing configuration objects.

Three small frozen dataclasses carry everything a run needs:
:class:`FlowConditions` (the gas and the operating point),
:class:`Discretization` (the mesh and polynomial order) and
:class:`SolverOptions` (how the pseudo-time march is driven).  The nozzle shape
lives separately in :class:`dgnozzle.geometry.NozzleGeometry`.

Non-dimensionalisation
----------------------
The reference state is the inlet stagnation condition, with
``total_pressure = 1``, ``total_temperature = 1``, ``Rgas = 0.4`` and
``gamma = 1.4``.  That gives a stagnation speed of sound
``a_t = sqrt(gamma R T_t) = 0.7483`` and stagnation density
``rho_t = gamma p_t / a_t^2 = 2.5``.  These are the units of the original MATLAB
solver, so results are directly comparable.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import Any

BACKENDS = ("numba", "numpy", "jax")


@dataclass(frozen=True)
class FlowConditions:
    """Gas properties and the operating point.

    Parameters
    ----------
    gamma
        Ratio of specific heats.
    Rgas
        Specific gas constant in the solver's non-dimensional units.
    total_temperature, total_pressure
        Inlet reservoir (stagnation) conditions.
    back_pressure_ratio
        ``p_back / total_pressure``, the static pressure imposed at the exit
        plane whenever the outflow is locally subsonic.  This is what sets the
        operating point: large values keep the nozzle subsonic, small values
        produce a shock in the diverging section, and at the design value the
        nozzle runs shock-free.  See
        :func:`dgnozzle.quasi1d.operating_regime`.
    inflow_angle
        Inflow direction in radians, measured from the ``+x`` axis.
    entropy_fix
        Harten-Hyman entropy-fix parameter of the Roe flux, as a fraction of the
        Roe-averaged sound speed.
    """

    gamma: float = 1.4
    Rgas: float = 0.4
    total_temperature: float = 1.0
    total_pressure: float = 1.0
    back_pressure_ratio: float = 0.15
    inflow_angle: float = 0.0
    entropy_fix: float = 0.05

    def __post_init__(self) -> None:
        if self.gamma <= 1.0:
            raise ValueError(f"gamma must exceed 1, got {self.gamma}")
        if self.Rgas <= 0.0 or self.total_pressure <= 0.0 or self.total_temperature <= 0.0:
            raise ValueError("Rgas, total_pressure and total_temperature must be positive")
        if not 0.0 < self.back_pressure_ratio <= 1.0:
            raise ValueError(
                f"back_pressure_ratio must lie in (0, 1], got {self.back_pressure_ratio}"
            )
        if self.entropy_fix < 0.0:
            raise ValueError("entropy_fix must be non-negative")

    @property
    def back_pressure(self) -> float:
        """Static pressure imposed at a subsonic exit."""
        return self.back_pressure_ratio * self.total_pressure

    @property
    def stagnation_sound_speed(self) -> float:
        return math.sqrt(self.gamma * self.Rgas * self.total_temperature)

    @property
    def stagnation_density(self) -> float:
        return self.gamma * self.total_pressure / self.stagnation_sound_speed**2

    def replace(self, **kwargs: Any) -> FlowConditions:
        return replace(self, **kwargs)


@dataclass(frozen=True)
class Discretization:
    """Mesh resolution and polynomial orders.

    Parameters
    ----------
    element
        ``'tri'`` or ``'quad'``.
    order
        Solution polynomial order ``p``.  ``p = 0`` is a finite-volume scheme;
        ``p = 1`` and ``p = 2`` give second- and third-order accuracy.
    geometry_order
        ``Q``.  ``Q = 1`` uses straight-sided elements, so the curved wall is
        approximated by chords; ``Q = 2`` represents it quadratically and is
        roughly two orders of magnitude more accurate on the wall geometry for
        the same element count.
    refine
        Uniform refinement level.  Each level halves the element size, so the
        element count grows by 4.
    nx, nr
        Base element counts along and across the channel, before refinement.
    x_spacing
        ``'throat'`` clusters axial nodes around the throat (the default, and
        correct for any ``throat_x``); ``'legacy'`` reproduces the original
        MATLAB inlet-clustered distribution exactly; ``'uniform'`` is uniform.
    cluster_strength, cluster_width
        For ``x_spacing='throat'``: the far-field-to-throat spacing ratio, and
        the width of the clustered region as a fraction of the length.
    """

    element: str = "tri"
    order: int = 1
    geometry_order: int = 1
    refine: int = 0
    nx: int = 14
    nr: int = 5
    x_spacing: str = "throat"
    cluster_strength: float = 3.0
    cluster_width: float = 0.12

    def __post_init__(self) -> None:
        if self.element not in ("tri", "quad"):
            raise ValueError(f"element must be 'tri' or 'quad', got {self.element!r}")
        if self.order < 0:
            raise ValueError(f"order (p) must be non-negative, got {self.order}")
        if self.geometry_order < 1:
            raise ValueError(f"geometry_order (Q) must be >= 1, got {self.geometry_order}")
        if self.refine < 0:
            raise ValueError(f"refine must be non-negative, got {self.refine}")

    @property
    def n_elements(self) -> int:
        """Element count implied by this discretisation."""
        cells = self.nx * self.nr * 4**self.refine
        return cells * (2 if self.element == "tri" else 1)

    @property
    def n_basis(self) -> int:
        from . import elements as el

        return el.n_basis(self.element, self.order)

    @property
    def n_dof(self) -> int:
        """Degrees of freedom per conserved variable."""
        return self.n_elements * self.n_basis

    def replace(self, **kwargs: Any) -> Discretization:
        return replace(self, **kwargs)


@dataclass(frozen=True)
class SolverOptions:
    """Pseudo-time march control.

    Parameters
    ----------
    cfl
        Multiplier on the order-dependent stable step
        ``dt = cfl * 2 A_e / ((2p + 1) * sum_f s_f l_f)``.
    tolerance
        Convergence threshold on the residual norm, measured against the
        problem's own physical scale ``rho_t a_t / L`` rather than against the
        residual of the initial guess.  This matters: a *relative* criterion
        silently demands a tighter absolute residual the better the initial
        guess is, which makes a good starting field look slower than a bad one
        and makes runs with different initial conditions incomparable.
    max_iterations
        Hard cap.  Reaching it is reported, never silently ignored.
    scheme
        ``'rk4'`` (classical four-stage, the legacy choice) or ``'ssprk3'``
        (three-stage strong-stability-preserving, better behaved with limiters).
    limiter
        ``'none'``, ``'positivity'`` (Zhang-Shu scaling, cheap and enough to
        keep the march alive) or ``'barth-jespersen'`` (also damps shock
        oscillations).  Ignored at ``p = 0``, which cannot oscillate within an
        element.
    initial_condition
        ``'quasi1d'`` projects the quasi-one-dimensional solution for this
        geometry and back pressure.  Measured against a uniform start on the
        reference case it saves only about 15% of the iterations at ``p = 0``
        and 3% at ``p = 1`` -- but at ``p = 2`` without ``p_continuation`` the
        uniform start *diverges* while the quasi-1D start converges.  Its value
        is robustness, not speed.  ``'uniform'`` reproduces the legacy
        ``M = 0.95`` freestream start.
    p_continuation
        Solve at ``p = 0`` first and interpolate upward one order at a time.
        Each stage warm-starts the next, which is cheaper than starting the
        target order cold.
    check_interval
        Iterations between convergence tests.  Also the chunk size handed to the
        backend, so it is what the JAX backend fuses into one compiled loop.
        Small values waste dispatch; large values overshoot the tolerance.
    print_interval
        Iterations between progress lines; ``0`` silences them.
    divergence_factor
        Abort if the residual grows beyond this multiple of its initial value.
    stall_window, stall_ratio
        Stop early when the residual has changed by less than
        ``1 - stall_ratio`` over ``stall_window`` consecutive convergence checks.
        A limiter switching on and off parks the residual at a fixed level, and
        without this the run would spend its whole budget on a limit cycle.
    """

    cfl: float = 1.0
    tolerance: float = 1e-6
    max_iterations: int = 200_000
    scheme: str = "rk4"
    limiter: str = "positivity"
    initial_condition: str = "quasi1d"
    p_continuation: bool = True
    check_interval: int = 50
    print_interval: int = 500
    divergence_factor: float = 1e4
    stall_window: int = 40
    stall_ratio: float = 0.98

    def __post_init__(self) -> None:
        if self.cfl <= 0.0:
            raise ValueError(f"cfl must be positive, got {self.cfl}")
        if self.tolerance <= 0.0:
            raise ValueError(f"tolerance must be positive, got {self.tolerance}")
        if self.max_iterations < 1:
            raise ValueError("max_iterations must be at least 1")
        if self.check_interval < 1:
            raise ValueError("check_interval must be at least 1")
        if self.stall_window < 2:
            raise ValueError("stall_window must be at least 2")
        if not 0.0 < self.stall_ratio < 1.0:
            raise ValueError(f"stall_ratio must lie in (0, 1), got {self.stall_ratio}")
        if self.scheme not in ("rk4", "ssprk3"):
            raise ValueError(f"scheme must be 'rk4' or 'ssprk3', got {self.scheme!r}")
        if self.limiter not in ("none", "positivity", "barth-jespersen"):
            raise ValueError(
                f"limiter must be 'none', 'positivity' or 'barth-jespersen', got {self.limiter!r}"
            )
        if self.initial_condition not in ("quasi1d", "uniform"):
            raise ValueError(
                f"initial_condition must be 'quasi1d' or 'uniform', got {self.initial_condition!r}"
            )

    def replace(self, **kwargs: Any) -> SolverOptions:
        return replace(self, **kwargs)
