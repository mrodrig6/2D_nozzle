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
``rho_t = gamma p_t / a_t^2 = 2.5``.  Every reported quantity is in these
units unless stated otherwise.
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
        Roe-averaged sound speed.  Read only when ``flux='roe'``.
    flux
        The interface flux between two element traces.

        ``'roe'``
            Roe's approximate Riemann solver with the Harten-Hyman entropy fix.
            The default, and what every number in the documentation was produced
            with.
        ``'hllc'``
            HLLC with Batten's wave speeds.  Three waves -- two acoustic and the
            contact -- instead of a full eigen-decomposition.  **Provably
            positivity-preserving** under a CFL condition, which the Roe flux is
            not, and which is the assumption the Zhang-Shu positivity limiter's
            theorem needs.  It also needs no entropy fix, because the HLL family
            cannot produce an expansion shock, so there is no constant to tune.
        ``'ausm'``
            Liou's AUSM+-up flux-vector splitting.  Not a Riemann solver at all:
            it splits the flux into a convective part carried by an interface
            mass flux and a pressure part, with no eigen-decomposition anywhere.
            Immune to the carbuncle the Roe flux admits, and more accurate as
            ``M -> 0``, which is where the inlet of this nozzle runs.  **Does not
            currently give a physical solution on this nozzle** -- see
            ``docs/theory.md``.

        The two agree to discretisation error on a smooth solution, so switching
        is a way to ask how much of an answer is the flux rather than the mesh.
        Neither one makes a shocked operating point converge.
    """

    gamma: float = 1.4
    Rgas: float = 0.4
    total_temperature: float = 1.0
    total_pressure: float = 1.0
    back_pressure_ratio: float = 0.15
    inflow_angle: float = 0.0
    entropy_fix: float = 0.05
    flux: str = "roe"
    ausm_cutoff_mach: float = 0.2
    hllc_low_mach: float = 0.0

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
        if self.flux not in ("roe", "hllc", "ausm"):
            raise ValueError(
                f"flux must be 'roe', 'hllc' or 'ausm', got {self.flux!r}"
            )
        if self.hllc_low_mach < 0.0:
            raise ValueError("hllc_low_mach must be non-negative")
        if not 0.0 <= self.ausm_cutoff_mach <= 1.0:
            raise ValueError(
                f"ausm_cutoff_mach must lie in [0, 1], got {self.ausm_cutoff_mach}"
            )

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
        correct for any ``throat_x``); ``'inlet'`` clusters them toward the inlet
        instead, which resolves a throat near ``x = 0`` but not one placed
        further downstream; ``'uniform'`` is uniform.
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
        Courant number in its usual sense: a multiplier on the order-dependent
        stable step,

        ``dt_e = cfl / (2p + 1) * 2 A_e / sum_f s_f l_f``

        with ``1/(2p+1)`` the standard restriction for explicit DG.

        ``None``, the default, means
        :func:`dgnozzle.assembly.recommended_cfl` -- 70% of the largest ``cfl``
        measured to converge at this order and scheme, so the *margin* is 30%
        whatever ``p`` is.  That matters because the textbook restriction is
        over-conservative above ``p = 0``: a fixed ``cfl = 1`` sits at 62% of the
        stable step at ``p = 0`` but only 38% at ``p = 1``, which left most of a
        factor of two unused.  :data:`dgnozzle.assembly.STABILITY_LIMIT` has the
        measured limits.
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
        ``'rk4'`` (classical four-stage) or ``'ssprk3'`` (three-stage
        strong-stability-preserving, better behaved with limiters).
    limiter
        ``'none'``, ``'positivity'`` (Zhang-Shu scaling, cheap and enough to
        keep the march alive) or ``'superbee'`` (Roe's TVD slope limiter with
        the Cockburn-Shu TVB threshold, which also damps shock oscillations).
        Ignored at ``p = 0``, which cannot oscillate within an element.
    tvb_constant
        The constant of the Cockburn-Shu TVB threshold, used by ``'superbee'``: a
        face whose increment is below ``tvb_constant * (A_e/A_Omega) * max|u_s|`` is left
        unlimited, which stops a TVD limiter toggling on and off at smooth
        extrema.  Only read when ``limiter='superbee'``.

        The default of ``50`` is measured, not guessed, and it is the *minimum*
        that works.  At a shock-free point the slope limiter has to be idle on
        the converged field or the residual cannot reach the tolerance: at
        ``tvb_constant = 0`` (the pure TVD limiter) 77 of 140 elements of an already
        converged field are still being clipped and the residual parks at
        ``3.1`` instead of ``10^-6``.  Scanned over the design, over-expanded
        and under-expanded points at ``(p, refine)`` of ``(1,0)``, ``(1,1)``
        and ``(2,0)``: ``10`` fails all nine, ``20`` fails the three at
        ``refine = 1``, and ``50`` converges all nine.

        Be aware of what that buys and what it costs.  At ``tvb_constant = 50`` the slope
        limiter is nearly inactive even *at a shock*: 200 steps into a shocked
        run it touches 0 of 140 elements at ``p_b/p_t = 0.50`` and 2 of 140 at
        ``0.70``, where ``tvb_constant = 0`` touches 72 and 92.  The window in which this
        limiter both leaves a smooth steady solution alone and still bites on a
        shock is, for this problem, empty.  That is a property of asking a TVD
        limiter to coexist with a steady pseudo-time march, not of Superbee:
        lower it and the limiter chatters and the residual parks; raise it and
        the limiter stops acting.  Shocked points do not converge at ``p >= 1``
        for this reason among others -- see the README.
    initial_condition
        ``'quasi1d'`` projects the quasi-one-dimensional solution for this
        geometry and back pressure.  Measured against a uniform start on the
        reference case it saves only about 15% of the iterations at ``p = 0``
        and 3% at ``p = 1`` -- but at ``p = 2`` without ``p_continuation`` the
        uniform start *diverges* while the quasi-1D start converges.  Its value
        is robustness, not speed.  ``'uniform'`` starts from a uniform
        ``M = 0.95`` freestream.
    p_continuation
        Solve at ``p = 0`` first and re-project upward one order at a time.

        Off by default, on measurement.  Re-projection is exact, so it never
        changes the answer -- but it is not cheaper: across ``p = 1`` and
        ``p = 2`` at two refinement levels it cost 2-28% more wall time than
        solving the target order directly, because the quasi-1D initial
        condition has already removed most of the transient that the ``p = 0``
        stage would otherwise remove.

        Turn it on as a *fallback*: it converges cases a direct high-order solve
        cannot start, such as ``p = 2`` from a uniform initial condition.
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

    cfl: float | None = None
    tolerance: float = 1e-6
    max_iterations: int = 200_000
    scheme: str = "rk4"
    limiter: str = "positivity"
    tvb_constant: float = 50.0
    initial_condition: str = "quasi1d"
    p_continuation: bool = False
    check_interval: int = 50
    print_interval: int = 500
    divergence_factor: float = 1e4
    stall_window: int = 40
    stall_ratio: float = 0.98

    def __post_init__(self) -> None:
        if self.cfl is not None and self.cfl <= 0.0:
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
        if self.tvb_constant < 0.0:
            raise ValueError(f"tvb_constant must be non-negative, got {self.tvb_constant}")
        if self.limiter not in ("none", "positivity", "superbee"):
            raise ValueError(
                f"limiter must be 'none', 'positivity' or 'superbee', got {self.limiter!r}"
            )
        if self.initial_condition not in ("quasi1d", "uniform"):
            raise ValueError(
                f"initial_condition must be 'quasi1d' or 'uniform', got {self.initial_condition!r}"
            )

    def replace(self, **kwargs: Any) -> SolverOptions:
        return replace(self, **kwargs)
