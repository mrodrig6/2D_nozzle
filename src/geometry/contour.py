"""Parameterised nozzle wall contours.

This is the *only* nozzle-specific part of the discretisation.  Everything in
``mesh``, ``operators``, ``physics`` and ``solver`` works on a generic mesh with
tagged boundaries and knows nothing about nozzles.

The nozzle is **planar** (a 2D channel of unit depth), so the one-dimensional
area ratio equals the height ratio::

    A(x) / A_throat = y_wall(x) / y_throat

Every function here is written against an injected array module ``xp`` (either
``numpy`` or ``jax.numpy``) and takes its design variables in a plain ``dict``.
That keeps the whole geometry pipeline differentiable, so ``jax.grad`` can be
taken straight through the contour, the mesh coordinates and the metric terms.

Contour families
----------------
``'bell'``     Cubic Hermite diverging section with prescribed initial and exit
               wall angles.  The default engineering contour.  A non-zero
               initial angle puts a **slope discontinuity** at the throat -- the
               sharp-corner expansion of classical minimum-length nozzle
               design, not a mistake.
``'smooth'``   ``'bell'`` with a zero initial wall angle, so the wall is C1
               across the throat.  Use this to study how throat smoothing
               changes the expansion fan.
``'conical'``  Straight diverging wall.
``'moc'``      Rao-type bell: ``'bell'`` with the classical 30 deg initial and
               0 deg exit wall angles.  Only monotone for long nozzles.
``'bezier'``   Cubic Bezier diverging section with two shape weights.  The most
               convenient family for gradient-based shape optimisation.
``'analytic'`` A fixed closed-form contour, smooth everywhere, kept as a
               verification geometry: it has no throat corner, so it reaches the
               scheme's design order of accuracy.  Ignores ``area_ratio`` and
               ``throat_x``.

The converging section is always a cubic Hermite from the inlet to the throat
with zero slope at both ends, so the throat really is a stationary point of
``y_wall`` and ``throat_x`` really is the throat location.
"""

from __future__ import annotations

import math
import warnings
from collections.abc import Mapping
from dataclasses import asdict, dataclass, replace
from typing import Any

import numpy as np

CONTOURS = ("bell", "smooth", "conical", "moc", "bezier", "analytic")

#: Design variables that ``jax.grad`` may be taken with respect to.
DESIGN_PARAMETERS = (
    "area_ratio",
    "throat_x",
    "throat_half_height",
    "inlet_half_height",
    "theta_initial",
    "theta_exit",
    "bezier_w1",
    "bezier_w2",
    "length",
)

#: Throat of the fixed analytic contour, located by Newton iteration below.
ANALYTIC_THROAT_X = 0.13880048
ANALYTIC_THROAT_Y = 0.13989434
ANALYTIC_INLET_Y = 0.15
ANALYTIC_EXIT_Y = 0.35
ANALYTIC_AREA_RATIO = ANALYTIC_EXIT_Y / ANALYTIC_THROAT_Y


# --------------------------------------------------------------------------
# The fixed analytic contour
# --------------------------------------------------------------------------
def analytic_contour(x, xp=np):
    r"""A closed-form bell contour on :math:`x \in [0, 1]`.

    .. math::
        t(x) &= \pi \ln\!\bigl(1 + (e^2 - 1)\, x\bigr) \\
        y(x) &= 0.01\,\bigl(4 - \cos\tfrac{t}{2}\bigr)
                \bigl(\cos t + 5 - \cos\tfrac{t}{2}\bigr)

    It runs from ``y = 0.15`` at the inlet through a throat of ``0.13989434`` at
    ``x = 0.138800`` to ``y = 0.35`` at the exit, an area ratio of 2.5019.
    """
    t = math.pi * xp.log1p((math.exp(2.0) - 1.0) * x)
    half = xp.cos(0.5 * t)
    return 0.01 * (4.0 - half) * (xp.cos(t) + 5.0 - half)


def _locate_analytic_throat() -> tuple[float, float]:
    """Newton-refine the minimum of the analytic contour (used for the constants)."""
    x = 0.14
    for _ in range(80):
        h = 1e-7
        d1 = (analytic_contour(x + h) - analytic_contour(x - h)) / (2 * h)
        d2 = (analytic_contour(x + h) - 2 * analytic_contour(x) + analytic_contour(x - h)) / h**2
        step = d1 / d2
        x -= step
        if abs(step) < 1e-14:
            break
    return float(x), float(analytic_contour(x))


# --------------------------------------------------------------------------
# Design variables
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class NozzleGeometry:
    """Design variables for a planar nozzle contour.

    Parameters
    ----------
    area_ratio
        Exit-to-throat area ratio ``y_exit / y_throat``.  Sets the design Mach
        number through the isentropic area relation.
    throat_x
        Throat location as a fraction of the nozzle length.
    throat_half_height
        Throat half-height in metres.  Scales the whole nozzle; the thrust
        coefficient is normalised by it, so changing it alone leaves the
        non-dimensional solution unchanged.
    inlet_half_height
        Inlet half-height in metres.  Must exceed ``throat_half_height``.
    length
        Axial length in metres.
    contour
        One of :data:`CONTOURS`.
    theta_initial_deg
        Wall angle immediately downstream of the throat, in degrees.  ``None``
        picks a monotone default (``atan(1.6 * mean slope)``).
    theta_exit_deg
        Wall angle at the exit plane, in degrees.  Zero gives axial exit flow.
    bezier_w1, bezier_w2
        Shape weights of the ``'bezier'`` contour.  ``(1/3, 2/3)`` reproduces
        the conical wall; larger ``w1`` expands earlier.
    """

    area_ratio: float = ANALYTIC_AREA_RATIO
    throat_x: float = ANALYTIC_THROAT_X
    throat_half_height: float = ANALYTIC_THROAT_Y
    inlet_half_height: float = ANALYTIC_INLET_Y
    length: float = 1.0
    contour: str = "bell"
    theta_initial_deg: float | None = None
    theta_exit_deg: float = 0.0
    bezier_w1: float = 0.55
    bezier_w2: float = 0.90

    def __post_init__(self) -> None:
        if self.contour not in CONTOURS:
            raise ValueError(f"unknown contour {self.contour!r}; choose from {CONTOURS}")
        if self.contour != "analytic":
            if not 0.0 < self.throat_x < 1.0:
                raise ValueError(f"throat_x must lie in (0, 1), got {self.throat_x}")
            if self.area_ratio <= 1.0:
                raise ValueError(f"area_ratio must exceed 1, got {self.area_ratio}")
            if self.inlet_half_height <= self.throat_half_height:
                raise ValueError(
                    "inlet_half_height must exceed throat_half_height "
                    f"({self.inlet_half_height} <= {self.throat_half_height})"
                )
        if self.throat_half_height <= 0.0:
            raise ValueError("throat_half_height must be positive")
        if self.length <= 0.0:
            raise ValueError("length must be positive")

    # -- design-variable plumbing -----------------------------------------
    def params(self) -> dict[str, float]:
        """The differentiable design variables as a flat ``dict``.

        Wall angles are returned in **radians** under the keys ``theta_initial``
        and ``theta_exit`` so that gradients are taken in consistent units.
        """
        return {
            "area_ratio": float(self.area_ratio),
            "throat_x": float(self.throat_x),
            "throat_half_height": float(self.throat_half_height),
            "inlet_half_height": float(self.inlet_half_height),
            "theta_initial": float(self._theta_initial_rad()),
            "theta_exit": math.radians(float(self.theta_exit_deg)),
            "bezier_w1": float(self.bezier_w1),
            "bezier_w2": float(self.bezier_w2),
            "length": float(self.length),
        }

    def with_params(self, params: Mapping[str, Any]) -> NozzleGeometry:
        """Rebuild the geometry from a (possibly partial) design-variable dict."""
        fields: dict[str, Any] = {}
        for key, value in params.items():
            if key not in DESIGN_PARAMETERS:
                raise ValueError(f"{key!r} is not a design parameter; use {DESIGN_PARAMETERS}")
            if key == "theta_initial":
                fields["theta_initial_deg"] = math.degrees(float(value))
            elif key == "length":
                fields["length"] = float(value)
            elif key == "theta_exit":
                fields["theta_exit_deg"] = math.degrees(float(value))
            else:
                fields[key] = float(value)
        return replace(self, **fields)

    def _theta_initial_rad(self) -> float:
        if self.contour == "smooth":
            return 0.0
        if self.theta_initial_deg is not None:
            return math.radians(float(self.theta_initial_deg))
        if self.contour == "moc":
            return math.radians(30.0)
        y_th = self.throat_half_height
        y_ex = y_th * self.area_ratio
        span = max(self.length * (1.0 - self.throat_x), 1e-12)
        return math.atan(1.6 * (y_ex - y_th) / span)

    def wall(self, x, xp=np):
        """Wall half-height at axial station(s) ``x``."""
        return wall_half_height(x, self.params(), self.contour, xp=xp)

    def throat_location(self) -> float:
        """Axial coordinate of the throat, in metres."""
        if self.contour == "analytic":
            return ANALYTIC_THROAT_X * self.length
        return float(self.throat_x) * self.length

    def throat_height(self) -> float:
        """Throat half-height, in metres."""
        if self.contour == "analytic":
            return ANALYTIC_THROAT_Y
        return float(self.throat_half_height)

    def realised_area_ratio(self) -> float:
        """Area ratio actually produced by the contour, ``y(L) / y_throat``."""
        return float(self.wall(np.asarray([self.length]))[0]) / self.throat_height()

    def describe(self) -> str:
        return (
            f"{self.contour} contour: AR={self.realised_area_ratio():.4f}, "
            f"throat at x={self.throat_location():.4f} m, "
            f"y_throat={self.throat_height():.6f} m, "
            f"y_exit={float(self.wall(np.asarray([self.length]))[0]):.6f} m"
        )

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


# --------------------------------------------------------------------------
# Contour evaluation
# --------------------------------------------------------------------------
def _hermite(s, y0, y1, m0, m1, xp):
    """Cubic Hermite through ``(0, y0)`` and ``(1, y1)`` with slopes ``m0, m1``."""
    s2 = s * s
    s3 = s2 * s
    h00 = 2.0 * s3 - 3.0 * s2 + 1.0
    h10 = s3 - 2.0 * s2 + s
    h01 = -2.0 * s3 + 3.0 * s2
    h11 = s3 - s2
    return h00 * y0 + h10 * m0 + h01 * y1 + h11 * m1


def wall_half_height(x, params: Mapping[str, Any], contour: str = "bell", xp=np):
    """Wall half-height ``y(x)`` for a contour family and design-variable dict.

    ``x`` may be any array the module ``xp`` understands.  All arithmetic goes
    through ``xp``, so passing ``jax.numpy`` and traced ``params`` yields a
    differentiable contour.
    """
    if contour not in CONTOURS:
        raise ValueError(f"unknown contour {contour!r}; choose from {CONTOURS}")

    x = xp.asarray(x)
    xi = x / params.get("length", 1.0)  # normalised axial coordinate in [0, 1]
    if contour == "analytic":
        return analytic_contour(xi, xp=xp)

    y_th = params["throat_half_height"]
    y_in = params["inlet_half_height"]
    y_ex = y_th * params["area_ratio"]
    x_th = params["throat_x"]

    # Converging section, normalised to [0, 1]; zero slope at inlet and throat.
    s_conv = xp.clip(xi / x_th, 0.0, 1.0)
    y_conv = _hermite(s_conv, y_in, y_th, 0.0, 0.0, xp)

    # Diverging section, normalised to [0, 1].  `span` is a fraction of the
    # length, so wall-angle slopes are converted with the physical dx/ds below.
    span = 1.0 - x_th
    s_div = xp.clip((xi - x_th) / span, 0.0, 1.0)
    dx_ds = span * params.get("length", 1.0)

    if contour == "conical":
        y_div = y_th + (y_ex - y_th) * s_div
    elif contour in ("bell", "smooth", "moc"):
        # theta_* are physical wall angles, so dy/ds = tan(theta) * dx/ds.
        m0 = xp.tan(params["theta_initial"]) * dx_ds
        m1 = xp.tan(params["theta_exit"]) * dx_ds
        y_div = _hermite(s_div, y_th, y_ex, m0, m1, xp)
    elif contour == "bezier":
        w1 = params["bezier_w1"]
        w2 = params["bezier_w2"]
        c1 = y_th + w1 * (y_ex - y_th)
        c2 = y_th + w2 * (y_ex - y_th)
        u = 1.0 - s_div
        y_div = u**3 * y_th + 3.0 * u**2 * s_div * c1 + 3.0 * u * s_div**2 * c2 + s_div**3 * y_ex
    else:  # pragma: no cover - guarded above
        raise AssertionError(contour)

    return xp.where(xi <= x_th, y_conv, y_div)


def check_contour(geom: NozzleGeometry, n: int = 401) -> dict[str, float]:
    """Sanity-check a contour and warn about shapes that will trouble the solver.

    Returns a small dictionary of diagnostics.  Warnings (never errors) are
    issued for a non-monotone diverging section or a throat that is not the
    global minimum, because both are legitimate — if unusual — designs.
    """
    x = np.linspace(0.0, geom.length, n)
    y = np.asarray(geom.wall(x))

    if np.any(y <= 0.0):
        raise ValueError(
            "contour has a non-positive wall height, which produces inverted "
            "elements; check area_ratio, theta_initial_deg and the Bezier weights"
        )

    i_min = int(np.argmin(y))
    x_throat = geom.throat_location()
    diverging = x >= x_throat
    dy = np.diff(y[diverging])
    monotone = bool(np.all(dy >= -1e-12))

    if not monotone:
        warnings.warn(
            "the diverging section is not monotone: the wall bulges and then "
            "contracts, so the flow will over-expand and re-compress. Reduce "
            "theta_initial_deg (or bezier_w1/w2) for a monotone bell.",
            stacklevel=2,
        )
    if abs(x[i_min] - x_throat) > 2.0 * geom.length / (n - 1):
        warnings.warn(
            f"the minimum wall height sits at x={x[i_min]:.4f} m but throat_x "
            f"implies x={x_throat:.4f} m; the reported area ratio and thrust "
            "normalisation use throat_x.",
            stacklevel=2,
        )

    eps = 1e-6 * geom.length
    slope_before = float(
        (
            geom.wall(np.asarray([x_throat - eps]))[0]
            - geom.wall(np.asarray([x_throat - 3 * eps]))[0]
        )
        / (2 * eps)
    )
    slope_after = float(
        (
            geom.wall(np.asarray([x_throat + 3 * eps]))[0]
            - geom.wall(np.asarray([x_throat + eps]))[0]
        )
        / (2 * eps)
    )

    return {
        "y_inlet": float(y[0]),
        "y_throat": float(y[i_min]),
        "x_throat": float(x[i_min]),
        "y_exit": float(y[-1]),
        "area_ratio": float(y[-1] / y[i_min]),
        "monotone_diverging": float(monotone),
        "throat_wall_angle_deg": float(np.degrees(np.arctan(slope_after))),
        "throat_slope_jump": float(slope_after - slope_before),
    }
