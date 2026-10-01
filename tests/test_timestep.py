"""The pseudo-time step and its stability calibration.

``dgnozzle.assembly.STABILITY_LIMIT`` is measured, not derived, so it is the kind
of constant that can quietly stop being true when the flux, the quadrature or the
limiter changes.  These run in the fast suite for that reason.
"""

from __future__ import annotations

import numpy as np
import pytest

from dgnozzle import NozzleGeometry
from dgnozzle.config import SolverOptions

pytestmark = pytest.mark.numba


@pytest.mark.parametrize("scheme", ["rk4", "ssprk3"])
@pytest.mark.parametrize("order", [0, 1, 2])
def test_a_step_at_the_calibrated_limit_stays_finite(order, scheme):
    """``cfl = 1`` is meant to sit at the measured edge of stability.

    :data:`dgnozzle.assembly.STABILITY_LIMIT` is a measurement, not a derivation,
    so it can silently stop being true -- a change to the flux, the quadrature or
    the limiter moves the real limit.  A few hundred steps at ``cfl = 1`` is
    enough to catch a limit that has become optimistic: an unstable explicit
    march does not decay slowly, it overflows.
    """
    from dgnozzle import build_case, initialize
    from dgnozzle.backends import get_backend

    case = build_case(NozzleGeometry(contour="smooth"), order=order)
    opts = SolverOptions(cfl=1.0, scheme=scheme)
    bk = get_backend("numba", case.operators, case.flow, opts)
    U = initialize.initial_state(case.operators, case.flow, case.geometry, "quasi1d")
    U, res = bk.run(bk.asarray(U), 400, scheme)
    assert np.isfinite(np.asarray(U)).all()
    assert np.isfinite(res)


@pytest.mark.parametrize("scheme", ["rk4", "ssprk3"])
def test_the_calibration_keeps_the_default_inside_the_limit(scheme):
    """The shipped default must be a genuine margin, not a rounding of the limit."""
    from dgnozzle import assembly as asm
    from dgnozzle.config import DEFAULT_CFL

    for order in range(4):
        limit = asm.stability_limit(order, scheme)
        assert asm.step_coefficient(order, DEFAULT_CFL, scheme) < 0.8 * limit


def test_the_step_coefficient_is_order_calibrated():
    """One ``cfl`` must mean the same fraction of the stable step at every order.

    That is the whole point of the change: under the old ``1/(2p+1)`` scaling the
    same number was 62% of the limit at ``p = 0`` and 38% at ``p = 1``.
    """
    from dgnozzle import assembly as asm

    fractions = [
        asm.step_coefficient(p, 0.5, "rk4") / asm.stability_limit(p, "rk4")
        for p in range(4)
    ]
    assert all(abs(f - 0.5) < 1e-12 for f in fractions)


def test_an_unknown_scheme_has_no_calibration():
    from dgnozzle import assembly as asm

    with pytest.raises(ValueError, match="unknown scheme"):
        asm.stability_limit(1, "midpoint")
