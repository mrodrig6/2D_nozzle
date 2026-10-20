"""The pseudo-time step and its stability calibration.

``src.timestepping.step.STABILITY_LIMIT`` is measured, not derived, so it is the kind
of constant that can quietly stop being true when the flux, the quadrature or the
limiter changes.  These run in the fast suite for that reason.
"""

from __future__ import annotations

import numpy as np
import pytest

from src import NozzleGeometry
from src.config import SCHEMES, SolverOptions

pytestmark = pytest.mark.numba


@pytest.mark.parametrize("scheme", SCHEMES)
@pytest.mark.parametrize("order", [0, 1, 2])
def test_a_step_at_the_measured_limit_stays_finite(order, scheme):
    """:data:`~src.timestepping.step.STABILITY_LIMIT` must still be the limit.

    It is a measurement, not a derivation, so it can silently stop being true --
    a change to the flux, the quadrature or the limiter moves the real limit, and
    the default ``cfl`` would then be chasing it.  A few hundred steps right at
    the tabulated value is enough to catch one that has become optimistic: an
    unstable explicit march does not decay slowly, it overflows.
    """
    from src import build_case
    from src.backends import get_backend
    from src.timestepping import initialize
    from src.timestepping import step as ts

    case = build_case(NozzleGeometry(contour="smooth"), order=order)
    opts = SolverOptions(cfl=ts.stability_limit(order, scheme), scheme=scheme)
    bk = get_backend("numba", case.operators, case.flow, opts)
    U = initialize.initial_state(case.operators, case.flow, case.geometry, "quasi1d")
    U, res = bk.run(bk.asarray(U), 400, scheme)
    assert np.isfinite(np.asarray(U)).all()
    assert np.isfinite(res)


@pytest.mark.parametrize("scheme", SCHEMES)
def test_the_default_keeps_a_real_margin_at_every_order(scheme):
    """The default must be a genuine margin, not a rounding of the limit."""
    from src.timestepping import step as ts

    for order in range(4):
        limit = ts.stability_limit(order, scheme)
        assert ts.recommended_cfl(order, scheme) < 0.8 * limit


def test_cfl_keeps_its_traditional_meaning():
    """``cfl`` multiplies ``1/(2p+1)``, exactly as the textbook bound does.

    This is the property the default must not quietly break: a student who reads
    the formula in ``docs/theory.md`` and passes ``cfl=1`` has to get the step
    that formula describes, not something the calibration rescaled.
    """
    from src.timestepping import step as ts

    for order in range(5):
        for cfl in (0.25, 1.0, 2.0):
            assert ts.step_coefficient(order, cfl) == pytest.approx(cfl / (2 * order + 1))


def test_the_default_holds_the_margin_constant_instead_of_the_number():
    """The default rises with order because the stable ``cfl`` does.

    Under a *fixed* ``cfl`` the margin swings between 38% and 62%, which is the
    defect the order-aware default fixes while leaving the definition alone.
    """
    from src.timestepping import step as ts

    margins = [
        ts.recommended_cfl(p, s) / ts.stability_limit(p, s)
        for s in ("euler", "ssprk3")
        for p in range(4)
    ]
    assert all(abs(m - ts.CFL_MARGIN) < 1e-12 for m in margins)
    # and it is genuinely order-dependent, not a constant in disguise
    assert ts.recommended_cfl(1, "ssprk3") > 1.5 * ts.recommended_cfl(0, "ssprk3")


def test_an_unknown_scheme_has_no_calibration():
    from src.timestepping import step as ts

    with pytest.raises(ValueError, match="unknown scheme"):
        ts.stability_limit(1, "midpoint")
