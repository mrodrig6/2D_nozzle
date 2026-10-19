r"""How big a step to take, and how converged the march is.

Separated from the residual because these answer a different question.
:mod:`src.spatial.assembly` says what :math:`M\,dU/dt` is; this module says how
far to move along it and when to stop.  Both the march and every backend need
them.

The step size carries the DG order explicitly.  A higher-order basis resolves
finer structure on the same element, so the explicit stability limit shrinks
with ``p`` -- the textbook factor is :math:`1/(2p+1)` -- and the measurements
below record what that factor actually is for this scheme rather than what the
textbook bound guarantees.
"""

from __future__ import annotations

import numpy as np

from ..config import FlowConditions
from ..physics import FLOOR
from ..spatial.operators import Operators

#: Largest ``cfl`` -- in the traditional sense below, so a multiplier on
#: :math:`1/(2p+1)` -- at which the march still converged, by polynomial order.
#: Measured by bisection to about 3%, over the ``bell`` and ``smooth`` contours
#: at refinement levels 0 and 1; the two contours agreed to within a bisection
#: step at every order, and the tighter refinement level is the one recorded.
#:
#: These are *limits*.  They are what makes the textbook ``1/(2p+1)``
#: restriction visibly over-conservative above ``p = 0``: ``cfl = 1`` sits at 62%
#: of the stable step at ``p = 0`` but only 38% at ``p = 1`` and 40% at
#: ``p = 2``, so the same ``cfl`` means a different safety margin at every order.
#: :data:`RECOMMENDED_CFL` is what the solver actually defaults to.
STABILITY_LIMIT: dict[str, tuple[float, ...]] = {
    "rk4": (1.625, 2.625, 2.500),
    "ssprk3": (1.437, 2.337, 2.240),
}

#: Beyond the measured orders, fall back to ``K (2p+1)/(p+1)``.  In terms of the
#: *step* rather than the ``cfl`` number, the stable coefficient falls as
#: 1 : 0.54 : 0.31 across ``p = 0, 1, 2``, which ``1/(p+1)`` (1 : 0.5 : 0.33)
#: fits far better than ``1/(2p+1)`` (1 : 0.33 : 0.2) -- so in ``cfl`` units the
#: limit *grows* with order by ``(2p+1)/(p+1)``.  ``K`` is the smallest the
#: measurements support, so the fallback is conservative where it is used.
#:
#: SSP-RK3 comes out at 0.885, 0.890 and 0.897 times RK4 at the three measured
#: orders -- near enough constant that the two schemes differ by a scheme factor
#: and not by their order dependence, which is a useful sanity check on both
#: columns.
STABILITY_FALLBACK: dict[str, float] = {"rk4": 1.5, "ssprk3": 1.32}

#: Fraction of the measured stability limit that :func:`recommended_cfl` takes.
#: A scan over two contours at one back pressure is a measurement, not a
#: stability proof, so the shipped default keeps 30% in hand.
CFL_MARGIN = 0.7


def stability_limit(order: int, scheme: str = "rk4") -> float:
    """The largest ``cfl`` measured to converge at this order and scheme."""
    try:
        table = STABILITY_LIMIT[scheme]
    except KeyError:
        raise ValueError(
            f"unknown scheme {scheme!r}; choose from {tuple(STABILITY_LIMIT)}"
        ) from None
    if order < len(table):
        return table[order]
    return STABILITY_FALLBACK[scheme] * (2 * order + 1) / (order + 1)


def recommended_cfl(order: int, scheme: str = "rk4") -> float:
    """The ``cfl`` the solver uses when none is given.

    Order-dependent *because* ``cfl`` keeps its traditional meaning: one number
    cannot carry the same safety margin at every order, and the margin is the
    thing worth holding constant.  Under the textbook ``1/(2p+1)`` restriction
    ``cfl = 1`` sat at 62% of the stable step at ``p = 0`` and 38% at ``p = 1``,
    which is why the old fixed default left most of a factor of two unused above
    ``p = 0``.
    """
    return CFL_MARGIN * stability_limit(order, scheme)


def step_coefficient(order: int, cfl: float | None, scheme: str = "rk4") -> float:
    r"""The scalar in :math:`\Delta t_e = c\,2 A_e / \sum_f s_f l_f`.

    ``cfl`` has its traditional meaning -- a multiplier on the order-dependent
    stable step,

    .. math::
        \Delta t_e = \frac{\mathrm{cfl}}{2p+1}\,
                     \frac{2 A_e}{\sum_f s_f \ell_f},

    with :math:`1/(2p+1)` the standard restriction for explicit DG.  ``None``
    means :func:`recommended_cfl`, which is where the measurements enter: the
    textbook restriction is over-conservative above ``p = 0``, so the default
    ``cfl`` rises with order to hold the *margin* constant at 30% instead of
    letting it swing between 38% and 62%.
    """
    if cfl is None:
        cfl = recommended_cfl(order, scheme)
    return cfl / (2 * order + 1)


def local_time_step(wave_sum, ops: Operators, order: int, cfl: float, scheme: str = "rk4", xp=np):
    r"""Element-local pseudo-time step.

    .. math::
        \Delta t_e = c(p, \text{scheme},\ \mathrm{cfl})\,
                     \frac{2\,A_e}{\sum_f s_f\, l_f}

    with :math:`c` from :func:`step_coefficient`.
    """
    denom = xp.maximum(wave_sum, FLOOR)
    return step_coefficient(order, cfl, scheme) * 2.0 * ops.elem_area / denom


def residual_scale(flow: FlowConditions, length: float) -> float:
    r"""The natural magnitude of :math:`\dot{U}` for this problem.

    ``R`` has the units of a rate of change of a conserved variable, so the
    reservoir state and the nozzle length give the scale
    :math:`\rho_t a_t / L`.  Dividing by it turns the residual into a
    dimensionless number that means the same thing on every mesh, at every
    order, and from every initial guess.

    This is the *mass* component's scale, and on its own it is not enough: the
    momentum and energy components carry extra powers of :math:`a_t`.  They are
    brought onto this same scale first, by
    :func:`~src.backends.base.component_weights`, and only then does dividing by
    one number mean anything.
    """
    return float(flow.stagnation_density * flow.stagnation_sound_speed / max(length, 1e-30))
