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
    "ssprk3": (1.437, 2.337, 2.240),
    "euler": (1.169, 1.337, 0.775),
}

#: Beyond the measured orders, and the two schemes extrapolate differently.
#:
#: ``ssprk3`` grows as ``K (2p+1)/(p+1)``.  In terms of the *step* rather than
#: the ``cfl`` number its stable coefficient falls as 1 : 0.54 : 0.31 across
#: ``p = 0, 1, 2``, which ``1/(p+1)`` (1 : 0.5 : 0.33) fits far better than
#: ``1/(2p+1)`` (1 : 0.33 : 0.2) -- so in ``cfl`` units the limit grows with
#: order by ``(2p+1)/(p+1)``.  ``K`` is the smallest the measurements support,
#: so the fallback is conservative where it is used.
#:
#: ``euler`` is held **flat** instead, below the smallest value measured for
#: it.  Its limit does not grow with order -- it *falls*, to 0.775 at
#: ``p = 2`` -- so the growth rule above would extrapolate it upward in
#: exactly the regime where it is already weakest.
STABILITY_FALLBACK: dict[str, tuple[str, float]] = {
    "ssprk3": ("grow", 1.32),
    "euler": ("flat", 0.70),
}

#: The ``euler`` row was measured the same way -- bisection on "the march
#: converged" -- but over one contour (``smooth``) at refinement 0 rather than
#: the four combinations behind the ``ssprk3`` row.  :data:`CFL_MARGIN` carries
#: the difference: across the ``ssprk3`` cases the most permissive contour sat
#: 6-10% above the tightest, which is well inside the 30% the margin holds back.
#:
#: **Euler's limit falls with order**, which ``ssprk3``'s does not: 1.169,
#: 1.337, then 0.775 at ``p = 2``, below where it started.  That is worth
#: knowing before reaching for ``euler`` at high order -- it is the cheap
#: scheme per step and the expensive one per unit of pseudo-time there.
#:
#: This row is also a caution about measuring stability cheaply.  Bisecting on
#: "still finite after 400 steps" instead gave (1.254, 1.963, 2.766) on this
#: contour -- 7% high at ``p = 0``, 47% high at ``p = 1``, and 3.6 times too
#: high at ``p = 2``, where the march stays finite for hundreds of steps and
#: then diverges.  A default built on those numbers diverged on its first run.
#: The cheap criterion is not a conservative approximation of the real one; it
#: is not an approximation of it at all.

#: Fraction of the measured stability limit that :func:`recommended_cfl` takes.
#: A scan over two contours at one back pressure is a measurement, not a
#: stability proof, so the shipped default keeps 30% in hand.
CFL_MARGIN = 0.7


def stability_limit(order: int, scheme: str = "ssprk3") -> float:
    """The largest ``cfl`` measured to converge at this order and scheme."""
    try:
        table = STABILITY_LIMIT[scheme]
    except KeyError:
        raise ValueError(
            f"unknown scheme {scheme!r}; choose from {tuple(STABILITY_LIMIT)}"
        ) from None
    if order < len(table):
        return table[order]
    rule, k = STABILITY_FALLBACK[scheme]
    return k if rule == "flat" else k * (2 * order + 1) / (order + 1)


def recommended_cfl(order: int, scheme: str = "ssprk3") -> float:
    """The ``cfl`` the solver uses when none is given.

    Order-dependent *because* ``cfl`` keeps its traditional meaning: one number
    cannot carry the same safety margin at every order, and the margin is the
    thing worth holding constant.  Under the textbook ``1/(2p+1)`` restriction
    ``cfl = 1`` sat at 62% of the stable step at ``p = 0`` and 38% at ``p = 1``,
    which is why the old fixed default left most of a factor of two unused above
    ``p = 0``.
    """
    return CFL_MARGIN * stability_limit(order, scheme)


def step_coefficient(order: int, cfl: float | None, scheme: str = "ssprk3") -> float:
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


def local_time_step(
    wave_sum, ops: Operators, order: int, cfl: float, scheme: str = "ssprk3", xp=np
):
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
