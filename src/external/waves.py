r"""Closed-form oblique shock and Prandtl-Meyer relations.

These are the classical quasi-one-dimensional relations, used here to describe
the wave structure **outside** the exit plane from the converged exit state.
Nothing in this module touches the DG solver: it is post-processing, and it is
deliberately separate so that a failure here -- a turn past maximum deflection
has no solution -- cannot spoil an otherwise good flow solve.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import brentq

__all__ = [
    "mach_angle",
    "max_deflection_angle",
    "oblique_shock_angle",
    "prandtl_meyer",
    "prandtl_meyer_inverse",
    "pressure_ratio_across_oblique_shock",
]


def mach_angle(mach: float) -> float:
    r"""Mach angle :math:`\mu = \arcsin(1/M)`, in radians."""
    if mach < 1.0:
        raise ValueError(f"the Mach angle needs supersonic flow, got M={mach}")
    return float(np.arcsin(1.0 / mach))


def prandtl_meyer(mach: float, gamma: float = 1.4) -> float:
    r"""The Prandtl-Meyer function :math:`\nu(M)`, in radians.

    .. math::
        \nu(M) = \sqrt{\frac{\gamma+1}{\gamma-1}}
                 \arctan\sqrt{\frac{\gamma-1}{\gamma+1}(M^2-1)}
               - \arctan\sqrt{M^2-1}

    :math:`\nu` is the angle through which a sonic stream must turn to reach
    ``mach``, so the turn between two states is the difference of their
    :math:`\nu`.
    """
    if mach < 1.0:
        raise ValueError(f"Prandtl-Meyer needs supersonic flow, got M={mach}")
    k = np.sqrt((gamma + 1.0) / (gamma - 1.0))
    m2 = mach * mach - 1.0
    return float(k * np.arctan(np.sqrt(m2) / k) - np.arctan(np.sqrt(m2)))


def prandtl_meyer_inverse(nu: float, gamma: float = 1.4) -> float:
    r"""The Mach number with Prandtl-Meyer angle ``nu``.

    :math:`\nu` is strictly increasing in :math:`M`, so this is a bracketed
    root-find rather than an iteration that can wander.
    """
    if nu < 0.0:
        raise ValueError(f"nu must be non-negative, got {nu}")
    nu_max = prandtl_meyer(1e6, gamma)
    if nu >= nu_max:
        raise ValueError(
            f"nu={nu:.4f} rad is at or beyond the vacuum limit {nu_max:.4f}; "
            "the flow would have to expand to infinite Mach number"
        )
    return float(brentq(lambda m: prandtl_meyer(m, gamma) - nu, 1.0 + 1e-12, 1e6))


def pressure_ratio_across_oblique_shock(
    mach: float, beta: float, gamma: float = 1.4
) -> float:
    r"""Static pressure ratio across an oblique shock at wave angle ``beta``.

    Only the normal component matters, so this is the normal-shock relation at
    :math:`M_n = M\sin\beta`.
    """
    mn2 = (mach * np.sin(beta)) ** 2
    if mn2 < 1.0:
        raise ValueError(
            f"M_n = {np.sqrt(mn2):.4f} is subsonic; beta must exceed the Mach angle"
        )
    return float(1.0 + 2.0 * gamma / (gamma + 1.0) * (mn2 - 1.0))


def deflection_from_wave_angle(mach: float, beta: float, gamma: float = 1.4) -> float:
    r"""Flow deflection :math:`\theta` for a given wave angle, the theta-beta-M relation.

    .. math::
        \tan\theta = 2\cot\beta\,
            \frac{M^2\sin^2\beta - 1}{M^2(\gamma + \cos 2\beta) + 2}
    """
    s2 = (mach * np.sin(beta)) ** 2
    num = 2.0 / np.tan(beta) * (s2 - 1.0)
    den = mach * mach * (gamma + np.cos(2.0 * beta)) + 2.0
    return float(np.arctan2(num, den))


def max_deflection_angle(mach: float, gamma: float = 1.4) -> tuple[float, float]:
    r"""The largest deflection an attached oblique shock can turn, and its ``beta``.

    Past this the shock detaches and no attached-shock solution exists, which is
    the one case where :func:`oblique_shock_angle` legitimately has no answer.
    """
    mu = mach_angle(mach)
    betas = np.linspace(mu + 1e-9, 0.5 * np.pi - 1e-9, 4000)
    thetas = np.array([deflection_from_wave_angle(mach, b, gamma) for b in betas])
    i = int(np.argmax(thetas))
    return float(thetas[i]), float(betas[i])


def oblique_shock_angle(
    mach: float, pressure_ratio: float, gamma: float = 1.4, *, weak: bool = True
) -> float:
    r"""Wave angle :math:`\beta` of the oblique shock that raises pressure by ``pressure_ratio``.

    This is the over-expanded case: the jet leaves below ambient pressure and an
    oblique shock turns it inward until the two match.  Inverting the normal-shock
    pressure relation for :math:`M_n` is exact, so no root-find is needed:

    .. math::
        M_n^2 = 1 + \frac{\gamma+1}{2\gamma}\left(\frac{p_2}{p_1} - 1\right),
        \qquad \sin\beta = M_n / M .

    ``weak`` is accepted for symmetry with the usual formulation but does not
    change the answer: a *pressure* ratio fixes :math:`M_n` and so fixes
    :math:`\beta` uniquely.  The weak/strong ambiguity belongs to the
    *deflection*-angle problem, where two wave angles give the same turn.
    """
    if mach <= 1.0:
        raise ValueError(f"an oblique shock needs supersonic flow, got M={mach}")
    if pressure_ratio < 1.0:
        raise ValueError(
            f"an oblique shock raises pressure, so the ratio must be >= 1, "
            f"got {pressure_ratio:.4f}"
        )
    mn2 = 1.0 + (gamma + 1.0) / (2.0 * gamma) * (pressure_ratio - 1.0)
    sin_beta = np.sqrt(mn2) / mach
    if sin_beta > 1.0:
        raise ValueError(
            f"M={mach:.3f} cannot produce a pressure ratio of {pressure_ratio:.3f} "
            f"through an attached oblique shock: it needs M_n={np.sqrt(mn2):.3f} > M. "
            "The shock would detach into a normal shock standing off the lip."
        )
    return float(np.arcsin(sin_beta))
