r"""Slope and positivity limiters for the high-order solution.

Why a limiter is needed
-----------------------
At ``p = 0`` the solution is constant in each element and cannot oscillate.  At
``p >= 1`` it can, and near a shock it does: the polynomial overshoots, density
or pressure goes negative at a quadrature point, and the Roe flux hits a
negative Roe-averaged sound speed.  Aborting on that is the wrong response in a
pseudo-time march, where such an excursion is a transient to be controlled
rather than a fatal error -- and it would put every shocked operating point out
of reach at ``p >= 1``.

Two limiters are provided.

``positivity``
    Zhang-Shu scaling.  The deviation from the cell average is shrunk by the
    largest factor ``theta in [0, 1]`` that keeps density and pressure positive
    at *every* point the scheme evaluates -- the volume quadrature points and the
    element's own face quadrature points.  Because the cell average is untouched,
    the scheme stays **conservative**, and because ``theta = 1`` wherever the
    solution is already positive, smooth regions keep full accuracy.

``barth-jespersen``
    Adds monotonicity: the deviation is further shrunk so the solution stays
    within the range of the neighbouring cell averages.  This damps the
    oscillations themselves rather than just their consequences, at the cost of
    clipping genuine extrema (the classical BJ shortcoming).

Both act by scaling the deviation from the mean:

.. math::
    U^{lim}_{e,i} = \bar{U}_e + \theta_e\,(U_{e,i} - \bar{U}_e),
    \qquad \theta_e \in [0, 1]

For a Lagrange basis this is equivalent in coefficient space and in value space,
because the basis reproduces constants exactly.

Differentiability note
----------------------
``min``, ``max`` and ``clip`` make the limiter piecewise smooth.  Gradients
exist almost everywhere and JAX returns a valid one-sided derivative on the
kinks, but a shape derivative taken across a point where ``theta`` switches will
be noisy.  For gradient-based design work, prefer shock-free operating points
(``limiter='positivity'``, which is inactive on a smooth solution and therefore
exactly differentiable there).
"""

from __future__ import annotations

from typing import NamedTuple

import numpy as np

from .config import FlowConditions
from .operators import Operators


class LimiterDiagnostics(NamedTuple):
    """What the limiter had to do, for reporting."""

    n_limited: int
    min_theta: float
    min_density: float
    min_pressure: float


def cell_means(U, ops: Operators, xp=np):
    r"""Cell averages ``(nelem, 4)``."""
    return xp.einsum("ei,eis->es", ops.mean_weights, U)


def probe_values(U, ops: Operators, xp=np):
    """Solution at every point the scheme evaluates within an element.

    That is the volume quadrature points plus the element's own trace at each of
    its face quadrature points -- shaped ``(nelem, n_probe, 4)``.  Guaranteeing
    positivity at exactly this set is what makes the positivity limiter sound:
    no other point is ever fed to a flux function.
    """
    vol = xp.einsum("iq,eis->eqs", xp.asarray(ops.ref.phi_vol), U)
    face = xp.einsum("efiq,eis->efqs", ops.face_basis, U)
    nelem = vol.shape[0]
    face = face.reshape(nelem, -1, 4)
    return xp.concatenate([vol, face], axis=1)


def reference_floors(flow: FlowConditions, fraction: float = 1e-8) -> tuple[float, float]:
    """Absolute density and pressure floors, scaled to the reservoir state."""
    return fraction * flow.stagnation_density, fraction * flow.total_pressure


def floor_cell_means(U, ops: Operators, flow: FlowConditions, *, fraction: float = 1e-8, xp=np):
    """Last-resort repair of a cell whose *average* has become non-physical.

    Zhang-Shu scaling can only work if the cell average is admissible: at
    ``theta = 0`` the solution collapses to that average, so a negative mean
    density is unfixable by any scaling.  A negative mean means the time step was
    too large or the state was already broken.

    Rather than abort, the mean is minimally repaired: density is raised to the
    floor and energy raised to whatever makes the pressure reach its floor, with
    momentum untouched.  This is **not conservative**, so the solver counts these
    repairs and reports them; a run that needed them should be rerun with a
    smaller ``cfl``.
    """
    rho_floor, p_floor = reference_floors(flow, fraction)
    gamma = flow.gamma
    ubar = cell_means(U, ops, xp=xp)

    rho = ubar[..., 0]
    mx, my = ubar[..., 1], ubar[..., 2]
    rhoE = ubar[..., 3]

    rho_new = xp.maximum(rho, rho_floor)
    e_min = p_floor / (gamma - 1.0) + 0.5 * (mx * mx + my * my) / rho_new
    rhoE_new = xp.maximum(rhoE, e_min)

    if xp is np and not (np.any(rho < rho_floor) or np.any(rhoE < e_min)):
        return U  # nothing to repair, which is the normal case

    shift = xp.stack(
        [rho_new - rho, xp.zeros_like(mx), xp.zeros_like(my), rhoE_new - rhoE], axis=-1
    )
    # add the correction to the mean only: a constant shift in value space is a
    # constant shift of every Lagrange coefficient
    return U + shift[:, None, :]


def _pressure_at(u, gamma, xp):
    rho = xp.maximum(u[..., 0], 1e-300)
    return (gamma - 1.0) * (u[..., 3] - 0.5 * (u[..., 1] ** 2 + u[..., 2] ** 2) / rho)


def _min_pressure(ubar, dev, theta, gamma, xp):
    """Minimum pressure over probe points for a given scaling ``theta``."""
    u = ubar[:, None, :] + theta[:, None, None] * dev
    return _pressure_at(u, gamma, xp).min(axis=1)


def _solve_theta(ubar, probes, eps_rho, eps_p, gamma, steps, xp):
    r"""Largest ``theta in [0, 1]`` keeping density and pressure above their floors.

    Density is linear in ``theta`` so its bound is solved in closed form.
    Pressure is quadratic, so it is bracketed by bisection -- a fixed step count
    rather than a convergence test, which keeps the routine branch free and
    therefore ``jit``-able and differentiable.
    """
    dev = probes - ubar[:, None, :]
    rho_bar = ubar[..., 0]

    rho_min = probes[..., 0].min(axis=1)
    gap = rho_bar - rho_min
    theta_rho = xp.where(
        rho_min < eps_rho,
        xp.clip((rho_bar - eps_rho) / xp.where(gap > 0.0, gap, 1.0), 0.0, 1.0),
        1.0,
    )

    lo = xp.zeros_like(theta_rho)
    hi = theta_rho
    ok_at_hi = _min_pressure(ubar, dev, hi, gamma, xp) >= eps_p
    for _ in range(steps):
        mid = 0.5 * (lo + hi)
        good = _min_pressure(ubar, dev, mid, gamma, xp) >= eps_p
        lo = xp.where(good, mid, lo)
        hi = xp.where(good, hi, mid)
    return xp.where(ok_at_hi, theta_rho, lo)


def positivity_limiter(
    U,
    ops: Operators,
    flow: FlowConditions,
    *,
    fraction: float = 1e-8,
    bisection_steps: int = 12,
    xp=np,
):
    r"""Zhang-Shu positivity limiter.

    Returns the limited coefficients.  ``fraction`` sets the margin: density and
    pressure are held at or above ``fraction`` times their cell-average value,
    and never below an absolute floor scaled to the reservoir state, so the bound
    is both scale free and finite.

    Cost
    ----
    On a converged smooth solution *no* element violates positivity, and the
    expensive part -- the pressure bisection -- is skipped entirely.  Under NumPy
    the skip is a plain early return, and when only some elements are affected
    the bisection runs on that subset alone.  Under JAX, where shapes must stay
    static, the same skip is expressed as a ``lax.cond``.  Without this the
    limiter costs about twenty times the residual evaluation and dominates the
    whole solve.
    """
    if ops.ref.order == 0:
        return U  # a constant cannot overshoot its own mean

    gamma = flow.gamma
    rho_floor, p_floor = reference_floors(flow, fraction)

    U = floor_cell_means(U, ops, flow, fraction=fraction, xp=xp)
    ubar = cell_means(U, ops, xp=xp)
    probes = probe_values(U, ops, xp=xp)

    rho_bar = ubar[..., 0]
    p_bar = (gamma - 1.0) * (
        ubar[..., 3] - 0.5 * (ubar[..., 1] ** 2 + ubar[..., 2] ** 2) / rho_bar
    )
    eps_rho = xp.maximum(fraction * rho_bar, 0.5 * rho_floor)
    eps_p = xp.maximum(fraction * p_bar, 0.5 * p_floor)

    violates = (probes[..., 0].min(axis=1) < eps_rho) | (
        _pressure_at(probes, gamma, xp).min(axis=1) < eps_p
    )

    if xp is np:
        idx = np.flatnonzero(violates)
        if idx.size == 0:
            return U
        theta = np.ones(U.shape[0])
        theta[idx] = _solve_theta(
            ubar[idx], probes[idx], eps_rho[idx], eps_p[idx], gamma, bisection_steps, np
        )
    else:
        import jax

        theta = jax.lax.cond(
            violates.any(),
            lambda: _solve_theta(ubar, probes, eps_rho, eps_p, gamma, bisection_steps, xp),
            lambda: xp.ones(U.shape[0], dtype=U.dtype),
        )

    return ubar[:, None, :] + theta[:, None, None] * (U - ubar[:, None, :])


def barth_jespersen_limiter(U, ops: Operators, flow: FlowConditions, *, xp=np, tol: float = 1e-12):
    r"""Barth-Jespersen slope limiter on the cell-average neighbourhood.

    For each conserved variable, the deviation is scaled by

    .. math::
        \theta_e = \min_{k}\ \min\!\left(1,\
        \frac{\bar{U}^{\max} - \bar{U}_e}{U_e(x_k) - \bar{U}_e}\ \text{or}\
        \frac{\bar{U}^{\min} - \bar{U}_e}{U_e(x_k) - \bar{U}_e}\right)

    over the probe points ``x_k``, where the min/max run over the element and its
    face neighbours.  Unlike the classical formulation this scales *all* the
    higher modes, not just the linear part, so the same code works at any ``p``.
    """
    if ops.ref.order == 0:
        return U

    ubar = cell_means(U, ops, xp=xp)
    nb = ops.topology.edges.face_neighbour  # (nelem, nface); self on boundaries
    nb_means = ubar[nb]  # (nelem, nface, 4)
    u_max = xp.maximum(ubar, nb_means.max(axis=1))
    u_min = xp.minimum(ubar, nb_means.min(axis=1))

    probes = probe_values(U, ops, xp=xp)
    dev = probes - ubar[:, None, :]

    allowed_up = (u_max - ubar)[:, None, :]
    allowed_dn = (u_min - ubar)[:, None, :]
    ratio = xp.where(
        dev > tol,
        allowed_up / xp.where(dev > tol, dev, 1.0),
        xp.where(dev < -tol, allowed_dn / xp.where(dev < -tol, dev, 1.0), 1.0),
    )
    theta = xp.clip(ratio.min(axis=1), 0.0, 1.0)  # (nelem, 4)
    return ubar[:, None, :] + theta[:, None, :] * (U - ubar[:, None, :])


def apply_limiter(U, ops: Operators, flow: FlowConditions, kind: str, *, xp=np):
    """Dispatch to the requested limiter.  ``'none'`` returns ``U`` unchanged."""
    if kind == "none" or ops.ref.order == 0:
        return U
    if kind == "positivity":
        return positivity_limiter(U, ops, flow, xp=xp)
    if kind == "barth-jespersen":
        U = barth_jespersen_limiter(U, ops, flow, xp=xp)
        # positivity is still enforced afterwards: BJ bounds the range but does
        # not by itself guarantee a positive pressure at every probe point
        return positivity_limiter(U, ops, flow, xp=xp)
    raise ValueError(f"unknown limiter {kind!r}")


def diagnose(U, ops: Operators, flow: FlowConditions, xp=np) -> LimiterDiagnostics:
    """Minimum density and pressure over all probe points (unlimited)."""
    probes = probe_values(U, ops, xp=xp)
    rho = probes[..., 0]
    p = (flow.gamma - 1.0) * (
        probes[..., 3] - 0.5 * (probes[..., 1] ** 2 + probes[..., 2] ** 2)
        / xp.where(rho != 0.0, rho, 1.0)
    )
    return LimiterDiagnostics(
        n_limited=0,
        min_theta=1.0,
        min_density=float(rho.min()),
        min_pressure=float(p.min()),
    )
