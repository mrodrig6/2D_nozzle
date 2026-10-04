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

``superbee``
    A TVD slope limiter: the deviation is further shrunk so that the element's
    own increment toward each neighbour is no larger than Roe's Superbee limiter
    function allows, given the jump in cell averages across that face.  This
    damps the oscillations themselves rather than just their consequences.

    Superbee is the most *compressive* of the second-order TVD limiters -- at the
    steepening end it permits twice the neighbour difference where ``minmod``
    permits one -- which keeps a captured shock sharp but makes it the most
    willing to switch on.  For a steady march that is the wrong instinct, so the
    TVB modification of Cockburn and Shu is applied with it: an increment below a
    threshold that scales as :math:`h^2` is left unlimited, which is what keeps
    the limiter from toggling on and off in smooth regions and parking the
    residual in a limit cycle.  The threshold is the deciding parameter and it
    cuts both ways -- see :func:`superbee_limiter` and ``tvb_constant`` in
    :class:`~src.config.SolverOptions`.

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


def _pressure_at(U, gamma, xp):
    rho = xp.maximum(U[..., 0], 1e-300)
    return (gamma - 1.0) * (U[..., 3] - 0.5 * (U[..., 1] ** 2 + U[..., 2] ** 2) / rho)


def _min_pressure(ubar, dev, theta, gamma, xp):
    """Minimum pressure over probe points for a given scaling ``theta``."""
    U = ubar[:, None, :] + theta[:, None, None] * dev
    return _pressure_at(U, gamma, xp).min(axis=1)


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
    p_bar = (gamma - 1.0) * (ubar[..., 3] - 0.5 * (ubar[..., 1] ** 2 + ubar[..., 2] ** 2) / rho_bar)
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


def _minmod(a, b, xp=np):
    """``minmod(a, b)``: the smaller magnitude, or zero if the signs differ."""
    return xp.where(a * b <= 0.0, 0.0, xp.where(xp.abs(a) < xp.abs(b), a, b))


def _maxmod(a, b, xp=np):
    """``maxmod(a, b)``: the larger magnitude, or zero if the signs differ."""
    return xp.where(a * b <= 0.0, 0.0, xp.where(xp.abs(a) > xp.abs(b), a, b))


def superbee(a, b, xp=np):
    r"""Roe's Superbee limiter, two-argument form.

    .. math::
        \mathrm{superbee}(a, b) = \mathrm{maxmod}\bigl(
            \mathrm{minmod}(2a,\, b),\ \mathrm{minmod}(a,\, 2b)\bigr)

    Equivalent to the ratio form :math:`\phi(r) = \max(0, \min(2r, 1),
    \min(r, 2))` with :math:`r = a/b`, and preferable here because it needs no
    division and so has nothing to guard when ``b`` vanishes.

    Of the second-order TVD limiters this is the most compressive: it sits on the
    upper edge of Sweby's TVD region, where ``minmod`` sits on the lower one.
    """
    return _maxmod(_minmod(2.0 * a, b, xp), _minmod(a, 2.0 * b, xp), xp)


def face_trace_means(U, ops: Operators, xp=np):
    """The element's own trace, averaged over each of its faces.

    Shaped ``(nelem, nface, 4)``.  A slope limiter needs one increment per face
    rather than one per quadrature point, and the face average is the natural
    choice: it is what the jump in cell averages across that face is comparable
    with.
    """
    w = xp.asarray(ops.ref.w_face)
    total = w.sum()
    return xp.einsum("efiq,q,eis->efs", ops.face_basis, w, U) / total


def superbee_limiter(
    U,
    ops: Operators,
    flow: FlowConditions,
    *,
    tvb_constant: float = 50.0,
    xp=np,
    tol: float = 1e-12,
):
    r"""Superbee TVD slope limiter on the cell-average neighbourhood.

    For each element and each of its faces, the element's own increment toward
    that face, :math:`d_f = \bar{u}_f^{\text{trace}} - \bar{u}_e`, is compared
    with the jump in cell averages across it,
    :math:`a_f = \bar{u}_{n(f)} - \bar{u}_e`.  Superbee gives the largest
    increment that jump admits, and the deviation is scaled to respect the
    tightest face:

    .. math::
        \theta_e = \min_{f,\,s} \min\!\left(1,\
        \frac{\mathrm{superbee}(a_{f,s},\, d_{f,s})}{d_{f,s}}\right),
        \qquad
        \mathbf{U}^{\mathrm{lim}}_{e,i} = \bar{\mathbf{U}}_e
            + \theta_e\bigl(\mathbf{U}_{e,i} - \bar{\mathbf{U}}_e\bigr)

    As with the positivity limiter the cell average is untouched, so this is
    conservative.  Scaling *all* the higher modes rather than only the linear
    part is what lets one implementation serve any ``p``.

    Three details matter more than they look.

    **Boundary faces are not limited against.**  ``face_neighbour`` reports an
    element as its own neighbour on a boundary, so the jump there is identically
    zero and a naive formula would drive :math:`\theta_e` to zero -- flattening
    the solution in exactly the elements that carry the wall and the exit plane.

    **One scalar factor, not one per component.**  The minimum in
    :math:`\theta_e` runs over the four conserved components as well as the
    faces, so a single number scales the whole element.  That is not tidiness,
    it is admissibility.  The set
    :math:`\{\rho > 0,\ p > 0\}` is *convex*, so
    :math:`\bar{\mathbf{U}}_e + \theta(\mathbf{U}_{e,i} - \bar{\mathbf{U}}_e)`
    is a convex combination of two admissible states and is itself admissible
    for every :math:`\theta \in [0, 1]`.  A per-component factor leaves that
    line -- it pairs a strongly limited density with a barely limited energy --
    and the result can fall outside the set even though both endpoints are
    inside it.  Measured on the shocked case at ``cfl = 0.3``, a per-component
    factor drove the *cell average* non-physical 10,246 times and the march to
    ``NaN``; the scalar factor needed no repairs at all and kept the minimum
    pressure positive throughout.

    **The TVB threshold.**  Following Cockburn and Shu, a face whose increment is
    small enough to be smooth data rather than an oscillation is left alone:

    .. math::
        |d_{f,s}| \le K_{\mathrm{TVB}} \,\frac{A_e}{A_\Omega}\,
            \max_{e'} |\bar{u}_{e',s}|
        \quad\Longrightarrow\quad
        \text{no limiting of component } s \text{ across } f .

    Their threshold is :math:`K_{\mathrm{TVB}} h^2` with that constant a
    bound on a second
    derivative, so *dimensional*: the same value means different things at
    different nozzle scales and in different non-dimensionalisations, and a value
    that works becomes a value that stalls when the problem is rescaled.  The
    form above is that threshold made dimensionless.  The element's share of the
    domain area stands in for :math:`(h/L)^2`, so the threshold still shrinks as
    :math:`h^2` under refinement -- which is what makes the limiter vanish in the
    limit and recovers the TVB argument -- and the state scale is supplied by the
    largest cell average of that component *anywhere in the domain*.

    That last word is not decoration.  The obvious choice, the element's own
    :math:`|\bar{u}_e|`, is what a first attempt used, and it fails outright: the
    transverse momentum passes through zero along the nozzle axis, so the
    threshold vanishes on precisely the elements where the field is smoothest and
    the limiter can never switch off.  Measured, the residual stalled at
    :math:`2.6\times 10^{-2}` after 4000 iterations even at ``tvb_constant = 200``.  A global
    scale has no zero to fall into.

    Without a threshold at all (``tvb_constant = 0``, the pure TVD limiter) the
    limiter stays marginally active wherever the solution has a smooth extremum,
    clipping it on some iterations and not others.  That is not a small effect:
    measured at the shock-free design point it clips 77 of 140 elements of an
    already converged field and the residual parks at 3.1 instead of reaching
    :math:`10^{-6}`.  It is the same failure mode that made the Barth-Jespersen
    limiter this replaced worse than no slope limiter at all.

    The shipped default ``tvb_constant = 50`` is the smallest value that converges every
    shock-free point tested, and at that value the limiter is nearly inactive
    even at a shock.  :class:`~src.config.SolverOptions` documents that
    trade-off, which is the reason shocked points do not converge at ``p >= 1``.
    """
    if ops.ref.order == 0:
        return U

    ubar = cell_means(U, ops, xp=xp)
    nb = ops.topology.edges.face_neighbour  # (nelem, nface); self on a boundary
    interior = (nb != np.arange(ops.n_elem)[:, None])[..., None]  # (nelem, nface, 1)

    jump = ubar[nb] - ubar[:, None, :]  # (nelem, nface, 4)
    dev = face_trace_means(U, ops, xp=xp) - ubar[:, None, :]

    allowed = superbee(jump, dev, xp)
    safe = xp.where(xp.abs(dev) > tol, dev, 1.0)
    theta_face = xp.clip(allowed / safe, 0.0, 1.0)

    # a face that is not limiting: no increment to limit, a domain boundary, or
    # an increment below the TVB threshold
    area = xp.asarray(ops.elem_area)
    share = (area / area.sum())[:, None, None]  # stands in for (h/L)^2
    scale = xp.abs(ubar).max(axis=0)[None, None, :]  # global, per component
    threshold = tvb_constant * share * scale
    quiet = (xp.abs(dev) <= tol) | ~interior | (xp.abs(dev) <= threshold)
    theta_face = xp.where(quiet, 1.0, theta_face)

    # one scalar per element: the minimum over faces *and* components, so the
    # limited state stays on the segment between U and its own cell mean and
    # therefore inside the convex admissible set
    theta = theta_face.min(axis=(1, 2))  # (nelem,)
    return ubar[:, None, :] + theta[:, None, None] * (U - ubar[:, None, :])


def apply_limiter(
    U,
    ops: Operators,
    flow: FlowConditions,
    kind: str,
    *,
    tvb_constant: float = 50.0,
    xp=np,
):
    """Dispatch to the requested limiter.  ``'none'`` returns ``U`` unchanged."""
    if kind == "none" or ops.ref.order == 0:
        return U
    if kind == "positivity":
        return positivity_limiter(U, ops, flow, xp=xp)
    if kind == "superbee":
        U = superbee_limiter(U, ops, flow, tvb_constant=tvb_constant, xp=xp)
        # Positivity is still enforced afterwards.  A slope limiter bounds the
        # *range* of the solution by its neighbours' averages; it does not by
        # itself guarantee a positive pressure at every probe point, and the Roe
        # flux needs that.
        return positivity_limiter(U, ops, flow, xp=xp)
    raise ValueError(f"unknown limiter {kind!r}")


def diagnose(U, ops: Operators, flow: FlowConditions, xp=np) -> LimiterDiagnostics:
    """Minimum density and pressure over all probe points (unlimited)."""
    probes = probe_values(U, ops, xp=xp)
    rho = probes[..., 0]
    p = (flow.gamma - 1.0) * (
        probes[..., 3]
        - 0.5 * (probes[..., 1] ** 2 + probes[..., 2] ** 2) / xp.where(rho != 0.0, rho, 1.0)
    )
    return LimiterDiagnostics(
        n_limited=0,
        min_theta=1.0,
        min_density=float(rho.min()),
        min_pressure=float(p.min()),
    )
