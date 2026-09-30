r"""Initial conditions for the pseudo-time march.

The march converges to the same steady state from any admissible start, so the
initial condition affects cost and robustness, not the answer.

Starting from the quasi-one-dimensional solution for the actual geometry and back
pressure is worth doing, but the honest accounting is narrower than it might
seem.  On the reference case it saves about 15% of the iterations at ``p = 0``
and 3% at ``p = 1``.  Where it earns its place is robustness: at ``p = 2``
without ``p``-continuation a uniform ``M = 0.95`` start -- what the legacy code
used -- diverges, its residual growing by four orders of magnitude, while the
quasi-1D start converges cleanly.

Note also that the saving is only visible because convergence is judged against
a *fixed physical scale* rather than against the first iteration's residual.  A
relative criterion demands a tighter absolute residual the better the initial
guess is, which makes a good starting field look slower than a bad one.

Projection
----------
An initial field is set by :math:`L^2` projection, not by interpolation:

.. math::
    U_e = M_e^{-1} \int_{\Omega_e} \phi_i\, u_0(\vec{x})\, d\Omega

This is the optimal representation of ``u_0`` in the discrete space and it is
exact whenever ``u_0`` lies in that space, which makes it the right tool both
here and for order-to-order continuation.
"""

from __future__ import annotations

import numpy as np

from .config import FlowConditions
from .geometry import NozzleGeometry
from .operators import Operators
from .quasi1d import Quasi1DSolution, solve_quasi1d


def project(values, ops: Operators, xp=np):
    r"""Project pointwise values at the volume quadrature points onto the basis.

    ``values`` has shape ``(nelem, n_qvol, 4)``.
    """
    rhs = xp.einsum("iq,eq,eqs->eis", xp.asarray(ops.ref.phi_vol), ops.weighted_det, values)
    return xp.einsum("eij,ejs->eis", ops.inv_mass, rhs)


def uniform_state(flow: FlowConditions, mach: float = 0.95) -> np.ndarray:
    """A single conserved state at the given Mach number, isentropic from the reservoir."""
    g = flow.gamma
    fac = 1.0 + 0.5 * (g - 1.0) * mach * mach
    p = flow.total_pressure * fac ** (-g / (g - 1.0))
    rho = flow.stagnation_density * fac ** (-1.0 / (g - 1.0))
    a = np.sqrt(g * p / rho)
    u = mach * a
    return np.array([rho, rho * u, 0.0, p / (g - 1.0) + 0.5 * rho * u * u])


def uniform_initial(ops: Operators, flow: FlowConditions, mach: float = 0.95, xp=np):
    """Project a uniform freestream.  Reproduces the legacy initial condition."""
    state = uniform_state(flow, mach)
    values = xp.broadcast_to(xp.asarray(state), (ops.n_elem, ops.ref.n_qvol, 4))
    return project(values, ops, xp=xp)


def quasi1d_initial(
    ops: Operators,
    flow: FlowConditions,
    geom: NozzleGeometry,
    *,
    solution: Quasi1DSolution | None = None,
    xp=np,
):
    """Project the quasi-1D solution, with a transverse velocity from the wall slope.

    The axial state is taken from quasi-1D theory at each quadrature point's
    ``x``.  The transverse velocity is set to ``v = u (y / y_wall) dy_wall/dx``,
    the leading-order streamline slope for a slowly diverging channel, which
    gives the march a start that already satisfies the wall condition
    approximately.
    """
    sol = solution if solution is not None else solve_quasi1d(geom, flow)
    xy = np.asarray(ops.xy_vol)
    x = xy[..., 0]
    y = xy[..., 1]

    rho = np.interp(x, sol.x, sol.density)
    u = np.interp(x, sol.x, sol.velocity)
    p = np.interp(x, sol.x, sol.pressure)

    y_wall = np.asarray(geom.wall(x))
    h = 1e-6 * geom.length
    slope = (np.asarray(geom.wall(x + h)) - np.asarray(geom.wall(np.maximum(x - h, 0.0)))) / (
        h + np.minimum(x, h)
    )
    v = u * np.clip(y / np.maximum(y_wall, 1e-30), 0.0, 1.0) * slope

    rhoE = p / (flow.gamma - 1.0) + 0.5 * rho * (u * u + v * v)
    values = np.stack([rho, rho * u, rho * v, rhoE], axis=-1)
    return project(xp.asarray(values), ops, xp=xp)


def initial_state(
    ops: Operators,
    flow: FlowConditions,
    geom: NozzleGeometry,
    kind: str = "quasi1d",
    *,
    xp=np,
):
    """Dispatch on :attr:`dgnozzle.config.SolverOptions.initial_condition`."""
    if kind == "quasi1d":
        return quasi1d_initial(ops, flow, geom, xp=xp)
    if kind == "uniform":
        return uniform_initial(ops, flow, xp=xp)
    raise ValueError(f"unknown initial_condition {kind!r}; use 'quasi1d' or 'uniform'")


def change_order(U, ops_from: Operators, ops_to: Operators, xp=np):
    r"""Re-project a solution from one polynomial order onto another, same mesh.

    Used by ``p``-continuation.  Raising the order is exact -- the coarse
    solution lies in the finer space -- so the continuation adds no error, only
    removes iterations.

    .. note::
       The legacy ``extrapolate.m`` read ``resdata.p`` as the *old* order, but
       ``main.m`` had already overwritten it with the *new* one before calling.
       It therefore built the operator for the wrong pair of orders and indexed
       past the end of the incoming solution.  It also assumed the mesh was
       unchanged while ``main.m`` rebuilt it at a new refinement level between
       calls, so the element counts disagreed as well.
    """
    if ops_from.n_elem != ops_to.n_elem:
        raise ValueError(
            f"change_order needs the same mesh: {ops_from.n_elem} vs {ops_to.n_elem} elements"
        )
    # Evaluate the source solution at the *target* quadrature points, then
    # project.  Going through physical values keeps this correct for any pair of
    # orders in either direction.
    from . import elements as el

    phi_src, _, _ = el.shape_functions(
        ops_from.ref.kind, ops_from.ref.order, ops_to.ref.points_vol
    )
    values = xp.einsum("iq,eis->eqs", xp.asarray(phi_src), U)
    return project(values, ops_to, xp=xp)
