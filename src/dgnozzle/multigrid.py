r"""FAS multigrid acceleration of the pseudo-time march.

The explicit march converges at a rate set by nothing but its time step: the
iteration count is, measured, exactly inversely proportional to ``cfl``.  Low
wavenumber error is what takes the longest to leave, and the only way to move it
faster is to represent it on a coarser space where the stable step is larger.

Full Approximation Storage
--------------------------
Write the semi-discrete problem in *rate* form,

.. math::
    L_h(\mathbf{U}) = -\mathbf{M}_h^{-1}\mathbf{R}_h(\mathbf{U}),

so a steady state is :math:`L_h(\mathbf{U}) = \mathbf{0}`.  Rates live in the
same space as the solution, which is what makes a single transfer operator serve
for both; restricting :math:`\mathbf{R}` directly would need the mass-weighted
transpose instead, and the two are easy to confuse.

One FAS cycle, from level :math:`h` to level :math:`H`, with :math:`\Pi` the
restriction and :math:`P` the prolongation:

1. smooth :math:`\nu_1` steps on :math:`h`;
2. restrict the state and the rate, :math:`\mathbf{V}_0 = \Pi\mathbf{U}_h` and
   :math:`\ell_H = \Pi L_h(\mathbf{U}_h)`;
3. form the defect source :math:`s_H = L_H(\mathbf{V}_0) - \ell_H` and solve
   :math:`L_H(\mathbf{V}) = s_H` on the coarse level, recursively;
4. correct, :math:`\mathbf{U}_h \mathrel{+}= P(\mathbf{V} - \mathbf{V}_0)`;
5. smooth :math:`\nu_2` steps on :math:`h`.

The source is what makes this *full approximation* rather than a correction
scheme: at step 3 the coarse level solves the full nonlinear problem, not a
linearisation, and :math:`s_H` is exactly the term that makes
:math:`\mathbf{V} = \mathbf{V}_0` the answer whenever the fine level is already
converged.  So a converged fine state produces a zero coarse correction, and the
cycle cannot move a solution away from the fine-level steady state it has
reached.  That is the property :func:`dgnozzle.multigrid.v_cycle` is tested on,
and it is why the converged answer is the single-grid answer: multigrid changes
how fast :math:`\mathbf{R}_h = \mathbf{0}` is reached, never what satisfies it.

Does it help here?  Measured, barely
------------------------------------
Not as much as the idea promises, and the measurement is the point of writing it
down.  Compared at the same ``cfl``, a V-cycle buys 1.1-1.3x of wall time at
best, and costs more than plain stepping at ``p = 2, refine = 1``.  Compared at
each method's *own* best ``cfl``, plain stepping wins outright.  The README
carries the full table.

It also has a stability floor.  ``mg_pre = mg_post = 1`` diverges outright, and
below ``cfl`` of roughly 0.55 the cycle stops converging at every order tested --
too little smoothing leaves the coarse correction injecting error the smoother
cannot take back out.  At ``p = 2, refine = 1`` even ``mg_pre = 2`` fails,
repairing cell averages as the correction pushes them non-physical, and the
settings that failed were the *fastest* ones: a wall-time table that does not
also check :attr:`~dgnozzle.solver.SolveResult.converged` will recommend them.

The reason is visible in the march itself: its iteration count is exactly
inversely proportional to the time step, which says the slow mode is the
acoustic transit of the nozzle rather than a spectrum of spatial wavenumbers.
Multigrid earns its keep where low-wavenumber *spatial* error is what takes the
longest to leave -- the elliptic case.  A steady supersonic march is dominated by
transport, and a coarse grid has little to offer it.  Driving ``mg_coarse`` from
8 to 40 bears that out: the coarse level converges completely and the fine-level
cycle count does not move, so the error multigrid is designed to remove is not
the error that is left.

The other half of the explanation is the smoother.  RK4 at 70% of its stability
limit is tuned for the largest stable step, which is not the same as damping the
wavenumbers the coarse level cannot see; per cycle the residual falls by only
about 0.92, where textbook multigrid expects 0.1-0.3.  A smoother designed for
damping -- the optimised multi-stage coefficients of Van Leer, Tai and Powell,
say -- is the missing piece, and is a separate piece of work.

So this is **off by default** and stays that way.  It is kept because it is
correct, because the measurement is worth having written down, and because it is
the foundation anything better would be built on.

Two hierarchies
---------------
``'p'``
    Same mesh, decreasing polynomial order.  The coarse space sits exactly
    inside the fine one, so the transfers are the exact :math:`L^2` projections
    :func:`dgnozzle.initialize.change_order` already provides, and prolongation
    of a coarse field is exact to round-off.  Needs ``order >= 1``.

``'h'``
    Same order, one refinement level coarser.  The logical grids nest by
    construction -- the node set at one refinement level is a subset of the next
    -- so every fine element has exactly one parent and the reference-space map
    between them is affine.  Needs ``refine >= 1``.

    The transfer is *not* exact here, and not because of the algebra: a curved
    wall meshed at two resolutions simply does not tile, since the fine elements
    hug the contour more closely than their parent does.  The mismatch is the
    geometry error of the coarse mesh, and it costs convergence rate rather than
    accuracy -- the fine-level fixed point is untouched by any transfer.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Protocol

import numpy as np

from . import elements as el
from .backends import Backend, get_backend
from .config import Discretization, FlowConditions, SolverOptions
from .operators import Operators, build_operators


# ==========================================================================
#  Transfer operators
# ==========================================================================
class Transfer(Protocol):
    """Prolongation and restriction between two nested DG spaces."""

    def prolong(self, V: np.ndarray) -> np.ndarray:
        """Coarse state or rate to the fine space."""

    def restrict(self, U: np.ndarray) -> np.ndarray:
        """Fine state or rate to the coarse space."""


@dataclass(frozen=True)
class OrderTransfer:
    r"""``p``-transfer: the exact :math:`L^2` projection between two orders.

    Algebraically this is :func:`dgnozzle.initialize.change_order`, and the test
    suite already pins that to be exact in the coarse-to-fine direction.  It is
    not called here, though, because it rebuilds the basis evaluation -- a
    Vandermonde solve -- on every call, and a V-cycle calls the transfer twice
    per level.  The two matrices are the same every time, so they are formed
    once and reduced to a pair of batched products.
    """

    fine: Operators
    coarse: Operators
    up: np.ndarray = field(repr=False, default=None)  # type: ignore[assignment]
    down: np.ndarray = field(repr=False, default=None)  # type: ignore[assignment]

    @classmethod
    def build(cls, fine: Operators, coarse: Operators) -> OrderTransfer:
        up = _projection_matrix(coarse, fine)
        down = _projection_matrix(fine, coarse)
        return cls(fine=fine, coarse=coarse, up=up, down=down)

    def prolong(self, V):
        return np.einsum("eij,ejs->eis", self.up, V)

    def restrict(self, U):
        return np.einsum("eij,ejs->eis", self.down, U)


def _projection_matrix(src: Operators, dst: Operators) -> np.ndarray:
    r"""The per-element matrix taking ``src`` coefficients to ``dst`` ones.

    The same :math:`L^2` projection :func:`dgnozzle.initialize.change_order`
    performs -- evaluate the source basis at the target quadrature points, then
    project -- written out so it can be applied as a single batched product.
    """
    phi_src, _, _ = el.shape_functions(src.ref.kind, src.ref.order, dst.ref.points_vol)
    phi_dst = np.asarray(dst.ref.phi_vol)
    # inv_mass . (phi_dst * w det) . phi_src
    rhs = np.einsum("iq,eq,jq->eij", phi_dst, np.asarray(dst.weighted_det), phi_src)
    return np.ascontiguousarray(np.einsum("eij,ejk->eik", np.asarray(dst.inv_mass), rhs))


def element_mass(ops: Operators) -> np.ndarray:
    r""":math:`(\mathbf{M}_e)_{ij} = \int_{\Omega_e}\phi_i\phi_j`, per element.

    :class:`~dgnozzle.operators.Operators` stores only the inverse, because that
    is all the march needs; the transfer blocks need the forward matrix.
    """
    phi = np.asarray(ops.ref.phi_vol)
    return np.einsum("iq,eq,jq->eij", phi, np.asarray(ops.weighted_det), phi)


@dataclass(frozen=True)
class MeshTransfer:
    """``h``-transfer between two refinement levels of the same logical grid.

    ``prolong_block[b]`` evaluates the parent's basis at the child's Lagrange
    nodes, which is exact in reference space because the child-to-parent map is
    affine there.  Restriction is the mass-weighted transpose summed over the
    children, :math:`\\tilde{\\mathbf{M}}_H^{-1}\\sum_k P_k^{T}\\mathbf{M}_{h,k}`
    with :math:`\\tilde{\\mathbf{M}}_H = \\sum_k P_k^{T}\\mathbf{M}_{h,k}P_k`; see
    :func:`build_mesh_transfer` for why that is the children's mass matrix rather
    than the coarse mesh's own.
    """

    parent: np.ndarray  # (n_fine,) coarse element holding each fine element
    children: np.ndarray  # (n_coarse, n_child) fine elements of each coarse one
    block_of: np.ndarray  # (n_fine,) which prolongation block each fine one uses
    prolong_block: np.ndarray  # (n_block, nbf, nbf) parent basis at child nodes
    restrict_block: np.ndarray  # (n_fine, nbf, nbf) per-element M_H^-1 P^T M_h

    up: np.ndarray = field(repr=False, default=None)  # type: ignore[assignment]
    """``prolong_block[block_of]``, materialised once: a cycle re-gathers it twice
    per level otherwise, and the gather allocates the whole array each time."""
    down: np.ndarray = field(repr=False, default=None)  # type: ignore[assignment]
    """``restrict_block[children]``, likewise."""

    def prolong(self, V):
        return np.einsum("eij,ejs->eis", self.up, V[self.parent])

    def restrict(self, U):
        return np.einsum("ekij,ekjs->eis", self.down, U[self.children])


#: Which of a coarse triangle's two sub-elements holds each fine sub-element,
#: indexed ``[di, dj, sub_fine]`` with ``(di, dj)`` the child cell's position in
#: its parent.  The cell diagonal runs from ``(1, 0)`` to ``(0, 1)``, so all four
#: child cells split along a line parallel to their parent's -- which is why
#: every fine triangle has a single parent at all.  Derived in
#: :func:`_child_to_parent_map`, and checked there against the geometry.
_TRI_PARENT_SUB = np.array([[[0, 0], [0, 1]], [[0, 1], [1, 1]]])


def _child_to_parent_map(kind: str, di: int, dj: int, sub: int) -> tuple[np.ndarray, int]:
    r"""Affine map from a child's reference element to its parent's, plus the parent sub-element.

    Returned as ``(A, parent_sub)`` with ``A`` of shape ``(2, 3)`` acting as
    :math:`\xi_H = A[:, :2]\,\xi_h + A[:, 2]`.

    Everything is worked out in the parent cell's own index coordinates, where
    the parent spans ``[0, 2] x [0, 2]`` in child-cell units and child cell
    ``(di, dj)`` spans ``[di, di+1] x [dj, dj+1]``.  That is the frame the two
    node grids actually nest in, which is what makes the map affine and exact.
    """
    if kind == "quad":
        # child ref -> cell index (di + xi, dj + eta) -> parent ref / 2
        return np.array([[0.5, 0.0, di / 2.0], [0.0, 0.5, dj / 2.0]]), 0

    # -- triangles.  Child sub 0 has corners (0,0),(1,0),(0,1) of its cell;
    #    sub 1 has (1,0),(1,1),(0,1).  In cell-index coordinates:
    if sub == 0:
        #  (i, j) = (di + xi, dj + eta)
        idx = np.array([[1.0, 0.0, float(di)], [0.0, 1.0, float(dj)]])
    else:
        #  (i, j) = (di + 1 - eta, dj + xi + eta)
        idx = np.array([[0.0, -1.0, di + 1.0], [1.0, 1.0, float(dj)]])

    parent_sub = int(_TRI_PARENT_SUB[di, dj, sub])
    if parent_sub == 0:
        # parent corners (0,0),(2,0),(0,2):  xi_H = i/2, eta_H = j/2
        to_ref = np.array([[0.5, 0.0, 0.0], [0.0, 0.5, 0.0]])
    else:
        # parent corners (2,0),(2,2),(0,2):  xi_H = (i + j - 2)/2, eta_H = 1 - i/2
        to_ref = np.array([[0.5, 0.5, -1.0], [-0.5, 0.0, 1.0]])

    # compose: to_ref acting on (i, j, 1), with (i, j, 1) = idx_ext @ (xi, eta, 1)
    idx_ext = np.vstack([idx, [0.0, 0.0, 1.0]])
    return to_ref @ idx_ext, parent_sub


def build_mesh_transfer(fine: Operators, coarse: Operators, disc: Discretization,
                        refine_fine: int) -> MeshTransfer:
    """Transfer between refinement level ``refine_fine`` and one level coarser."""
    kind = disc.element
    nref = 2**refine_fine
    ncx_f, ncy_f = disc.nx * nref, disc.nr * nref
    ncx_c, ncy_c = ncx_f // 2, ncy_f // 2
    if ncx_c * 2 != ncx_f or ncy_c * 2 != ncy_f:
        raise ValueError("the fine mesh is not a 2x refinement of a coarser one")
    n_sub = 2 if kind == "tri" else 1
    cells_f, cells_c = ncx_f * ncy_f, ncx_c * ncy_c
    n_fine, n_coarse = n_sub * cells_f, n_sub * cells_c
    if fine.n_elem != n_fine or coarse.n_elem != n_coarse:
        raise ValueError(
            f"element counts do not match the logical grid: fine {fine.n_elem} vs "
            f"{n_fine}, coarse {coarse.n_elem} vs {n_coarse}"
        )

    # -- the distinct reference-space maps, one per (di, dj, sub)
    nodes_f = el.reference_nodes(kind, fine.ref.order)
    blocks: list[np.ndarray] = []
    parent_sub: list[int] = []
    key: dict[tuple[int, int, int], int] = {}
    for di in (0, 1):
        for dj in (0, 1):
            for sub in range(n_sub):
                A, psub = _child_to_parent_map(kind, di, dj, sub)
                mapped = nodes_f @ A[:, :2].T + A[:, 2]
                _check_inside(kind, mapped, di, dj, sub)
                phi, _, _ = el.shape_functions(kind, coarse.ref.order, mapped)
                key[(di, dj, sub)] = len(blocks)
                blocks.append(np.ascontiguousarray(phi.T))  # (nbf_f, nbf_c)
                parent_sub.append(psub)

    # -- per-element parent and block indices, from pure index arithmetic
    e = np.arange(n_fine)
    sub_f = e // cells_f
    c_f = e % cells_f
    ci_f, cj_f = c_f // ncy_f, c_f % ncy_f
    di, dj = ci_f % 2, cj_f % 2
    block_of = np.array([key[(int(a), int(b), int(s))]
                         for a, b, s in zip(di, dj, sub_f, strict=True)])
    sub_c = np.array([parent_sub[b] for b in block_of])
    parent = sub_c * cells_c + (ci_f // 2) * ncy_c + (cj_f // 2)

    counts = np.bincount(parent, minlength=n_coarse)
    if counts.min() != counts.max():
        raise AssertionError(
            f"uneven refinement: coarse elements hold between {counts.min()} and "
            f"{counts.max()} children"
        )
    order = np.argsort(parent, kind="stable")
    children = order.reshape(n_coarse, counts[0])

    # -- restriction as the mass-weighted transpose of prolongation
    M_f = element_mass(fine)
    P = np.asarray(blocks)[block_of]  # (n_fine, nbf_f, nbf_c)

    # The coarse mass matrix used here is the one the *children* imply,
    # sum_k P_k^T M_h,k P_k, not the coarse mesh's own.  On a curved wall the two
    # differ -- the children hug the contour more closely than their parent, so
    # they do not tile it, and on the default nozzle the gap reaches 4% of the
    # coarse mass matrix.  Using the coarse mesh's own matrix there makes
    # restriction stop being a projection: measured, it sent a uniform fine field
    # to a coarse field 12% away from uniform, which hands the coarse level a
    # state the fine level never had.  The children-consistent matrix restores
    # both identities exactly, for any geometry:
    #
    #     restrict(prolong(V)) = V        and        restrict(1) = 1
    #
    # The price is that the restricted field's integral now matches the fine one
    # only to the coarse mesh's geometry error, where before it was exact.  That
    # is the better trade: FAS corrects an inconsistent coarse *operator* through
    # its source term, but nothing corrects a distorted coarse *state*.
    PtMP = np.einsum("ekj,ekl,elm->ejm", P, M_f, P)  # (n_fine, nbf_c, nbf_c)
    M_tilde = np.zeros((coarse.n_elem,) + PtMP.shape[1:])
    np.add.at(M_tilde, parent, PtMP)
    inv_M_c = np.linalg.inv(M_tilde)

    restrict_block = np.einsum("eij,ekj,ekl->eil", inv_M_c[parent], P, M_f)
    prolong_block = np.asarray(blocks)
    restrict_block = np.ascontiguousarray(restrict_block)
    return MeshTransfer(
        parent=parent,
        children=children,
        block_of=block_of,
        prolong_block=prolong_block,
        restrict_block=restrict_block,
        up=np.ascontiguousarray(prolong_block[block_of]),
        down=np.ascontiguousarray(restrict_block[children]),
    )


def _check_inside(kind: str, pts: np.ndarray, di: int, dj: int, sub: int) -> None:
    """A child's nodes must land inside its parent, or the map is wrong."""
    tol = 1e-12
    bad = (pts < -tol).any() or (pts > 1.0 + tol).any()
    if kind == "tri":
        bad = bad or (pts.sum(axis=1) > 1.0 + tol).any()
    if bad:
        raise AssertionError(
            f"child-to-parent map for {kind} cell ({di},{dj}) sub {sub} sends nodes "
            f"outside the parent reference element:\n{pts}"
        )


# ==========================================================================
#  Levels and the cycle
# ==========================================================================
@dataclass
class Level:
    """One level of the hierarchy, plus the transfer down to the next."""

    operators: Operators
    backend: Backend
    transfer: Transfer | None = None
    label: str = ""
    smoothing_steps: int = 0
    """Fine-level-equivalent work done here, accumulated for reporting."""


def build_hierarchy(
    kind: str,
    ops: Operators,
    flow: FlowConditions,
    geom,
    disc: Discretization,
    opts: SolverOptions,
    *,
    backend: str = "numba",
    max_levels: int = 0,
) -> list[Level]:
    """Build the level list, finest first.

    ``kind`` is ``'p'`` or ``'h'``.  A hierarchy of length 1 means no coarse
    level was available -- ``order = 0`` for ``'p'``, ``refine = 0`` for
    ``'h'`` -- which the caller should treat as "no multigrid".
    """
    if kind == "p":
        return _p_hierarchy(ops, flow, disc, opts, backend, max_levels)
    if kind == "h":
        return _h_hierarchy(ops, flow, geom, disc, opts, backend, max_levels)
    raise ValueError(f"unknown multigrid kind {kind!r}; use 'p' or 'h'")


def _p_hierarchy(ops, flow, disc, opts, backend, max_levels) -> list[Level]:
    orders = list(range(disc.order, -1, -1))
    if max_levels:
        orders = orders[:max_levels]
    levels: list[Level] = []
    ops_at = {disc.order: ops}
    for p in orders:
        if p not in ops_at:
            ops_at[p] = build_operators(ops.topology, ops.node_coords, p)
        levels.append(
            Level(
                operators=ops_at[p],
                backend=get_backend(backend, ops_at[p], flow, opts),
                label=f"p={p}",
            )
        )
    for fine, coarse in zip(levels, levels[1:], strict=False):
        fine.transfer = OrderTransfer.build(fine.operators, coarse.operators)
    return levels


def _h_hierarchy(ops, flow, geom, disc, opts, backend, max_levels) -> list[Level]:
    from .api import build_case

    refines = list(range(disc.refine, -1, -1))
    if max_levels:
        refines = refines[:max_levels]

    levels: list[Level] = []
    ops_at: dict[int, Operators] = {disc.refine: ops}
    discs: dict[int, Discretization] = {disc.refine: disc}
    for r in refines:
        if r not in ops_at:
            # a coarse level is an ordinary case at a lower refinement, so it
            # gets the real boundary tags and metric terms rather than anything
            # agglomerated by hand
            case = build_case(geom, flow, disc.replace(refine=r), check=False)
            ops_at[r] = case.operators
            discs[r] = case.discretization
        levels.append(
            Level(
                operators=ops_at[r],
                backend=get_backend(backend, ops_at[r], flow, opts),
                label=f"refine={r}",
            )
        )
    for rf, fine, coarse in zip(refines, levels, levels[1:], strict=False):
        fine.transfer = build_mesh_transfer(fine.operators, coarse.operators, discs[rf], rf)
    return levels


def v_cycle(
    levels: Sequence[Level],
    U: np.ndarray,
    *,
    pre: int = 2,
    post: int = 2,
    coarse: int = 8,
    scheme: str = "rk4",
    depth: int = 0,
) -> tuple[np.ndarray, float]:
    """One FAS V-cycle.  Returns ``(U, fine-level residual norm)``.

    The residual is whatever the finest level's pre-smoothing last measured --
    the norm at the state one step before the coarse visit -- so it is the
    ordinary single-grid residual norm, computed by the same code, and a
    multigrid run's convergence history is directly comparable with a plain
    one's.  It lags the returned state by a few smoothing steps, which only ever
    makes the convergence test pessimistic.
    """
    lvl = levels[depth]
    bk = lvl.backend
    last = depth + 1 >= len(levels)

    n_pre = coarse if last else pre
    U, res = bk.run(U, n_pre, scheme)
    lvl.smoothing_steps += n_pre
    if last:
        return U, res

    transfer = lvl.transfer
    assert transfer is not None  # built by build_hierarchy
    nxt = levels[depth + 1]

    # -- restrict the state and the rate this level currently sees
    rate, _ = bk.rate(U)
    V0 = np.asarray(transfer.restrict(U))
    ell = np.asarray(transfer.restrict(np.asarray(rate)))

    # -- the FAS source: what makes V0 the coarse answer when U is converged
    nxt.backend.rate_source = None
    rate_coarse, _ = nxt.backend.rate(V0)
    nxt.backend.rate_source = np.asarray(rate_coarse) - ell
    try:
        V, _ = v_cycle(levels, V0.copy(), pre=pre, post=post, coarse=coarse,
                       scheme=scheme, depth=depth + 1)
    finally:
        # never leave a source behind: the next cycle recomputes it, and a stale
        # one would quietly solve the wrong coarse problem
        nxt.backend.rate_source = None

    # -- correct, then limit: a coarse correction can push a cell non-physical
    U = np.asarray(U) + np.asarray(transfer.prolong(V - V0))
    U = bk.limit(U)

    U, _ = bk.run(U, post, scheme)
    lvl.smoothing_steps += post
    return U, res


@dataclass
class CycleStepper:
    """Adapts :func:`v_cycle` to the ``run(U, n, scheme)`` shape ``march`` wants.

    One "iteration" of the march is one V-cycle, so the iteration counts a
    multigrid run reports are cycles and are *not* comparable with a single-grid
    run's steps.  :attr:`work` carries the fine-level-equivalent smoothing count
    for when they need to be.
    """

    levels: list[Level]
    pre: int = 2
    post: int = 2
    coarse: int = 8
    cycles: int = 0
    _hist: list[float] = field(default_factory=list, repr=False)

    @property
    def top(self) -> Backend:
        return self.levels[0].backend

    @property
    def work(self) -> int:
        """Smoothing steps weighted by each level's cost relative to the finest."""
        base = self.levels[0].operators.n_elem * self.levels[0].operators.ref.n_basis
        total = 0.0
        for lvl in self.levels:
            size = lvl.operators.n_elem * lvl.operators.ref.n_basis
            total += lvl.smoothing_steps * size / base
        return int(round(total))

    def run(self, U, n_steps: int, scheme: str):
        res = float("nan")
        for _ in range(n_steps):
            U, res = v_cycle(self.levels, np.asarray(U), pre=self.pre, post=self.post,
                             coarse=self.coarse, scheme=scheme)
            self.cycles += 1
        return U, res

    # -- the rest of the Backend surface `march` and `solve_steady` touch
    def limiter_activity(self) -> float:
        return self.top.limiter_activity()

    def asarray(self, A):
        return self.top.asarray(A)

    def to_numpy(self, A) -> np.ndarray:
        return self.top.to_numpy(A)

    @property
    def n_mean_repaired(self) -> int:
        return sum(lvl.backend.n_mean_repaired for lvl in self.levels)
