"""FAS multigrid: the transfers, the cycle's fixed point, and the answer.

The property that matters most is the last one.  Multigrid is an accelerator: it
may change how fast ``R_h = 0`` is reached but never what satisfies it, so a
multigrid solve and a single-grid solve of the same problem must agree to the
convergence tolerance.  Everything else here exists to localise a failure of
that one.
"""

from __future__ import annotations

import numpy as np
import pytest

from dgnozzle import NozzleGeometry, build_case
from dgnozzle import multigrid as mg
from dgnozzle.config import FlowConditions, SolverOptions
from dgnozzle.operators import build_operators

pytestmark = pytest.mark.numba

KINDS = ("p", "h")


def hierarchy(kind, *, order=1, refine=1, element="tri", geometry_order=1, opts=None):
    case = build_case(
        NozzleGeometry(contour="smooth"), FlowConditions(),
        order=order, refine=refine, element=element, geometry_order=geometry_order,
    )
    levels = mg.build_hierarchy(
        kind, case.operators, case.flow, case.geometry, case.discretization,
        opts or SolverOptions(),
    )
    return case, levels


# ==========================================================================
#  Transfers
# ==========================================================================
@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("element", ["tri", "quad"])
@pytest.mark.parametrize("order", [1, 2])
def test_prolong_then_restrict_is_the_identity(kind, element, order):
    """The coarse space sits inside the fine one, so this must be exact.

    It is the defining property of a projection pair, and it is what tells you
    the parent map and the reference-space maps are both right -- a wrong parent
    still produces plausible-looking numbers, but not this.
    """
    _, levels = hierarchy(kind, order=order, element=element)
    tr, coarse = levels[0].transfer, levels[1].operators
    rng = np.random.default_rng(0)
    V = rng.standard_normal((coarse.n_elem, coarse.ref.n_basis, 4))
    back = np.asarray(tr.restrict(tr.prolong(V)))
    assert np.abs(back - V).max() / np.abs(V).max() < 1e-12


@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("element", ["tri", "quad"])
def test_a_constant_field_survives_both_transfers(kind, element):
    """A uniform flow must stay uniform, or the coarse level solves a fiction."""
    _, levels = hierarchy(kind, element=element)
    tr, fine, coarse = levels[0].transfer, levels[0].operators, levels[1].operators
    up = np.asarray(tr.prolong(np.full((coarse.n_elem, coarse.ref.n_basis, 4), 3.5)))
    down = np.asarray(tr.restrict(np.full((fine.n_elem, fine.ref.n_basis, 4), 3.5)))
    assert np.abs(up - 3.5).max() < 1e-12
    assert np.abs(down - 3.5).max() < 1e-12


@pytest.mark.parametrize("geometry_order", [1, 2])
def test_every_child_lies_inside_its_parent(geometry_order):
    """Checked geometrically, not just in reference space.

    ``build_mesh_transfer`` asserts the reference-space maps land inside the
    parent element; that holds even if the *parent index* is wrong.  Comparing
    element areas catches the index: a parent's area must match its children's
    sum to the coarse mesh's own geometry error, which is a few percent on a
    curved wall and would be tens of percent if a child were assigned elsewhere.
    """
    _, levels = hierarchy("h", geometry_order=geometry_order)
    tr, fine, coarse = levels[0].transfer, levels[0].operators, levels[1].operators
    af, ac = np.asarray(fine.elem_area), np.asarray(coarse.elem_area)
    summed = af[tr.children].sum(axis=1)
    assert np.abs(summed - ac).max() / ac.max() < 0.05


def test_the_child_to_parent_maps_are_exact_on_a_straight_duct():
    """With a flat wall the meshes tile exactly, so nothing may be approximate.

    This separates the two error sources.  On the nozzle the ``h`` transfer
    carries the coarse mesh's geometry error, because a curved wall meshed twice
    does not tile; flattening the wall removes that and leaves only the index
    arithmetic, which must be exact.
    """
    discs, ops = {}, {}
    for refine in (1, 0):
        case = build_case(
            NozzleGeometry(contour="smooth"), FlowConditions(),
            order=1, refine=refine, x_spacing="uniform",
        )
        coords = np.array(case.node_coords)
        wall = np.asarray(case.geometry.wall(coords[:, 0]))
        coords[:, 1] = coords[:, 1] / np.maximum(wall, 1e-30) * 0.14
        ops[refine] = build_operators(case.topology, coords, 1)
        discs[refine] = case.discretization

    tr = mg.build_mesh_transfer(ops[1], ops[0], discs[1], 1)
    af, ac = np.asarray(ops[1].elem_area), np.asarray(ops[0].elem_area)
    assert np.abs(af[tr.children].sum(axis=1) - ac).max() / ac.max() < 1e-13

    m_fine = mg.element_mass(ops[1])
    m_coarse = mg.element_mass(ops[0])
    blocks = tr.prolong_block[tr.block_of]
    implied = np.zeros_like(m_coarse)
    np.add.at(implied, tr.parent, np.einsum("ekj,ekl,elm->ejm", blocks, m_fine, blocks))
    assert np.abs(implied - m_coarse).max() / np.abs(m_coarse).max() < 1e-13


def test_a_misdirected_parent_would_be_caught():
    """The identity test is only convincing if it can actually fail."""
    _, levels = hierarchy("h")
    tr = levels[0].transfer
    broken = mg.MeshTransfer(
        parent=np.roll(tr.parent, 1),
        children=tr.children,
        block_of=tr.block_of,
        prolong_block=tr.prolong_block,
        restrict_block=tr.restrict_block,
        up=tr.up,
        down=tr.down,
    )
    coarse = levels[1].operators
    V = np.random.default_rng(0).standard_normal((coarse.n_elem, coarse.ref.n_basis, 4))
    back = np.asarray(broken.restrict(broken.prolong(V)))
    assert np.abs(back - V).max() / np.abs(V).max() > 1e-3


# ==========================================================================
#  The cycle
# ==========================================================================
@pytest.mark.parametrize("kind", KINDS)
def test_a_converged_state_is_a_fixed_point_of_the_cycle(kind):
    """FAS must not move a solution that already satisfies the fine problem.

    This is the whole reason for the source term: with ``s_H`` in place the
    coarse level's answer is the restricted fine state itself, so the correction
    prolonged back is zero.  Drop the source and this test fails -- which is how
    you tell a correction scheme from full approximation storage.
    """
    from dgnozzle import solve_nozzle

    result = solve_nozzle(
        contour="smooth", order=1, refine=1, tolerance=1e-10,
        verbose=False, print_interval=0,
    )
    assert result.converged

    _, levels = hierarchy(kind)
    before = np.asarray(result.U)
    after, _ = mg.v_cycle(levels, before.copy(), pre=1, post=1, coarse=2)
    moved = np.abs(np.asarray(after) - before).max() / np.abs(before).max()
    assert moved < 1e-6, moved


@pytest.mark.parametrize("kind", KINDS)
def test_the_cycle_leaves_no_source_behind(kind):
    """A stale source would quietly make the next cycle solve a different problem."""
    case, levels = hierarchy(kind)
    from dgnozzle import initialize as ini

    U = ini.initial_state(levels[0].operators, case.flow, case.geometry, "quasi1d")
    mg.v_cycle(levels, np.asarray(U), pre=1, post=1, coarse=2)
    assert all(lvl.backend.rate_source is None for lvl in levels)


def test_an_unknown_hierarchy_is_rejected():
    case = build_case(NozzleGeometry(contour="smooth"), FlowConditions(), order=1)
    with pytest.raises(ValueError, match="unknown multigrid kind"):
        mg.build_hierarchy(
            "q", case.operators, case.flow, case.geometry, case.discretization,
            SolverOptions(),
        )


def test_p_multigrid_at_order_zero_has_nothing_to_coarsen():
    case = build_case(NozzleGeometry(contour="smooth"), FlowConditions(), order=0)
    levels = mg.build_hierarchy(
        "p", case.operators, case.flow, case.geometry, case.discretization,
        SolverOptions(),
    )
    assert len(levels) == 1


def test_h_multigrid_without_refinement_has_nothing_to_coarsen():
    case = build_case(NozzleGeometry(contour="smooth"), FlowConditions(),
                      order=1, refine=0)
    levels = mg.build_hierarchy(
        "h", case.operators, case.flow, case.geometry, case.discretization,
        SolverOptions(),
    )
    assert len(levels) == 1


@pytest.mark.parametrize("kind", KINDS)
def test_the_solver_falls_back_quietly_when_no_coarse_level_exists(kind):
    """Asking for multigrid where none is possible must still solve."""
    from dgnozzle import solve_nozzle

    bad = {"p": dict(order=0, refine=1), "h": dict(order=1, refine=0)}[kind]
    result = solve_nozzle(
        contour="smooth", multigrid=kind, verbose=False, print_interval=0, **bad
    )
    assert result.converged
    assert result.multigrid == "none"
    assert result.work_equivalent == 0


# ==========================================================================
#  The answer
# ==========================================================================
@pytest.mark.slow
@pytest.mark.parametrize("kind", KINDS)
def test_multigrid_reaches_the_single_grid_answer(kind):
    """An accelerator changes the path, never the destination."""
    from dgnozzle import performance, solve_nozzle

    common = dict(contour="smooth", order=1, refine=1, tolerance=1e-8,
                  verbose=False, print_interval=0)
    plain = solve_nozzle(**common)
    accel = solve_nozzle(multigrid=kind, **common)
    assert plain.converged and accel.converged
    assert accel.multigrid == kind
    assert accel.work_equivalent > 0

    # the states agree to rather better than the tolerance they were marched to
    assert np.abs(accel.U - plain.U).max() / np.abs(plain.U).max() < 1e-5
    pa, pb = performance(plain), performance(accel)
    assert abs(pa.thrust - pb.thrust) / abs(pa.thrust) < 1e-5
    assert abs(pa.exit_mach_area_averaged - pb.exit_mach_area_averaged) < 1e-4


@pytest.mark.slow
@pytest.mark.parametrize("kind", KINDS)
def test_multigrid_does_less_work_than_plain_stepping(kind):
    """The point of the exercise, measured in fine-level-equivalent steps.

    Wall time is not asserted, for two reasons.  The transfers and the extra rate
    evaluation per cycle are real costs the work count does not include, and
    measured they eat the whole saving -- the README has the times and the
    recommendation not to turn this on.  And the comparison is machine- and
    load-dependent in a way an assertion should not be.

    Nor is ``p = 2, refine = 1`` used here, even though it is the largest case
    the suite can afford: ``multigrid='p'`` does not converge there at the default
    cycle, repairing cell averages as the coarse correction pushes them
    non-physical.  That is a genuine limitation rather than a flaky test, and the
    README states it.
    """
    from dgnozzle import solve_nozzle

    common = dict(contour="smooth", order=1, refine=2, verbose=False, print_interval=0)
    plain = solve_nozzle(**common)
    accel = solve_nozzle(multigrid=kind, mg_pre=3, mg_post=3, mg_coarse=20, **common)
    assert plain.converged, plain.message
    assert accel.converged, accel.message
    assert accel.work_equivalent < plain.iterations
