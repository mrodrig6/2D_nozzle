"""Parameter sweeps: axis handling, warm starting, and failure reporting."""

from __future__ import annotations

import numpy as np
import pytest

from dgnozzle import METRICS, sweep
from dgnozzle.sweep import _is_axis, _serpentine, _split_grids

pytestmark = pytest.mark.slow


def test_iterables_are_axes_and_scalars_are_settings():
    """sweep(area_ratio=[...], order=1) must sweep one and fix the other."""
    assert _is_axis([1.0, 2.0])
    assert _is_axis(np.linspace(0, 1, 3))
    assert not _is_axis(1)
    assert not _is_axis(1.0)
    assert not _is_axis("smooth")
    assert not _is_axis(True)

    axes, fixed = _split_grids({"area_ratio": [2.0, 3.0], "order": 1, "contour": "smooth"})
    assert set(axes) == {"area_ratio"}
    assert fixed == {"order": 1, "contour": "smooth"}


def test_serpentine_walk_keeps_successive_points_adjacent():
    """Warm starting is only useful if the previous point is a nearby design."""
    order = _serpentine([np.arange(3), np.arange(4)])
    assert len(order) == 12
    assert len(set(order)) == 12
    # order[1:] is one shorter on purpose: we compare consecutive pairs
    for a, b in zip(order, order[1:], strict=False):
        assert sum(abs(x - y) for x, y in zip(a, b, strict=True)) == 1


def test_a_sweep_with_no_axis_is_rejected():
    with pytest.raises(ValueError, match="at least one iterable"):
        sweep(order=1)


def test_one_dimensional_sweep():
    table = sweep(area_ratio=np.linspace(2.0, 3.0, 3), order=0,
                  contour="smooth", progress=False)
    assert table.n_points == 3
    assert table.shape == (3,)
    assert table.all_converged, table.failures()
    assert np.allclose(table["area_ratio"], [2.0, 2.5, 3.0])
    assert set(table.values) == set(METRICS)
    assert np.all(table["thrust"] > 0.0)
    # a bigger nozzle expands more, so the exit Mach number must rise
    assert np.all(np.diff(table["exit_mach"]) > 0.0)


def test_warm_starting_changes_cost_but_not_the_answer():
    kw = dict(area_ratio=np.linspace(2.0, 3.0, 4), order=1,
              contour="smooth", progress=False)
    warm = sweep(warm_start=True, **kw)
    cold = sweep(warm_start=False, **kw)
    assert warm.all_converged and cold.all_converged

    # Two states converged to a scaled residual of 1e-6 agree on a functional to
    # roughly that residual times the functional's sensitivity.  Measured here
    # that is about 1.1e-5 -- so a 1e-5 tolerance fails for reasons that have
    # nothing to do with warm starting.  Tighten `tolerance` and the difference
    # shrinks with it.
    assert np.allclose(warm["thrust"], cold["thrust"], rtol=1e-4)
    assert warm["iterations"].sum() <= cold["iterations"].sum()


def test_two_dimensional_sweep_reshapes():
    table = sweep(area_ratio=[2.0, 2.5], back_pressure_ratio=[0.12, 0.18],
                  order=0, contour="smooth", progress=False)
    assert table.shape == (2, 2)
    assert table.reshape("thrust").shape == (2, 2)
    assert table.all_converged, table.failures()


def test_indexing_by_parameter_and_metric():
    table = sweep(area_ratio=[2.0, 2.5], order=0, contour="smooth", progress=False)
    assert table["area_ratio"].shape == (2,)
    assert table["thrust"].shape == (2,)
    with pytest.raises(KeyError, match="neither a swept parameter"):
        table["nonsense"]


def test_failures_are_recorded_not_hidden():
    """A point that fails must leave NaN and an explanation, not silence."""
    table = sweep(area_ratio=[2.5], order=1, contour="smooth",
                  cfl=80.0, max_iterations=600, progress=False)
    if not table.all_converged:
        assert np.isnan(table["thrust"][0])
        failures = table.failures()
        assert len(failures) == 1
        assert failures[0][1]


def test_csv_round_trip(tmp_path):
    table = sweep(area_ratio=[2.0, 2.5], order=0, contour="smooth", progress=False)
    path = tmp_path / "sweep.csv"
    table.to_csv(str(path))
    lines = path.read_text().strip().splitlines()
    assert len(lines) == 3
    header = lines[0].split(",")
    assert "area_ratio" in header and "thrust" in header and "converged" in header


def test_table_renders():
    table = sweep(area_ratio=[2.0, 2.5], order=0, contour="smooth", progress=False)
    text = table.table()
    assert "area_ratio" in text
    assert len(text.splitlines()) == 4
