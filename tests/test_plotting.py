"""The plotting contract: which quantities can be drawn, and against what.

These do not check what a figure *looks* like -- that is not something a test
can usefully assert.  They check the part that silently rots: that every
quantity the module advertises can actually be drawn, and that one it does not
advertise is refused rather than failing somewhere inside Matplotlib with a
``KeyError`` that names nothing a caller recognises.
"""

from __future__ import annotations

import pytest

matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402

from src.api import solve_nozzle  # noqa: E402
from src.plotting import (  # noqa: E402
    CENTRELINE_QUANTITIES,
    plot_centreline,
    plot_field,
)
from src.postprocess import SCALARS, centreline_profile  # noqa: E402


@pytest.fixture(scope="module")
def solved():
    return solve_nozzle(order=1, refine=0, verbose=False)


@pytest.mark.parametrize("quantity", sorted(CENTRELINE_QUANTITIES))
def test_every_advertised_centreline_quantity_draws(solved, quantity):
    fig, ax = plt.subplots()
    plot_centreline(solved, ax=ax, quantity=quantity)
    plt.close(fig)


def test_an_unadvertised_centreline_quantity_is_refused(solved):
    with pytest.raises(ValueError, match="quantity must be one of"):
        plot_centreline(solved, quantity="entropy")


def test_centreline_quantities_are_all_available_in_the_profile(solved):
    """The table cannot name a quantity the profile does not carry.

    This is the failure the table was added to prevent: ``temperature`` was
    derivable by ``scalar_field`` and present in ``SCALARS``, but absent from
    the sampled profiles, so asking for it raised ``KeyError`` deep inside the
    plotting call.
    """
    cl = centreline_profile(solved)
    for quantity in CENTRELINE_QUANTITIES:
        assert quantity in cl, f"{quantity} is advertised but not sampled"


@pytest.mark.parametrize("name", ["mach", "temperature", "pressure"])
def test_field_contours_draw_for_the_headline_scalars(solved, name):
    """Mach is the default and the one the overview figure shows."""
    assert name in SCALARS
    fig, ax = plt.subplots()
    plot_field(solved, name, ax=ax, subdivisions=1, levels=8)
    plt.close(fig)
