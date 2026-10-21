"""Rebuild current solver figures beside this source.

From the repository root: python docs/figures/rebuild.py --output /tmp/nozzle-figures
Historical bytes and timings are not promised; all plotted states must converge.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from src import NozzleGeometry, solve_nozzle, sweep  # noqa: E402
from src.external import (  # noqa: E402
    jet_wave_cells,
    plot_jet_cells,
    plot_jet_field,
    plot_nozzle_and_plume,  # noqa: E402
)
from src.post.plotting import (  # noqa: E402
    plot_contour,
    plot_dg_vs_quasi1d,
    plot_exit_profile,
    plot_field,
    plot_mesh,
    plot_sweep,
    plot_wall,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--only", choices=("all", "overview"), default="all")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    def save(name):
        plt.gcf().savefig(args.output / name, dpi=130, bbox_inches="tight")
        plt.close("all")
        print(name)

    result = solve_nozzle(
        contour="smooth", area_ratio=2.5, order=1, geometry_order=2, verbose=False
    )
    assert result.converged, result.message
    _, axes = plt.subplots(2, 2, figsize=(11, 7), constrained_layout=True)
    plot_field(result, "mach", ax=axes[0, 0])
    plot_mesh(result, ax=axes[0, 1])
    plot_wall(result, ax=axes[1, 0])
    plot_exit_profile(result, ax=axes[1, 1])
    save("overview_design.png")
    if args.only == "overview":
        return
    plot_mesh(result)
    save("mesh.png")
    plot_dg_vs_quasi1d(result, "mach")
    save("stitched.png")
    _, axes = plt.subplots(3, 2, figsize=(10, 7))
    for contour, ax in zip(
        ("bell", "smooth", "conical", "moc", "bezier", "analytic"), axes.flat, strict=True
    ):
        plot_contour(NozzleGeometry(contour=contour), ax=ax)
        ax.set_title(contour)
    save("contours.png")
    table = sweep(
        area_ratio=np.linspace(2.0, 4.0, 9),
        contour="smooth",
        order=1,
        refine=1,
        back_pressure_ratio=0.15,
        progress=False,
    )
    assert table.converged.all(), list(table.failures())
    table.to_csv(args.output / "area_ratio_sweep.csv")
    plot_sweep(table, "thrust_coefficient")
    save("area_ratio_sweep.png")
    choked = sweep(
        back_pressure_ratio=np.linspace(0.05, 0.4, 8),
        contour="smooth",
        area_ratio=2.5,
        order=1,
        progress=False,
    )
    assert choked.converged.all(), list(choked.failures())
    plot_sweep(choked, "mass_flow_in")
    save("choked_mass_flow.png")
    external = solve_nozzle(order=2, back_pressure_ratio=0.064, verbose=False)
    assert external.converged, external.message
    cells = jet_wave_cells(external, ambient_pressure_ratio=0.03, cells=3)
    plot_jet_cells(external, cells)
    save("jet_cells.png")
    _, axes = plt.subplots(2, 1, figsize=(10, 6))
    for ax, quantity in zip(axes, ("pressure", "velocity"), strict=True):
        plot_jet_field(external, cells, quantity=quantity, ax=ax)
    save("jet_fields.png")
    plot_nozzle_and_plume(external, quantity="mach", ambient_pressure_ratio=0.03)
    save("nozzle_plume.png")


if __name__ == "__main__":
    main()
