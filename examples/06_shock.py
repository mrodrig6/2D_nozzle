#!/usr/bin/env python3
"""Example 6 --- driving a shock through the nozzle with back pressure.

Run:  ./dg2d.sh run shock

Sweeps the back pressure across all four operating regimes and tracks where the
shock stands.

What to notice
--------------
* Above the first critical ratio the nozzle is *not choked* and the mass flow
  still responds to back pressure.  Below it, the mass flow is frozen -- that is
  the definition of choking, and it is visible in the table.
* Between the second and first critical ratios a normal shock stands in the
  diverging section, and it moves downstream as the back pressure falls.
* The band with a shock *inside* the diverging section is **refused** by the
  solver, and the sweep records those points as failures rather than pretending
  to solve them.  That band is not what a nozzle-design exercise wants anyway:
  sizing a nozzle means avoiding a shock in the diverging section, and the
  interesting wave structure -- oblique shocks when over-expanded, a
  Prandtl-Meyer fan when under-expanded -- forms *outside* the exit plane.
* So the DG points below are the shock-free ones, and quasi-1D theory supplies
  the rest of the operating map.  `shock_free_range(area_ratio)` is where the
  boundary comes from.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from src import (
    FlowConditions,
    build_case,
    critical_ratios,
    is_shock_free,
    shock_free_range,
    solve_quasi1d,
    sweep,
)
from src.plotting import plot_sweep


def main() -> None:
    area_ratio = 2.5
    crit = critical_ratios(area_ratio)
    print(f"AR = {area_ratio}:  {crit.describe()}")
    print()

    # the full operating map from quasi-1D theory, including the refused band
    print("quasi-1D operating map across the whole range:")
    for pb in np.linspace(0.05, 0.97, 13):
        q = solve_quasi1d(
            build_case(contour="smooth", area_ratio=area_ratio).geometry,
            FlowConditions(back_pressure_ratio=float(pb)),
        )
        mark = " " if is_shock_free(area_ratio, float(pb)) else "*"
        print(f"  {mark} p_b/p_t={pb:5.3f}  {q.regime.value}")
    print("  (* = shock inside the diverging section; the DG solver refuses it)")
    print()

    # the DG sweep, over the shock-free set only
    lo, _hi = shock_free_range(area_ratio)
    table = sweep(
        back_pressure_ratio=np.linspace(0.02, lo[1] * 0.97, 16),
        area_ratio=area_ratio,
        contour="smooth",
        order=1,  # the set is shock free, so p>=1 converges
        refine=1,
        scheme="ssprk3",  # SSP stages match the positivity limiter
        max_iterations=200_000,
    )

    print()
    print(table.table(("thrust_coefficient", "mass_flow_in", "exit_mach", "quasi1d_shock_x")))
    print()
    print("regimes encountered:", sorted(set(r for r in table.regimes if r)))

    for point, why in table.failures():
        print(f"FAILED at {point}: {why}")

    fig, axs = plt.subplots(1, 3, figsize=(14, 4), constrained_layout=True)
    metrics = ("thrust_coefficient", "mass_flow_in", "quasi1d_shock_x")
    for ax, metric in zip(axs, metrics, strict=True):
        plot_sweep(table, metric, ax=ax)
        for value, label in (
            (crit.third, "design"),
            (crit.second, "shock at exit"),
            (crit.first, "choking"),
        ):
            ax.axvline(value, ls="--", lw=0.8, color="0.55")
            ax.annotate(
                label,
                (value, ax.get_ylim()[1]),
                rotation=90,
                fontsize=7,
                va="top",
                ha="right",
                color="0.4",
            )
    axs[1].set_title("mass flow is frozen once choked", fontsize=9)
    fig.savefig("example_06.png", dpi=130, bbox_inches="tight")
    print("\nwrote example_06.png")


if __name__ == "__main__":
    main()
