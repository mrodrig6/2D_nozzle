#!/usr/bin/env python3
"""Example 6 --- driving a shock through the nozzle with back pressure.

Run:  python examples/06_shock_in_nozzle.py

Sweeps the back pressure across all four operating regimes and tracks where the
shock stands.

What to notice
--------------
* Above the first critical ratio the nozzle is *not choked* and the mass flow
  still responds to back pressure.  Below it, the mass flow is frozen -- that is
  the definition of choking, and it is visible in the table.
* Between the second and first critical ratios a normal shock stands in the
  diverging section, and it moves downstream as the back pressure falls.
* Shocked cases need `limiter='barth-jespersen'` and `scheme='ssprk3'`.  With no
  limiter the p=1 solution overshoots into negative pressure.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from dgnozzle import critical_ratios, sweep
from dgnozzle.plotting import plot_sweep


def main() -> None:
    area_ratio = 2.5
    crit = critical_ratios(area_ratio)
    print(f"AR = {area_ratio}:  {crit.describe()}")
    print()

    table = sweep(
        back_pressure_ratio=np.linspace(0.05, 0.97, 24),
        area_ratio=area_ratio,
        contour="smooth",
        order=1,
        refine=0,
        limiter="barth-jespersen",   # a shock at p>=1 needs this
        scheme="ssprk3",             # SSP stages match the positivity limiter
        max_iterations=200_000,
    )

    print()
    print(table.table(("thrust_coefficient", "mass_flow_in", "exit_mach",
                       "quasi1d_shock_x")))
    print()
    print("regimes encountered:", sorted(set(r for r in table.regimes if r)))

    for point, why in table.failures():
        print(f"FAILED at {point}: {why}")

    fig, axs = plt.subplots(1, 3, figsize=(14, 4), constrained_layout=True)
    metrics = ("thrust_coefficient", "mass_flow_in", "quasi1d_shock_x")
    for ax, metric in zip(axs, metrics, strict=True):
        plot_sweep(table, metric, ax=ax)
        for value, label in ((crit.third, "design"),
                             (crit.second, "shock at exit"),
                             (crit.first, "choking")):
            ax.axvline(value, ls="--", lw=0.8, color="0.55")
            ax.annotate(label, (value, ax.get_ylim()[1]), rotation=90, fontsize=7,
                        va="top", ha="right", color="0.4")
    axs[1].set_title("mass flow is frozen once choked", fontsize=9)
    fig.savefig("example_06.png", dpi=130, bbox_inches="tight")
    print("\nwrote example_06.png")


if __name__ == "__main__":
    main()
