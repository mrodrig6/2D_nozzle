#!/usr/bin/env python3
"""Example 2 --- sweeping the design space.

Run:  ./dg2d.sh run sweep

Sweeps the area ratio at a fixed back pressure and finds the thrust optimum.

What to notice
--------------
There *is* an optimum, and it is not the largest nozzle.  Increasing the area
ratio raises the exit Mach number, which helps; but past the matched condition
the exit pressure falls below the back pressure and the nozzle is over-expanded,
which hurts.  The two effects cross.

The sweep warm-starts each point from the previous one, so the whole study costs
much less than the sum of its parts.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from dgnozzle import sweep
from dgnozzle.plotting import plot_sweep


def main() -> None:
    area_ratios = np.linspace(2.0, 4.5, 11)

    table = sweep(
        area_ratio=area_ratios,   # iterable -> a sweep axis
        contour="smooth",         # scalars  -> fixed settings
        order=1,
        refine=1,                 # ref=0 under-resolves the largest area ratios
        back_pressure_ratio=0.15,
    )

    print()
    print(table.table(("thrust_coefficient", "exit_mach", "quasi1d_exit_mach",
                       "thrust_efficiency")))

    for point, why in table.failures():
        print(f"\nFAILED at {point}: {why}")

    ok = table.converged
    if ok.any():
        best = int(np.argmax(np.where(ok, table["thrust_coefficient"], -np.inf)))
        print(f"\nbest thrust coefficient at area_ratio = "
              f"{table['area_ratio'][best]:.3f}: "
              f"c_F = {table['thrust_coefficient'][best]:.5f}")

    table.to_csv("sweep_area_ratio.csv")
    print("wrote sweep_area_ratio.csv")

    fig, axs = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
    plot_sweep(table, "thrust_coefficient", ax=axs[0])
    plot_sweep(table, "exit_mach", ax=axs[1], reference="quasi1d_exit_mach")
    fig.savefig("example_02.png", dpi=130, bbox_inches="tight")
    print("wrote example_02.png")


if __name__ == "__main__":
    main()
