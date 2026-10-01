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
* The shocked points are run at **p=0**, and that is deliberate.  A shock inside
  the nozzle does not converge at p>=1 in this solver: the march leaves the
  physical state and the solver stops it.  Watch for the `FAILED at ...` lines
  below -- they are the honest output, not a bug in your setup.  See the "Known
  limitation" section of the README for the measurements behind that, and use
  `solve_quasi1d` when you want shock physics rather than a 2D field.
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
        order=0,                     # a shock in the nozzle does not converge
                                     # at p>=1 -- see the docstring above
        refine=1,
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
