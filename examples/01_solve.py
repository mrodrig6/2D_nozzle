#!/usr/bin/env python3
"""Example 1 --- a single design point, start to finish.

Run:  ./dg2d.sh run solve

What to notice
--------------
* The DG result and quasi-1D theory agree closely at the exit but diverge in the
  expansion, where the streamlines are no longer parallel.  That gap *is* the
  two-dimensionality of the flow.
* Thrust is reported twice, from two mathematically identical integrals.  Their
  difference is a free estimate of discretisation error -- watch it fall when you
  raise `refine`.
"""

from __future__ import annotations

import matplotlib.pyplot as plt

from src import critical_ratios, performance, solve_nozzle, solve_quasi1d
from src.plotting import overview


def main() -> None:
    # --- what operating points are available for this area ratio? ----------
    area_ratio = 2.5
    print("Operating map for AR =", area_ratio)
    print(" ", critical_ratios(area_ratio).describe())
    print()

    # --- solve --------------------------------------------------------------
    result = solve_nozzle(
        contour="smooth",
        area_ratio=area_ratio,
        back_pressure_ratio=0.15,  # over-expanded: shock-free inside the nozzle
        order=1,
        geometry_order=2,  # curved elements: far better wall geometry
        refine=0,
    )

    if not result.converged:
        raise SystemExit(f"did not converge: {result.message}")

    # --- performance --------------------------------------------------------
    perf = performance(result)
    print()
    print(perf.summary())

    # --- compare against theory --------------------------------------------
    q1d = solve_quasi1d(result.geometry, result.flow)
    print()
    print("Against quasi-1D theory:")
    print(f"  exit Mach   DG {perf.exit_mach_area_averaged:.4f}   theory {q1d.exit_mach:.4f}")
    print(f"  mass flow   DG {perf.mass_flow_in:.6f}   theory {q1d.mass_flow:.6f}")
    print(f"  exit p/p_t  DG {perf.exit_pressure_ratio:.5f}   theory {q1d.exit_pressure:.5f}")

    overview(result)
    plt.savefig("example_01.png", dpi=130, bbox_inches="tight")
    print("\nwrote example_01.png")


if __name__ == "__main__":
    main()
