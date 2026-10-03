#!/usr/bin/env python3
"""Example 5 --- order-of-accuracy study.

Run:  ./dg2d.sh run convergence

Measures the observed order of accuracy from the entropy error, which is a true
error measure for shock-free flow (the exact solution is isentropic).

What to notice
--------------
The contour family decides what order you can reach.  `'smooth'` is C1 across
the throat and converges at the design rate p+1.  `'bell'` has a deliberate
slope discontinuity there -- the sharp-corner expansion of minimum-length nozzle
design -- which puts a Prandtl-Meyer singularity in the exact solution.  Its
error *plateaus*, and no amount of refinement recovers the rate.

That is physics, not a solver defect, and it is the reason a convergence study
must use `'smooth'` or `'analytic'`.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from src import performance, solve_nozzle
from src.plotting import plot_convergence_study


def study(contour: str, order: int, refines=(0, 1, 2)):
    dofs, errors = [], []
    for refine in refines:
        result = solve_nozzle(
            contour=contour, order=order, refine=refine,
            geometry_order=2,            # curved: otherwise geometry error dominates
            back_pressure_ratio=0.15,
            tolerance=1e-9, verbose=False, max_iterations=300_000,
        )
        if not result.converged:
            print(f"  skipped refine={refine}: {result.message}")
            continue
        dofs.append(result.discretization.n_dof)
        errors.append(performance(result).entropy_error)
        rate = ""
        if len(errors) > 1:
            rate = f"   rate {np.log(errors[-2] / errors[-1]) / np.log(2.0):5.2f}"
        print(f"  {contour:9s} p={order} refine={refine}  "
              f"DOF={dofs[-1]:6d}  entropy error {errors[-1]:.4e}{rate}")
    return np.array(dofs), np.array(errors)


def main() -> None:
    fig, ax = plt.subplots(figsize=(6.0, 4.6))
    for contour, order, expected in (
        ("smooth", 1, 2.0),
        ("smooth", 2, 3.0),
        ("bell", 1, 2.0),
    ):
        refines = (0, 1) if order == 2 else (0, 1, 2)
        dofs, errors = study(contour, order, refines)
        if len(dofs) > 1:
            plot_convergence_study(dofs, errors, ax=ax,
                                   label=f"{contour}, p={order}",
                                   expected_rate=expected if contour == "smooth" else None)
        print()

    ax.set_title("entropy error, curved elements (Q=2)", fontsize=10)
    fig.savefig("example_05.png", dpi=130, bbox_inches="tight")
    print("wrote example_05.png")


if __name__ == "__main__":
    main()
