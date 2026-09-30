#!/usr/bin/env python3
"""Example 4 --- gradient-based shape optimisation.

Run:  python examples/04_shape_optimisation.py     (needs JAX and SciPy)

Maximises thrust over a Bezier wall using L-BFGS-B with adjoint gradients.

What to notice
--------------
* Each iteration costs one flow solve plus one adjoint solve, regardless of how
  many shape variables there are.  Add more Bezier weights and the cost per
  iteration barely moves.
* The optimiser is working on the *discrete* problem.  Refine the mesh and the
  optimum shifts slightly -- that shift is discretisation error, and checking it
  is part of doing this honestly.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize

from dgnozzle import NozzleGeometry, differentiable_case
from dgnozzle.plotting import plot_contour

NAMES = ("area_ratio", "bezier_w1", "bezier_w2")
BOUNDS = [(1.8, 4.5), (0.05, 0.95), (0.30, 0.99)]


def main() -> None:
    dcase = differentiable_case(
        contour="bezier", order=1, refine=0,
        back_pressure_ratio=0.15, tolerance=1e-10,
    )

    history: list[tuple[np.ndarray, float]] = []

    def objective(x: np.ndarray):
        params = dict(zip(NAMES, x, strict=True))
        value, grad, _ = dcase.value_and_gradient("thrust", params, names=NAMES)
        history.append((x.copy(), value))
        print(f"  thrust = {value:.7f}  at  " +
              ", ".join(f"{n}={v:.4f}" for n, v in params.items()))
        return -value, -np.array([grad[n] for n in NAMES])

    x0 = np.array([dcase.default_params()[n] for n in NAMES])
    print("optimising thrust over", NAMES)
    result = minimize(objective, x0, jac=True, method="L-BFGS-B",
                      bounds=BOUNDS, options={"maxiter": 25, "ftol": 1e-10})

    print(f"\n{result.message}")
    print(f"  {len(history)} objective evaluations, {result.nit} iterations")
    print(f"  thrust {history[0][1]:.7f} -> {-result.fun:.7f} "
          f"({100 * (-result.fun / history[0][1] - 1):+.2f}%)")
    for name, value in zip(NAMES, result.x, strict=True):
        print(f"    {name:12s} {dict(zip(NAMES, x0, strict=True))[name]:.4f} -> {value:.4f}")

    # --- before and after --------------------------------------------------
    fig, ax = plt.subplots(figsize=(9, 3.2))
    for x, label, style in ((x0, "initial", "--"), (result.x, "optimised", "-")):
        geom = NozzleGeometry(contour="bezier", **dict(zip(NAMES, x, strict=True)))
        plot_contour(geom, ax=ax, mirror=False, ls=style, label=label,
                     color="0.4" if label == "initial" else "tab:blue")
    ax.legend(fontsize=9, frameon=False)
    ax.set_title("thrust-optimised wall", fontsize=10)
    fig.savefig("example_04.png", dpi=130, bbox_inches="tight")
    print("\nwrote example_04.png")


if __name__ == "__main__":
    main()
