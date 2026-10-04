#!/usr/bin/env python3
"""Example 3 --- exact design sensitivities by the discrete adjoint.

Run:  ./dg2d.sh run sensitivity     (needs JAX)

The adjoint gives dJ/da for *every* design variable at the cost of one extra
solve, where finite differences need two flow solves per variable.  This script
computes both and compares them, which is the check you should run once yourself
before trusting any gradient.

What to notice
--------------
* Agreement to ~1e-6 or better.  The adjoint is exact for the discrete problem.
* The cost gap grows with the number of variables: the adjoint is flat, finite
  differences are linear.
* d(thrust)/d(back_pressure) is *zero* at this operating point -- the exit is
  supersonic, so back pressure cannot influence the flow inside the nozzle.
  A gradient of exactly zero is physics, not a bug.
"""

from __future__ import annotations

import time

from src import check_gradient, differentiable_case


def main() -> None:
    dcase = differentiable_case(
        contour="smooth",  # shock-free and C1: the right setting for AD
        order=1,
        refine=0,
        back_pressure_ratio=0.15,
        tolerance=1e-11,  # the adjoint needs a tightly converged state
    )

    names = ("area_ratio", "throat_x", "theta_exit", "inlet_half_height", "back_pressure")
    print("design variables available:", sorted(dcase.default_params()))
    print()

    t0 = time.perf_counter()
    check_gradient(dcase, "thrust", names=names, step=1e-4)
    print(
        f"\n(verification took {time.perf_counter() - t0:.1f} s, "
        f"most of it in the {2 * len(names)} finite-difference solves)"
    )

    # The adjoint alone, which is what you would use in an optimiser
    t0 = time.perf_counter()
    value, grad, _ = dcase.value_and_gradient("thrust", names=names)
    print(f"\nadjoint only: {time.perf_counter() - t0:.1f} s for all {len(names)} derivatives")
    print(f"  thrust = {value:.8f}")
    for name in names:
        print(f"    d(thrust)/d({name:20s}) = {grad[name]:+.6e}")


if __name__ == "__main__":
    main()
