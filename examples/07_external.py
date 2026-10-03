#!/usr/bin/env python3
"""Example 7 --- the flow outside the nozzle, where the waves actually are.

Run:  ./dg2d.sh run external

The mesh stops at the lip. A nozzle designed the way example 6 argues for --
shock free inside the diverging section -- therefore has its entire wave system
*outside* the computed domain, and this example evaluates that in closed form
from the converged exit state.

What to notice
--------------
* The nozzle is solved **once**. Everything below is post-processing at three
  different ambient pressures, because for a supersonic exit the back pressure
  cannot reach back upstream -- the interior solution is identical in all three.
  That is not an approximation here; it is why the exit state is enough.
* The regime is set by ``p_e / p_amb``: a Prandtl-Meyer fan turning the flow
  outward when it exceeds 1, an oblique shock turning it inward when it is below,
  and nothing at all when it equals 1.
* ``jet_wave_cells`` extends the domain **downstream**, marching that first wave
  through its reflections off the symmetry axis and off the constant-pressure jet
  boundary. The pattern repeats -- those are the shock diamonds in a photograph
  of a rocket plume -- and the cell length is a number you can measure off one.
* The jet never settles to ambient. It rings about it: the axis pressure
  overshoots below ambient, comes back above, and repeats. In this model it rings
  forever, because every wave is treated as isentropic. A real jet's compressions
  steepen into shocks, lose total pressure, and the cells fade.
* Push the mismatch far enough and the march **stops**, because the regular
  reflection it assumes becomes a Mach disc. It says so rather than drawing a
  pattern that does not exist.
"""

from __future__ import annotations

import matplotlib.pyplot as plt

from src.api import solve_nozzle
from src.external import exit_wave_structure, jet_wave_cells, plot_jet_cells
from src.postprocess import performance


def main() -> None:
    result = solve_nozzle(order=2, refine=0, back_pressure_ratio=0.0640, verbose=False)
    perf = performance(result)
    design = float(perf.exit_pressure_ratio)
    print(f"exit: M = {perf.exit_mach_area_averaged:.4f}, p_e/p_t = {design:.5f}")
    print(
        f"this nozzle's own design point is p_amb/p_t = {design:.5f}, "
        "not the quasi-1D 0.0640 -- see src/external/plume.py for why\n"
    )

    cases = [
        ("strongly under-expanded", 0.030),
        ("at the design point", design),
        ("over-expanded", 0.100),
    ]
    for label, amb in cases:
        print(f"--- {label}: p_amb/p_t = {amb:.5f} ---")
        print("  " + exit_wave_structure(result, ambient_pressure_ratio=amb).describe())
        try:
            cells = jet_wave_cells(result, ambient_pressure_ratio=amb, cells=3)
        except ValueError as exc:
            print(f"  no cells to march: {exc}\n")
            continue
        print("  " + cells.describe().replace("\n", "\n  ") + "\n")

    # the two that have a wave pattern, side by side
    fig, axes = plt.subplots(
        4,
        1,
        figsize=(10.0, 10.5),
        height_ratios=(2.4, 1, 2.4, 1),
        constrained_layout=True,
    )
    for amb, (ax, ax_p) in zip(
        (0.030, 0.100),
        ((axes[0], axes[1]), (axes[2], axes[3])),
        strict=True,
    ):
        cells = jet_wave_cells(result, ambient_pressure_ratio=amb, cells=3)
        plot_jet_cells(result, cells, ax=ax, ax_pressure=ax_p)
        ax_p.set_xlim(*ax.get_xlim())
    fig.suptitle(
        "Extended downstream domain: the shock-cell pattern outside the nozzle",
        fontsize=12,
    )
    fig.savefig("example_external.png", dpi=150)
    print("wrote example_external.png")


if __name__ == "__main__":
    main()
