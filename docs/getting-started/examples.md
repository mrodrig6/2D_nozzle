# Examples

Run these scripts from the repository root with `[all]` installed. The scripts are the complete recipes; the linked workflow pages explain their choices.

| Script | Command | Additional requirements | Purpose |
|---|---|---|---|
| [01_solve.py](../../examples/01_solve.py) | `./nozzle.sh example solve` | plot | a single design point, start to finish. |
| [02_sweep.py](../../examples/02_sweep.py) | `./nozzle.sh example sweep` | plot | sweeping the design space. |
| [03_sensitivity.py](../../examples/03_sensitivity.py) | `./nozzle.sh example sensitivity` | jax | exact design sensitivities by the discrete adjoint. |
| [04_optimise.py](../../examples/04_optimise.py) | `./nozzle.sh example optimise` | plot, jax, SciPy | gradient-based shape optimisation. |
| [05_convergence.py](../../examples/05_convergence.py) | `./nozzle.sh example convergence` | plot | order-of-accuracy study. |
| [06_shock.py](../../examples/06_shock.py) | `./nozzle.sh example shock` | plot | driving a shock through the nozzle with back pressure. |
| [07_external.py](../../examples/07_external.py) | `./nozzle.sh example external` | plot | the flow outside the nozzle, where the waves actually are. |

## Output and convergence

The solve, sweep, optimization and convergence studies write `example_*.png`; the sweep also writes `sweep_area_ratio.csv`; external flow writes `example_external.png` and `example_external_fields.png`. The sensitivity and shock examples print tables. Some examples intentionally record refused shock-band points.

For headless runs, set `MPLBACKEND=Agg`. Treat reported nonconvergence as a failed operating point even when a study script exits zero. Full adaptation, optimization and convergence studies cost more than a single solve; hardware and first-use compilation affect wall time.

## Common recipes

```bash
# Run the test suite before changing a study.
./nozzle.sh test
# Inspect a geometry before a full solve.
./nozzle.sh geometry contour=smooth
# Recheck tools after changing the environment.
./nozzle.sh check
```

See [solving](../running/solving.md), [sweeps](../running/sweeps.md), [sensitivities](../running/sensitivity.md), and [optimization](../running/optimization.md).
