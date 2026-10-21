# Extending the solver

Add a component at the existing source interfaces or postprocess a returned result. The worked metric below leaves the numerical implementation unchanged.

## Where to add things

- **A new approximate Riemann solver** → `physics/fluxes.py`, then the
  dispatch in `spatial/assembly.py` and `FLUXES` in `config.py`. If it should
  be fast, it also needs a kernel in `backends/_numba_kernels.py` and a row in
  `tests/test_backends.py`.
- **A new boundary condition** → `physics/boundary.py`, a tag in
  `geometry/mesh.py`, and a branch in `spatial/assembly.py`.
- **A new limiter** → `spatial/limiter.py`, and a name in `config.py`.
- **A new time stepper** → `timestepping/step.py` for its stability limit,
  `backends/base.py` for its stage combination, and a name in `config.py`.
- **A new element kind** → `reference/elements.py` and `reference/quadrature.py`;
  nothing above layer 2 should need to change, and that is the property those
  modules exist to protect.
- **A new contour family** → `geometry/contour.py` and `CONTOURS`.
- **A new derived quantity** → `post/postprocess.py`, and a figure in
  `post/plotting.py` if it is worth looking at.
- **A new entry point** → the top level, next to `api.py`.

Anything that needs a *new* layer probably belongs in `spatial/` or
`timestepping/`; the test for which is whether it would still make sense if
the time stepper changed.

## Worked extension: a custom performance metric

Add a postprocessor in your own study script; it needs no registry or solver edit. This example uses the same `Performance` values reported by `src.post.postprocess.performance`.

```python
from src import performance, solve_nozzle


def thrust_per_inlet_mass_flow(result):
    assert result.converged, result.message
    values = performance(result)
    return values.thrust / values.mass_flow_in


result = solve_nozzle(contour="smooth", verbose=False)
print(f"thrust / inlet mass flow = {thrust_per_inlet_mass_flow(result):.6f}")
```

The ratio carries the solver's velocity reference scale; it is not a value in SI units. To add a flux or time integrator, follow the module locations above and check every backend. This repository exposes source-level extension points, not a runtime plugin loader.
