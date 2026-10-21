# Library usage

Use `src.api.solve_nozzle` for one operating point. It routes keyword overrides to geometry, flow, discretization and solver objects.

```python
from src import performance, solve_nozzle

result = solve_nozzle(contour="smooth", area_ratio=2.5, order=1, verbose=False)
assert result.converged, result.message
print(performance(result).summary())
```

| Entry point / result | Purpose |
|---|---|
| `solve_nozzle(**overrides)` | Build and solve one operating point |
| `build_case(...)` | Prepare topology, mesh and operators |
| `Case.solve(options, backend=..., U0=...)` | Reuse a prepared case and optionally warm-start |
| `SolveResult.converged`, `.message` | Check whether the flow result is usable |
| `performance(result)` | Thrust, conservation, entropy and exit conditions |
| `sweep(...)` | Parameter grids with failure records |
| `differentiable_case(...)` | Prepare discrete-adjoint derivatives |

See [parameters](../model/parameters.md) for defaults and [the code map](../reference/code-map.md) for module locations. Keyword overrides take precedence over supplied configuration objects. An incompatible warm-start shape is rejected.

For a whole workflow, use the complete [example scripts](../getting-started/examples.md). Keep one repository on the Python import path: the companion airfoil package also uses the name `src`.
