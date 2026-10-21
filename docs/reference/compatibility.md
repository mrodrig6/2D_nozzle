# Compatibility

Use this matrix to distinguish an implemented feature from a measured or untested combination.

| Combination | Status / restriction |
|---|---|
| NumPy reference solver | ✓ |
| Numba compiled solver | ✓; first run compiles kernels |
| JAX differentiation | ✓; CPU float64 is used for gradient checks |
| Triangles and quadrilaterals | ✓ |
| Curved geometry | ✓; use `geometry_order=2` for accuracy studies |
| Euler / SSP-RK3 | ✓; default is order-dependent |
| RK4 | ✗; removed |
| Internal normal shock | ✗ by default; explicit opt-in only for a nonconverged experiment |
| `analytic` shape design | Restricted: fixed contour ignores selected geometric parameters |
| External plume | Restricted: idealized regular reflections; no Mach-disc solution or dissipative plume march |
| Viscosity, turbulence, unsteady physical time | ✗; outside the model |
| Linux / Python 3.12 CPU | ✓; local validation environment |
| Python 3.10 / newest stable | CI targets; results depend on the workflow run |
| Windows / macOS / GPU | Not validated by this change |

See [assumptions](../model/problem.md), [parameters](../model/parameters.md), and [backend behavior](../method/backends.md).
