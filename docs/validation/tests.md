# Test-suite claims

The existing suite checks the claims listed here. Slow tests run converged flow solves; optional backend tests require the matching extra.

## Exactness checks

The following hold to machine precision and are asserted in the test suite:

- **Geometric conservation.** For a uniform state with the exact normal flux
  imposed on every edge, the total residual is $\mathcal{O}(10^{-17})$ — the
  volume gradient term and the face term cancel identically. This is the
  strongest available check on the metric terms.
- **Discrete divergence theorem.**
  $\sum_f \mathbf{n}_f \ell_f = \mathbf{0}$ per element to $10^{-12}$, for
  triangles and quadrilaterals, $Q = 1$ and $Q = 2$.
- **Flux consistency and conservation.** $\hat{\mathbf{F}}(\mathbf{U}, \mathbf{U}, \mathbf{n}) = \mathbf{F}_n(\mathbf{U})$, and $\hat{\mathbf{F}}(\mathbf{U}^{+}, \mathbf{U}^{-}, \mathbf{n}) = -\hat{\mathbf{F}}(\mathbf{U}^{-}, \mathbf{U}^{+}, -\mathbf{n})$.
- **Inflow boundary condition.** A uniform isentropic state is reproduced
  exactly, to $10^{-15}$, for $M \in [0.05, 0.9]$.
- **Backend agreement.** The Numba and NumPy residuals agree to a relative
  $4\times10^{-15}$.
- **Order continuation.** Re-projection $p \to p+1 \to p$ returns the original
  coefficients to $10^{-15}$, and preserves cell averages exactly.
- **Limiter conservation.** Cell averages are bit-identical before and after
  limiting whenever the average is admissible.

## Running the suite

```bash
# Run all scientific and interface assertions.
./dg2d.sh test
# Run fast tests only.
./dg2d.sh test --fast
# Run slow solves and gradient checks only.
./dg2d.sh test --slow
```

Install `[all,dev]` to include Numba and JAX coverage. A missing optional backend produces skips; `--fast` deselects slow tests. Neither outcome is a pass for the omitted checks. CI requires numerical backends to import before the full run.

## Claims and implementation

- All three backends must compute the same residual. [test_backends.py](../../tests/test_backends.py).
- Command-line interface. [test_cli.py](../../tests/test_cli.py).
- The ``dg2d.sh`` launcher. [test_dg2d.py](../../tests/test_dg2d.py).
- The documentation's maths has to survive GitHub's Markdown, not just LaTeX. [test_docs.py](../../tests/test_docs.py).
- Lagrange bases must interpolate, reproduce constants, and order edges consistently. [test_elements.py](../../tests/test_elements.py).
- Closed-form external wave structure at the nozzle lip. [test_external.py](../../tests/test_external.py).
- Contour families must hit their design targets and stay differentiable. [test_geometry.py](../../tests/test_geometry.py).
- The limiter must restore positivity, preserve cell averages, and stay inactive on smooth solutions. [test_limiter.py](../../tests/test_limiter.py).
- Mesh topology and the metric terms. [test_mesh_operators.py](../../tests/test_mesh_operators.py).
- Is the solver actually non-dimensional, or only documented as such? [test_nondimensional.py](../../tests/test_nondimensional.py).
- Fluxes and boundary conditions. [test_physics.py](../../tests/test_physics.py).
- The plotting contract: which quantities can be drawn, and against what. [test_plotting.py](../../tests/test_plotting.py).
- Performance metrics, line-outs and field sampling. [test_postprocess.py](../../tests/test_postprocess.py).
- Quadrature rules must integrate polynomials exactly and have positive weights. [test_quadrature.py](../../tests/test_quadrature.py).
- Quasi-1D theory: the reference solution the DG result is checked against. [test_quasi1d.py](../../tests/test_quasi1d.py).
- Adjoint design sensitivities, verified against finite differences. [test_sensitivity.py](../../tests/test_sensitivity.py).
- The operating set a nozzle-design exercise actually uses. [test_shock_free.py](../../tests/test_shock_free.py).
- End-to-end solves: convergence, conservation, and agreement with theory. [test_solver.py](../../tests/test_solver.py).
- Parameter sweeps: axis handling, warm starting, and failure reporting. [test_sweep.py](../../tests/test_sweep.py).
- The pseudo-time step and its stability calibration. [test_timestep.py](../../tests/test_timestep.py).
