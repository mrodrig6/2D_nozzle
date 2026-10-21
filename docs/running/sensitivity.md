# Sensitivity analysis

Compute derivatives of a converged discrete solution with JAX. Check the forward convergence and gradient before using it in a design loop.

## 2. Sensitivity analysis

```bash
./dg2d.sh run sensitivity    # examples/03_sensitivity.py
```

Exact derivatives by the **discrete adjoint** — one solve, any number of design
variables:

```python
from src import differentiable_case

dc = differentiable_case(order=1, contour="smooth", back_pressure_ratio=0.15, tolerance=1e-11)

value, grad, _ = dc.value_and_gradient(
    "thrust",
    names=("area_ratio", "throat_x", "bezier_w1", "bezier_w2", "theta_exit"),
)
print(value, grad)
```

Continue with the same `dc` to check it against finite differences:

Continue in the same Python session:

```python
from src import check_gradient

check_gradient(dc, "thrust", names=("area_ratio", "throat_x"))
```

See the captured comparison in [gradient validation](../validation/gradients.md).

**Thrust is the default objective, and it is differentiable end to end.** The
same number `performance(result).thrust` reports is what the adjoint
differentiates, so there is no separate "optimisation thrust" that could drift
from the reported one:

| `objective=` | What it maximises |
|---|---|
| `thrust` *(default)* | axial thrust, momentum form, per unit depth |
| `thrust_coefficient` | `thrust / (p_t A_throat)` |
| `exit_mach` | area-averaged exit Mach number |
| `mass_flow` | inlet mass flow |
| `exit_pressure` | area-averaged exit static pressure |



Design variables: every geometric one, plus `back_pressure`.

> The gradient is of the **discrete** problem, which is what optimisation needs.
> Use a shock-free operating point: a limiter switching on and off introduces
> kinks, and a shock's position is only piecewise differentiable.
