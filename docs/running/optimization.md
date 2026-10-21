# Shape optimization

Use a constrained SciPy optimizer with the discrete-adjoint gradient. The worked script below contains the complete objective, bounds and stopping criteria.

## 3. Shape optimisation

```bash
./dg2d.sh run optimise       # examples/04_optimise.py
```

The gradient plugs straight into SciPy:

```python
import numpy as np
from scipy.optimize import minimize
from src import differentiable_case

dc = differentiable_case(order=1, contour="bezier")
names = ("area_ratio", "bezier_w1", "bezier_w2")


def negative_thrust(x):
    params = dict(zip(names, x))
    value, grad, _ = dc.value_and_gradient("thrust", params, names=names)
    return -value, -np.array([grad[n] for n in names])


x0 = np.array([dc.default_params()[n] for n in names])
best = minimize(
    negative_thrust, x0, jac=True, method="L-BFGS-B", bounds=[(1.5, 5.0), (0.0, 1.0), (0.0, 1.0)]
)
```

All three live in [`examples/`](../../examples), alongside a first solve, a
convergence study and a back-pressure sweep. `./dg2d.sh list` names them.
