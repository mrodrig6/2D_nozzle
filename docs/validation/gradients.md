# Gradient comparisons

This comparison uses the smooth, shock-free case in `examples/03_sensitivity.py`. It compares derivatives of the discrete solver, not of the exact Euler solution.

```bash
# Print all five adjoint and central-difference columns.
./dg2d.sh run sensitivity
```

Captured numerical output:

```text
objective thrust = 0.0458513361
parameter                     adjoint    finite diff    rel err
area_ratio               9.772082e-03   9.772082e-03   2.11e-08
throat_x                -1.490300e-02  -1.490298e-02   1.62e-06
theta_exit               7.602688e-03   7.602688e-03   5.40e-08
inlet_half_height       -1.524949e+00  -1.524949e+00   6.96e-08
back_pressure            0.000000e+00   0.000000e+00   0.00e+00
```

The relative errors depend on round-off and the finite-difference step. `tests/test_sensitivity.py` asserts agreement for selected variables and objectives. The supersonic exit makes the interior independent of back pressure, so its derivative is zero.

The [adjoint derivation](../theory/adjoint.md) explains the frozen logical grid. The [sensitivity workflow](../running/sensitivity.md) lists objectives and caveats.

## Additional recorded geometry check

The former theory page also recorded a length derivative at p=1, smooth contour and back-pressure ratio 0.15. Its adjoint and central difference both rounded to `1.353855e-2`, with relative difference `5.6e-8`. This is retained historical evidence; the captured five-variable run above does not include length.
