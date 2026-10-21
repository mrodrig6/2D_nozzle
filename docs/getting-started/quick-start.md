# Quick start

Run this from the checkout root after installation. The displayed output was captured with Python 3.12 and the Numba backend.

```bash
# Solve one operating point and check the command's exit status.
./dg2d.sh solve contour=smooth ar=2.5 pb=0.15 p=1 Q=2 --quiet
```

```text
thrust        0.046033  (c_F = 0.164527, 90.50% of ideal)
  wall form   0.045675  (imbalance 7.78e-03)
mass flow     in 0.302020, out 0.302313  (imbalance 9.70e-04, C_d = 0.576995)
exit          M = 2.4517, p/p_t = 0.06691, T/T_t = 0.45613
entropy error 1.5005e-03
```

- `thrust` is the full-channel momentum balance per unit depth; `c_F` divides it by reservoir pressure and throat area.
- `wall form` is an independent discrete pressure integral. Its imbalance estimates discretization error.
- `mass flow` reports inlet/outlet values; `C_d` normalizes inlet mass flow by the reservoir density, sound speed and throat area.
- `exit` gives area-averaged Mach number, pressure ratio and temperature ratio.
- `entropy error` measures departure from isentropic flow. It is numerical error only when the true flow is shock-free.

`--quiet` removes the march history. The CLI still exits nonzero if the solve does not converge. Do not interpret a nonzero-exit run as a design result.

See [output definitions](../running/output.md), [configuration](../running/configuration.md), and [the complete examples](examples.md).
