# Configuration

A case file is a JSON object of `name: value` pairs for one run. Pass its path where the arguments go, and later arguments override its values.

## Case files

A case lives in `cases/` and holds the same names the Python API uses:

```json
{
  "description": "the design point: shock-free, fully expanded, the case to start from",
  "area_ratio": 2.5,
  "back_pressure_ratio": 0.064,
  "order": 1,
  "geometry_order": 2,
  "refine": 0
}
```

The key `description` is for people. The launcher skips it, and `./nozzle.sh list` prints it.

Booleans become `yes` or `no`, and lists become comma-separated values, so `[-4, 8, 2]` is passed as `alphas=-4,8,2`.

## Using a case file

A case file is passed by its path, so a case you run often does not have to live in your shell history:

```bash
./nozzle.sh run cases/design.json                 # the values in cases/design.json
./nozzle.sh run cases/design.json pb=0.12         # the same case, with one value changed
./nozzle.sh sweep back_pressure_ratio 0.05 0.3 12 cases/design.json
./nozzle.sh bench cases/converged.json
```

**Later arguments win.** That is what makes `cases/design.json pb=0.12` an override rather than a conflict. It is a property of the underlying parser, so it holds for every key.

A case file does not include another. Where one case differs from another by a single value, it repeats the other's values in full.

| Case file | What it is |
|---|---|
| `cases/design.json` | the design point: shock-free, fully expanded |
| `cases/overexpanded.json` | exit plane below ambient, still shock-free |
| `cases/converged.json` | the design point at `order=2`, `refine=1` — quote numbers from this |
| `cases/shocked.json` | a shock in the diverging section, at `order=0` because `order>=1` diverges |
| `cases/contour.json` | geometry only, for `geometry` |
| `cases/default.json` | the default operating point, with the AUSM⁺-up flux and SSP-RK3 |

A case file works with any subcommand that accepts the keys it holds. `run`, `sweep` and `bench` take the full set, while `geometry` takes only the geometry keys. It reports `unrecognized arguments` if handed a case with solver options. A typo gets the same report rather than being silently ignored.

Case files cover one operating point on one mesh. For a sweep with a loop in it, a sensitivity study or an optimisation, write a Python case file instead. That is what [`examples/`](../../examples) contains, and the names are the same either way, so nothing has to be unlearned when you outgrow a case file.

## The shock case

The shipped `cases/shocked.json` does not opt into the internal-shock band. To study the failure deliberately, use a literal Boolean flag:

```bash
# Intentionally returns nonzero if the experiment does not converge.
./nozzle.sh run cases/shocked.json --allow-shock-in-nozzle
```

The Python spelling is `allow_shock_in_nozzle=True`. Use `solve_quasi1d` for the shock reference; this opt-in is not a convergence remedy.
