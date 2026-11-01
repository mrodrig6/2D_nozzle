# Configuration

A config is a JSON file of `name: value` pairs for one run. Pass it with `--deck name`, and later arguments override its values.

## Config files

A config lives in `cases/` and holds the same names the Python API uses:

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

The key `description` is for people. The launcher skips it and `./nozzle.sh list` prints it.

Booleans become `yes` or `no`, and lists become comma-separated values, so `[-4, 8, 2]` is passed as `alphas=-4,8,2`.

## Using a config

A `--deck name` argument is replaced by the values in [`cases/name.json`](../../cases), so a case you run often does not have to live in your shell history:

```bash
./nozzle.sh solve --deck design              # the values in cases/design.json
./nozzle.sh solve --deck design pb=0.12      # the same config, with one value changed
./nozzle.sh sweep back_pressure_ratio 0.05 0.3 12 --deck design
./nozzle.sh bench --deck converged
```

**Later arguments win.** That is what makes `--deck design pb=0.12` an override rather than a conflict. It is a property of the underlying parser, so it holds for every key.

A config does not include another config. Where one case differs from another by a single value, it repeats the other's values in full.

| Config | What it is |
|---|---|
| `--deck design` | the design point: shock-free, fully expanded |
| `--deck overexpanded` | exit plane below ambient, still shock-free |
| `--deck converged` | the design point at `order=2`, `refine=1` — quote numbers from this |
| `--deck shocked` | a shock in the diverging section, at `order=0` because `order>=1` diverges |
| `--deck contour` | geometry only, for `geometry` |
| `--deck default` | the default operating point, with the AUSM⁺-up flux and SSP-RK3 |

A config works with any subcommand that accepts the keys it holds. `solve`, `sweep` and `bench` take the full set. `geometry` takes only the geometry keys, and it reports `unrecognized arguments` if handed a config with solver options. A typo gets the same report rather than being silently ignored.

Configs cover one operating point on one mesh. For a sweep with a loop in it, a sensitivity study or an optimisation, write a Python case file instead. That is what [`examples/`](../../examples) contains, and the names are the same either way, so nothing has to be unlearned when you outgrow a config.

## The shock config

The shipped `--deck shocked` config does not opt into the internal-shock band. To study the failure deliberately, use a literal Boolean flag:

```bash
# Intentionally returns nonzero if the experiment does not converge.
./nozzle.sh solve --deck shocked --allow-shock-in-nozzle
```

The Python spelling is `allow_shock_in_nozzle=True`. Use `solve_quasi1d` for the shock reference; this opt-in is not a convergence remedy.
