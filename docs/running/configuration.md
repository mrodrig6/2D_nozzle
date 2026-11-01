# Configuration

Store repeatable CLI arguments in a case deck. Later arguments override earlier entries, including arguments expanded from a deck.

## Case files you pass as an argument

A case you run often does not have to live in your shell history. A `--deck name`
argument is replaced by the lines of [`cases/name.dg`](../../cases), so:

```bash
./nozzle.sh solve --deck design              # exactly the lines in cases/design.dg
./nozzle.sh solve --deck design pb=0.12      # the same deck, with one value changed
./nozzle.sh sweep back_pressure_ratio 0.05 0.3 12 --deck design
./nozzle.sh bench --deck converged
```

A deck is the same `name=value` lines you would have typed — there is no second
vocabulary and no schema to keep in step with the code:

```
# the design point: shock-free, fully expanded, the case to start from
ar=2.5
pb=0.064
p=1
Q=2
ref=0
```

`#` starts a comment, blank lines are ignored, and **a deck may name another
deck** to build on it, which is all `cases/overexpanded.dg` is:

```
# over-expanded: the exit plane sits below ambient, still shock-free
--deck design
pb=0.15
```

**Later arguments win.** That is what makes `--deck design pb=0.12` an override rather
than a conflict, and it is a property of the underlying parser rather than
something the launcher arranges, so it holds for every key.

| Deck | What it is |
|---|---|
| `--deck design` | the design point: shock-free, fully expanded |
| `--deck overexpanded` | exit plane below ambient, still shock-free |
| `--deck converged` | the design point at `p=2`, `refine=1` — quote numbers from this |
| `--deck shocked` | a shock in the diverging section, at `p=0` because `p>=1` diverges |
| `--deck contour` | geometry only, for `geometry` |

Because this is argument expansion and nothing more, a deck works with any
subcommand that accepts the keys it holds: `solve`, `sweep` and `bench` take the
full set, while `geometry` takes only the geometry keys and will say
`unrecognized arguments` if handed a deck carrying solver options — which is
also what a typo gets, rather than being silently ignored.

Decks cover one operating point on one mesh. For a sweep with a loop in it, a
sensitivity study or an optimisation, write a Python case file instead — that is
what [`examples/`](../../examples) is, and the names are the same either way, so
nothing has to be unlearned when you outgrow a deck.

## Refused shock deck

The shipped `--deck shocked` deck does not opt into the internal-shock band. To study the failure deliberately, use a literal Boolean flag:

```bash
# Intentionally returns nonzero if the experiment does not converge.
./nozzle.sh solve --deck shocked --allow-shock-in-nozzle
```

The Python spelling is `allow_shock_in_nozzle=True`. Use `solve_quasi1d` for the shock reference; this opt-in is not a convergence remedy.
