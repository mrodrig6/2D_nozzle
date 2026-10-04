# `legacy/` — handover to the airfoil project

**This folder is not part of the solver.** Nothing in `src/` imports it, no test
runs it, and no documentation depends on it. It is a self-contained package of
work that belongs to the *next* codebase, kept here only until it is handed over.
Delete it once it has been.

## What is here

| file | what |
|---|---|
| `riemann_solvers.py` | All four interface fluxes this project implemented, verbatim, with their full derivations and an explanation of why two were removed |

`riemann_solvers.py` depends on nothing but NumPy and runs as-is:

```python
import sys

sys.path.insert(0, "legacy")
import numpy as np
import riemann_solvers as rs

g = 1.4
UL = np.array([1.0, 0.6, 0.0, 2.5])  # (rho, rho vx, rho vy, rho E)
UR = np.array([0.8, 0.4, 0.0, 2.0])
print(rs.hllc_flux(UL, UR, 1.0, 0.0, g).flux)
```

All four were checked for consistency (`F(U, U) == F(U)·n`) after extraction.

## Why this is being handed over rather than reused

The airfoil problem has a **shock on the body surface**. This nozzle solver does
not: it refuses any operating point that puts a shock inside the diverging
section, so its domain is shock free by construction. That difference is not a
detail — it inverts which flux is the right choice, and the file documents the
measurement behind that claim.

The short version: SLAU2 and AUSM⁺-up are *designed* for robustness at a
captured strong shock, and on the standard carbuncle test they score 20/20 and
16/20 while Roe-with-entropy-fix scores **0**. In this shock-free solver the
ranking is exactly reversed — Roe-with-entropy-fix is the only thing that
converges reliably, and SLAU2 does not converge at all.

**So do not carry this project's conclusion across.** "Use Roe or HLLC" is a
statement about a shock-free nozzle marched to steady state with explicit
high-order DG. For an airfoil with a surface shock, the schemes removed here are
plausibly the better starting point. Carry the measurement and the reasoning;
re-measure the conclusion.

## The two papers

Both are worth giving to the next project, and they are the source for the
claims above:

- K. Kitamura and E. Shima, "Towards shock-stable and accurate hypersonic
  heating computations: a new pressure flux for AUSM-family schemes",
  *J. Comput. Phys.* **245**, 62–83, 2013. — derives SLAU2 and AUSM⁺-up2;
  Eqs. (2.3) and (3.5) are what `slau2_flux` implements, and Table 2 is the
  shock-robustness comparison.
- K. Kitamura, "Assessment of SLAU2 and other flux functions with slope limiters
  in hypersonic shock-interaction heating", *Computers and Fluids* **129**,
  134–145, 2016. — Table 2 scores twelve fluxes on the 1.5D carbuncle test;
  Table 1 is the thin/broad and total-enthalpy classification.

They are not committed here; supply the PDFs directly.

## One caveat to pass along

The failure mode seen here — a scheme that reaches the right answer and then
sits in a bounded limit cycle instead of converging — was **never explained**,
after six tested-and-refuted hypotheses. If the airfoil solver is also an
explicit steady march and an AUSM-family flux stalls the same way, that is the
same unresolved problem, not a new one. `docs/theory.md` in this repository
records every measurement, including the ones that came back null.
