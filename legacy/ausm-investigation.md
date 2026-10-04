# Why the AUSM-family fluxes were removed

This is the investigation record, moved out of `docs/theory.md` when SLAU2 was
removed from the solver. It is kept with `legacy/riemann_solvers.py` because it
is the part that matters to a project with a shock on a body surface: it says
what was measured, what was refuted, and exactly how far the conclusion
generalises (not far).

The solver this was measured in refuses any operating point with a shock inside
the nozzle, so its domain is shock free. That is the single fact that makes the
conclusion below local rather than general.

---

### The AUSM family: why SLAU2 ships, and why it is not recommended

`flux='slau2'` selects SLAU2, the parameter-free low-dissipation AUSM-family
flux. An AUSM-family scheme does not solve a Riemann problem; it splits the
interface into a *mass* flux carrying the convective field and a *pressure* flux
carrying the acoustic one,

```math
\hat{\mathbf{F}} = \frac{\dot{m} + |\dot{m}|}{2}\,\boldsymbol{\Psi}_L
                 + \frac{\dot{m} - |\dot{m}|}{2}\,\boldsymbol{\Psi}_R
                 + \tilde{p}\,\mathbf{N},
\qquad
\boldsymbol{\Psi} = (1,\ v_x,\ v_y,\ H)^T .
```

SLAU2 was chosen over AUSM⁺-up2 for one reason: **it has no tunable constants.**
AUSM⁺-up carries $K_p$, $K_u$ and a cutoff Mach number $M_{co}$, and this package
carried that scheme briefly and removed it. One of the two failures found along
the way was traceable to $M_{co}$ — read as an epsilon and floored at `1e-8` it
makes $K_p/f_a$ diverge, turning a 1% pressure difference in still air into a
mass flux of $-9.8\times10^4$. SLAU2 builds its switches ($g$, $\chi$) from the
states themselves, so there is no constant to get wrong. `src/physics.py` carries
the full formulation, checked equation by equation against Kitamura & Shima's
Eqs. (2.3d)–(2.3l) and (3.5).

**The flux evaluation is sound.** It is consistent, conservative under swapping
the two sides and the normal, preserves a contact discontinuity exactly, upwinds
every convected quantity in the supersonic limit, and its low-Mach dissipation
scales as $O(M^2)$ — measured at 4.00 per halving of $M$. In the still-air
configuration that broke AUSM⁺-up it agrees with Roe to 3%. All of that is
pinned by tests in `tests/test_physics.py`.

**And it still does not converge.** At the design point it reaches the right
answer — thrust and area-averaged exit Mach within 0.7% of Roe — and then sits
there with a scaled residual of 4.8 against Roe's 9.6e-7, which never decays.
That is the same failure AUSM⁺-up showed. Five hypotheses were tested and four
were refuted outright:

| Hypothesis | Measurement | Verdict |
|---|---|---|
| Starved dissipation on flow-aligned faces | binned by $\lvert v_n\rvert/a$ on a converged Roe field: 1.01× on tangential faces, 0.49× only in the transonic band | refuted |
| An un-smoothed kink at $v_n = 0$ (Roe's entropy fix smooths its own) | smoothed each of the three kinks ($g$, $\lvert\overline{v_n}\rvert$, the $\dot m$ split) Harten–Hyman style | no change to 4 digits |
| No dissipation on the shear mode | at $v_n = 0$: SLAU2, HLLC and un-fixed Roe all give **exactly zero**; only fixed Roe gives 2.96e-4 | refuted — HLLC has zero too and converges |
| Under-dissipation generally | added Rusanov dissipation at 0.02/0.05/0.20, and separately on each of the four components | none converges; **more dissipation makes the residual worse** |
| A limiter artifact, or a frozen state | the limiter changes the state by exactly 0; the state keeps moving, amplitude saturating at 1.7e-3 (8e-4 relative) | not a limiter artifact — it is a bounded limit cycle |
| A linear instability: undamped modes in the semi-discrete operator | full 1680×1680 Jacobian of $-M^{-1}R$ by central differences at a common base state converged to $\lVert R\rVert_\infty = 2.1\times10^{-17}$ | **refuted** — see below |

The residual is also *distributed*, not localised: total $\lVert R\rVert$ is 1.13
against Roe's 6.9e-5, and the ten worst elements carry only 8.3% of it, spread
through the supersonic section rather than piled at one feature.

The spectrum is worth stating in full, because it rules out the explanation one
would reach for first:

| flux | span (most negative $\mathrm{Re}\,\lambda$) | $\max \mathrm{Re}\,\lambda$ | modes with $\mathrm{Re} \ge 0$ |
|---|---|---|---|
| `roe`   | −2.62e+02 | −2.18 | 0 of 1680 |
| `hllc`  | −2.61e+02 | −2.54 | 0 of 1680 |
| `slau2` | −3.56e+02 | −2.57 | 0 of 1680 |

**All three operators are strictly damped, and SLAU2's is indistinguishable from
HLLC's.** There is no linear instability to find. Whatever drives the limit
cycle is therefore not visible to a linearisation — which points at the scheme's
*switches*, the places where the flux is continuous but its derivative is not and
a finite-difference Jacobian quietly averages across the corner. That is a
coherent reading, but it is not a demonstration: the three switches that can be
smoothed ($g$, $\lvert\overline{v_n}\rvert$, the $\dot m$ split) were smoothed,
and the residual did not move.

**So the mechanism is not identified.** What is established is where it is *not*:
not the tuning constants (SLAU2 has none), not the flux evaluation (verified
against five properties), not under-dissipation (adding any amount, on any
component, makes it worse), not the limiter, not a localised feature, and not a
linear instability. Writing that down is more useful than a sixth hypothesis
offered without a measurement behind it.

One more variant was tried, and it is the one the source paper itself suggests.
Kitamura & Shima's whole argument is that the interfacial speed of sound $c_{1/2}$
matters: their Table 2 scores SLAU at 33 with the arithmetic mean (Eq. 2.3h, the
default implemented here) and 35 with the critical-speed form (Eq. 2.5g), and the
SLAU2 section calls the latter "slightly more robust". Swapping it in makes the
residual **worse** here, 10.0 against 4.8. Six variants, six refutations.

#### Why the published robustness results do not transfer

This is worth stating precisely, because the papers' own numbers make the point
better than any argument. Kitamura's 2016 assessment scores twelve fluxes on the
1.5D steady-normal-shock test — the standard carbuncle probe, $M_\infty = 6$, ten
shock positions, 20 points maximum:

| flux | shock-robustness score | captured shock |
|---|---|---|
| **Roe (E-fix)** | **0** | thin |
| Roe | 8 | thin |
| **HLLC** | **8** | thin |
| AUSM⁺-up | 16 | thin |
| HLLE | 16 | broad |
| AUSMPW+ | 17 | thin |
| van Leer FVS | 20 | broad |
| Hänel FVS | 20 | broad |
| **SLAU2** | **20** | broad |
| AUSM⁺-up2 | 20 | broad |

Read that against what happens in *this* solver, and the ranking is not merely
different — it is **inverted**. The three fluxes that converge here are Roe
(8), Roe with the entropy fix (**0**, the single worst scheme in the table) and
HLLC (8). The one that fails here, SLAU2, is tied for best (20). The entropy fix
that makes Roe usable in this DG march is precisely the modification that takes
Roe from 8 to 0 on their test, and the 2013 paper says so explicitly: "too much
dissipation addition to the flux yields 1D stability but in expense of Multi-D
stability, as reported in [4] for Roe flux with entropy-fix".

So the user's reading of the van Leer row was right — van Leer FVS really does
score 20, beating Roe and HLLC. It is just that the property being scored is
**robustness against carbuncle at a captured strong shock**, and this solver
refuses the shocked band outright: there is no strong shock in the domain for
that property to apply to. Meanwhile the thing the table does *not* score — a
damped steady state of an explicit high-order DG march — is the only thing that
matters here, and on the evidence of these ten rows it is anti-correlated with
what the table does score.

The "captured shock" column suggests why the two might genuinely trade off.
Every scheme that scores 20 is *broad*: it spreads a discontinuity over more
cells. Broadening is what kills the carbuncle. In a shock-free nozzle on a
high-order basis there is nothing to broaden, so the mechanism buys nothing —
and the schemes that are *thin*, which is what a high-order method wants, are
exactly the ones that converge. That is a hypothesis, not a demonstration: it is
consistent with all ten rows and with the six refutations above, but nothing here
tests it directly.

The practical conclusion is narrow. **Use `roe` or `hllc`.** `slau2` is shipped
because reproducing the AUSM-family failure with a *parameter-free* member of the
family rules out the tuning constants as the cause, and because having it
runnable is what made the comparison against the papers' own tables possible.
