# Numerical fluxes

The interface flux supplies the upwinding between neighboring states. Three fluxes ship: Roe, HLLC and AUSM⁺-up.

## Numerical flux

The interface flux is Roe's approximate Riemann solver,

```math
\hat{\mathbf{F}}(\mathbf{U}^{+}, \mathbf{U}^{-}, \mathbf{n})
= \tfrac{1}{2}\left(\mathbf{F}_n(\mathbf{U}^{+})
                  + \mathbf{F}_n(\mathbf{U}^{-})\right)
- \tfrac{1}{2}\left|\hat{\mathbf{A}}\right|
  \left(\mathbf{U}^{-} - \mathbf{U}^{+}\right),
```

where $\hat{\mathbf{A}}$ is the flux Jacobian evaluated at the Roe-averaged
state

```math
\hat{v}_x = \frac{\sqrt{\rho^{+}}\,v_x^{+} + \sqrt{\rho^{-}}\,v_x^{-}}
               {\sqrt{\rho^{+}} + \sqrt{\rho^{-}}},
\quad \text{(and likewise for } \hat{v}_y,\ \hat{H}\text{)},
\qquad
\hat{a}^{2} = (\gamma - 1)\left(\hat{H} - \tfrac{1}{2}|\hat{\mathbf{v}}|^{2}\right).
```

The dissipation term is evaluated in factored form rather than by assembling
$|\hat{\mathbf{A}}|$: with $\Delta(\cdot) = (\cdot)^{-} - (\cdot)^{+}$,

```math
\begin{aligned}
  G_1 &= (\gamma-1)\left(\tfrac{1}{2}|\hat{\mathbf{v}}|^{2}\Delta\rho
         - \hat{v}_x\,\Delta(\rho v_x) - \hat{v}_y\,\Delta(\rho v_y)
         + \Delta(\rho E)\right),\\
  G_2 &= -\hat{v}_n \Delta\rho + \Delta(\rho v_x)\, n_x
         + \Delta(\rho v_y)\, n_y,\\
  s_1 &= \tfrac{1}{2}\left(|\lambda_1| + |\lambda_2|\right), \qquad
  s_2 = \tfrac{1}{2}\left(|\lambda_1| - |\lambda_2|\right),\\
  C_1 &= \frac{G_1}{\hat{a}^{2}}\left(s_1 - |\lambda_3|\right)
         + \frac{G_2}{\hat{a}} s_2,
  \qquad
  C_2 = \frac{G_1}{\hat{a}} s_2 + \left(s_1 - |\lambda_3|\right) G_2 .
\end{aligned}
```

### Entropy fix

An unmodified Roe flux admits a non-physical expansion shock where an
eigenvalue passes through zero — which happens at the throat of every choked
nozzle. The Harten–Hyman fix replaces small eigenvalues by a smooth quadratic,

```math
|\lambda| \leftarrow
\begin{cases}
  \dfrac{\lambda^{2} + \epsilon^{2}}{2\epsilon}, & |\lambda| < \epsilon,\\
  |\lambda|, & \text{otherwise},
\end{cases}
\qquad \epsilon = 0.05\,\hat{a}.
```

Note what this fix is and is not. It is a *local repair* that rules out one
specific non-physical solution, and it proves nothing globally. It is **not**
entropy stability in the modern sense — a provable discrete entropy inequality,
built from an entropy-conservative two-point flux (Ismail and Roe; Tadmor) on
summation-by-parts operators. Roe contributed to both lines of work two decades
apart, which makes them easy to conflate, but only the second is a theorem.

### HLLC, and why it is the robust choice

`flux='hllc'` selects HLLC with Batten's wave-speed estimates: three waves —
two acoustic and the contact — instead of a full eigen-decomposition.

```math
\hat{\mathbf{F}} = \begin{cases}
    \mathbf{F}_L, & 0 \le S_L,\\
    \mathbf{F}_L + S_L(\mathbf{U}^{*}_L - \mathbf{U}_L), & S_L \le 0 \le S_M,\\
    \mathbf{F}_R + S_R(\mathbf{U}^{*}_R - \mathbf{U}_R), & S_M \le 0 \le S_R,\\
    \mathbf{F}_R, & S_R \le 0,
\end{cases}
```

with the star states carrying the contact's normal velocity and each side's own
tangential velocity,

```math
\rho^{*}_K = \rho_K \frac{S_K - v_{nK}}{S_K - S_M}, \qquad
\mathbf{v}^{*}_K = \mathbf{v}_K + (S_M - v_{nK})\,\mathbf{n},
```

```math
E^{*}_K = E_K + (S_M - v_{nK})
    \left(S_M + \frac{p_K}{\rho_K (S_K - v_{nK})}\right),
```

and Batten's estimates, which are what make the scheme positivity-preserving:

```math
S_L = \min(v_{nL} - a_L,\ \tilde{v}_n - \tilde{a}), \qquad
S_R = \max(v_{nR} + a_R,\ \tilde{v}_n + \tilde{a}).
```

Restoring the contact wave is what separates HLLC from HLL, whose missing middle
wave smears every shear layer.

**It supplies an assumption this code was relying on without having.** The
Zhang–Shu positivity limiter has a theorem, and that theorem *assumes* the
underlying first-order flux is positivity-preserving. HLLC with Batten's speeds
provably is; the Roe flux, with or without the entropy fix, is not. Under
`flux='roe'` the limiter has therefore been running without its central
hypothesis.

**And there is nothing to tune.** The HLL family cannot produce an
entropy-violating expansion shock, so HLLC needs no entropy fix and carries no
constant — `entropy_fix` is simply unread. A test checks the mass flux passes
smoothly through $M = 1$, which is exactly where the Roe flux needs its patch.

Measured against Roe on the shock-free design point, the answers agree to about
1% in thrust, HLLC converges in 451 iterations against Roe's 751 at $p = 0$, and
Roe is consistently the **more accurate** of the two — entropy error
$3.8\times10^{-3}$ against $6.2\times10^{-3}$ at $p = 1$, and
$1.3\times10^{-3}$ against $3.7\times10^{-3}$ at $p = 2$. That is the expected
trade: HLLC's three-wave model is more dissipative than Roe's full
decomposition. On the shocked point at $p = 0$ HLLC stalls at
$8.8\times10^{-2}$ where Roe stalls at $1.4\times10^{-1}$.

`hllc_low_mach` enables the low-Mach correction of Fleischmann et al., scaling
the acoustic wave speeds by
$\phi = \min(1,\ \max(|M_L|, |M_R|)/M_{\mathrm{lim}})$ while leaving the contact
untouched. `0` disables it and gives standard HLLC, which is the verified path;
the form follows the paper, so check the cutoff constant against it before
relying on that branch quantitatively.

### AUSM+-up, a flux with no wave decomposition

`flux='ausmup'` selects AUSM+-up (Liou, *J. Comput. Phys.* **214**, 137,
2006). It belongs to a different family from the other two. Roe and HLLC model
the jump as waves; AUSM-type schemes instead split the flux into a
*convective* part, carried by a single interface velocity, and a separate
*pressure* part:

```math
\hat{\mathbf{F}} = u_{1/2}\,\boldsymbol{\Psi}_{\text{upwind}} + p_{1/2}\,\mathbf{N},
\qquad
\boldsymbol{\Psi} = (\rho,\ \rho v_x,\ \rho v_y,\ \rho H)^T,
\qquad
\mathbf{N} = (0,\ n_x,\ n_y,\ 0)^T,
```

with $\boldsymbol{\Psi}$ upwinded on the sign of $u_{1/2}$. Because the pressure
is carried entirely by $p_{1/2}$, the convective part has no pressure
dissipation, and a contact discontinuity is reproduced exactly.

Both interface quantities come from polynomial splittings of each side's
normal Mach number $M_K = v_{n,K}/a_{1/2}$, plus the two "-up" terms the scheme
is named for:

```math
\begin{aligned}
u_{1/2} &= a_{1/2}\bigl[M^+_{(4)}(M_L) + M^-_{(4)}(M_R)\bigr]
  + K_p\,\frac{\max(1-\bar M^2,\,0)\,(p_L - p_R)}{\rho_{1/2}\,a_{1/2}},\\
p_{1/2} &= P^+_{(5)}(M_L)\,p_L + P^-_{(5)}(M_R)\,p_R
  + K_u\,P^+_{(5)}P^-_{(5)}\,\rho_{1/2}\,a_{1/2}\,(v_{n,L} - v_{n,R}),
\end{aligned}
```

with $a_{1/2}$ and $\rho_{1/2}$ arithmetic means and
$\bar M^2 = \tfrac12(M_L^2 + M_R^2)$. The splittings are pinned by two exact
identities, $M^+_{(4)} + M^-_{(4)} = M$ and $P^+_{(5)} + P^-_{(5)} = 1$, which
the tests check. Their constants $\beta = 1/8$ and $\alpha = 3/16$ are not
tunable; $K_p = 0.4$ and $K_u = 0.2$ are, and the values used are the
benchmarked ones carried over from the `approximate_riemann_solver` project.
That provenance matters: a version of this flux reconstructed from memory once
had $K_u$ effectively 7.5 times too large, which made the scheme
*anti-dissipative* on a moving contact — an error the identity tests cannot
catch.

The $K_p$ term is the pressure–velocity coupling that plain AUSM+ lacks. Without
it the mass flux vanishes whenever $v_{n,L} = v_{n,R} = 0$, however large the
pressure jump, so AUSM+ alone cannot start a flow from rest.

**Measured on the default design point** ($p_b/p_t = 0.15$):

| $p$ | flux | iterations | $c_F$ | $M_e$ |
|---|---|---|---|---|
| 1 | `roe` | 1001 | 0.17823 | 2.4646 |
| 1 | `ausmup` | 901 | 0.17674 | 2.4512 |
| 2 | `roe` | 1951 | 0.17703 | 2.4581 |
| 2 | `ausmup` | 1501 | 0.17604 | 2.4539 |

It converges, in fewer iterations than Roe, and the gap to Roe on $c_F$ falls
from 0.8% to 0.6% as the order rises — which is how a discretisation
difference behaves.

That was not a foregone conclusion. This project previously implemented
SLAU2, a sibling from the same family, and it reached the right answer and then
sat in a bounded limit cycle rather than converging; six hypotheses for why were
tested and refuted, including a full Jacobian eigenvalue study that found no
unstable modes. AUSM+-up does not inherit that behaviour. One visible structural
difference is the $K_u$ velocity-diffusion term in $p_{1/2}$, which SLAU2's
pressure flux lacks — a plausible lead, not a demonstrated cause.

Roe remains the default because every other number in this documentation was
produced with it. Switching is the cheapest way to ask how much of an answer
is the flux rather than the mesh.
