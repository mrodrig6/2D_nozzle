# Accuracy and curved geometry

These retained measurements compare entropy and geometry error under refinement. A throat singularity can limit the observed order even when the discretization is correct.

## Order of accuracy

Entropy error on uniformly refined meshes at the shock-free design point
($p_b/p_t = 0.15$), with curved ($Q = 2$) elements. The expected rates are
$p+1$.

| Contour | $p$ | DOF | $\varepsilon_s$ | observed rate |
|---|---|---|---|---|
| `analytic` | 1 | 420 | $1.904\times10^{-3}$ | — |
| | 1 | 1680 | $4.996\times10^{-4}$ | 1.93 |
| | 1 | 6720 | $1.280\times10^{-4}$ | 1.96 |
| | 2 | 840 | $1.871\times10^{-4}$ | — |
| | 2 | 3360 | $2.872\times10^{-5}$ | 2.70 |
| `smooth` | 1 | 420 | $1.506\times10^{-3}$ | — |
| | 1 | 1680 | $3.835\times10^{-4}$ | 1.97 |
| | 1 | 6720 | $9.941\times10^{-5}$ | 1.95 |
| | 2 | 840 | $1.587\times10^{-4}$ | — |
| | 2 | 3360 | $2.635\times10^{-5}$ | 2.59 |
| `bell` | 1 | 420 | $3.835\times10^{-3}$ | — |
| | 1 | 1680 | $7.780\times10^{-4}$ | 2.30 |
| | 1 | 6720 | $7.731\times10^{-4}$ | **0.01** |

The $C^1$ contours achieve their design rates. The `'bell'` contour *stalls*:
its $21.3^{\circ}$ throat corner launches a Prandtl–Meyer expansion fan, a
genuine singularity in the exact solution, and no amount of mesh refinement
recovers the asymptotic rate. This is physics, not a solver defect — but it
means *a convergence study must use* `'smooth'` *or* `'analytic'`.

The thrust functional converges cleanly in all cases. The gap between the two
forms of the thrust integral falls as
$9.7\times10^{-3} \to 2.7\times10^{-3} \to 6.6\times10^{-4}$ for $p=1$, $Q=1$
under refinement (rate $\approx 1.9$), and reaches $6.7\times10^{-5}$ at $p=2$,
$Q=2$.

## Effect of curved elements

At fixed element count, raising the geometry order from $Q = 1$ to $Q = 2$
reduces the error in the discrete domain area from $1.10\times10^{-3}$ to
$8.76\times10^{-6}$ — a factor of 125 — because the curved wall is represented
quadratically rather than by chords. At $p = 2$ the entropy error improves
correspondingly, from $1.30\times10^{-3}$ to $2.91\times10^{-4}$. With $Q = 1$
the geometry error dominates and caps the observed rate near 1.6 regardless of
$p$.
