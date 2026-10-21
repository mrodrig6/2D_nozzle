# Reading results

Read convergence before loads or performance numbers. This page defines the reported fields, normalization and saved outputs.

## Performance metrics

### Thrust

For steady inviscid flow the axial momentum balance over the nozzle interior
closes exactly, giving *two* expressions for the same force:

```math
F
= \underbrace{\int_{\mathrm{exit}}\!\left(\rho v_x^{2} + p\right)\mathrm{d}A
  - \int_{\mathrm{inlet}}\!\left(\rho v_x^{2} + p\right)\mathrm{d}A}_{\text{momentum form}}
= \underbrace{-\oint_{\mathrm{wall}} p\, n_x\, \mathrm{d}s}_{\text{wall form}} .
```

They are identical in exact arithmetic but *discretely* distinct, so the gap
between them is a free measure of discretisation error. Both are computed and
reported, together with

```math
c_F = \frac{F}{p_t A_{\mathrm{throat}}},
\qquad A_{\mathrm{throat}} = 2 y_{\mathrm{th}} .
```

> **The throat height must come from the geometry.** Dividing by a fixed value
> silently mis-normalises every coefficient computed at a different
> `area_ratio` or `throat_x`. For the same reason the wall integral runs over
> the contoured wall alone: folding the symmetry axis into it is harmless only
> while that axis stays horizontal.

### Entropy error

For shock-free flow the exact solution is isentropic, so

```math
\varepsilon_s = \sqrt{\frac{1}{A}\int_{\Omega}
  \left(\frac{s}{s_t} - 1\right)^{2}\mathrm{d}\Omega},
\qquad
s = \frac{p}{\rho^{\gamma}},
\qquad
s_t = p_t^{1-\gamma}\left(R T_t\right)^{\gamma},
```

is a genuine error measure that converges at the scheme's design rate. Across a
shock, entropy rises physically, and the number then reflects the flow rather
than the discretisation.
