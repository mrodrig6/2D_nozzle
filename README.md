# 2D Euler DG Nozzle Solver

A 2D compressible Euler solver using a discontinuous Galerkin (DG) method,
for use in the nozzle design lab.

## Quick start

1. Clone this repository
2. Download the MEX binary for your platform from the [Releases page](../../releases/latest)
3. Place the binary in the same folder as `main.m`
4. Open `main.m`, set your design parameters, and run

## Design parameters (top of `main.m`)

```matlab
area_ratio   = 2.50;      % nozzle exit-to-throat area ratio
throat_x     = 0.14;      % throat x-location (m)
contour_type = 'smooth';  % 'smooth' | 'conical' | 'moc'
p_back_ratio = 0.15;      % back pressure / total inlet pressure
ref_max      = 1;         % mesh refinement level (0=coarse, 1=medium, 2=fine)
p_max        = 1;         % max polynomial order (0 or 1)
```

## Using the MEX binary (faster)

Download from the [Releases page](../../releases/latest):

| File | Platform |
|------|----------|
| `rk4_mex_p0q1r0.mexa64`    | Linux 64-bit |
| `rk4_mex_p0q1r0.mexw64`    | Windows 64-bit |
| `rk4_mex_p0q1r0.mexmaci64` | macOS Intel (pre-2021 Mac) |
| `rk4_mex_p0q1r0.mexmaca64` | macOS Apple Silicon (M1/M2/M3) |

> **Not sure which Mac?** Apple menu → About This Mac.
> "Apple M1/M2/M3" → download `mexmaca64`. "Intel" → download `mexmaci64`.

In `main.m`, find the solver call and swap to the MEX version:

```matlab
% Slow (pure MATLAB — works without downloading anything):
% U = rk4(resdata, U);

% Fast (requires MEX binary from Releases):
U = rk4_mex_p0q1r0(resdata, U);
```

MATLAB selects the correct binary automatically — no other changes needed.

## Contour types

| `contour_type` | Description |
|----------------|-------------|
| `'smooth'`     | Analytic bell nozzle (default) |
| `'conical'`    | Straight-walled diverging section |
| `'moc'`        | Parabolic approximation to MOC minimum-length contour |

The converging section is always a smooth cubic Hermite.

## Refinement guide

| `ref_max` | Elements | Est. runtime (pure MATLAB) |
|-----------|----------|---------------------------|
| 0         | ~140     | ~30 s                     |
| 1         | ~560     | ~10 min                   |
| 2         | ~2240    | hours                     |

Use `ref_max = 0` for design exploration. Use `ref_max = 1` with MEX for
publication-quality results.
