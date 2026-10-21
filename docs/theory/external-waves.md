# External waves

The external model evaluates waves from a converged supersonic exit state. It does not add an exterior CFD mesh.

`src.external.waves` implements the Prandtl–Meyer expansion relation and weak attached oblique-shock relations. `src.external.plume.exit_wave_structure` classifies the pressure mismatch; `src.external.jet.jet_wave_cells` marches regular reflections.

| Exit static pressure / ambient pressure | Lip response |
|---|---|
| Above one | Expansion fan turns the flow outward |
| One | Matched exit; no wave cells to march |
| Below one | Compression turns the flow inward |

For an expansion, the turning angle is the difference in Prandtl–Meyer angles:

```math
\nu(M)=\sqrt{\frac{\gamma+1}{\gamma-1}}\tan^{-1}\sqrt{\frac{\gamma-1}{\gamma+1}(M^2-1)}-\tan^{-1}\sqrt{M^2-1}.
```

The repeating cell march treats every wave as isentropic, so its pattern does not decay. Real compression waves steepen into shocks and lose total pressure. Strong mismatches can require a Mach disc; the regular-reflection model refuses that case instead of computing it.

The stitched plot uses quasi-1D theory inside and the computed two-dimensional exit state outside. A small discontinuity at the join is the difference between those models, not a plotting error. See [external-flow recipes](../running/external-flow.md) for the ratios and color scales.

`tests/test_external.py` checks expansion inversion, oblique-shock limits, periodic cells, ambient boundary states, and the stitched-field offset.
