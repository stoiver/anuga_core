# Subgrid terrain: literature notes

Notes gathered on 2026-11-06 for the design note (#508). Only the first paper
was read in detail (sections 1–4.7 and 5.1–5.3 verbatim, the rest from a
summary); the others are from abstracts, vendor documentation and search
results. Check the originals before relying on any detail.

## Closest to ANUGA: Bitsch, Heilmann, Engsig-Karup, Sørensen & Grooss (2026)

"An unstructured second-order subgrid method for the shallow water equations",
arXiv:2601.01895 (DHI and DTU; submitted to IJNMF).
https://arxiv.org/html/2601.01895v1

An explicit, cell-centred finite volume scheme on unstructured triangles, like
ANUGA, with the coarse triangle split uniformly, like the two-level idea in
the design.

1. **Subgrid level.**
   - Each coarse triangle is split into n_sg² equal subcells, and each edge
     into n_sg subfaces (Sec. 3, Eq. 5–9).
   - The bed is piecewise constant per subcell, even with P1 reconstruction
     (Sec. 3.1).
   - Subcell depth: h_ℓ = max(0, η̄ + ∇η·(x_ℓ − x_m) + d_ℓ) (Eq. 13, 35).
     Their d is depth below datum, so η + d is the water depth.
   - **No precomputed tables:** everything is evaluated per subcell or
     subface on the fly.
2. **State and water level.**
   - The coarse average is (h̄, hū, hv̄) (Eq. 10–11).
   - The water level comes from matching volume over wet subcells,
     |T|h̄ = Σ|T_ℓ|h_ℓ (Eq. 26–29). Partly wet cells use Newton–Raphson on
     f(η̄) (Eq. 32–33).
   - Second order freezes ∇η and shifts η̄ (Sec. 4.2.1).
   - The effective cell bed is recomputed as d̄ = h̄ − η̄ (Eq. 34). This is
     the trick that lets the coarse scheme run unchanged on η.
3. **Faces.** Three cases:
   - Both sides wet: a Chen–Noelle-type reconstruction when depth < bed jump
     (Eq. 36–38), otherwise the averaged bed (Eq. 37).
   - One side dry: a wall if η_l < −d_r (Eq. 39–40).
   - Both dry: a wall (Eq. 41).

   The Riemann solver is Toro's HLLC with the dry solution (Sec. 4.4). The
   wall flux is pressure only, ½g(η*² − 2η*d*)n (Eq. 45). The paper is
   inconsistent about Case 2.1 (HLLC in Sec. 4.4, a wall in Eq. 39–40).
4. **Bed source, well balanced.**
   - S_g = (1/|T|)(g/2) Σ(η* + η̄)(d* − d̄)|E| n, a line integral over
     subfaces (Eq. 46–47).
   - Lake at rest holds to machine precision with an emerged bump
     (Eq. 50–52, Table 1). It needs η* = η̄ at the faces.
   - Cost scales with 3·n_sg faces, not n_sg² subcells.
5. **Friction.**
   - The authors rejected Volp's matching-gradient assumption as "not
     general enough" (Sec. 4.6).
   - Instead they use a wet-area-weighted average of g/(M²h_ℓ^(1/3)) over wet
     subcells, times the coarse velocity (Eq. 54–55).
   - It is implicit, by Newton iteration on the coarse velocity (Eq. 57).
   - Velocity u = hu/h, set to zero below 1e-4 m (Eq. 15).
6. **Time stepping.**
   - Two-stage TVD Runge–Kutta with implicit friction (Eq. 56).
   - CFL: Δt ≤ CFL·min(√|T| / λ), with λ = |u| + √(g·h_max) and h_max the
     deepest subcell (Eq. 58–59). CFL 0.4–0.5.
7. **Tests:** lake at rest, a small perturbation, MacDonald steady friction
   flow (n_sg = 2, 4, 8), a parabolic bowl, and a qualitative meandering
   channel on a floodplain.
   - **No run times, speedups or resolved-mesh comparison reported.**
   - At n_sg = 8 the subgrid scheme beats plain second order in L2(h).
   - At n_sg = 2 it is worse than plain second order (bowl), and first-order
     subgrid gives no gain on MacDonald.
   - A small water leak near the channel in the floodplain test (from the
     summary only).
8. **Their future work:** Central-WENO for higher order, path-conservative
   well-balancing.

**For ANUGA:** the effective-bed trick (point 2) and the subface line-integral
bed source (point 4) map directly onto ANUGA's centroid stage, edge
extrapolation and hydrostatic reconstruction. The design note prefers tables
for a first version, for speed; their on-the-fly approach is the fallback
where tables fall short (sloped sub-cell surface, bed change).

## Others

| Work | Key point | Reference |
|---|---|---|
| Casulli 2009 | Semi-implicit; cell volume and wet area against level from subgrid bathymetry; mass-conserving wetting and drying with no depth threshold | IJNMF 60(4), 391–408, doi:10.1002/fld.1896 |
| Sehili, Lang & Lippert 2014 | UnTRIM operational Elbe model; grid generation, filtering small subcells, porosity, volume and wet area | Ocean Dynamics 64, 519–535, doi:10.1007/s10236-014-0693-x |
| Volp, van Prooijen & Stelling 2013 | FV on a coarse grid with subgrid bathymetry and roughness; "friction depth" from matching gradients | Water Resources Research 49(7), 4126–4135, doi:10.1002/wrcr.20324 |
| Volp et al. 2016 | Morphodynamics with subgrid: hydrodynamics on the coarse grid, bed change on a separate high-resolution bed grid | Adv. Water Resour. 93, 105–117, doi:10.1016/j.advwatres.2015.07.013 |
| Woodruff, Dietrich, Wirasaet, Kennedy et al. 2021 | Subgrid corrections added to ADCIRC (unstructured continuous Galerkin), starting with a "level 0" wet/dry-front closure; Hurricane Matthew; slides quote 33× mesh coarsening for comparable results | Ocean Modelling 167, 101887, doi:10.1016/j.ocemod.2021.101887 |
| Woodruff et al. 2023 | The same, extended to ocean scale (South Atlantic Bight) | Natural Hazards (2023) |
| Sanders, Schubert & Gallegos 2008 | Integral porosity: volumetric porosity per cell for storage, areal porosity per edge for conveyance; for buildings | J. Hydrol. 362, 19–38, doi:10.1016/j.jhydrol.2008.08.009 |
| Guinot, Sanders & Schubert 2017 | Dual integral porosity, better than the 2008 anisotropic model | (follow-up) |
| Shamkhalchian & de Almeida (2019, later) | Godunov FV upscaling with two nested meshes; runtime on the coarse mesh | JHR 2019, Southampton eprints 443830 |
| HEC-RAS 2D | Volume–elevation table per cell; area, wetted perimeter and conveyance tables per face; faces act as small cross-sections. Recommends aligning faces with key terrain. | https://therassolution.kleinschmidtgroup.com/ras-post/hec-ras-subgrid-bathymetry-theory-and-application |
| TUFLOW HPC "SGS" | Cell volume–depth and face conveyance curves (2020 release); 3–4 cells across a flow path with SGS, 6–8 without; velocity and hazard detail still need smaller cells | https://docs.tuflow.com/classic-hpc/manual/2026.0/ModelResolutionDiscretisation-2.html, https://wiki.tuflow.com/Quadtree_and_Sub-Grid_Sampling_FAQ |

## Local time stepping (companion work)

The LTS literature review (Dazzi et al. 2018, Dumbser 2014, Boscheri et al.
2015, MPAS-Ocean, Kesserwani & Liang 2015, Tan et al. 2004) is in the comments
of [#466](https://github.com/anuga-community/anuga_core/issues/466). The PDFs
and per-paper reviews are in `~/anuga_core/doc/lts_review/` (untracked, in the
main checkout).
