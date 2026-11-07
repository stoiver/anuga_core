# Subgrid terrain: design note

Copy of the public design note [anuga_core#508](https://github.com/anuga-community/anuga_core/issues/508), posted 2026-11-06. Keep this file and the issue in step when the design changes.

## Summary

ANUGA gives each triangle one bed elevation, so it represents terrain only at the scale of the mesh. A channel narrower than the cells gets either too much storage (a conditioned trench, #453) or a mesh fine enough to resolve it. A resolved mesh needs an order of magnitude more triangles, and those small, deep cells then set the global timestep.

This note proposes **subgrid terrain**: keep the coarse mesh and the explicit finite volume scheme, but let each coarse cell and edge know the fine terrain inside it. The idea starts from a two-level picture: split each coarse triangle uniformly into sub-triangles at about DTM-pixel size, run the standard flux on the coarse level, and treat the sub-level separately. Over one coarse step the water surface inside a coarse cell is close to level. So most of what the coarse scheme needs from the sub-level depends only on the cell's water level, and can be computed before the run instead of solved during it. A local sub-level solve is kept as an option for the cells where a level surface is a poor assumption.

The note also compares this with local time stepping (LTS, #466). The two solve different problems, and they can be combined.

## How other codes do it

| Code / paper | Mesh and scheme | What the sub-level supplies |
|---|---|---|
| Casulli 2009 (IJNMF 60:391) | semi-implicit, orthogonal unstructured grids | cell volume and wet area against water level; mass-conserving wetting and drying without a depth threshold |
| Sehili, Lang & Lippert 2014 (Ocean Dynamics 64:519) | UnTRIM, staggered C-grid, operational Elbe model | the same, plus grid generation and filtering of small subcells |
| Volp, van Prooijen & Stelling 2013 (WRR 49:4126) | finite volume on a coarse grid | storage, plus a friction treatment that assumes the bed and free surface gradients match inside a cell |
| Woodruff, Dietrich, Wirasaet, Kennedy et al. 2021 (Ocean Modelling 167:101887) | ADCIRC, continuous Galerkin, unstructured | subgrid corrections to water levels and velocities, starting with a "level 0" closure at the wet/dry front; conference slides quote a factor of 33 in mesh coarsening for comparable results |
| Sanders, Schubert & Gallegos 2008 (J. Hydrol. 362:19) and later porosity models | finite volume, unstructured | volumetric porosity for storage and areal porosity on edges for conveyance, for buildings rather than terrain |
| HEC-RAS 2D | implicit, polygonal cells | volume–elevation table per cell; area, wetted perimeter and conveyance table per face |
| TUFLOW HPC "SGS" | explicit, Cartesian/quadtree | cell volume–depth and face conveyance curves; the manual asks for 3–4 cells across a flow path with SGS against 6–8 without |
| Bitsch, Heilmann, Engsig-Karup, Sørensen & Grooss 2026 (arXiv 2601.01895) | **explicit, cell-centred finite volume, unstructured triangles, HLLC, second order** | each triangle split uniformly into n² subcells with piecewise-constant bed; computed on the fly, not tabulated |

The last paper is closest to ANUGA, and to the two-level picture above. Its main points:
- **State:** the coarse average (h̄, hū, hv̄). The water level η̄ comes from matching volume over the wet subcells, by Newton iteration in partly wet cells. The coarse scheme then sees an effective bed d̄ = η̄ − h̄.
- **Edges:** each edge is split into n subfaces, with wet/wet, wet/dry and dry/dry cases at the face.
- **Bed source:** a line integral over the subfaces that is well balanced at the subgrid level (lake at rest to machine precision).
- **Friction:** a wet-area-weighted average over the wet subcells. The authors found Volp's matching-gradient assumption not general enough.
- **CFL:** based on the deepest subcell.
- **Limits:** tests are analytical (lake at rest, MacDonald, parabolic bowl) plus a qualitative floodplain case. It reports no runtimes or comparison with a resolved mesh. With n = 2, its second-order subgrid scheme did worse than plain second order, so the gain needs n of about 4–8 or more.

## Proposed design

### The sub-level
- Each coarse triangle is split uniformly into n² sub-triangles (the usual 4-way split, repeated), with n chosen so that sub-triangles are about the DTM pixel size.
- Each sub-triangle gets the area-weighted mean of the DTM pixels it covers, so stored volume is exact for a level water surface.
- Each coarse edge is split into n subfaces with their own bed elevations and Manning's n.

### What the coarse scheme gets

| Coarse need | From the sub-level | Notes |
|---|---|---|
| Water level from volume | V(η) and wet area A(η) per cell, tabulated at about 20–30 levels | conserve volume (mean depth h̄ = V/area), and set the effective bed d̄ = η(h̄) − h̄ each step so the existing reconstruction works on η unchanged |
| Edge flux | per-subface depths from the reconstructed edge η, or tabulated wetted width, flow area and ½g∫(η−z)² per edge | the subface pressure terms must balance the bed source, so lake at rest stays exact |
| Bed source | line integral over subfaces, as in Bitsch et al. Eq. 46–47 | this is what keeps partly wet cells well balanced |
| Friction | conveyance K(η) = Σ hᵢ^(5/3) wᵢ/nᵢ per edge, or the wet-subcell average | inside the existing semi-implicit friction step |
| Timestep | CFL from the coarse inradius with the deepest wet subcell's depth | coarse cells, so far fewer and larger steps than a resolved mesh |
| Output | coarse stage and velocity as now; optionally depth downscaled to the sub-level (η − z_sub) for flood maps | velocity and hazard stay at the coarse resolution |

**Tables or on-the-fly?** Tables (HEC-RAS, ADCIRC) are cheap at run time, and their cost doesn't depend on n. On-the-fly subcells (Bitsch et al.) cost O(n²) per cell, but they allow a sloping water surface inside a cell and a changing bed. A first version should use tables; on-the-fly subcells can come later where they are needed.

### The optional sub-level solve
Where a level surface is wrong, a few implicit diffusion-wave iterations on the sub-triangles could redistribute water, with the coarse edge fluxes as boundary conditions. The candidates are:
- a wetting front crossing the cell;
- a channel inside a floodplain cell filling or draining ahead of the rest;
- steep cells.

Bitsch et al. use a cheaper alternative: the coarse η gradient carries into the subcells, giving a sloped surface without a solve. This step should only be built if the synthetic tests show the level-surface version falling short.

### What it touches in ANUGA
- **Flux and extrapolation kernels:**
  - the effective bed d̄(h̄) and η from volume per cell;
  - subface fluxes at the edges and the subface bed source;
  - both compute modes (mode 1 C, mode 2 GPU).
  
  This is best added as a separate flow algorithm (`DE_subgrid`), so existing results stay bit-for-bit unchanged.
- **Quantity setters and operators:** anything that converts between depth and volume must go through the tables. That includes `set_quantity('stage')`, `Inlet_operator`, `Rate_operator` (rain), culverts and structures, set-stage operators, and `get_water_volume`.
- **Not compatible at first:**
  - Bed change (erosion and sediment, `set_elevation` during a run) invalidates tables. Volp et al. 2016 handle morphodynamics with a separate bed grid; that is later work.
  - Riverwalls need their heights reconciled with subface beds.
- **Parallel and I/O:** tables partition with their cells and are copied to ghosts. The SWW file stores the effective bed, with sub-level terrain optionally kept for downscaled output.
- **Memory:** with 25 levels, one cell table plus three edge tables is about 100 floats per triangle. That fits easily on a GPU at the coarser triangle counts this allows.

## Subgrid compared with local time stepping (#466)

| | Subgrid terrain | Local time stepping |
|---|---|---|
| Problem it solves | **cells too coarse for the terrain**: storage and conveyance errors in channels narrower than the cells | **a few small or fast cells set the global step** on a graded mesh |
| How it saves time | fewer cells (an order of magnitude on narrow-channel problems) and a larger global step | the same cells, but large slow cells update less often |
| Accuracy | an approximation: a level surface in each cell, no sub-cell velocity, no channel–floodplain momentum exchange inside a cell | reproduces the fine-mesh solution; with one level it is the current scheme exactly |
| Fixes #453-type storage error on a coarse mesh | **yes** (storage tables alone) | no; it only makes the resolved mesh cheaper |
| Estimated gain | not yet measured for ANUGA: the codes above report large mesh coarsening (ADCIRC slides quote 33×), less a per-cell cost for subfaces that I estimate at 2–4×; Phase 0 measures it | 4–9× on graded meshes, about 2× on near-uniform ones (#466 estimate, ideal), practical 50–80% of that |
| GPU fit | good: fewer cells and more arithmetic per cell, same kernel launches | harder: per-level launches on subsets, more synchronisation, main risk |
| Code touched | flux, extrapolation, depth–volume conversions in operators | time loop, every operator's dt, boundaries, MPI ghost exchange per level |
| Velocity and hazard detail | coarse only | full resolution |
| Bed change (sediment) | conflicts with tables | carries over |
| First useful step | storage-only tables, small and self-contained | edge-based flux kernel at parity (Phase 0 of #466) |

**They are complementary:**
- Subgrid takes most small cells out of narrow-channel models. LTS then helps whatever grading remains, for example structures, riverwalls or a refined town.
- LTS Phase 0, solving each edge once from an edge list, is also the natural place for subgrid's per-edge subface work, so that step serves both.
- Where users need full-resolution velocities (hazard, structures, sediment), LTS on a resolved mesh stays the right tool.

## Plan, each phase ending in a go/no-go decision

**Phase 0: synthetic benchmark (also usable for LTS).**
- A trapezoidal channel narrower than the coarse cells, set in a floodplain with small depressions, driven by a flood hydrograph.
- The reference is ANUGA on a resolved mesh. Metrics:
  - storage against water level;
  - the steady rating curve;
  - flood-wave arrival and attenuation downstream;
  - peak water levels and flood extent;
  - runtime.
- It also reports how the conditioned-trench approach compares.

**Phase 1: storage only, mode 1.**
- Tables V(η) and A(η) per cell, the volume-conserving update and the effective bed, with edges and friction unchanged.
- Go if:
  - lake at rest holds to round-off with partly wet cells;
  - mass is conserved to round-off;
  - storage against level matches the reference;
  - the storage error of a coarse mesh drops substantially.

**Phase 2: edges.**
- Subface fluxes, the subface bed source, and conveyance-based friction.
- Go if the rating curve and wave travel time on the coarse mesh approach the resolved reference.

**Phase 3: production.**
- The mode 2 GPU kernels, the operator audit (inlets, rain, culverts, set-stage), parallel partitioning of the tables, and SWW output with optional downscaled depth.
- A user-facing way to build the sub-level from a DEM, alongside `create_domain_from_dem` (#461) and the size field (#436).

**Phase 4 (optional).** The sub-level relaxation, or a sloped sub-cell surface, if Phases 1–2 show level-surface errors in wetting fronts.

## Open questions
- **Tables or on-the-fly subcells for edges:** tables are faster, subcells more general. Phase 2 can measure both.
- **Second order:** Bitsch et al. lose accuracy at small n. Is first order inside the cell, with second order across cells, the better trade?
- **Sub-cell velocities:** HEC-RAS and TUFLOW leave them out. Downscaling velocity with a conveyance split is possible but needs validation.
- **Mesh guidance:** with subgrid, do edges still need to follow banks (as HEC-RAS recommends), or is a size field enough?

## References
- Casulli, V. (2009). A high-resolution wetting and drying algorithm for free-surface hydrodynamics. IJNMF 60(4), 391–408. doi:10.1002/fld.1896
- Sehili, A., Lang, G., Lippert, C. (2014). High-resolution subgrid models: background, grid generation, and implementation. Ocean Dynamics 64, 519–535. doi:10.1007/s10236-014-0693-x
- Volp, N.D., van Prooijen, B.C., Stelling, G.S. (2013). A finite volume approach for shallow water flow accounting for high-resolution bathymetry and roughness data. Water Resources Research 49(7), 4126–4135. doi:10.1002/wrcr.20324
- Volp, N.D., van Prooijen, B.C., Pietrzak, J.D., Stelling, G.S. (2016). A subgrid based approach for morphodynamic modelling. Advances in Water Resources 93, 105–117. doi:10.1016/j.advwatres.2015.07.013
- Woodruff, J.L., Dietrich, J.C., Wirasaet, D., Kennedy, A.B., et al. (2021). Subgrid corrections in finite-element modeling of storm-driven coastal flooding. Ocean Modelling 167, 101887. doi:10.1016/j.ocemod.2021.101887
- Sanders, B.F., Schubert, J.E., Gallegos, H.A. (2008). Integral formulation of shallow-water equations with anisotropic porosity for urban flood modeling. Journal of Hydrology 362, 19–38. doi:10.1016/j.jhydrol.2008.08.009
- Bitsch, Heilmann, Engsig-Karup, Sørensen, Grooss (2026). An unstructured second-order subgrid method for the shallow water equations. arXiv:2601.01895
- HEC-RAS 2D subgrid bathymetry: https://therassolution.kleinschmidtgroup.com/ras-post/hec-ras-subgrid-bathymetry-theory-and-application
- TUFLOW HPC sub-grid sampling: https://docs.tuflow.com/classic-hpc/manual/2026.0/ModelResolutionDiscretisation-2.html

Related: #453 (narrow channels), #466 (local time stepping), #436 (size field), #461 (`create_domain_from_dem`).
