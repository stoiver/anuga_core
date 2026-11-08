# Subgrid terrain (#508): status of the code

State on 2026-11-08. Branch `develop_subgrid` at the commits listed at the end.
Read `SUBGRID_START_HERE.md` for setup and rules, and `SUBGRID_DESIGN.md` for the plan.

## Summary

- **Phases 0, 1 and most of 2 are done** as a prototype in mode 1 (legacy, CPU). The
  storage tables, effective bed, conveyance friction, subface edge fluxes, a
  draining-time limiter and a conveyance-weighted subface velocity all run end to end
  on the synthetic benchmark.
- **The best variants are the EKLC runs.** Against the 569 000-triangle reference
  (band025TU, 100 min on the GPU):
  - `coarse500_sub8EKLC` (3082 triangles, 99 s on 8 CPU threads): flood extent CSI
    0.96, outlet peak 52.3 m³/s against 52.1, mean |peak-stage error| 2.7 cm;
  - `coarse2000_sub8EKLC` (780 triangles, 14 s): CSI 0.91 (0.93 with the switch on),
    outlet peak 52.5–52.8, mean |peak-stage error| 2.7–3.7 cm;
  - peak timing within 2 minutes of the reference for both.
- **Reference correction:** the reference runs were biased. ANUGA's flux diffused
  momentum across the channel–bank shear, making resolved channels too resistive.
  That is now fixed upstream as an opt-in switch (#509, PR #510, merged), and the
  references have been rerun with it.
- **Still to do:** the production steps of Phase 3. The tables live in a Python
  operator in the bench, not in a flow algorithm. There is no GPU (mode 2) path, the
  operators (inlets, rain, culverts) haven't been audited, and there is no parallel
  support.

## Where the code is

### In ANUGA (`anuga/shallow_water`), committed in 18c1c851

All of this is inactive unless the subgrid attributes below are set on the domain, so
default runs are unchanged. The full suite passes: 4062 passed, 6 skipped, with MPI.

| Piece | Where | What it does |
|---|---|---|
| `core_compute_fluxes_subgrid` | `gpu/core_kernels.c` (CPU-only build branch) | splits each edge into `nsub` subfaces with their own bed; sums the central flux and hydrostatic pressure over the subfaces; replaces the flux + gravity terms, like the central kernel |
| conveyance split | inside the same kernel | with `subgrid_edge_invn` and `subgrid_cell_neff`, the subface velocity is u·n_eff/n_j·(h_j/h_edge)^(2/3); a uniform cell gives weight 1 exactly, and boundary sides mirror the interior weights |
| `core_subgrid_drain_limit` | `gpu/core_kernels.c` | draining-time limiter: scales a cell's outgoing fluxes so it can't go below dry within the step; uses per-slot fluxes the kernel records |
| Cython wrappers | `sw_domain_openmp_ext.pyx`: `compute_fluxes_ext_subgrid`, `subgrid_drain_limit` | shape checks, pass-through |
| Domain hooks | `shallow_water_domain.py`: `compute_fluxes`, `update_conserved_quantities` | switch to the subgrid kernel when `domain.subgrid_edge_z` is set (mode 1 only, otherwise `NotImplementedError`); apply the limiter when `domain.subgrid_drain_limiter` is set |

The domain attributes are set by the bench operator:
- `subgrid_edge_z`: (3n, nsub) subface beds;
- `subgrid_edge_invn`: (3n, nsub) 1/n per subface;
- `subgrid_cell_neff`: (n,) effective Manning, refreshed in place every step;
- `subgrid_drain_limiter`: bool.

### In the bench (`subgrid_bench/`), committed in 9feda8ac

All synthetic. Run outputs (`results/`, `*.npz`, logs, plots) stay local through
`subgrid_bench/.gitignore`.

| File | Role |
|---|---|
| `terrain.py` | analytic valley: 2000 × 500 m, slope 1:1000, a meandering trapezoidal channel (10 m base, 2 m deep, 1:2 banks), a floodplain with hollows, and an outlet control at x = 1900. `SUBGRID_AMPLITUDE=0` straightens the channel |
| `subgrid_tables.py` | the method: `Subgrid_tables` (sorted sub-triangle beds, V(η), η(h), Kc(η)), `edge_subface_beds` (subface beds and 1/n), and `Subgrid_storage_operator` (the per-step update) |
| `run_phase0.py` | hydrograph runs (5 → 60 → 5 m³/s, 5.5 h) for every variant; `--gpu`, `--diag`; environment switches `SUBGRID_TAG` and `SUBGRID_LOW_FROUDE` |
| `evaluate_phase0.py` | scores the runs against a reference: flood-extent CSI, peak Q and its timing, arrival time, peak stage |
| `test_phase1.py`, `test_phase2.py` | lake at rest with partly wet cells, mass conservation, dam breaks (run as scripts; all pass) |
| `static_storage.py`, `static_conveyance.py`, `static_edges.py` | static checks of storage, conveyance and edge area/conveyance against the DTM |
| `steady.py`, `trap_conv.py` | steady runs against Manning normal depth; the straight-trapezoid check that found the reference bias |
| `launch_*.sh` | the tmux batch scripts used for each round of runs |

## The method as built

- **Storage (Phase 1):**
  - Each coarse triangle is split into n² equal sub-triangles, each holding the mean
    of its 0.5 m DTM pixels. Sorted beds and their prefix sums give V(η) exactly, and
    η(h) by a search plus one division.
  - After every step the operator keeps the volume and sets η = η(h), the effective
    bed d = η − h, and stage w = η.
  - A dry cell's bed is its lowest sub-triangle, so a channel crossing the cell sets
    the cell's bed.
- **Conveyance friction ("K"):** Kc(η) = (1/m) Σ max(η − z_k, 0)^(5/3)/n_k. It goes
  through the existing friction kernel as n_eff = h^(5/3)/Kc, which is exactly the
  conveyance form −g h|q|q/Kc².
- **Edge fluxes ("E"):** 16 subfaces per edge, with beds sampled along the edge. Each
  subface is clipped below at the higher adjacent cell's lowest sub-triangle, so the
  flux can't drain a cell below its lowest point.
- **Timestep and limiter ("L"):** a partly wet cell's level moves by 1/f times what the
  coarse scheme assumes (f is the wet fraction). The first fix was a CFL cap scaled by
  f, which was very costly (1.2 million steps). The draining-time limiter keeps depths
  non-negative for any step, so the cap drops the f factor (`EDGE_CAP_P = 0`). That cut
  the steps 30×.
- **Conveyance split ("C"):** without it, every subface of an edge gets the cell's mean
  velocity. Shallow floodplain subfaces then carry too much water and the deep
  channel subfaces too little, which attenuated the flood peak (46.7 against 51.5
  m³/s). The split gives each subface the velocity its own conveyance supports.
  - It needs a cap CFL of 0.5: at 0.75, partly wet channel cells near the inlet
    oscillate.

## Results

The reference is `band025TU`: a 0.25 m² channel band and 25 m² floodplain (569 004
triangles), run with LF1 + tangential upwind on the GPU (5998 s). The references
converge: band1LF4 (1 m², same flux) is within 1.8 cm mean peak stage and 0.3 m³/s
outlet peak, and band025 (old flux) within 2.0 cm.

| run | triangles | wall | CSI | outlet peak Q | peak time | mean peak stage error (mean abs) |
|---|---|---|---|---|---|---|
| **reference band025TU** | 569 004 | 5998 s GPU | 1 | 52.1 | 3.69 h | 0 |
| band1LF4 | 188 198 | 1032 s GPU | 0.944 | 51.8 | 3.73 h | +0.018 m |
| band025 (old flux) | 569 004 | 5653 s GPU | 0.938 | 51.5 | 3.74 h | +0.020 m |
| band1 (old flux) | 188 198 | 1063 s GPU | 0.845 | 50.9 | 3.79 h | +0.053 m |
| coarse500 centroid bed | 3082 | 3 s | 0.395 | 52.4 | 3.94 h | +0.459 m |
| coarse500 trench | 3082 | 3 s | 0.644 | 46.0 | 4.12 h | +0.093 (0.222) m |
| coarse500_sub8 (storage only) | 3082 | 19 s | 0.441 | 52.5 | 3.91 h | +0.367 m |
| coarse500_sub8EKL | 3082 | 61 s | 0.764 | 46.7 | 4.06 h | +0.035 (0.110) m |
| **coarse500_sub8EKLC** | 3082 | 99 s | **0.957** | 52.3 | 3.66 h | −0.019 (**0.027**) m |
| **coarse2000_sub8EKLC** | 780 | 14 s | **0.906** | 52.8 | 3.66 h | +0.010 (**0.037**) m |
| coarse500_sub8EKLC + TU | 3082 | 99 s | 0.955 | 52.3 | 3.66 h | −0.026 (0.029) m |
| coarse2000_sub8EKLC + TU | 780 | 15 s | 0.933 | 52.5 | 3.67 h | −0.004 (0.027) m |

- Wall times are on 8 CPU threads unless marked GPU.
- "+ TU" means the subgrid runs also use LF1 + tangential upwind. The switch barely
  changes them, because inside a coarse cell the channel conveyance comes from the
  tables.
- Arrival times (Q passing half the peak) of all EKLC runs are within 0.04 h of the
  reference at every section; the storage-only and K-only variants arrive
  0.6–0.9 h late at the outlet.
- Score with `python evaluate_phase0.py band025TU`.

Floodplain storage, in thousands of m³:

| | near-floodplain peak | far-floodplain peak | near-floodplain at 4.5 h |
|---|---|---|---|
| band1 (old flux) | 43.3 | 12.8 | 4.5 |
| band1LF4 | 36.6 | 8.7 | 0.8 |
| **band025TU (reference)** | **33.6** | **7.3** | **0.7** |
| EKLC 500 / 2000 | 33.8 / 37.7 | 6.0 / 8.3 | 0.1 / 0.1 |
| EKLC 500 / 2000 + TU | 32.4 / 35.2 | 5.4 / 6.8 | 0.1 / 0.0 |

The earlier "floodplain drains too fast" finding was mostly error in the reference.

### Steady checks (straight channel)

- **EKLC 500:** within 1–5 cm of Manning normal depth at Q = 10–55 m³/s, and the cell
  momentum balance holds: (q/Kc)/√S ≈ 1.
- **Older variants are off:** EKL is 15 cm high at Q = 55, and band1 (old flux) is
  12 cm high at Q = 25.

### The reference bias (#509, fixed in #510)

The central-upwind flux diffuses edge-tangential momentum at about ½√(gh)·Δx. On a
straight trapezoid the resolved channel carries only part of its true conveyance:

| cell size | old flux | LF1 + tangential upwind |
|---|---|---|
| 4 m² | 0.63 | 0.79 |
| 1 m² | 0.83 | 0.93 |
| 0.25 m² | 0.92 | — |

Breaklines and flow-aligned meshes don't fix it. Turn the switch on with
`domain.set_tangential_upwind()` after `set_low_froude`; in the bench, use
`SUBGRID_LOW_FROUDE=5`. Even the corrected references are about 7% short of true
conveyance, so the Manning normal depth stays the cleanest steady check.

## Known issues and open items

1. **Diagnostic split is stale:** `run_phase0.py --diag` still looks for
   `subgrid_cell_kc`, but the operator now sets `subgrid_cell_neff`. So the channel /
   floodplain discharge split in the diagnostics uses the uniform closure for C
   variants. Fix: Kc = h^(5/3)/n_eff. This affects diagnostics only.
2. **Rising limb:** there are small unsteady differences (for example, at Q = 20 the
   level is 1.28 against 1.46 on the 500 mesh), partly a loop-rating effect. The
   residual near-floodplain storage at 4.5 h is 0.1 against 0.7 thousand m³.
3. **Cap CFL:** 0.5 for the C variants costs about 1.5× the steps of 0.75. A wet-fraction
   cap also works but is slower.
4. **Prototype limits:**
   - Kc is evaluated directly, O(m) per cell per step, not tabulated.
   - The operator runs in Python.
   - Mode 1 only.
   - The edge-flux kernel is per cell, so each edge is computed twice.
5. **Not yet done (Phase 3):**
   - a `DE_subgrid` flow algorithm in place of the operator;
   - mode 2 / GPU kernels;
   - an audit of anything that converts between depth and volume (inlets, rain,
     culverts, set-stage, `get_water_volume`);
   - parallel partitioning of the tables;
   - SWW output with downscaled depth;
   - a DEM-based builder.
6. **Private real-world check:** the Serna case (#453) hasn't been re-scored yet. Its
   data and results stay outside the repo.

## How to run

```bash
conda activate anuga_subgrid_env_3.14      # CPU build, has mpi4py
cd subgrid_bench
OMP_NUM_THREADS=8 python run_phase0.py coarse2000_sub8EKLC --diag
SUBGRID_LOW_FROUDE=5 SUBGRID_TAG=_tu OMP_NUM_THREADS=8 python run_phase0.py coarse500_sub8EKLC --diag
python evaluate_phase0.py band025TU        # score everything in results/ against a reference
PYTHONPATH=$PWD python test_phase2.py      # tests run as scripts
python trap_conv.py 1 LF=5                 # straight trapezoid check
```

For GPU reference runs, use `anuga_subgrid_gpu_env_3.14`. That build doesn't track
header changes: after editing a header, touch the `.c`/`.pyx` files that include it
and run `ninja` in `build-gpu` with nvc on PATH.

## Commits on develop_subgrid

| commit | content |
|---|---|
| 30073834 | handoff notes |
| 18c1c851 | subgrid edge flux kernel, draining-time limiter, conveyance split |
| 9feda8ac | bench code tracked (outputs ignored) |
| 131d7fb5 | merge develop: tangential upwind flux (#510), breakline area limit (#512) |
| b9f22192 | bench uses `set_low_froude` for the switch |
| b52906f0 | this report; bench runs with the switch on |

The branch is not pushed to origin.
