# Subgrid terrain (#508): what we are building and where it stands

State on 2026-11-08. Branch `develop_subgrid` at the commits listed at the end.
Read `SUBGRID_START_HERE.md` for setup and rules, and `SUBGRID_DESIGN.md` for the plan.
The figures are in `claude/figures/subgrid/`. Regenerate them from the bench results with
`python subgrid_bench/report_figures.py`.

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

| Phase | State |
|---|---|
| 0 · benchmark: synthetic valley, flood hydrograph, fine references, scoring | done |
| 1 · storage: volume–level tables per cell, effective bed, lake at rest and mass tests | done |
| 2 · conveyance: conveyance friction, subface edge fluxes, draining limiter, conveyance split | mostly done (still a Python operator) |
| 3 · production: DE_subgrid flow algorithm, GPU, operator audit, parallel, output, DEM builder | not started |

## The problem

Floodplain models are often run on triangles of hundreds to thousands of square metres,
because that is what fits the time budget. A river channel is much narrower. In the test
valley the channel is 18 m wide at the top and 2 m deep, so a 2000 m² triangle can't
resolve it.

A coarse cell has to pick one bed level:
- If it takes the terrain at its centroid, most cells see floodplain and the channel
  vanishes.
- If it carves a trench into every cell the channel crosses, the channel becomes far too
  wide and too smooth.

Either way, both the storage and the conveyance are wrong. The subgrid method keeps the
coarse mesh for the flow equations, but gives every cell and every edge lookup tables
built from the fine terrain. A cell knows how much water it holds at any level and how
easily that water flows; an edge knows the bed along its length.

![The test valley and the same 150 m patch on three meshes](figures/subgrid/problem.png)

*Top: the 2000 × 500 m valley at a 1:1000 slope, coloured by height relative to the
floodplain level. The dark meander is the trapezoidal channel (10 m base, 1:2 banks, 2 m
deep); the spots are floodplain hollows, and the floodplain rises 1 m over 250 m away
from the channel. The terrain is an analytic function sampled on a 0.5 m grid (4 million
pixels). A flood hydrograph of 5 → 60 → 5 m³/s enters at the left over 4 hours. Bottom:
the orange patch on the band1 reference mesh (1 m² triangles along the channel, 25 m² on
the floodplain) and on the coarse500 and coarse2000 meshes, with the channel shaded. A
single coarse triangle holds a stretch of channel plus its banks.*

## Inside a coarse cell: storage and conveyance tables

Each coarse triangle is split into 8 × 8 = 64 equal sub-triangles, each holding the mean
of the 0.5 m terrain pixels inside it. That is the only terrain information the model
keeps. From the 64 bed levels z_k, sorted, it builds three functions of the water level η:

```
mean depth      h(η)  = (1/m) Σ max(η − z_k, 0)
wet fraction    f(η)  = (number of z_k < η) / m
conveyance      Kc(η) = (1/m) Σ max(η − z_k, 0)^(5/3) / n_k
```

h is the cell's water volume per unit area, f the share of the cell that is wet, and Kc
the conveyance per unit width (the discharge a unit friction slope drives). With the
sub-beds sorted, h(η) is piecewise linear, and its inverse η(h) is a search plus one
division.

![One coarse2000 cell crossed by the channel, and its tables](figures/subgrid/cell.png)

*A real 1278 m² triangle from the coarse2000 mesh. Its centroid lands on the floodplain
at 9.04 m, but the channel crosses one corner and the lowest sub-triangle is at 6.95 m,
2.1 m lower. A centroid-bed model would see no channel here at all. At the flood peak in
the EKLC 2000 run this cell reaches η = 9.25 m: the tables give a mean depth of 0.31 m
with the whole cell wet, where a flat bed at the centroid would hold 0.21 m. The
conveyance is 6.4 m²/s against 1.5 m²/s, because the water is concentrated in the deep,
smoother channel.*

How the tables enter the coarse scheme:
- **Storage.** ANUGA keeps conserving the mean depth h = stage − bed, so volume is exact.
  After every step the subgrid operator sets the level to η(h), then moves the cell's
  bed to d = η − h and its stage to η. A dry cell's bed is its lowest sub-triangle, so a
  channel crossing the cell sets where water first collects.
- **Conveyance friction ("K").** ANUGA's Manning friction is −g n² |q| q / h^(7/3).
  Setting the cell's n to n_eff = h^(5/3)/Kc(η) turns it into exactly −g h |q| q / Kc²,
  the friction of the cell's real cross-section. A cell whose water is mostly in a deep
  channel gets a small n_eff and flows faster.

## Across an edge: subfaces and the conveyance split

Storage and friction alone aren't enough. A coarse edge that crosses the channel must pass
water through its channel part even when the rest of the edge is dry.

- **Edge subfaces ("E").** Every edge is split into 16 subfaces, each with its own bed
  sampled along the edge. The flux kernel computes the central-upwind flux and
  hydrostatic pressure on each subface with its own depth h_j = η − z_j, and sums them.
  Subfaces are clipped below at the lowest point of the higher neighbouring cell, so a
  flux can never draw water from below a cell's lowest point.
- **Conveyance split ("C").** Without it, every subface gets the cell's mean velocity, so
  discharge goes with depth: shallow floodplain subfaces carry too much and the deep
  channel subfaces too little. Over the whole valley about a third of the flow went over
  the floodplain, which flattened the flood peak (46.7 against 52.1 m³/s). With the
  split, each subface gets the velocity its own depth and roughness support under the
  same friction slope, u_j = u · (n_eff/n_j) · (h_j/h)^(2/3). A uniform cell gets weight 1
  everywhere, so nothing changes there. It needs a cap CFL of 0.5: at 0.75, partly wet
  channel cells near the inlet oscillate.

![Subface beds along an edge and the discharge each subface carries](figures/subgrid/edge.png)

*Top: the bed along a 59 m edge of the example cell, the 16 subface beds the kernel uses,
and the water level at the flood peak. Bottom: how the edge's discharge is shared
between subfaces, with both distributions scaled to the same total. With the cell's mean
velocity on every subface, the channel subfaces carry 72% of it. With the conveyance
split they carry 95%, because deep water in the smoother channel (n = 0.035 against
0.05) flows faster.*

## One time step

```mermaid
flowchart LR
  subgraph C["C kernels in anuga/shallow_water (mode 1)"]
    E["Edge fluxes (E, C)<br/>16 subfaces per edge<br/>h_j = η − z_j, u_j weighted by n_eff/n_j"] --> K["Friction (K)<br/>Manning kernel with n = n_eff"]
    K --> L["Update (L)<br/>draining limiter: no cell<br/>goes below empty this step"]
  end
  subgraph P["Python operator in the bench"]
    O["Subgrid_storage_operator<br/>1. η = η(h)<br/>2. bed d = η − h, stage = η<br/>3. n_eff = h^5/3 / Kc(η)<br/>4. time-step cap"]
  end
  L -- "new h" --> O
  O -- "bed, stage, n_eff for the next step" --> E
  S["fixed at start-up:<br/>subface beds z_j and 1/n_j"] -.-> E
```

The C pieces are switched on only when the domain carries subgrid arrays. The table update
is a Python operator that runs after the inlet each step. Letters match the run names:
E = edge subfaces, K = conveyance friction, L = draining limiter, C = conveyance split.
Storage tables are always on in a subgrid run.

**Time step and limiter ("L").** A partly wet cell has a free surface of only f·A, so a
given volume change moves its level by 1/f times what the coarse scheme expects. The first
fix was a time-step cap scaled by f, which needed over a million steps. The draining-time
limiter keeps every depth non-negative for any step, so the cap can drop the f factor
(`EDGE_CAP_P = 0`), and the step count fell 30-fold.

![Time steps per variant](figures/subgrid/steps.png)

## Results

The reference is `band025TU`: *band* is a refined band along the channel, *025* means
0.25 m² triangles in it (25 m² on the floodplain, 569 004 triangles), and *TU* means the
corrected flux (LF1 + tangential upwind, see below), run on the GPU (5998 s). The
references have converged: band1LF4 (1 m², same flux) is within 1.8 cm mean peak stage and
0.3 m³/s outlet peak, and band025 (old flux) within 2.0 cm.

![Maximum flood extent against the reference](figures/subgrid/extent.png)

*Coarse cells' peak levels compared with the terrain on a 2.5 m grid: a point is wet if
its cell was wet and the cell's peak level is above the ground there. The centroid-bed
model floods the floodplain far too widely, because its channel is missing. EKL keeps the
channel but still pushes too much flow onto the floodplain. EKLC matches nearly
everywhere. CSI = both wet / either wet.*

![Outlet discharge and near-floodplain storage](figures/subgrid/hydrographs.png)

![Peak level along the channel minus the reference](figures/subgrid/profile.png)

*EKLC stays within a few centimetres along most of the valley. All the coarse runs run
low below x ≈ 1600 m, approaching the steep outlet reach that starts at x = 1900 m, where
a coarse cell straddles the change of slope. EKLC 500 is 0.38 m low at x = 1925 m. The
centroid-bed model, not drawn, is 0.46 m high on average and up to 0.96 m.*

![Accuracy against cost](figures/subgrid/cost.png)

*Wall time on a log scale; filled points ran on 8 CPU threads, hollow points on the GPU
(RTX 5070). The GPU references would take several times longer on the CPU.*

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

- Wall times are on 8 CPU threads unless marked GPU. Discharge in m³/s.
- "+ TU" means the subgrid runs also use LF1 + tangential upwind. The switch barely
  changes them, because inside a coarse cell the channel conveyance comes from the
  tables.
- Arrival times (Q passing half the peak) of all EKLC runs are within 0.04 h of the
  reference at every section; the storage-only and K-only variants arrive
  0.6–0.9 h late at the outlet.
- Score with `python evaluate_phase0.py band025TU`.

Floodplain storage, in thousands of m³ (near floodplain = 9–60 m from the centreline):

| | near-floodplain peak | far-floodplain peak | near-floodplain at 4.5 h |
|---|---|---|---|
| band1 (old flux) | 43.3 | 12.8 | 4.5 |
| band1LF4 | 36.6 | 8.7 | 0.8 |
| **band025TU (reference)** | **33.6** | **7.3** | **0.7** |
| EKLC 500 / 2000 | 33.8 / 37.7 | 6.0 / 8.3 | 0.1 / 0.1 |
| EKLC 500 / 2000 + TU | 32.4 / 35.2 | 5.4 / 6.8 | 0.1 / 0.0 |

The earlier "floodplain drains too fast" finding was mostly error in the reference.

Channel share of the discharge at peak (3.7 h), at x = 250 / 500 / 1000 / 1500 m. The
`--diag` split rebuilds Kc = h^(5/3)/n_eff from `subgrid_cell_neff`, which matches the
kernel's subface velocities:

| run | channel share |
|---|---|
| band025TU (reference) | 0.94 / 0.92 / 0.91 / 0.96 |
| EKLC 500 | 0.93 / 0.94 / 0.93 / 0.96 |
| EKLC 2000 | 0.91 / 0.92 / 0.89 / 0.94 |
| EKL 500 (no split) | 0.66 / 0.61 / 0.64 / 0.93 |

Without the split, a third of the flow goes over the floodplain, which is why EKL
attenuates the peak.

### Steady checks (straight channel)

- **EKLC 500:** within 1–5 cm of Manning normal depth at Q = 10–55 m³/s, and the cell
  momentum balance holds: (q/Kc)/√S ≈ 1.
- **Older variants are off:** EKL is 15 cm high at Q = 55, and band1 (old flux) is
  12 cm high at Q = 25.

## The reference bias (#509, fixed in #510)

Partway through, the subgrid runs seemed to drain the floodplain too fast. The cause was in
the references. ANUGA's central-upwind flux mixes the momentum *along* an edge between the
two cells at roughly the gravity-wave speed: the dissipation term is about ½√(gh)·Δ(vh).
In the exact Riemann problem that component only moves with the flow *across* the edge,
which is near zero along a channel bank. So the old flux dragged the fast channel water
against the slow bank water: resolved channels came out too rough, the flood wave slowed,
and too much water spilled onto the floodplain.

The switch replaces only the tangential momentum flux with the HLLC form, mass flux ×
the upwind cell's tangential velocity; mass and normal momentum are unchanged. On a
straight trapezoid the resolved channel carries this share of its true conveyance:

![Conveyance of a resolved straight channel by cell size](figures/subgrid/bias.png)

| cell size | old flux | LF1 + tangential upwind |
|---|---|---|
| 4 m² | 0.63 | 0.79 |
| 1 m² | 0.83 | 0.93 |
| 0.25 m² | 0.92 | — |

Breaklines and flow-aligned meshes don't fix it. Turn the switch on with
`domain.set_tangential_upwind()` after `set_low_froude`, or `set_low_froude(5)`; in the
bench, use `SUBGRID_LOW_FROUDE=5`. Even the corrected references are about 7% short of
true conveyance, so the Manning normal depth stays the cleanest steady check. A side fix
for breakline meshes ignoring the maximum triangle area is in PR #512 (issue #511).

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
| `subgrid_tables.py` | the method: `SubgridTables` (sorted sub-triangle beds, h(η), η(h), f(η), Kc(η)), `edge_subface_beds` (subface beds and 1/n), and `Subgrid_storage_operator` (the per-step update) |
| `run_phase0.py` | hydrograph runs (5 → 60 → 5 m³/s, 5.5 h) for every variant; `--gpu`, `--diag`; environment switches `SUBGRID_TAG` and `SUBGRID_LOW_FROUDE` |
| `evaluate_phase0.py` | scores the runs against a reference: flood-extent CSI, peak Q and its timing, arrival time, peak stage |
| `report_figures.py` | the figures in this report, from the `results/*_diag*.npz` files |
| `test_phase1.py`, `test_phase2.py` | lake at rest with partly wet cells, mass conservation, dam breaks (run as scripts; all pass) |
| `static_storage.py`, `static_conveyance.py`, `static_edges.py` | static checks of storage, conveyance and edge area/conveyance against the DTM |
| `steady.py`, `trap_conv.py` | steady runs against Manning normal depth; the straight-trapezoid check that found the reference bias |
| `launch_*.sh` | the tmux batch scripts used for each round of runs |

## Known issues and open items

1. **Outlet reach:** every coarse run is low approaching the steep outlet reach
   (x > 1600 m); EKLC 500 is 0.38 m low at x = 1925 m, where coarse cells straddle the
   change of slope. Not yet investigated.
2. **Unsteady differences:** on the rising limb the level at Q = 20 is 1.28 against
   1.46 m on the 500 mesh, partly a loop-rating effect. On the falling limb the near
   floodplain holds 0.1 against 0.7 thousand m³ at 4.5 h.
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
python report_figures.py                   # redraw claude/figures/subgrid/*.png
PYTHONPATH=$PWD python test_phase2.py      # tests run as scripts
python trap_conv.py 1 LF=5                 # straight trapezoid check
```

`report_figures.py` needs the `--diag` results of band025 (TU), band1 (old flux and
LF4), coarse500_sub8EKL, coarse500/2000_sub8EKLC and coarse500_centroid.

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
| b52906f0 | status report; bench runs with the switch on |
| d9d8bbe3 | status rescored against band025TU |

The branch is not pushed to origin.
