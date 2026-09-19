# ANUGA 4.1.0 (unreleased)

## Breaking changes

* **The legacy forcing-function classes have been removed**, as announced in
  4.0.0: `Rainfall`, `Inflow`, `Wind_stress`, `Wind_stress_fast`,
  `Barometric_pressure`, `Barometric_pressure_fast` and `General_forcing` are
  gone from `anuga.shallow_water.forcing`, and `anuga.Inflow`, `anuga.Rainfall`
  and `anuga.Wind_stress` are no longer exported. Use `Rate_operator.rainfall()`
  / `Rate_operator.inflow()`, `Wind_stress_operator` and
  `Barometric_pressure_operator`, which work in both compute modes. Note the
  rainfall units change from mm/s to mm/hr. See `UPGRADING_TO_4.0.md` §2.
  `Cross_section` stays in `anuga.shallow_water.forcing`.

## Selected fixes

* Line regions (`Region(line=...)`, `Set_quantity(line=...)`, the exchange
  lines of the structure operators, and the edges of an expanded polygon
  region) selected a compiler-dependent set of triangles when the line lay on
  mesh lines: the same script gave a 68% larger inlet area under one compiler
  than another (#231). The segment-triangle test is now tolerance-aware
  clipping with one rule: a triangle is selected when the segment passes
  through or along it with positive length, and not when it only touches a
  vertex or ends on an edge from outside. Selections change wherever a line
  touched triangles at a point: a 10 m exchange line on a 5 m mesh line now
  selects the 8 triangles along it rather than 32 (or 54).
* Two runs writing the same SWW file used to interleave their frames on the
  shared time dimension, giving a file that opened fine and held plausible
  data with a non-monotonic time variable (#232). The writer now holds a
  sidecar `<name>.sww.lock` for the life of the run and a second run that
  would create the same file stops with `SWWFileInUseError`; stale locks from
  dead processes are taken over with a warning. It also refuses to append a
  frame earlier than the last one on file that rewrites no existing frame
  (`SWWTimeOrderError`); checkpoint resumes, which rewrite frames, still work.
* Sediment transport: the bed slope for the `'depth_slope'` shear closure
  (and the surface slope for `'energy_slope'`) is now the least-squares
  gradient of the centroid values over the cell and its neighbours. It was
  read from the edge values, which the DE algorithms rebuild every step
  through the hydrodynamic limiter: cells along a reflective wall saw no
  slope and never eroded, and the slope was reduced wherever the limiter
  engaged. Results change for those closures near boundaries, kinks and
  steps; `'quadratic_drag'` (the default) is unaffected.

## Smaller improvements

* `Domain.get_water_volume()` and `compute_total_volume()` take `region=` (an
  `anuga.Region`) or `indices=` to sum the water in part of the domain, as
  `Quantity.get_integral()` already did (#22). Regional queries do not enter
  `volume_history`.
* `Wind_stress_operator` and `Barometric_pressure_operator` accept
  `use_coordinates=False` with a `file_function`, replacing the file-driven
  wind and pressure fields of the removed `_fast` classes. The file's
  precomputed time series is interpolated for every point at once
  (`anuga.utilities.function_utils.evaluate_file_function_all_points`).

---

# ANUGA 4.0.0

First release on the 4.x line, and the first release of the work that has been
accumulating on `develop` since 3.3.10 — 993 commits across 935 files.

The headline is a **GPU/multicore solver**: the shallow-water kernels now run
through a single unified C implementation that executes on the CPU or offloads
to a GPU, with mode-1 and mode-2 agreeing bit-for-bit on the paths that matter.
Alongside that are a decade-old correctness fix in the structure operators, a
substantial set of parallel and wet/dry fixes, and container images.

**Existing scripts keep working.** The default compute mode is unchanged
(`'legacy'`), and both the unified mode and GPU offload are opt-in. What makes this a major version is the
removal of long-deprecated code, not a change in day-to-day behaviour.

---

## Highlights

### The unified compute mode

The solver and its operators now have a single C implementation that runs
CPU-multicore, and can offload to a GPU on a suitably built install.

```python
domain.set_compute_mode('unified')     # per domain; CPU-multicore by default
```

or process-wide with `ANUGA_DEFAULT_COMPUTE_MODE=unified`.

* **No GPU required.** 'unified' is a compute mode, not a GPU switch: on an
  ordinary build it runs the same unified C kernels on the CPU. GPU offload is a
  separate, process-wide opt-in — `anuga.set_gpu_offload(True)` on a build made
  with the NVIDIA HPC SDK. `gpu_offload_supported()` reports whether a build can.
* **One implementation of the physics.** `culvert_compute_one()` is shared by
  both paths, so 'legacy' and 'unified' give bit-identical culvert results.
* On GPU: fluxes, extrapolation, boundaries, riverwalls, culverts and operators
  all execute on device, with multi-GPU via MPI and device-side halo exchange.
* New controls: `set_gpu_offload()`, `gpu_offload_enabled()`,
  `gpu_offload_supported()`, `set_omp_num_threads()`, `get_omp_num_threads()`.

`set_multiprocessor_mode(1|2)` still works as a thin wrapper, but
`set_compute_mode('legacy'|'unified')` is the preferred API for new code.

### New timestepping method

* **`DE_ader2`** — a fused ADER-2 predictor/extrapolation step, measured **1.75×
  faster than DE1** at equivalent accuracy. Select with
  `domain.set_flow_algorithm('DE_ader2')`.

### Rainfall and forcing

* `Raster_rate_operator` — apply gridded rainfall rasters directly.
* Australian Rainfall & Runoff support: `ARR_rate_operator`, `Arr_hub_rain`,
  `Arr_ifd_rain`, `ARR_point_rainfall_patterns`, `Arr_grd`.
* `Wind_stress_operator` and `Barometric_pressure_operator` — operator
  equivalents of the legacy forcing classes (see Deprecations).

### Boundaries

* `Absorbing_wave_boundary` and `Characteristic_wave_boundary`.

### Meshes and memory

* `sequential_mesh_dump` / `sequential_mesh_load` — partition once, run many
  scenarios; NetCDF, self-describing, safe to share.
* `uniform_refine_domain`, `sequential_mesh_refine`, `create_parallel_mesh`.
* ~58% reduction in quantity memory; `quantity_memory_stats()`,
  `domain_memory_stats()` to inspect it.

### Tooling

* Container images: CPU, GPU (NVHPC/nvc), and GPU+CUDA-aware-MPI, published to
  GHCR on release.
* GUIs: `anuga_sww_gui`, `anuga_animate_gui`.
* `anuga_toml_run` for the TOML scenario interface (which itself shipped in
  3.3.x; new here are `[[mesh.interior_holes]]` and `[[erosion]]` sections).
* `anuga_run_isolated_tests` — per-test process isolation, needed on GPU builds.

---

## Breaking changes

* **`anuga.culvert_flows` is removed** (7 790 lines). It was superseded years ago.
  Migrate to the structure operators — `Boyd_box_operator`, `Boyd_pipe_operator`,
  `Weir_orifice_trapezoid_operator`. `examples/structures/run_open_slot_wide_bridge.py`
  shows the equivalent setup.
* **Local-timestepping attributes removed**: `max_flux_update_frequency`,
  `flux_update_frequency`, `update_next_flux`, `update_extrapolation`,
  `edge_timestep`, `allow_timestep_increase`. The 3.1.9 implementation was not
  GPU-compatible and had been dead code; see `claude/FUTURE_WORK.md` P3.1 for the
  redesign.
* **Structure operators now write back a level water surface** (see below). This
  changes results for any structure whose inlet sits on a sloping bed.

### The structure write-back change (#229)

Structure operators wrote their transfer back as a **uniform depth** across each
inlet. On a sloping bed that tilts a level water surface onto the bed, so a lake
at rest was disturbed by roughly half the bed elevation range across the inlet —
every timestep, at zero discharge.

The write-back now applies the volume change by **levelling the stage** (water
finds its level), clamping cells at their bed.

What to expect:

| case | change |
|---|---|
| flat bed under the inlet | **none** — bit-identical by construction |
| sloping bed | results move; magnitude scales with the bed range across the inlet |

Measured on the Towradgi catchment (22 culverts on real terrain, inlet bed
spread median 0.77 m): peak stage difference **0.32 m**, 99th percentile 1.5 mm.
Three of four flat-bed validation cases were bit-identical.

If you calibrated a model against pre-4.0.0 results with structures on sloping
ground, re-check it.

---

## Deprecations

The legacy forcing classes in `anuga.shallow_water.forcing` now all emit
`DeprecationWarning` and **will be removed in 4.1**:

| deprecated | use instead |
|---|---|
| `Rainfall` | `Rate_operator.rainfall()` |
| `Inflow` | `Rate_operator.inflow()` |
| `Wind_stress`, `Wind_stress_fast` | `Wind_stress_operator` |
| `Barometric_pressure`, `Barometric_pressure_fast` | `Barometric_pressure_operator` |

These are **silently skipped** in the `'unified'` compute mode, which applies
forcing in C and handles only Manning friction; a warning is issued when that
happens. Migrating to the operators is required to use them with 'unified'.

`manning_friction_semi_implicit` is *not* deprecated — it is an in-step
semi-implicit term, not an operator.

---

## Selected fixes

* **Parallel inlet mass balance** (#193) and **startup mass loss** (#200) — both
  affected conserved volume in ordinary runs.
* **Culvert stack overflow** with more than 64 culverts (#217) — buffers were
  sized by a constant that was only the initial capacity.
* **GPU culvert inlet cap** (#225) — an inlet was limited to 64 triangles.
* **Riverwall crest changes reach the device** (#224) — runtime
  `set_elevation()` was silently ignored under GPU offload.
* **DE0 non-GPU boundary fallback** — Euler steps silently ignored
  Python-evaluated boundary types in the 'unified' mode.
* Degenerate-timestep protection now warns rather than silently not running
  in the 'unified' mode (#189).

---

## Smaller improvements

* Importing anuga from an unbuilt source tree now explains itself. It used to
  fail with a bare `ModuleNotFoundError: No module named 'anuga._version'`,
  which pointed at neither cause: `_version.py` is generated at build time, and
  because Python searches the working directory first, an unbuilt tree also
  shadows a correctly installed anuga whenever you run from the repository
  root. The error now names the directory it imported from and says how to fix
  it (#237).

## Requirements

* Python 3.10 – 3.14, numpy ≥ 2.0
* GPU offload requires the NVIDIA HPC SDK (`nvc`); the standard build is
  unaffected and needs no GPU.

## Known issues

* Windows CI pins the conda-forge mingw sysroot chain to build 10. Build 11
  (2026-08-20) produces a toolchain whose binaries abort in the stack protector;
  tracked at conda-forge/m2w64-sysroot-feedstock#23. This pins only the CI
  environment — released wheels are unaffected.
* A structure sitting in near-still water amplifies roundoff; see
  `claude/KNOWN_ISSUES.md`.

## Thanks

Stephen Roberts, Jorge Luis Gálvez Vallejo, wangshuo, Samir Shaikh, and everyone
who reported issues against 3.3.x.

The GPU and multicore solver was developed in collaboration with the Centre for
Development of Advanced Computing (C-DAC), India, as part of a catchment flood
prediction project aimed at two-day forecasts, and the National Computational
Infrastructure (NCI), Australia, whose Gadi supercomputer provided the GPU
systems it was developed, tested and benchmarked on. Thanks to Samir Shaikh,
Shweta Das and Rutvik Gulhane (C-DAC) and Jorge Luis Gálvez Vallejo (NCI).
