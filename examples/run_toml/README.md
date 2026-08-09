# Running ANUGA from a TOML file with `anuga_toml_run`

`anuga_toml_run` runs a complete ANUGA simulation from a single plain-text
[TOML](https://toml.io) configuration file — no Python scripting required.
You describe the mesh (including voids for buildings and other obstructions),
initial conditions, boundaries, rainfall, inlets, hydraulic structures and bed
erosion declaratively, and the runner builds the domain, evolves it, and writes
the SWW output (plus GeoTIFFs of the peak quantities).

This directory has three scenarios, smallest first:

| Folder | Scenario | What it shows |
|--------|----------|---------------|
| [`simple/`](simple/) | Dam break in a flat box | The minimum viable TOML: mesh from a CSV polygon, constant initial conditions, reflective walls. |
| [`complex/`](complex/) | Sloping floodplain with inflow + rain | A "Cairns-like" run: sloped bed from an x,y,z file, a refined interior region, an inlet hydrograph, rainfall, a prescribed-stage outlet, and evolve-loop reporting hooks. |
| [`cairns/`](cairns/) | Real Cairns DEM tsunami | The full real-world dataset: elevation from a DEM raster, a shapefile bounding polygon, a tsunami wave boundary, and one each of a bridge, culvert, pipe culvert, weir, and pumping station. |

`simple/` and `complex/` run out of the box — every input is a small CSV
committed alongside the TOML, so no external datasets are needed. `cairns/`
references a real DEM that is shared (once) under
[`../data/cairns/`](../data/cairns/); its `cairns_example.toml` is heavily
commented and documents every available TOML section.

A legacy **Excel front-end** to the same Cairns scenario is kept under
[`../cairns_toml_excel/`](../cairns_toml_excel/) for users of the older `.xlsx`
interface; new scenarios should use the TOML runner shown here.

## How to run

`anuga_toml_run` is installed as a console command (via meson). In a source
checkout you can also run the script directly with
`python <repo>/scripts/anuga_toml_run.py`.

```bash
# Simple scenario (serial)
cd simple
anuga_toml_run dam_break.toml

# Complex scenario (serial)
cd ../complex
anuga_toml_run floodplain.toml

# Either scenario in parallel (e.g. 4 processes)
mpiexec -np 4 anuga_toml_run floodplain.toml

# Dry run: preview the scenario as an HTML summary in your browser,
# without building a mesh or running the simulation.
anuga_toml_run floodplain.toml --dry-run
```

The runner `cd`s into the TOML's directory first, so all paths inside the TOML
are relative to the TOML file. Output lands in
`OUTPUT/RUN_<timestamp>_<scenario>/` (SWW, a `Simulation_logfile.log`, a copy of
the inputs under `code/`, and peak-quantity rasters).

### Dry run — preview a scenario

`--dry-run` (`-n`) parses the TOML and writes a self-contained HTML summary —
project settings, mesh resolution and refinement, boundary conditions, a
Manning's-*n* friction breakdown, a structures table, and, for rainfall, a
catchment-mean **hyetograph** (15-minute intensity bars with a cumulative-depth
curve) read straight from the gauge timeseries — then opens it in the default
browser. Nothing is simulated, so it is a fast way to sanity-check a config.

```bash
anuga_toml_run floodplain.toml --dry-run                 # write + open in browser
anuga_toml_run floodplain.toml --dry-run --no-browser    # write the file only
anuga_toml_run floodplain.toml --dry-run --summary-output preview.html
```

The summary defaults to `<config>_summary.html` next to the TOML.

### Viewing the raw TOML — `anuga_toml_view`

To read the config *text* in the terminal, `anuga_toml_view` syntax-highlights
it and collapses long runs of repeated array-of-table blocks (the many
one-per-file friction / culvert / rainfall entries) to the first block plus a
"N more" marker, so a big scenario stays legible:

```bash
anuga_toml_view towradgi.toml            # highlighted, collapsed, paged
anuga_toml_view towradgi.toml --full     # expand every block
anuga_toml_view towradgi.toml --no-color # plain text
```

(A source checkout without the console scripts installed can run
`python <repo>/scripts/anuga_toml_view.py …`.)

## Anatomy of a scenario directory

```
simple/
  dam_break.toml        # the scenario definition
  bounding_polygon.csv  # x,y polygon for the domain outline
  dam.csv               # x,y polygon for the initial reservoir

complex/
  floodplain.toml       # the scenario definition
  make_inputs.py        # regenerates the CSVs below (optional; they are committed)
  bounding_polygon.csv  # domain outline
  elevation.csv         # x,y,z sloped bed (header row: x,y,z)
  refine_channel.csv    # interior region polygon (finer resolution)
  inlet_line.csv        # cross-section line the inlet injects across
  inlet_hydrograph.csv  # time,discharge timeseries (header row)
  rain.csv              # time,rain_mm_hr timeseries (header row)
  downstream_stage.csv  # time,stage timeseries for the outlet boundary
  user_functions.py     # optional evolve-loop reporting hooks

cairns/
  cairns_example.toml   # the scenario definition (documents every section)
  cairns_mesh/          # shapefile outline (+ sidecars) and structure lines
  cairns_boundarycond/  # tsunami wave + mean-sea-level time,stage CSVs
  user_functions.py     # optional evolve-loop reporting hooks
  # the DEM raster is shared, at ../data/cairns/cairns.asc
```

## File-format quick reference

- **Polygon / line CSVs** (`bounding_polygon.csv`, `dam.csv`, `inlet_line.csv`,
  region polygons): plain `x,y` rows, **no header**.
- **Timeseries CSVs** (rainfall, inlet hydrograph, boundary stage): a **header
  row** followed by `time,value` rows.
- **Elevation point CSV** (`elevation.csv`): `x,y,z` rows; a header is optional.
- **Boundary tags for a CSV bounding polygon**: because a CSV polygon carries no
  tag attributes, the `[mesh]` section lists `[[mesh.boundary_tags]]` entries
  mapping each 0-based polygon **edge index** to a tag name. (A `.shp` bounding
  polygon instead carries the tags as a feature attribute — see the Cairns
  example.) Every tag must then appear under `[boundary_conditions]`.

## Boundary types

`type` in `[[boundary_conditions.boundaries]]` is one of:

- `Reflective` — solid wall (zero flux); no extra fields.
- `Stage` — transmissive momentum with prescribed stage from a `time,stage`
  file (GPU-friendly: `Transmissive_n_momentum_zero_t_momentum_set_stage`).
- `Flather_Stage` — Flather radiation boundary with a prescribed stage file.

`Stage` / `Flather_Stage` need `file = "...csv"` and an optional `start_time`.
