# Subgrid terrain development: start here

This clone (`~/Projects/anuga_core_subgrid`, branch `develop_subgrid`) is the
separate home for developing **subgrid terrain** in ANUGA: sub-cell storage and
conveyance so a coarse mesh can carry channels narrower than its cells.
Read this file first, then `claude/SUBGRID_DESIGN.md` (the design and phased
plan) and `claude/SUBGRID_LITERATURE.md` (prior work, closest paper first).

Handed over on 2026-11-07 from a session working in `~/anuga_core`.

## Setup

| Item | Value |
|---|---|
| Clone | `~/Projects/anuga_core_subgrid` |
| Branch | `develop_subgrid`, created from `anuga-community/develop` at b202bbab (#505 merged) |
| Remotes | `origin` = **stoiver/anuga_core** (the user's fork, push here); `upstream` = **anuga-community/anuga_core** (never push; PRs only) |
| Conda env | `anuga_subgrid_env_3.14`, from `environments/environment_3.14.yml`, editable **CPU** build (gcc, `gpu_offload=false`) |

Note the remote names are the reverse of `~/anuga_core`, where `origin` is
anuga-community and the fork is called `stoiver`. In this clone, `git push
origin develop_subgrid` goes to the fork, which is right.

```bash
conda activate anuga_subgrid_env_3.14
cd ~/Projects/anuga_core_subgrid
pip install --no-build-isolation -e .        # rebuild after C/Cython changes
cd /tmp && OMP_NUM_THREADS=1 pytest --pyargs anuga -m 'not slow' -q   # fast suite, ~5 min
```

- The CPU build runs the whole suite in one process, including the mode-2
  (unified) tests, whose `omp target` regions run on the host. Use it for
  Phases 0–2.
- A GPU (nvc) build is only needed for Phase 3. Build it in a separate
  build dir with `-Csetup-args=-Dgpu_offload=true` and the NVHPC compilers on PATH
  (`/opt/nvidia/hpc_sdk/Linux_x86_64/26.3/compilers/bin`). Read CLAUDE.md's
  "Testing a GPU-offload (nvc) build" first: the one-process suite aborts on a
  GPU build.
- Keep the branch up to date with `git fetch upstream && git merge upstream/develop`
  (merge, not rebase; avoid force-pushes).

## Working rules carried over from the main project

- **Synthetic data only in the repo.** Tests, validation cases, docs and issues
  use synthetic geometry. See "Private reference data" below for what must
  never be committed or posted.
- **Ask before** merging any PR, closing issues, posting comments or issues,
  and before pushing to a new remote branch.
- Commit messages end with
  `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`; PR bodies end with
  `🤖 Generated with [Claude Code](https://claude.com/claude-code)`.
- Simulations: `OMP_NUM_THREADS=8`, at most two at a time; GPU runs one at a
  time; pytest always with `OMP_NUM_THREADS=1`.
- Long runs on the laptop: wrap them in
  `systemd-inhibit --what=handle-lid-switch:sleep:idle` so a closed lid
  doesn't suspend them.
- New algorithm code goes behind its own flow algorithm (`DE_subgrid`), so
  existing algorithms stay bit-for-bit unchanged. Prove that with the existing
  suite.

## Where things stand

- **Design:** public design note
  [anuga_core#508](https://github.com/anuga-community/anuga_core/issues/508),
  copied to `claude/SUBGRID_DESIGN.md`. The user also has a private page with
  diagrams: https://claude.ai/artifact/AAxpHGqGzj5dSYTDz3Bpvv
- **Companion work, local time stepping:**
  [#466](https://github.com/anuga-community/anuga_core/issues/466) (estimate,
  implementation plan, literature review); private page
  https://claude.ai/artifact/TrhdUofmoDmDGCzyqMGi4V. LTS and subgrid are
  complementary. Both need one flux per edge (see "Code map").
- **Motivating issue:** [#453](https://github.com/anuga-community/anuga_core/issues/453),
  channels narrower than the cells. The public thread can be discussed; the
  reporter's data cannot (see below).
- **Status (2026-11-08):** Phases 0, 1 and most of 2 are built as a mode-1
  prototype. **Read `claude/SUBGRID_STATUS.md`** for what exists, results, known
  issues and how to run it.

## The real-world case

The real-world case behind this work is the reporter's Serna model from #453,
compared against HEC-RAS. What the earlier runs showed, and why HEC-RAS does
well on coarse cells, is written up privately in
`~/Projects/serna_anuga_hecras/claude_analysis/SERNA_FINDINGS_FOR_SUBGRID.md`.
Read it, but keep its contents out of this repo (see the rules below).

## Private reference data: rules

- **Serna package:** `~/Projects/serna_anuga_hecras` (not a git repo).
  - The DTM is licensed to the reporter (PMacAddress), who cannot authorise
    publication.
  - Use it, and every file that carries its elevations (`.sww`, `.hdf`, meshes
    with elevations set), **only for our own analysis**.
  - Never commit, upload, post or redistribute any of it, or numbers derived
    from it, in this repo, issues, PRs or any web service.
- **Previous analysis, copied for you:** `~/Projects/serna_anuga_hecras/claude_analysis/`
  (3.2 GB). In it:
  - `flow/`:
    - the 72 h runs: `asgiven`, `nocut`, `candidate`, `res64`, `res16`,
      `resolved` (4 m²), `res4_b492`, `res2_b492`;
    - the scripts `run_resolved.py` (env vars `RIVER_AREA`, `HOURS`) and
      `run_candidate.py`;
    - the scoring scripts `evaluate.py` and `evaluate_fine.py` (against
      HEC-RAS) and `pits.py` and `pits_fine.py` (trapped water in pits);
    - the results in `eval*.txt` and `pits*.txt`.
  - `static_storage.py`, `variants.py`, `ledger.py`: static stage–volume
    checks.
  - `reply_453_draft.md`: an unposted reply to the reporter. The user reads
    any reply before it is posted; never post it yourself.
  - The run directories under `flow/` are useful as resolved references for
    checking `DE_subgrid` on the real case, privately.
- **Also confidential, unrelated to subgrid:** Bill Mitchell's tide and surge
  benchmark kit and preprint (`~/Projects/bill-mitchell`). These are under
  embargo until Bill gives the OK, so keep them out of everything.
- **TUFLOW and FMA Challenge data:** no licence to publish. Don't use them for
  public cases (#440 was closed for this reason).

## Code map (on develop at b202bbab)

- `anuga/shallow_water/shallow_water_domain.py`:
  - `set_flow_algorithm` (line ~4787; `DE_ader2` is the default);
  - `compute_fluxes` (~5549), `distribute_to_vertices_and_edges` (~5901),
    `update_conserved_quantities` (~6154);
  - `evolve_one_euler_step` (~6650), `set_compute_mode` (~8875).
- `anuga/shallow_water/gpu/core_kernels.c`: the shared kernels for both
  compute modes (CPU `parallel for simd` with `CPU_ONLY_MODE`, else `omp target`).
  - **Extrapolation:** `core_extrapolate_centroid_pass` (25),
    `core_extrapolate_edge_pass_on` (196); the ADER predictor
    `core_ader_ck_predictor_edge` (4509) is fused into the edge pass.
  - **Fluxes:** `core_compute_fluxes_central` (3412) is the default,
    cell-based, and solves each edge twice. `core_compute_fluxes_scatter_on`
    (4210) solves each edge once and scatters with atomics.
    `core_compute_fluxes_edge_based` (3850) is a two-kernel "slot" variant.
  - **Update and friction:** `core_forcing_and_update_on` (2590),
    `core_manning_friction_*_semi_implicit` (3031–3203), `core_protect` (2784).
- `anuga/shallow_water/gpu/gpu_kernels.c`: `gpu_flux_mode` (line ~300) picks
  between the cell-based, scatter and slot flux modes.
- `anuga/shallow_water/sw_domain_openmp.c` and `sw_domain_openmp_ext.pyx`:
  the mode-1 (legacy) C path and its Cython glue.
- `anuga/shallow_water/sw_domain.h`: `struct domain`, the arrays both modes
  share. New per-cell or per-edge tables go here.

**Measured flux-kernel facts** (commit cf2d53be, Jorge Galvez, 2026-08-25,
16M-triangle dam break, V100 and H200):
- Scatter mode (one Riemann solve per edge, `omp atomic`, at most 3 adds per
  entry) with ADER2 reached 656 Mcell-steps/s on V100. It is a win, and it is
  digit-identical across GPU vendors.
- The two-kernel slot variant (`core_compute_fluxes_edge_based`) was 15%
  *slower*: its slot records cost more memory bandwidth than the duplicate
  solves.
- Scatter mode is opt-in (`reconstruct_edge_bed = 2` plus owned edges) and
  doesn't yet handle riverwalls or tracers.

For subgrid, the per-edge subface work belongs in a scatter-style kernel, so
each edge is computed once.

## Suggested first tasks

1. **Phase 0:** build the synthetic benchmark (a trapezoidal channel narrower
   than the coarse cells, a floodplain with small hollows, and a hydrograph),
   with a converged resolved-mesh reference and the metrics listed in
   `SUBGRID_DESIGN.md`. Put it under
   `validation_tests/` only if it stays small; otherwise keep it in a local
   `subgrid_bench/` directory that isn't committed.
2. **Phase 1 prototype:** per-cell V(η) and A(η) tables plus the
   volume-conserving update, in mode 1 only. Check lake at rest with partly
   wet cells first.
3. Privately, re-score the Serna case on the reporter's coarse mesh without
   the trench, with `evaluate.py` and `pits.py`, once Phase 1 works.
