# tools/

Helper scripts for installing ANUGA and a few developer utilities.

> **Authoritative install source:** the conda environment files in
> [`../environments/`](../environments/) and the build/test instructions in
> [`../CLAUDE.md`](../CLAUDE.md). The scripts here are conveniences and can drift;
> when in doubt, follow those.

## Recommended install

For most users/developers:

```bash
# 1. Miniforge + a conda env from environments/environment_<pyver>.yml
bash tools/install_miniforge.sh          # Linux/macOS
tools\install_miniforge_windows.bat      # Windows

# 2. Build ANUGA (editable)
conda activate anuga_env_3.14
pip install --no-build-isolation -e .
```

`install_miniforge*.{sh,bat}` are the only scripts here referenced by the docs
(`docs/source/installation/install_anuga_developers.rst`).

## GPU build

```bash
bash tools/install_anuga_nvc.sh          # NVIDIA HPC SDK (nvc) GPU build
```

Builds the GPU-offload extension (`-Dgpu_offload=true`) with `nvc`, runs the
isolated GPU test suite, and clears a stale build dir so the compiler switch
takes effect. Env vars: `PY` (default 3.14), `GPU_ARCH` (default `cc120`),
`NVHPC_ROOT`. See `../CLAUDE.md` → "Testing a GPU-offload (nvc) build".

## Contents

| Script | Purpose |
|--------|---------|
| `install_miniforge.sh` / `install_miniforge_windows.bat` | **Recommended** — install Miniforge and create the conda env. Referenced by the docs. |
| `install_anuga_nvc.sh` | GPU-offload build via the NVIDIA HPC SDK (nvc). |
| `install_ubuntu_22_04.sh`, `install_ubuntu_24_04.sh` | Install OS-level build dependencies on current Ubuntu LTS. |
| `count_lines.py` | Report total lines of code in ANUGA (dev utility). |
| `clear_for_master.sh` | Remove generated `*.c`/`*.so` build artifacts (dev utility). |

### Legacy / unreferenced (candidates for removal)

Kept for now but referenced by nothing in CI or docs; several predate the move
to GitHub Actions and conda `environment_*.yml` files:

- `install_conda.sh`, `install_conda_macos.sh` — Travis-era (invoked by the old
  `.travis.yml`).
- `install_conda_ubuntu.sh`, `install_conda_ubuntu_22_04.sh`, `install_ubuntu.sh`
  — older conda/apt install variants superseded by `install_miniforge*` +
  `environments/*.yml`.
- `find_repeated_old_div.sh` — a Python-2→3 (`old_div`) migration helper, obsolete.

## Notes

- Minimum Python is **3.10** (`pyproject.toml`: `requires-python = ">=3.10, <3.15"`).
- CI uses GitHub Actions (`../.github/workflows/`), not the scripts here.
