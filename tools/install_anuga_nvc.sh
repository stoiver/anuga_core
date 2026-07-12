#!/bin/bash
#
# Install ANUGA with GPU offloading support using the NVIDIA HPC SDK (nvc).
#
# Run this script as follows:
#
#   bash /path/to/anuga_core/tools/install_anuga_nvc.sh
#
# The script locates nvc in the NVIDIA HPC SDK, activates the conda
# environment, and builds the GPU-enabled anuga package.
#
# Environment variables (all optional):
#
#   PY          Python version to use (default: 3.14)
#   GPU_ARCH    GPU compute capability, e.g. cc120 (RTX 5070 Blackwell),
#               cc90 (H100), cc80 (A100), cc70 (V100).  Default: cc120
#   NVHPC_ROOT  Override path to NVIDIA HPC SDK root if auto-detection fails.
#               e.g. /opt/nvidia/hpc_sdk/Linux_x86_64/26.3
#
# Example — build for an A100 with Python 3.13:
#
#   PY=3.13 GPU_ARCH=cc80 bash /path/to/anuga_core/tools/install_anuga_nvc.sh
#
# Prerequisites:
#   - NVIDIA HPC SDK installed (see KNOWN_ISSUES.md for apt install recipe)
#   - conda environment anuga_env_${PY} created via install_miniforge.sh

PY=${PY:-"3.14"}
GPU_ARCH=${GPU_ARCH:-"cc120"}

set -e

trap 'echo ""; echo "#====================================================="; echo "# Installation failed at line $LINENO"; echo "#====================================================="; exit 1' ERR

SCRIPT=$(realpath "$0")
SCRIPTPATH=$(dirname "$SCRIPT")
ANUGA_CORE_PATH=$(realpath "$SCRIPTPATH/..")

echo "#============================================================"
echo "# ANUGA NVC GPU build  (PY=${PY}  GPU_ARCH=${GPU_ARCH})"
echo "#============================================================"
echo " "

# ------------------------------------------------------------------
# Locate nvc
# ------------------------------------------------------------------
if [ -n "$NVHPC_ROOT" ]; then
    NVC="$NVHPC_ROOT/compilers/bin/nvc"
else
    # Search /opt/nvidia/hpc_sdk/Linux_x86_64/ for the newest installed version
    NVHPC_BASE="/opt/nvidia/hpc_sdk/Linux_x86_64"
    if [ -d "$NVHPC_BASE" ]; then
        # Pick the highest version directory
        NVHPC_VER=$(ls "$NVHPC_BASE" | grep -E '^[0-9]+\.[0-9]+$' | sort -V | tail -1)
        NVC="$NVHPC_BASE/$NVHPC_VER/compilers/bin/nvc"
    else
        NVC=""
    fi
fi

if [ -z "$NVC" ] || [ ! -x "$NVC" ]; then
    echo "#=====================================================";
    echo "# ERROR: nvc not found."
    echo "#"
    echo "# Install the NVIDIA HPC SDK first:"
    echo "#"
    echo "#   curl -fsSL https://developer.download.nvidia.com/hpc-sdk/ubuntu/DEB-GPG-KEY-NVIDIA-HPC-SDK \\"
    echo "#     | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-hpcsdk-archive-keyring.gpg"
    echo "#   echo 'deb [signed-by=/usr/share/keyrings/nvidia-hpcsdk-archive-keyring.gpg] \\"
    echo "#     https://developer.download.nvidia.com/hpc-sdk/ubuntu/amd64 /' \\"
    echo "#     | sudo tee /etc/apt/sources.list.d/nvhpc.list"
    echo "#   sudo apt-get update -y && sudo apt-get install -y nvhpc"
    echo "#"
    echo "# Or set NVHPC_ROOT=/path/to/hpc_sdk/Linux_x86_64/<version>"
    echo "#=====================================================";
    exit 1
fi

echo "# nvc found: $NVC"
"$NVC" --version
echo " "

# ------------------------------------------------------------------
# Select the conda environment
#   Prefer an already-activated environment (this is what the user is working
#   in — likely with anuga already installed). Otherwise fall back to a
#   miniforge install in $HOME with env anuga_env_$PY.
# ------------------------------------------------------------------
if [ -n "$CONDA_PREFIX" ] && [ -n "$CONDA_DEFAULT_ENV" ] \
        && [ "$CONDA_DEFAULT_ENV" != "base" ]; then
    ENV_NAME="$CONDA_DEFAULT_ENV"
    CONDA_RUN=""   # build/test run in the current, already-activated shell
    echo "# Using the active conda environment: ${ENV_NAME}  (${CONDA_PREFIX})"
else
    CONDA_BIN="$HOME/miniforge3/bin"
    if [ ! -f "$CONDA_BIN/conda" ]; then
        echo "#=====================================================";
        echo "# ERROR: no conda environment is active and miniforge3 was not"
        echo "#        found at $HOME/miniforge3."
        echo "#"
        echo "# Either activate your anuga environment first"
        echo "#   (e.g. conda activate anuga_env_${PY}), or run"
        echo "#   install_miniforge.sh to create one."
        echo "#=====================================================";
        exit 1
    fi
    ENV_NAME="anuga_env_${PY}"
    if ! "$CONDA_BIN/conda" env list | grep -q "${ENV_NAME}"; then
        echo "#=====================================================";
        echo "# ERROR: conda environment '${ENV_NAME}' not found."
        echo "# Run install_miniforge.sh first (PY=${PY}), or activate your env."
        echo "#=====================================================";
        exit 1
    fi
    CONDA_RUN="$CONDA_BIN/conda run -n ${ENV_NAME}"
    echo "# conda environment: ${ENV_NAME}  (via $HOME/miniforge3)"
fi
echo " "

# ------------------------------------------------------------------
# Preflight: build backend + build requirements
#
# The build below uses `pip install --no-build-isolation`, so pip does NOT
# create a temporary build environment — the meson-python backend (module
# `mesonpy`) and the rest of pyproject's build-system.requires must already be
# installed in the target environment.  If they are missing, pip dies with an
# opaque "BackendUnavailable: Cannot import 'mesonpy'" traceback that never
# mentions meson-python.  Check up front and say exactly what to install.
#
# This also reports the environment's *actual* Python version: when an env is
# already activated we use it and ignore $PY, so the banner's PY can differ.
# ------------------------------------------------------------------
echo "# Preflight: checking build backend and build requirements"

PREFLIGHT_PY='
import importlib.util, shutil, sys
missing = []
for mod, pkg in (("mesonpy", "meson-python"), ("Cython", "cython"),
                 ("pybind11", "pybind11"), ("numpy", "numpy")):
    if importlib.util.find_spec(mod) is None:
        missing.append(pkg)
for exe in ("meson", "ninja"):
    if shutil.which(exe) is None:
        missing.append(exe)
print("PREFLIGHT|%d.%d|%s" % (sys.version_info[0], sys.version_info[1],
                              " ".join(missing)))
'
PREFLIGHT_OUT=$($CONDA_RUN python -c "$PREFLIGHT_PY" 2>/dev/null || true)
PREFLIGHT_LINE=$(printf '%s\n' "$PREFLIGHT_OUT" | grep '^PREFLIGHT|' | tail -1 || true)

if [ -z "$PREFLIGHT_LINE" ]; then
    echo "#====================================================="
    echo "# ERROR: could not run python in environment '${ENV_NAME}'."
    echo "#        Is the environment usable?  Try:  ${CONDA_RUN} python -V"
    echo "#====================================================="
    exit 1
fi

ENV_PY_VER=$(printf '%s' "$PREFLIGHT_LINE" | cut -d'|' -f2)
MISSING=$(printf '%s' "$PREFLIGHT_LINE" | cut -d'|' -f3)

echo "#   environment '${ENV_NAME}' is Python ${ENV_PY_VER}"

if [ -n "$MISSING" ]; then
    echo "#====================================================="
    echo "# ERROR: environment '${ENV_NAME}' is missing build requirements:"
    echo "#"
    echo "#     ${MISSING}"
    echo "#"
    echo "# This build uses 'pip install --no-build-isolation', so the"
    echo "# meson-python backend and its build requirements must already be"
    echo "# installed in the environment.  Without them pip fails with an"
    echo "# opaque \"BackendUnavailable: Cannot import 'mesonpy'\"."
    echo "#"
    echo "# Fix - install them into this environment:"
    echo "#"
    echo "#   conda install -c conda-forge ${MISSING}"
    echo "#"
    echo "# Or recreate the environment with everything already in it:"
    echo "#"
    echo "#   conda env create -n anuga_env_${ENV_PY_VER} \\"
    echo "#       -f environments/environment_${ENV_PY_VER}.yml"
    echo "#   conda activate anuga_env_${ENV_PY_VER}"
    echo "#====================================================="
    exit 1
fi

echo "#   ok - meson-python, meson, ninja, cython, pybind11, numpy all present"
echo " "

# ------------------------------------------------------------------
# Build ANUGA with GPU offloading
# ------------------------------------------------------------------
echo "#============================================================"
echo "# Building ANUGA with GPU offloading"
echo "#   CC=$NVC"
echo "#   gpu_offload=true  gpu_arch=${GPU_ARCH}"
echo "#============================================================"
echo " "

cd "${ANUGA_CORE_PATH}"

# meson-python reuses the build/cp<ver> directory and only honours CC on the
# FIRST configure of a dir — a subsequent build just runs `meson setup
# --reconfigure`, which keeps the originally detected compiler. A leftover dir
# configured with gcc (e.g. a prior CPU build) therefore stays on gcc and the
# gpu_offload=true guard in meson.build rejects it ("not supported with gcc").
# Remove any stale build dir so CC=nvc takes effect on a clean configure.
echo "# Removing any stale meson build directory (build/cp*) for a clean nvc configure"
rm -rf "${ANUGA_CORE_PATH}"/build/cp*
echo " "

$CONDA_RUN bash -c \
    "CC='$NVC' pip install --no-build-isolation -v -e . \
     -Csetup-args=-Dgpu_offload=true \
     -Csetup-args=-Dgpu_arch=${GPU_ARCH}"

echo " "
echo "#============================================================"
echo "# Running GPU test suite (isolated runner)"
echo "#   One fresh process per test.  A plain 'pytest' on this file"
echo "#   auto-skips on a GPU build: the NVHPC OpenMP-target runtime"
echo "#   aborts once many mode-2 GPU domains are created in a single"
echo "#   process, so the tests must each run in their own process."
echo "#   scripts/anuga_run_isolated_tests.py defaults to test_DE_gpu_omp.py"
echo "#   and opts in via ANUGA_GPU_TESTS_ISOLATED=1.  Run the script"
echo "#   directly (not the installed console command, which an editable"
echo "#   'pip install -e .' does not place on PATH)."
echo "#============================================================"
echo " "

$CONDA_RUN \
    python "${ANUGA_CORE_PATH}/scripts/anuga_run_isolated_tests.py"

echo " "
echo "#=================================================================="
echo "# Congratulations! ANUGA GPU build succeeded."
echo "#"
echo "# To use GPU mode activate the environment and set multiprocessor_mode=2:"
echo "#"
echo "#   conda activate ${ENV_NAME}"
echo "#   python -c \\"
echo "#     \"import anuga; d = anuga.rectangular_cross_domain(100,100); \\"
echo "#      d.set_boundary({b: anuga.Reflective_boundary(d) for b in d.get_boundary_tags()}); \\"
echo "#      d.set_multiprocessor_mode(2)\""
echo "#=================================================================="
