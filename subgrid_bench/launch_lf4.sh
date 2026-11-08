#!/bin/bash
# band1 reference with LF1 + tangential upwind (#510; low_froude 5), GPU.
# (results/band1_diag_lf4.npz was made with the pre-merge experimental build, same flux.)
cd "$(dirname "$0")"
source ~/miniforge3/etc/profile.d/conda.sh
S=results/lf4_status.log
echo "started $(date '+%F %T')" >> $S
conda activate anuga_subgrid_gpu_env_3.14
export PATH=/opt/nvidia/hpc_sdk/Linux_x86_64/26.3/compilers/bin:$PATH OMP_NUM_THREADS=8
SUBGRID_LOW_FROUDE=5 SUBGRID_TAG=_lf4 python run_phase0.py band1 --gpu --diag > results/band1_diag_lf4.log 2>&1
echo "$(date '+%F %T') band1 LF4 (GPU) exit $?" | tee -a $S
