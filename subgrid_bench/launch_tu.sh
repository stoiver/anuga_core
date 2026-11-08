#!/bin/bash
# Bench with LF1 + tangential upwind (#510, low_froude 5): band025 reference on
# the GPU, the two EKLC subgrid runs on the CPU (one per 8-core chiplet).
# Writes results/*_diag_tu.npz
cd "$(dirname "$0")"
source ~/miniforge3/etc/profile.d/conda.sh
S=results/tu_status.log
echo "started $(date '+%F %T')" >> $S
export SUBGRID_LOW_FROUDE=5 SUBGRID_TAG=_tu
( conda activate anuga_subgrid_gpu_env_3.14
  export PATH=/opt/nvidia/hpc_sdk/Linux_x86_64/26.3/compilers/bin:$PATH OMP_NUM_THREADS=8
  python run_phase0.py band025 --gpu --diag > results/band025_diag_tu.log 2>&1
  echo "$(date '+%F %T') band025 TU (GPU) exit $?" | tee -a $S ) &
( conda activate anuga_subgrid_env_3.14; export OMP_NUM_THREADS=8
  python run_phase0.py coarse500_sub8EKLC --diag > results/coarse500_sub8EKLC_diag_tu.log 2>&1
  echo "$(date '+%F %T') coarse500_sub8EKLC TU exit $?" | tee -a $S ) &
( conda activate anuga_subgrid_env_3.14; export OMP_NUM_THREADS=8
  python run_phase0.py coarse2000_sub8EKLC --diag > results/coarse2000_sub8EKLC_diag_tu.log 2>&1
  echo "$(date '+%F %T') coarse2000_sub8EKLC TU exit $?" | tee -a $S ) &
wait
echo "$(date '+%F %T') ALL DONE" | tee -a $S
