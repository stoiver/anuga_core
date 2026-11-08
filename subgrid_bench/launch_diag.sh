#!/bin/bash
# Attenuation diagnosis: band1 reference on the GPU, subgrid runs on the CPU
# (one GPU run, at most two sims at once). Writes results/*_diag.npz
cd "$(dirname "$0")"
source ~/miniforge3/etc/profile.d/conda.sh
S=results/diag_status.log
echo "started $(date '+%F %T')" >> $S
( conda activate anuga_subgrid_gpu_env_3.14
  export PATH=/opt/nvidia/hpc_sdk/Linux_x86_64/26.3/compilers/bin:$PATH OMP_NUM_THREADS=8
  python run_phase0.py band1 --gpu --diag > results/band1_diag.log 2>&1
  echo "$(date '+%F %T') band1 (GPU) exit $?" | tee -a $S ) &
( conda activate anuga_subgrid_env_3.14; export OMP_NUM_THREADS=8
  for v in coarse500_sub8EKL coarse2000_sub8EKL; do
    python run_phase0.py $v --diag > results/${v}_diag.log 2>&1
    echo "$(date '+%F %T') $v exit $?" | tee -a $S
  done ) &
wait
echo "$(date '+%F %T') ALL DONE" | tee -a $S
