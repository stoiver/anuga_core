#!/bin/bash
# Straight-channel test (meander amplitude 0): band1 on the GPU, subgrid on the CPU
cd "$(dirname "$0")"
source ~/miniforge3/etc/profile.d/conda.sh
export SUBGRID_AMPLITUDE=0 SUBGRID_TAG=_straight OMP_NUM_THREADS=8
S=results/straight_status.log
echo "started $(date '+%F %T')" >> $S
( conda activate anuga_subgrid_gpu_env_3.14
  export PATH=/opt/nvidia/hpc_sdk/Linux_x86_64/26.3/compilers/bin:$PATH
  python run_phase0.py band1 --gpu --diag > results/band1_diag_straight.log 2>&1
  echo "$(date '+%F %T') band1 (GPU) exit $?" | tee -a $S ) &
( conda activate anuga_subgrid_env_3.14
  for v in coarse2000_sub8EKLC coarse500_sub8EKLC coarse500_sub8EKL; do
    python run_phase0.py $v --diag > results/${v}_diag_straight.log 2>&1
    echo "$(date '+%F %T') $v exit $?" | tee -a $S
  done ) &
wait
echo "$(date '+%F %T') ALL DONE" | tee -a $S
