#!/bin/bash
# Phase 0 runs (1.5 h spin-up): GPU queue + one CPU queue. Logs in results/*.log
cd "$(dirname "$0")"
source ~/miniforge3/etc/profile.d/conda.sh
gpu() { ( conda activate anuga_subgrid_gpu_env_3.14
          export PATH=/opt/nvidia/hpc_sdk/Linux_x86_64/26.3/compilers/bin:$PATH OMP_NUM_THREADS=8
          for v in "$@"; do python run_phase0.py $v --gpu > results/$v.log 2>&1; echo "$v exit $?"; done ) }
cpu() { ( conda activate anuga_subgrid_env_3.14; export OMP_NUM_THREADS=8
          for v in "$@"; do python run_phase0.py $v > results/$v.log 2>&1; echo "$v exit $?"; done ) }
gpu band1 band025 &
cpu resolved coarse2000_centroid coarse2000_trench coarse500_centroid coarse500_trench &
wait
echo ALL DONE
