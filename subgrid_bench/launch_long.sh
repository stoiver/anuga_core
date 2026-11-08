#!/bin/bash
cd "$(dirname "$0")"
source ~/miniforge3/etc/profile.d/conda.sh
( conda activate anuga_subgrid_gpu_env_3.14
  export PATH=/opt/nvidia/hpc_sdk/Linux_x86_64/26.3/compilers/bin:$PATH OMP_NUM_THREADS=8
  python run_phase0.py fine --gpu > results/fine.log 2>&1 ) &
( conda activate anuga_subgrid_env_3.14; export OMP_NUM_THREADS=8
  python run_phase0.py resolved > results/resolved.log 2>&1 ) &
wait
echo ALL DONE
