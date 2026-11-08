#!/bin/bash
cd "$(dirname "$0")"
source ~/miniforge3/etc/profile.d/conda.sh
conda activate anuga_subgrid_env_3.14; export OMP_NUM_THREADS=8
( for v in coarse2000_sub8 coarse2000_sub16; do python run_phase0.py $v > results/$v.log 2>&1; echo "$v exit $?"; done ) &
( for v in coarse500_sub8 coarse500_sub16; do python run_phase0.py $v > results/$v.log 2>&1; echo "$v exit $?"; done ) &
wait; echo ALL DONE
