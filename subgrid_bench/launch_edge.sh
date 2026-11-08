#!/bin/bash
# Phase 2 edge-flux runs: two queues, at most two sims at once
cd "$(dirname "$0")"
source ~/miniforge3/etc/profile.d/conda.sh
conda activate anuga_subgrid_env_3.14; export OMP_NUM_THREADS=8
S=results/edge_status.log
echo "started $(date '+%F %T')" >> $S
run() { for v in "$@"; do python run_phase0.py $v > results/$v.log 2>&1; echo "$(date '+%F %T') $v exit $?" | tee -a $S; done; }
run coarse500_sub8EK coarse2000_sub8EK &
run coarse500_sub8E &
wait
echo "$(date '+%F %T') ALL DONE" | tee -a $S
