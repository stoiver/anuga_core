#!/bin/bash
# Rerun after the 2026-11-07 16:26 desktop crash: band025 (GPU) and the four
# Phase 1 subgrid runs (CPU). Two queues run at once, so at most two sims run
# at a time:
#   queue A: band025 (GPU), then coarse500_sub8, coarse500_sub16 (CPU)
#   queue B: coarse2000_sub8, coarse2000_sub16 (CPU)
# Logs in results/*.log; per-run exit codes in results/relaunch_status.log.
cd "$(dirname "$0")"
source ~/miniforge3/etc/profile.d/conda.sh
STATUS=results/relaunch_status.log
echo "started $(date '+%F %T')" >> $STATUS

gpu() { ( conda activate anuga_subgrid_gpu_env_3.14
          export PATH=/opt/nvidia/hpc_sdk/Linux_x86_64/26.3/compilers/bin:$PATH OMP_NUM_THREADS=8
          for v in "$@"; do python run_phase0.py $v --gpu > results/$v.log 2>&1
                            echo "$(date '+%F %T') $v exit $?" | tee -a $STATUS; done ) }
cpu() { ( conda activate anuga_subgrid_env_3.14; export OMP_NUM_THREADS=8
          for v in "$@"; do python run_phase0.py $v > results/$v.log 2>&1
                            echo "$(date '+%F %T') $v exit $?" | tee -a $STATUS; done ) }

( gpu band025; cpu coarse500_sub8 coarse500_sub16 ) &
cpu coarse2000_sub8 coarse2000_sub16 &
wait
echo "$(date '+%F %T') ALL DONE" | tee -a $STATUS
