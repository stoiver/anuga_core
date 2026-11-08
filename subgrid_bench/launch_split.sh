#!/bin/bash
# Conveyance-weighted subface velocity (EKLC) runs with diagnostics, two at once
cd "$(dirname "$0")"
source ~/miniforge3/etc/profile.d/conda.sh
conda activate anuga_subgrid_env_3.14
export OMP_NUM_THREADS=8
S=results/split_status.log
echo "started $(date '+%F %T')" >> $S
for v in coarse500_sub8EKLC coarse2000_sub8EKLC; do
  ( python run_phase0.py $v --diag > results/${v}_diag.log 2>&1
    echo "$(date '+%F %T') $v exit $?" | tee -a $S ) &
done
wait
echo "$(date '+%F %T') ALL DONE" | tee -a $S
