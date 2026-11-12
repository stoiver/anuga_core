#!/bin/bash
cd "$(dirname "$0")"; source ~/miniforge3/etc/profile.d/conda.sh; conda activate anuga_subgrid_env_3.14; export OMP_NUM_THREADS=8
python flat_plain.py --terrain flat --area 4 > results/flat_logs/p2_flat4.log 2>&1 &
python flat_plain.py --terrain ridged --area 4 > results/flat_logs/p2_ridged4.log 2>&1 &
wait
python flat_plain.py --terrain rough --area 4 > results/flat_logs/p2_rough4.log 2>&1 &
for t in flat ridged rough; do for a in 1000 6000; do python flat_plain.py --terrain $t --area $a --subgrid; python flat_plain.py --terrain $t --area $a; done; done > results/flat_logs/p2_coarse.log 2>&1
wait
echo "$(date '+%F %T') flat2 exit" >> results/flat_logs/status.log
