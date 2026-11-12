#!/bin/bash
# flat-plain front-speed bench: two lanes, OMP 8 each
cd "$(dirname "$0")"; source ~/miniforge3/etc/profile.d/conda.sh; conda activate anuga_subgrid_env_3.14; export OMP_NUM_THREADS=8
mkdir -p results/flat_logs
lane() { for a in "$@"; do set -- $a; python flat_plain.py --terrain $1 --area $2 $3 > results/flat_logs/$1_$2_${3:-plain}.log 2>&1; done; }
lane "flat 4" "ridged 4" "rough 4" "ridged 1" "flat 1000" "ridged 1000" "rough 1000" &
lane "flat 1000 --subgrid" "ridged 1000 --subgrid" "rough 1000 --subgrid" "flat 6000 --subgrid" "ridged 6000 --subgrid" "rough 6000 --subgrid" "ridged 250 --subgrid" "rough 250 --subgrid" &
wait
echo "$(date '+%F %T') flat exit" >> results/flat_logs/status.log
