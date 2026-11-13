#!/bin/bash
# meandering-channel plain: fine references on the GPU (one at a time), then the coarse runs on the CPU
cd "$(dirname "$0")"; source ~/miniforge3/etc/profile.d/conda.sh
mkdir -p results/flat_logs
export OMP_NUM_THREADS=8 SUBGRID_LOW_FROUDE=5
conda activate anuga_subgrid_gpu_env_3.14
for r in "meander 0.5" "mrough 0.5" "meander 1"; do set -- $r
  python flat_plain.py --terrain $1 --area 4 --band $2 > results/flat_logs/$1_4_R$2.log 2>&1
done
echo "$(date '+%F %T') meander refs exit" >> results/flat_logs/status.log
conda activate anuga_subgrid_env_3.14
for t in meander mrough; do
  for a in 1000 6000; do for v in "" "--dir" "--tensor --tring 20"; do python flat_plain.py --terrain $t --area $a --subgrid $v; done; done
  python flat_plain.py --terrain $t --area 1000
done > results/flat_logs/meander_coarse.log 2>&1
python flat_channel.py > results/flat_logs/meander_summary.txt
