#!/bin/bash
# Outlet-reach investigation: steady Q = 50 m3/s, 5 h.
#   straight channel (SUBGRID_AMPLITUDE=0): EKLC 500 / 2000, EKL 500, EKLC 500 with the
#   outlet reach (x > 1800) refined to 25 m2, and band1 (LF1 + tangential upwind) on the GPU;
#   meandering channel: EKLC 500, EKLC 500 refined, band1 on the GPU.
# Writes results/steady_*.npz; status in results/outlet_status.log
cd "$(dirname "$0")"
source ~/miniforge3/etc/profile.d/conda.sh
S=results/outlet_status.log
Q=50
echo "started $(date '+%F %T')" >> $S
run() {   # tag amplitude outlet_area variant [GPU]
  local tag=$1 amp=$2 area=$3 var=$4 gpu=$5
  SUBGRID_TAG=$tag SUBGRID_AMPLITUDE=$amp SUBGRID_OUTLET_AREA=$area \
    python steady_outlet.py $var $Q 18000 $gpu > results/steady_${var}_Q${Q}${tag}.log 2>&1
  echo "$(date '+%F %T') $var$tag exit $?" | tee -a $S
}
( conda activate anuga_subgrid_env_3.14; export OMP_NUM_THREADS=8
  run _straight 0 0 coarse500_sub8EKLC
  run _straight 0 0 coarse500_sub8EKL
  run _meander 40 0 coarse500_sub8EKLC ) &
( conda activate anuga_subgrid_env_3.14; export OMP_NUM_THREADS=8
  run _straight 0 0 coarse2000_sub8EKLC
  run _straight_out25 0 25 coarse500_sub8EKLC
  run _meander_out25 40 25 coarse500_sub8EKLC ) &
( conda activate anuga_subgrid_gpu_env_3.14
  export PATH=/opt/nvidia/hpc_sdk/Linux_x86_64/26.3/compilers/bin:$PATH OMP_NUM_THREADS=8
  export SUBGRID_LOW_FROUDE=5
  run _straight 0 0 band1 GPU
  run _meander 40 0 band1 GPU ) &
wait
echo "$(date '+%F %T') ALL DONE" | tee -a $S
