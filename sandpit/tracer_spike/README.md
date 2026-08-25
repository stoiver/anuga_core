# Tracer spike — test scripts

Verification for the generic-tracer work on `spike/tracer-flux-ns0-benchmark`.
Run from a directory OUTSIDE the anuga source tree (these import `anuga`).

    OMP_NUM_THREADS=1 python test_recon.py       # 6/6  edge reconstruction
    OMP_NUM_THREADS=1 python test_tracer_ns1.py  # 5/5  flux kernel, Ns=1
    OMP_NUM_THREADS=1 python test_ns2.py         # 3/3  multi-class striding
    OMP_NUM_THREADS=1 python test_time_int.py    # 6/6  time integration via evolve()

Benchmark (Ns=0 regression gate):

    OMP_NUM_THREADS=1 python bench_tracer.py --size large --mode 2 --repeats 5 \
        --label baseline --out baseline.json
    python compare.py baseline.json prototype.json 1.0

`--size large` is 360k triangles. Mode 1 = legacy CPU, mode 2 = unified/GPU.
Run baseline and prototype on the SAME build configuration — see the rebuild
warning in ../../../Projects/Sediment_Transport/HANDOVER.md.
