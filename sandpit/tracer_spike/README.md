# Tracer spike — test scripts

Verification for the generic-tracer work on `spike/tracer-flux-ns0-benchmark`.
Run from a directory OUTSIDE the anuga source tree (these import `anuga`).

    OMP_NUM_THREADS=1 python test_recon.py       # 6/6  edge reconstruction
    OMP_NUM_THREADS=1 python test_tracer_ns1.py  # 5/5  flux kernel, Ns=1
    OMP_NUM_THREADS=1 python test_ns2.py         # 3/3  multi-class striding
    OMP_NUM_THREADS=1 python test_time_int.py    # 6/6  time integration via evolve()
    OMP_NUM_THREADS=1 python test_add_tracer.py  # 20/20 add_tracer() registration API

Benchmark (Ns=0 regression gate):

    OMP_NUM_THREADS=1 python bench_tracer.py --size large --mode 2 --repeats 5 \
        --label baseline --out baseline.json
    python compare.py baseline.json prototype.json 1.0

`test_add_tracer.py` covers the registration API rather than the kernels: name
-> index mapping, array shapes/dtype/contiguity, growth from Ns=1 to Ns=2
without losing the existing tracer, `c`/`m = h*c` consistency, and the guards.
Its section B is the regression test for the cached-C-struct trap — a tracer
added after `evolve()` has built the struct must still move.

The four kernel scripts above still wire the six arrays by hand; that is
deliberate, since they are the verification record for the kernels themselves.
New work should use `domain.add_tracer()`.

`--size large` is 360k triangles. Mode 1 = legacy CPU, mode 2 = unified/GPU.
Run baseline and prototype on the SAME build configuration — see the rebuild
warning in ../../../Projects/Sediment_Transport/HANDOVER.md.

## Where the sediment tests went

The sediment suites that used to live here are now proper pytest modules in
`anuga/shallow_water/tests/test_sediment_*.py`, so CI runs them and pointing
pytest at this directory no longer matters:

    pytest anuga/shallow_water/tests/test_sediment_*.py
    pytest anuga/shallow_water/tests/test_sediment_*.py --run-fast   # skip the
                                                                    # convergence
                                                                    # studies

The mode 1 / mode 2 comparisons are collected in `test_sediment_gpu.py`, which
skips itself on a GPU-offload build unless `ANUGA_GPU_TESTS_ISOLATED=1` is set
-- the NVHPC runtime aborts a process that builds many mode-2 domains.

What remains here is the tracer work: the five kernel suites above, the
registration API test, and the Ns=0 benchmark harness (`bench_tracer.py`,
`compare.py`), which is a TIMING gate rather than a correctness test and so
does not belong in the pytest suite. Its equivalents are converted on the
`develop_tracers` branch.
