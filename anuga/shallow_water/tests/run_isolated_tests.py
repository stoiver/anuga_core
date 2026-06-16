#!/usr/bin/env python
"""Run each collected test in its own fresh pytest process.

Why
---
On a GPU-offload build (nvc, ``-Dgpu_offload=true``) the NVHPC OpenMP-target
runtime aborts — and occasionally hangs — once many mode-2 GPU domains have been
created in a single process (see ``claude/KNOWN_ISSUES.md``). Per-test process
isolation sidesteps the accumulation entirely: every test runs in a fresh
interpreter with a clean CUDA context, so one test can never poison the next.
A per-test timeout also converts a genuine hang into a reported failure instead
of a stuck run.

``pytest --forked`` does **not** work here: CUDA contexts are fork-unsafe, so
forking from a GPU-initialised parent poisons every child. This harness spawns
fresh ``python -m pytest <nodeid>`` subprocesses instead.

On a CPU build the harness still works (just slower than a single in-process
run); it is only *required* on a GPU build.

Usage
-----
    python run_isolated_tests.py [TARGET ...] [options]

TARGET defaults to this directory's ``test_DE_gpu_omp.py``. It may be any pytest
target: a file, a directory, a ``path::Class::test`` node id, or (with
``--pyargs``) a dotted module like ``anuga.shallow_water``.

Options
    --timeout SECONDS   per-test wall-clock limit (default 180; a test exceeding
                        it is recorded as TIMEOUT and its process is killed)
    -k EXPR             pytest -k expression to select tests
    --pyargs            interpret TARGETs as importable modules
    -x / --exitfirst    stop at the first non-pass
    -v / --verbose      echo each failing test's captured output
    -j N / --jobs N     run N tests concurrently (default 1; keep 1 on a GPU —
                        concurrent tests contend for the same device)

Exit status is 0 only if every test passed or was skipped.

Examples
    python run_isolated_tests.py
    python run_isolated_tests.py test_DE_gpu_omp.py --timeout 120
    python run_isolated_tests.py --pyargs anuga.shallow_water -k riverwall
"""

import argparse
import concurrent.futures
import os
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
DEFAULT_TARGET = str(HERE / "test_DE_gpu_omp.py")

# Status -> short label used in the live log and summary
PASS, FAIL, SKIP, ERROR, CRASH, TIMEOUT, NOTESTS = (
    "PASS", "FAIL", "SKIP", "ERROR", "CRASH", "TIMEOUT", "NOTESTS")
BAD = {FAIL, ERROR, CRASH, TIMEOUT}


def _base_env():
    env = dict(os.environ)
    # Opt back in to the GPU tests, which skip themselves in a normal in-process
    # run on a GPU build (see the module-level skip in test_DE_gpu_omp.py).
    env["ANUGA_GPU_TESTS_ISOLATED"] = "1"
    env.setdefault("OMP_NUM_THREADS", "1")
    return env


def collect(targets, pyargs, k_expr):
    """Return the list of test node ids pytest would run for the targets."""
    cmd = [sys.executable, "-m", "pytest", "--collect-only", "-q",
           "-p", "no:cacheprovider"]
    if pyargs:
        cmd.append("--pyargs")
    if k_expr:
        cmd += ["-k", k_expr]
    cmd += targets
    proc = subprocess.run(cmd, capture_output=True, text=True, env=_base_env())
    ids = []
    for line in proc.stdout.splitlines():
        line = line.strip()
        # node-id lines contain '::'; the trailing summary lines ("123 tests
        # collected", "no tests ran", warnings, etc.) do not.
        if "::" in line and not line.startswith(("<", "=", "-")):
            ids.append(line.split(" ")[0])
    if not ids and proc.returncode not in (0, 5):
        sys.stderr.write(proc.stdout + proc.stderr)
    return ids


def classify(returncode, stdout):
    """Map a single-test pytest run to one of the status labels."""
    if returncode is None:
        return TIMEOUT
    tail = "\n".join(stdout.strip().splitlines()[-4:]).lower()
    if returncode == 5:
        return NOTESTS
    if returncode < 0:                       # killed by a signal (segfault/abort)
        return CRASH
    if "failed" in tail:
        return FAIL
    if "error" in tail and "passed" not in tail:
        return ERROR
    if returncode == 0:
        if "passed" in tail:
            return PASS
        if "skipped" in tail and "passed" not in tail:
            return SKIP
        return PASS                          # collected & green, no count word
    # Non-zero exit with no recognisable failure summary == hard crash
    # (this is the NVHPC-runtime abort signature: exit 1, no pytest summary).
    return CRASH


def run_one(nodeid, timeout):
    cmd = [sys.executable, "-m", "pytest", nodeid, "-p", "no:cacheprovider",
           "-q", "--no-header", "-o", "addopts="]
    start = time.time()
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True,
                              env=_base_env(), timeout=timeout)
        rc, out = proc.returncode, proc.stdout + proc.stderr
    except subprocess.TimeoutExpired as e:
        rc, out = None, (e.stdout or "") + (e.stderr or "")
        if isinstance(out, bytes):
            out = out.decode("utf-8", "replace")
    return classify(rc, out), out, time.time() - start


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("targets", nargs="*", default=[DEFAULT_TARGET])
    ap.add_argument("--timeout", type=float, default=180.0)
    ap.add_argument("-k", dest="k_expr", default=None)
    ap.add_argument("--pyargs", action="store_true")
    ap.add_argument("-x", "--exitfirst", action="store_true")
    ap.add_argument("-v", "--verbose", action="store_true")
    ap.add_argument("-j", "--jobs", type=int, default=1)
    args = ap.parse_args(argv)
    targets = args.targets or [DEFAULT_TARGET]

    print(f"Collecting tests from: {' '.join(targets)}", flush=True)
    ids = collect(targets, args.pyargs, args.k_expr)
    if not ids:
        print("No tests collected.", file=sys.stderr)
        return 2
    print(f"Running {len(ids)} tests, one per process "
          f"(timeout {args.timeout:g}s, jobs {args.jobs}).\n", flush=True)

    counts = {s: 0 for s in (PASS, FAIL, SKIP, ERROR, CRASH, TIMEOUT, NOTESTS)}
    bad = []
    width = len(str(len(ids)))

    def record(i, nodeid, status, out, dt):
        counts[status] += 1
        mark = status if status in BAD else status.lower()
        print(f"[{i:>{width}}/{len(ids)}] {mark:<7} {dt:5.1f}s  {nodeid}",
              flush=True)
        if status in BAD:
            bad.append((nodeid, status))
            if args.verbose and out.strip():
                print("    " + out.strip().replace("\n", "\n    "), flush=True)

    if args.jobs <= 1:
        for i, nodeid in enumerate(ids, 1):
            status, out, dt = run_one(nodeid, args.timeout)
            record(i, nodeid, status, out, dt)
            if args.exitfirst and status in BAD:
                print("\nStopping at first failure (-x).", flush=True)
                break
    else:
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.jobs) as ex:
            futs = {ex.submit(run_one, nid, args.timeout): (i, nid)
                    for i, nid in enumerate(ids, 1)}
            for fut in concurrent.futures.as_completed(futs):
                i, nodeid = futs[fut]
                status, out, dt = fut.result()
                record(i, nodeid, status, out, dt)

    print("\n" + "=" * 60)
    print("  ".join(f"{s.lower()}={counts[s]}" for s in
                    (PASS, SKIP, FAIL, ERROR, CRASH, TIMEOUT, NOTESTS)
                    if counts[s]))
    if bad:
        print(f"\n{len(bad)} test(s) did not pass:")
        for nodeid, status in bad:
            print(f"  {status:<7} {nodeid}")
        return 1
    print("All tests passed (or were skipped).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
