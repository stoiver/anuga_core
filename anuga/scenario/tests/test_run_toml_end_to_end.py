"""
End-to-end smoke test for the ``anuga_run_toml`` runner script.

Unlike the other scenario tests (which exercise the ``setup_*`` modules in
isolation against synthetic domains), this drives the *whole* runner the way a
user does: it copies the shipped ``examples/run_toml/simple`` dam-break
scenario to a temporary directory and invokes ``scripts/anuga_run_toml.py`` as a
subprocess, then validates the SWW it produces.

The simple scenario is self-contained (two tiny CSV polygons + the TOML) and
short (30 s of model time on ~500 triangles), so the whole run takes a few
seconds.  Marked slow because it builds a mesh, evolves, and writes GeoTIFFs.

It is skipped automatically when run against an installed-only tree (e.g.
``pytest --pyargs anuga`` in site-packages) where the ``scripts/`` and
``examples/`` directories are not present.
"""

import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from glob import glob
from pathlib import Path

import pytest

try:
    import numpy as np
    from anuga.file.netcdf import NetCDFFile
    HAS_MODULE = True
    SKIP_REASON = ''
except ImportError as _e:  # pragma: no cover - import guard
    HAS_MODULE = False
    SKIP_REASON = str(_e)


# ---------------------------------------------------------------------------
# Locate the source tree (runner script + example data)
# ---------------------------------------------------------------------------

def _find_source_tree():
    """Walk up from this file looking for a checkout containing both the
    runner script and the simple example.  Returns (runner, example_dir) or
    (None, None) when not found (installed-only environment)."""
    runner_rel = os.path.join('scripts', 'anuga_run_toml.py')
    example_rel = os.path.join('examples', 'run_toml', 'simple', 'dam_break.toml')
    for parent in Path(__file__).resolve().parents:
        runner = parent / runner_rel
        example = parent / example_rel
        if runner.is_file() and example.is_file():
            return str(runner), str(example.parent)
    return None, None


RUNNER, EXAMPLE_DIR = _find_source_tree()
HAS_SOURCE = RUNNER is not None


@unittest.skipUnless(HAS_MODULE, SKIP_REASON)
@unittest.skipUnless(HAS_SOURCE, 'source tree (scripts/ + examples/) not found')
class TestRunTomlEndToEnd(unittest.TestCase):

    @pytest.mark.slow
    def test_simple_dam_break_produces_valid_sww(self):
        with tempfile.TemporaryDirectory() as work:
            # Copy the self-contained simple example into the work dir so the
            # runner writes its OUTPUT/ there rather than into the repo.
            for name in os.listdir(EXAMPLE_DIR):
                src = os.path.join(EXAMPLE_DIR, name)
                if os.path.isfile(src):
                    shutil.copy(src, work)

            # Run the script the way a user would, serially.
            env = dict(os.environ, OMP_NUM_THREADS='1')
            proc = subprocess.run(
                [sys.executable, RUNNER, 'dam_break.toml'],
                cwd=work, env=env,
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, timeout=600)

            tail = '\n'.join(proc.stdout.splitlines()[-25:])
            self.assertEqual(
                proc.returncode, 0,
                'anuga_run_toml exited %d. Output tail:\n%s'
                % (proc.returncode, tail))

            # The runner creates OUTPUT/RUN_<timestamp>_dam_break/dam_break.sww
            matches = glob(os.path.join(
                work, 'OUTPUT', 'RUN_*_dam_break', 'dam_break.sww'))
            self.assertEqual(
                len(matches), 1,
                'expected exactly one SWW, found %r. Output tail:\n%s'
                % (matches, tail))
            sww = matches[0]
            self.assertGreater(os.path.getsize(sww), 0, 'SWW is empty')

            self._check_sww(sww)

    def _check_sww(self, sww):
        fid = NetCDFFile(sww, 'r')
        try:
            # --- time vector: finaltime 30 s, yieldstep 2 s -> 16 frames ---
            time = fid.variables['time'][:]
            self.assertEqual(len(time), 16)
            self.assertAlmostEqual(float(time[0]), 0.0, places=6)
            self.assertAlmostEqual(float(time[-1]), 30.0, places=6)

            # --- stage: finite, and the 4 m reservoir column is present ---
            stage = fid.variables['stage'][:]
            self.assertTrue(np.all(np.isfinite(stage)), 'non-finite stage')
            self.assertGreater(float(np.max(stage)), 3.5)
            self.assertLess(float(np.max(stage)), 4.5)

            # --- CRS metadata: projection_information = -56 must propagate ---
            # to the SWW as UTM zone 56 (southern) / EPSG:32756. This guards
            # the runner's set_epsg() step (commit c4ef89f5).
            self.assertEqual(int(fid.zone), 56)
            self.assertEqual(int(fid.epsg), 32756)
        finally:
            fid.close()


if __name__ == '__main__':
    unittest.main()
