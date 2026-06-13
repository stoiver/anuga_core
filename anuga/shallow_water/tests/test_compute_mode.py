"""Tests for the compute-backend selector (set_compute_mode / get_compute_mode).

These exercise the mode-resolution logic that maps the user-facing
'legacy'/'cpu'/'gpu' backends onto the internal multiprocessor_mode (1/2)
dispatch, including the graceful fall-back from 'gpu' to 'cpu' on a build
without GPU offload. Robust to both CPU-only (gpu_offload=false) and GPU
(gpu_offload=true) builds.
"""

import unittest
import warnings

import anuga
from anuga import Domain, Reflective_boundary, rectangular_cross
from anuga.config import MULTIPROCESSOR_OPENMP, MULTIPROCESSOR_GPU
from anuga.shallow_water.sw_domain_gpu_ext import gpu_available


def _make_domain():
    points, vertices, boundary = rectangular_cross(6, 6, len1=6.0, len2=6.0)
    domain = Domain(points, vertices, boundary)
    domain.set_quantity('elevation', 0.0)
    domain.set_quantity('stage', 1.0)
    R = Reflective_boundary(domain)
    domain.set_boundary({'left': R, 'right': R, 'top': R, 'bottom': R})
    return domain


class Test_compute_mode(unittest.TestCase):

    def test_default_is_legacy(self):
        domain = _make_domain()
        self.assertEqual(domain.get_compute_mode(), 'legacy')
        self.assertEqual(domain.multiprocessor_mode, MULTIPROCESSOR_OPENMP)
        self.assertFalse(domain.use_c_rk_loop)

    def test_legacy_maps_to_mode1(self):
        domain = _make_domain()
        domain.set_compute_mode('legacy')
        self.assertEqual(domain.get_compute_mode(), 'legacy')
        self.assertEqual(domain.multiprocessor_mode, MULTIPROCESSOR_OPENMP)
        self.assertFalse(domain.use_c_rk_loop)

    def test_cpu_maps_to_mode2(self):
        domain = _make_domain()
        domain.set_compute_mode('cpu')
        self.assertEqual(domain.get_compute_mode(), 'cpu')
        self.assertEqual(domain.multiprocessor_mode, MULTIPROCESSOR_GPU)
        self.assertTrue(domain.use_c_rk_loop)
        self.assertIsNotNone(domain.gpu_interface)
        # CPU mode never reports offload as active.
        self.assertFalse(domain.gpu_offload_active)

    def test_gpu_request_resolves_against_build(self):
        domain = _make_domain()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            domain.set_compute_mode('gpu')
            fell_back = any('falling back' in str(w.message) for w in caught)

        self.assertEqual(domain.requested_compute_mode, 'gpu')
        if gpu_available():
            # GPU build with a device: request honoured.
            self.assertEqual(domain.get_compute_mode(), 'gpu')
            self.assertFalse(fell_back)
        else:
            # CPU-only build: warn + fall back to cpu, never hard-fail.
            self.assertEqual(domain.get_compute_mode(), 'cpu')
            self.assertTrue(fell_back)
        # Either way we end up in mode 2.
        self.assertEqual(domain.multiprocessor_mode, MULTIPROCESSOR_GPU)

    def test_invalid_mode_raises(self):
        domain = _make_domain()
        with self.assertRaises(ValueError):
            domain.set_compute_mode('bogus')

    def test_legacy_int_api_maps_to_compute_mode(self):
        domain = _make_domain()
        domain.set_multiprocessor_mode(1)
        self.assertEqual(domain.get_compute_mode(), 'legacy')
        self.assertEqual(domain.multiprocessor_mode, MULTIPROCESSOR_OPENMP)

        domain.set_multiprocessor_mode(2)
        # mode 2 auto-resolves to gpu when the build supports it, else cpu.
        expected = 'gpu' if gpu_available() else 'cpu'
        self.assertEqual(domain.get_compute_mode(), expected)
        self.assertEqual(domain.multiprocessor_mode, MULTIPROCESSOR_GPU)

    def test_invalid_int_mode_raises(self):
        domain = _make_domain()
        with self.assertRaises(ValueError):
            domain.set_multiprocessor_mode(3)

    def test_switch_back_to_legacy(self):
        domain = _make_domain()
        domain.set_compute_mode('cpu')
        domain.set_compute_mode('legacy')
        self.assertEqual(domain.get_compute_mode(), 'legacy')
        self.assertEqual(domain.multiprocessor_mode, MULTIPROCESSOR_OPENMP)
        self.assertFalse(domain.use_c_rk_loop)

    def test_compute_capabilities(self):
        domain = _make_domain()
        caps = domain.compute_capabilities()
        self.assertEqual(set(caps), {'gpu_offload', 'num_gpu_devices', 'mpi', 'modes'})
        self.assertIn('legacy', caps['modes'])
        self.assertIn('cpu', caps['modes'])
        # 'gpu' is selectable iff the build can offload.
        self.assertEqual('gpu' in caps['modes'], caps['gpu_offload'])
        self.assertEqual(caps['gpu_offload'], gpu_available())

    def test_parallel_without_mpi_build_falls_back_to_legacy(self):
        # Simulate a multi-rank run on a gpu_ext built WITHOUT MPI: mode 2's
        # C ghost exchange would be a silent no-op, so the selector must fall
        # back to 'legacy' rather than compute wrong parallel results.
        domain = _make_domain()
        domain._mode2_mpi_available = lambda: False
        saved = anuga.numprocs
        try:
            anuga.numprocs = 2
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter('always')
                domain.set_compute_mode('cpu')
                warned = any('Python MPI exchange' in str(w.message) for w in caught)
        finally:
            anuga.numprocs = saved

        self.assertEqual(domain.get_compute_mode(), 'legacy')
        self.assertEqual(domain.multiprocessor_mode, MULTIPROCESSOR_OPENMP)
        self.assertTrue(warned)

    def test_parallel_with_mpi_build_keeps_mode2(self):
        # With an MPI-enabled build, a multi-rank mode-2 request is honoured.
        domain = _make_domain()
        domain._mode2_mpi_available = lambda: True
        saved = anuga.numprocs
        try:
            anuga.numprocs = 2
            domain.set_compute_mode('cpu')
        finally:
            anuga.numprocs = saved
        self.assertEqual(domain.get_compute_mode(), 'cpu')
        self.assertEqual(domain.multiprocessor_mode, MULTIPROCESSOR_GPU)


if __name__ == '__main__':
    unittest.main()
