"""Generic passive tracers: the legacy (mode 1) and unified (mode 2) paths agree.

Both compute modes call the same tracer code in `gpu/core_kernels.c`, so a
disagreement between them is a mapping or binding fault rather than a physics
one -- most often a `D->member` load inside an `omp target` region, which reads
a host address on the device and makes the work silently not happen.

The properties tested here are exactly the ones such a fault breaks: the two
modes must agree on the hydrodynamics, on each tracer's concentration and
conserved mass, and on the total; and two tracers must stay distinct on the
device rather than one aliasing the other.

Deliberately NO `sync_from_device()` after the mode-2 evolve. The comparison
reads what an ordinary user's script would see when it inspects centroid values
after `evolve`, which is the thing that has to be right.
"""
import os
import warnings

import numpy as np
import pytest

import anuga
from anuga import Reflective_boundary, rectangular_cross_domain

LEN = 500.0
NXY = 20
FINALTIME = 20.0

_gpu_error = None
_gpu_avail = None


def gpu_available():
    """Check if the GPU OpenMP interface is importable."""
    global _gpu_error, _gpu_avail
    if _gpu_avail is not None:
        return _gpu_avail
    try:
        from anuga.shallow_water.sw_domain_gpu_ext import init_gpu_domain  # noqa: F401
        _gpu_avail = True
    except Exception as e:
        _gpu_avail = False
        _gpu_error = '%s: %s' % (type(e).__name__, e)
    return _gpu_avail


# On a GPU-offload build (nvc, -Dgpu_offload=true) the NVHPC OpenMP-target
# runtime aborts the process once many mode-2 domains have been created in it,
# so files that build them are skipped in a normal in-process run and are
# executed one class per fresh process by run_gpu_tests_isolated.sh, which sets
# ANUGA_GPU_TESTS_ISOLATED=1. On a CPU build the omp-target regions run on the
# host and this file is fine in one process. Mirrors test_DE_gpu_omp.py.
if (gpu_available() and anuga.gpu_offload_supported()
        and not os.environ.get('ANUGA_GPU_TESTS_ISOLATED')):
    _skip_reason = (
        "GPU-offload build: run this file via "
        "anuga/shallow_water/tests/run_gpu_tests_isolated.sh (one fresh process "
        "per class) - running it in one process aborts the NVHPC OpenMP-target "
        "runtime. Set ANUGA_GPU_TESTS_ISOLATED=1 to force in-process collection.")
    warnings.warn(_skip_reason, stacklevel=1)
    pytest.skip(_skip_reason, allow_module_level=True)


def _build(mode):
    d = rectangular_cross_domain(NXY, NXY, len1=LEN, len2=LEN)
    d.set_flow_algorithm('DE0')
    d.set_low_froude(0)
    d.store = False
    d.set_quantity('elevation', 0.0)
    d.set_quantity('stage', lambda x, y: np.where(x < LEN / 2, 2.0, 0.5))
    d.set_quantity('xmomentum', 0.0)
    d.set_quantity('ymomentum', 0.0)
    d.set_boundary({t: Reflective_boundary(d) for t in d.get_boundary_tags()})
    # One uniform tracer (tests consistency) and one structured (tests that the
    # device sees per-slot data rather than aliasing slot 0).
    d.add_tracer('uniform', beta=1.0)
    d.add_tracer('wedge', beta=1.0)
    x = d.centroid_coordinates[:, 0]
    d.set_tracer('uniform', 1.0)
    d.set_tracer('wedge', np.where(x < LEN / 2, 1.0, 0.0))
    d.set_multiprocessor_mode(mode)
    return d


@pytest.fixture(scope='module')
def evolved():
    """One mode-1 and one mode-2 run of the same problem."""
    if not gpu_available():
        pytest.skip('GPU OpenMP interface not available: %s' % _gpu_error)

    cpu = _build(1)
    cpu.evolve_to_end(finaltime=FINALTIME)

    gpu = _build(2)
    if getattr(gpu, 'multiprocessor_mode', None) != 2:
        # Comparing mode 1 against a silent mode-1 fallback would be a false
        # green, so refuse rather than pass.
        pytest.fail('mode 2 did not engage: multiprocessor_mode=%r compute_mode=%r'
                    % (getattr(gpu, 'multiprocessor_mode', None),
                       getattr(gpu, 'compute_mode', None)))

    from anuga.shallow_water.sw_domain_gpu_ext import sync_to_device
    sync_to_device(gpu.gpu_interface.gpu_dom)
    gpu.evolve_to_end(finaltime=FINALTIME)
    return cpu, gpu


@pytest.mark.parametrize('name', ['stage', 'xmomentum', 'ymomentum'])
def test_hydrodynamics_agree(evolved, name):
    cpu, gpu = evolved
    a = cpu.quantities[name].centroid_values
    b = gpu.quantities[name].centroid_values
    assert np.abs(a - b).max() < 1e-10


@pytest.mark.parametrize('slot,name', [(0, 'uniform'), (1, 'wedge')])
def test_tracer_fields_agree(evolved, slot, name):
    cpu, gpu = evolved
    ma = cpu.tracer_conserved_values[slot]
    mb = gpu.tracer_conserved_values[slot]
    assert np.abs(ma - mb).max() < 1e-10, 'conserved m disagrees for %r' % name

    ha = (cpu.quantities['stage'].centroid_values
          - cpu.quantities['elevation'].centroid_values)
    hb = (gpu.quantities['stage'].centroid_values
          - gpu.quantities['elevation'].centroid_values)
    wet = (ha > 1e-3) & (hb > 1e-3)
    assert np.abs(ma[wet] / ha[wet] - mb[wet] / hb[wet]).max() < 1e-10

    tot_a = float((ma * cpu.areas).sum())
    tot_b = float((mb * gpu.areas).sum())
    assert abs(tot_a - tot_b) < 1e-10 * max(abs(tot_a), 1.0)


def test_the_two_tracers_stay_distinct_on_the_device(evolved):
    """Guards against slot 1 aliasing slot 0 in the device mapping."""
    _, gpu = evolved
    assert not np.allclose(gpu.tracer_conserved_values[0],
                           gpu.tracer_conserved_values[1])


def test_depth_is_consistent_with_stage_in_both_modes(evolved):
    """A tracer must not corrupt the hydrodynamic state it rides on."""
    for nm, d in zip(('mode 1', 'mode 2'), evolved):
        h = (d.quantities['stage'].centroid_values
             - d.quantities['elevation'].centroid_values)
        assert h.min() > -1e-10, '%s: negative depth %.3e' % (nm, h.min())
