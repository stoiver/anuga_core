"""An operator may invalidate the GPU interface mid-timestep.

apply_fractional_steps decides `gpu_mode` before running the operators, then
syncs the host-side work of any CPU-only operator back to the device
afterwards. An operator that calls set_tracer_source in between sets
domain.gpu_interface = None -- deliberately, since rebinding the source array
means the C struct and the device mapping must be rebuilt -- so that trailing
sync used to dereference None.

A manufactured-solution source does this on every step, which is how it was
found: the sediment MMS tests failed under ANUGA_DEFAULT_COMPUTE_MODE=unified
with 'NoneType' object has no attribute 'sync_to_device'.
"""

import numpy as num
import pytest

import anuga
from anuga.operators.base_operator import Operator


class _ResetsTheInterface(Operator):
    """A CPU-only fractional-step operator that rebinds the tracer source."""

    def __init__(self, domain):
        Operator.__init__(self, domain)
        self.calls = 0

    def __call__(self):
        self.calls += 1
        n = len(self.domain)
        # Only the FIRST call invalidates the interface (that is when the
        # source array is allocated); calling every step is what a
        # manufactured source does and costs nothing after that.
        self.domain.set_tracer_source('mms', num.full(n, 1.0e-4))

    def parallel_safe(self):
        return True

    def statistics(self):
        return 'resets the interface'

    def timestepping_statistics(self):
        return ''


def _domain(mode):
    d = anuga.rectangular_cross_domain(6, 6, len1=60.0, len2=60.0)
    d.set_quantity('elevation', 0.0)
    d.set_quantity('stage', 1.0)
    b = anuga.Reflective_boundary(d)
    d.set_boundary({'left': b, 'right': b, 'top': b, 'bottom': b})
    # A sediment CLASS, not a bare tracer: tracer_external_source is applied
    # by the sediment source kernel, which does nothing with no classes
    # registered -- so a plain tracer would never show the source at all.
    d.add_sediment_class('mms', diameter=1.0e-4, initial_concentration=0.0)
    d.store = False
    op = _ResetsTheInterface(d)
    d.set_multiprocessor_mode(mode)
    return d, op


@pytest.mark.parametrize('mode', [1, 2])
def test_it_does_not_crash_when_the_interface_is_invalidated(mode):
    """The regression this file is for: it used to raise AttributeError."""
    d, op = _domain(mode)
    for _ in d.evolve(yieldstep=0.5, finaltime=2.0):
        pass
    assert op.calls > 0, 'the operator never ran, so nothing was exercised'


def test_the_source_reaches_the_solver_in_legacy_mode():
    """Mode 1 only, deliberately.

    In mode 2 the source currently does NOT take effect, and this test does
    not paper over it: Sediment_operator picks its path with

        on_gpu = (multiprocessor_mode == 2 and gpu_interface is not None)

    so an operator that nulls the interface earlier in the same
    fractional-step pass -- which set_tracer_source does on its first call
    -- pushes the sediment source onto the CPU path while the conserved
    state is on the device. On a CPU build both paths share host memory and
    it is harmless; on a GPU build the update is simply lost. Tracked
    separately; the fix belongs with set_tracer_source, which should not
    tear the interface down mid-run.
    """
    d, _ = _domain(1)
    for _ in d.evolve(yieldstep=0.5, finaltime=2.0):
        pass
    assert d.get_tracer('mms').max() > 0.0, \
        'the tracer source never took effect'
