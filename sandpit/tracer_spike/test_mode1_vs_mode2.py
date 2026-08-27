"""Phase 2 oracle: tracers must give the same answer in mode 1 and mode 2.

This is the free oracle the plan calls for -- the CPU path is already verified
(test_recon / test_tracer_ns1 / test_ns2 / test_time_int), so mode 2 only has
to agree with it.

UNTIL PHASE 2 MAPPING LANDS THIS SCRIPT IS EXPECTED TO FAIL at check 0, with
"mode 2 did not engage". That is the point: the mode-2 tracer arrays are not
mapped to the device, so sw_domain_gpu_ext refuses (NotImplementedError) and
the GPU interface falls back to mode 1.

Check 0 exists because WITHOUT IT THIS TEST PASSES FALSELY. The fallback is
silent enough that a naive mode-1-vs-mode-2 diff would be comparing mode 1
against mode 1 and reporting a green agreement to 0.0.

Run in its own process (mode-2 domains must not accumulate):
    OMP_NUM_THREADS=1 python test_mode1_vs_mode2.py
"""
import numpy as np
import anuga
from anuga import Reflective_boundary, rectangular_cross_domain

LEN = 500.0
NXY = 20
FINALTIME = 6.0
_fail = [0]


def check(label, ok, detail=''):
    print('  [%s] %s' % ('PASS' if ok else 'FAIL', label))
    if detail:
        print('         ' + detail)
    if not ok:
        _fail[0] += 1
    return ok


def build(mode):
    d = rectangular_cross_domain(NXY, NXY, len1=LEN, len2=LEN)
    d.set_flow_algorithm('DE0')
    d.set_low_froude(0)
    d.store = False
    d.set_quantity('elevation', 0.0)
    d.set_quantity('stage', lambda x, y: np.where(x < LEN / 2, 2.0, 0.5))
    d.set_quantity('xmomentum', 0.0)
    d.set_quantity('ymomentum', 0.0)
    d.set_boundary({t: Reflective_boundary(d) for t in d.get_boundary_tags()})
    # Two tracers: one uniform (tests consistency), one structured (tests that
    # the device sees per-slot data rather than aliasing slot 0).
    d.add_tracer('uniform', beta=1.0)
    d.add_tracer('wedge', beta=1.0)
    x = d.centroid_coordinates[:, 0]
    d.set_tracer('uniform', 1.0)
    d.set_tracer('wedge', np.where(x < LEN / 2, 1.0, 0.0))
    d.set_multiprocessor_mode(mode)
    return d


print(__doc__)
print('  gpu_offload_enabled:', anuga.gpu_offload_enabled())

cpu = build(1)
for _ in cpu.evolve(yieldstep=2.0, finaltime=FINALTIME):
    pass

gpu = build(2)
engaged = check(
    '0. mode 2 engaged (guards against a false green via CPU fallback)',
    getattr(gpu, 'multiprocessor_mode', None) == 2,
    'multiprocessor_mode = %r, compute_mode = %r'
    % (getattr(gpu, 'multiprocessor_mode', None),
       getattr(gpu, 'compute_mode', None)))

if not engaged:
    print('\n  Stopping: comparing mode 1 against a mode-1 fallback would be')
    print('  meaningless. This is the expected state until the tracer arrays')
    print('  are mapped to the device.')
    raise SystemExit(1)

from anuga.shallow_water.sw_domain_gpu_ext import sync_to_device, sync_from_device
sync_to_device(gpu.gpu_interface.gpu_dom)
for _ in gpu.evolve(yieldstep=2.0, finaltime=FINALTIME):
    pass
sync_from_device(gpu.gpu_interface.gpu_dom)

# Device fp evaluation order differs (fma, reduction order), so agreement is
# ~1e-9 on a real GPU build rather than machine precision.
atol = 1e-8 if anuga.gpu_offload_enabled() else 1e-12

for q in ('stage', 'xmomentum', 'ymomentum'):
    a = gpu.quantities[q].centroid_values
    b = cpu.quantities[q].centroid_values
    check('1. hydrodynamics agree: %s' % q,
          np.allclose(a, b, rtol=0, atol=atol),
          'max|diff| = %.3e  (atol %.0e)' % (np.abs(a - b).max(), atol))

for name in ('uniform', 'wedge'):
    a = gpu.get_tracer(name)
    b = cpu.get_tracer(name)
    check('2. tracer concentration agrees: %r' % name,
          np.allclose(a, b, rtol=0, atol=atol),
          'max|diff| = %.3e' % np.abs(a - b).max())

for i, name in enumerate(('uniform', 'wedge')):
    a = gpu.tracer_conserved_values[i]
    b = cpu.tracer_conserved_values[i]
    check('3. conserved m = h*c agrees: %r' % name,
          np.allclose(a, b, rtol=0, atol=atol),
          'max|diff| = %.3e' % np.abs(a - b).max())

for i, name in enumerate(('uniform', 'wedge')):
    ma = float((gpu.tracer_conserved_values[i] * gpu.areas).sum())
    mb = float((cpu.tracer_conserved_values[i] * cpu.areas).sum())
    check('4. total tracer mass agrees: %r' % name,
          abs(ma - mb) <= 1e-9 * max(abs(mb), 1.0),
          'mode2 %.10e vs mode1 %.10e' % (ma, mb))

n = 1 + 3 + 2 + 2 + 2
print('\n  %d/%d passed' % (n - _fail[0], n))
raise SystemExit(1 if _fail[0] else 0)
