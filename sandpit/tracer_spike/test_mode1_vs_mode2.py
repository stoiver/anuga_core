"""Phase 2 oracle: tracers must give the same answer in mode 1 and mode 2.

This is the free oracle the plan calls for -- the CPU path is already verified
(test_recon / test_tracer_ns1 / test_ns2 / test_time_int), so mode 2 only has
to agree with it.

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

from anuga.shallow_water.sw_domain_gpu_ext import sync_to_device
sync_to_device(gpu.gpu_interface.gpu_dom)
for _ in gpu.evolve(yieldstep=2.0, finaltime=FINALTIME):
    pass

# DELIBERATELY no sync_from_device() here.
#
# In mode 2 distribute_to_vertices_and_edges() is the OUTPUT path: it syncs the
# device centroids to the host and then runs the host protect + extrapolate,
# which re-derives every DERIVED quantity -- height, and c from m. evolve()
# already calls it before yielding, so the domain is correct and self-consistent
# on return.
#
# Calling sync_from_device() afterwards UNDOES that: it copies raw device
# centroids back over the freshly derived host values, leaving derived
# quantities one substep behind the conserved ones. Conserved quantities
# (stage, xmomentum, ymomentum, m) are identical either way, which is why
# test_rk3_mode1_vs_mode2_dam_break can sync after evolve safely -- it compares
# only conserved quantities. Derived quantities are not safe that way.

# Device fp evaluation order differs (fma, reduction order), so agreement is
# ~1e-9 on a real GPU build rather than machine precision.
atol = 1e-8 if anuga.gpu_offload_enabled() else 1e-12

for q in ('stage', 'xmomentum', 'ymomentum'):
    a = gpu.quantities[q].centroid_values
    b = cpu.quantities[q].centroid_values
    check('1. hydrodynamics agree: %s' % q,
          np.allclose(a, b, rtol=0, atol=atol),
          'max|diff| = %.3e  (atol %.0e)' % (np.abs(a - b).max(), atol))

# ---------------------------------------------------------------------------
# 2. Derived concentration c.
#
# c is DERIVED (c = m/h), so unlike m it is only correct where the derived
# quantities have been refreshed. In mode 2 that refresh is
# distribute_to_vertices_and_edges(), which evolve() calls before yielding --
# see the note above about not calling sync_from_device() afterwards.
# ---------------------------------------------------------------------------
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

# ---------------------------------------------------------------------------
# 5. Self-consistency of the DERIVED quantities in each mode.
#
# Guards the trap that produced a spurious 7e-2 disagreement in check 2 while m
# agreed to 6.7e-16: an explicit sync_from_device() after evolve copies raw
# device centroids over the host values that mode 2's output path had already
# derived, leaving height and c one substep behind stage and m. Conserved
# quantities survive that; derived ones do not. If this check fails, suspect a
# stray sync before suspecting the physics.
# ---------------------------------------------------------------------------
for nm, d in (('mode 1', cpu), ('mode 2', gpu)):
    hcv = d.quantities['height'].centroid_values
    st = np.maximum(d.quantities['stage'].centroid_values
                    - d.quantities['elevation'].centroid_values, 0.0)
    check('5. %s: height is consistent with stage' % nm,
          np.allclose(hcv, st, rtol=0, atol=atol),
          'max|height_cv - (stage-bed)| = %.3e' % np.abs(hcv - st).max())

n = 1 + 3 + 2 + 2 + 2 + 2
print('\n  %d/%d passed' % (n - _fail[0], n))
raise SystemExit(1 if _fail[0] else 0)
