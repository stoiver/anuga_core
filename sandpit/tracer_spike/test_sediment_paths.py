"""Guard: the sediment kernels are reached on every path, under every algorithm.

The sediment physics runs as a fractional step, dispatched by
Sediment_operator to three kernels -- the suspended exchange, bedload, and
angle-of-repose relaxation. Each is reached through a chain of conditions: the
operator has to be registered, the right branch of the mode 1 / mode 2 dispatch
has to be taken, and each kernel's own enable flag has to be read correctly.

Every link in that chain fails SILENTLY. A sediment class that is registered
but never exchanged simply behaves as an inert tracer: no error, no warning,
the bed just does not move. The individual physics is covered by the other
suites in this directory; what is covered HERE is that the physics is reached
at all.

  A  the source, bedload and repose kernels each change the answer when
     enabled -- so a dispatch that silently skipped one would fail here
  B  sediment works under every flow algorithm, not just the DE0 all the other
     suites use
  C  the operator is registered automatically, and stays on the GPU-safe list
  D  a tripwire for transport paths that do not exist yet

D mirrors test_tracers.py in anuga/shallow_water/tests on the develop_tracers
branch. anuga-community#241 proposes alternative flux and update kernels that
carry no tracers; the same question will apply to the sediment fractional step.
"""
import numpy as np
import anuga
from anuga import Reflective_boundary, rectangular_cross_domain

_fail = [0]

# Domain attributes, not yet present, that select an alternative flux or update
# kernel. Named so their arrival is noticed. See section D.
FUTURE_PATH_SELECTORS = ('reconstruct_edge_bed',)

FLOW_ALGORITHMS = ('DE0', 'DE1', 'DE2')


def check(label, ok, detail=''):
    print('  [%s] %s' % ('PASS' if ok else 'FAIL', label))
    if detail:
        print('         ' + detail)
    if not ok:
        _fail[0] += 1
    return ok


def build(algorithm='DE0', bedload=False, repose=None, mode=1,
          sediment=True):
    d = rectangular_cross_domain(30, 8, len1=60.0, len2=16.0)
    d.set_flow_algorithm(algorithm)
    d.set_low_froude(0)
    d.store = False
    x = d.centroid_coordinates[:, 0]
    d.set_quantity('elevation', -0.01 * x, location='centroids')
    d.set_quantity('stage', -0.01 * x + 0.6, location='centroids')
    d.set_quantity('xmomentum', 1.2)
    d.set_quantity('friction', 0.03)
    d.set_boundary({t: Reflective_boundary(d) for t in d.get_boundary_tags()})
    if sediment:
        d.set_sediment_parameters(porosity=0.3)
        if bedload:
            d.set_bedload('wong_parker_eq24')
        d.add_sediment_class('sand', diameter=3.0e-4)
        if repose is not None:
            d.set_angle_of_repose(repose, max_sweeps=50)
    if mode != 1:
        d.set_multiprocessor_mode(mode)
    return d


def evolve(d, t=30.0):
    z0 = d.quantities['elevation'].centroid_values.copy()
    d.evolve_to_end(finaltime=t)
    z1 = d.quantities['elevation'].centroid_values
    ns = getattr(d, 'n_sediment_classes', 0)
    m = sum(float((d.tracer_conserved_values[s] * d.areas).sum())
            for s in range(ns))
    bed = float(((1.0 - 0.3) * (z1 - z0) * d.areas).sum())
    return dict(z0=z0, z1=z1, scour=float((z0 - z1).max()),
                suspended=m, budget=m + bed)


print(__doc__)

# ---------------------------------------------------------------- A
print('A. each kernel changes the answer when enabled')

plain = evolve(build())
check('A1. the suspended exchange runs: the bed moves and sediment appears',
      plain['scour'] > 1e-4 and plain['suspended'] > 0.0,
      'scour %.4f m, suspended %.4e m3' % (plain['scour'], plain['suspended']))
check('A2. and it closes its budget, so it is the exchange doing the work '
      'rather than something else moving the bed',
      abs(plain['budget']) < 1e-10 * max(plain['suspended'], 1.0),
      'budget %+.3e m3' % plain['budget'])

with_bedload = evolve(build(bedload=True))
check('A3. the bedload kernel is reached: enabling it changes the bed',
      not np.allclose(plain['z1'], with_bedload['z1'], rtol=0, atol=1e-14),
      'max |dz| between bedload off and on = %.3e m'
      % np.abs(plain['z1'] - with_bedload['z1']).max())

# Repose needs a bed steep enough to relax, so give it a cone rather than the
# gently sloping channel above; on the channel it would correctly do nothing
# and the comparison would prove only that.
def cone_domain(repose=None, algorithm='DE0', mode=1):
    d = rectangular_cross_domain(40, 12, len1=60.0, len2=16.0)
    d.set_flow_algorithm(algorithm)
    d.set_low_froude(0)
    d.store = False
    x, y = d.centroid_coordinates[:, 0], d.centroid_coordinates[:, 1]
    r = np.sqrt((x - 30.0) ** 2 + (y - 8.0) ** 2)
    d.set_quantity('elevation', np.where(r < 8.0, 6.0 * (1.0 - r / 8.0), 0.0),
                   location='centroids')
    d.set_quantity('stage', -1.0)
    d.set_quantity('friction', 0.03)
    d.set_boundary({t: Reflective_boundary(d) for t in d.get_boundary_tags()})
    d.set_sediment_parameters(porosity=0.3)
    d.add_sediment_class('sand', diameter=3.0e-4)
    if repose is not None:
        d.set_angle_of_repose(repose, max_sweeps=400)
    if mode != 1:
        d.set_multiprocessor_mode(mode)
    return d


no_rep = cone_domain()
no_rep.evolve_to_end(finaltime=1.0)
rep = cone_domain(repose=30.0)
rep.evolve_to_end(finaltime=1.0)
zn = no_rep.quantities['elevation'].centroid_values
zr = rep.quantities['elevation'].centroid_values
check('A4. the repose kernel is reached: enabling it changes the bed',
      not np.allclose(zn, zr, rtol=0, atol=1e-14),
      'max |dz| between repose off and on = %.4f m' % np.abs(zn - zr).max())
check('A5. and it moved material rather than destroying it',
      abs(float(((zr - zn) * rep.areas).sum())) < 1e-9
      * max(float((np.abs(zr - zn) * rep.areas).sum()), 1.0),
      'net bed volume difference %+.3e m3 against %.3e moved'
      % (float(((zr - zn) * rep.areas).sum()),
         float((np.abs(zr - zn) * rep.areas).sum())))

# ---------------------------------------------------------------- B
print('B. every flow algorithm, not just DE0')

ref = None
for alg in FLOW_ALGORITHMS:
    r = evolve(build(algorithm=alg, bedload=True, repose=33.0))
    ok = (r['scour'] > 1e-4 and r['suspended'] > 0.0
          and abs(r['budget']) < 1e-9 * max(r['suspended'], 1.0))
    check('B. %s: sediment runs and the budget closes' % alg, ok,
          'scour %.4f m, suspended %.4e m3, budget %+.2e'
          % (r['scour'], r['suspended'], r['budget']))
    if ref is None:
        ref = r
    else:
        # Not identical -- the algorithms differ -- but the same problem, so
        # a wildly different answer would mean sediment saw a different flow.
        rel = abs(r['suspended'] - ref['suspended']) / max(ref['suspended'], 1e-30)
        check('B. %s: and agrees with DE0 to within a few percent' % alg,
              rel < 0.05, 'suspended differs by %.2f%%' % (100 * rel))

# ---------------------------------------------------------------- C
print('C. the operator is wired, and stays GPU-safe')

d = build()
ops = [type(o).__name__ for o in d.fractional_step_operators]
check('C1. add_sediment_class registers Sediment_operator automatically',
      'Sediment_operator' in ops, 'operators: %s' % ops)

d_none = build(sediment=False)
check('C2. and a domain without sediment does not get one',
      'Sediment_operator' not in [type(o).__name__
                                  for o in d_none.fractional_step_operators])

check('C3. the sediment operator does not force CPU-only fractional steps',
      not d._has_cpu_only_fractional_operators(),
      'a host-writing operator here would silently drop the whole run off '
      'the GPU path')

# ---------------------------------------------------------------- D
print('D. tripwire for transport paths that do not exist yet')

present = [n for n in FUTURE_PATH_SELECTORS if hasattr(build(sediment=False), n)]
check('D1. no uncovered alternative flux or update kernel has appeared',
      not present,
      ('Domain now has %s. Check the sediment fractional step is still reached '
       'on that path -- an alternative flux or update kernel that does not '
       'call it leaves sediment silently inert -- then extend FLOW_ALGORITHMS '
       'or add a parametrisation and remove the name from '
       'FUTURE_PATH_SELECTORS.' % ', '.join(repr(n) for n in present))
      if present else 'watching for: %s'
      % ', '.join(repr(n) for n in FUTURE_PATH_SELECTORS))

print('\n%d checks failed' % _fail[0])
raise SystemExit(1 if _fail[0] else 0)
