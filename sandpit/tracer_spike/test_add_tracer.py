"""Phase 1: domain.add_tracer() -- registration API.

The properties that matter:

  A. REGISTRATION.  Names map to indices; the six arrays get the right
     shapes, dtype and contiguity.
  B. THE TRAP.  A tracer added AFTER the C struct is built must still be
     seen by the kernels. The struct is cached and evolve() never passes
     update_domain_c_struct=True, so add_tracer must invalidate it.
  C. GROWTH.  Adding a second tracer reallocates; tracer 0's data survives.
  D. SEEDING.  set_tracer keeps c and m = h*c consistent.
  E. GUARDS.  Duplicate names, bad shapes, conflicting beta are rejected.
"""
import numpy as np
import anuga
from anuga import Reflective_boundary, rectangular_cross_domain

LEN = 500.0
_fail = [0]


def check(label, ok, detail=''):
    print('  [%s] %s' % ('PASS' if ok else 'FAIL', label))
    if detail:
        print('         ' + detail)
    if not ok:
        _fail[0] += 1


def build(nxy=10):
    d = rectangular_cross_domain(nxy, nxy, len1=LEN, len2=LEN)
    d.set_flow_algorithm('DE0')
    d.store = False
    d.set_quantity('elevation', 0.0)
    d.set_quantity('stage', lambda x, y: np.where(x < LEN / 2, 2.0, 0.5))
    d.set_boundary({t: Reflective_boundary(d) for t in d.get_boundary_tags()})
    return d


print(__doc__)

# --- A. registration -------------------------------------------------------
d = build()
N, BL = d.number_of_elements, d.boundary_length
check('A0. a fresh domain has no tracers',
      d.number_of_tracers == 0 and d.get_tracer_names() == [])

i = d.add_tracer('sand', beta=1.0)
check('A1. add_tracer returns index 0 and registers the name',
      i == 0 and d.get_tracer_names() == ['sand']
      and d.get_tracer_index('sand') == 0)

expected = {'tracer_centroid_values': (1, N), 'tracer_edge_values': (1, 3 * N),
            'tracer_boundary_values': (1, BL), 'tracer_explicit_update': (1, N),
            'tracer_conserved_values': (1, N), 'tracer_backup_values': (1, N)}
bad = [(a, getattr(d, a).shape, s) for a, s in expected.items()
       if getattr(d, a).shape != s]
check('A2. all six arrays have the right (ns, ...) shapes', not bad, str(bad))

bad = [a for a in expected
       if not (getattr(d, a).dtype == np.float64
               and getattr(d, a).flags['C_CONTIGUOUS'])]
check('A3. all six are C-contiguous float64 (the kernel indexes s*N+k)',
      not bad, str(bad))

# --- B. the trap -----------------------------------------------------------
d = build()
d.set_multiprocessor_mode(1)
d.evolve_to_end(finaltime=0.2)          # forces the C struct to be built
check('B1. the C struct exists after an evolve', d._Domain_C_struct is not None)

d.add_tracer('mud', beta=1.0)
check('B2. add_tracer invalidates the cached C struct',
      d._Domain_C_struct is None,
      'without this the tracer is silently invisible to the kernels')

d.set_tracer('mud', 1.0)
m0 = float((d.tracer_conserved_values[0] * d.areas).sum())
d.evolve_to_end(finaltime=2.0)
eu = np.abs(d.tracer_explicit_update[0]).max()
m1 = float((d.tracer_conserved_values[0] * d.areas).sum())
check('B3. the late-registered tracer actually moves', eu > 0.0,
      'max|dm/dt| = %.6e  (zero would mean the struct was still stale)' % eu)
check('B4. and is conserved while it moves',
      abs(m1 - m0) <= 1e-12 * max(abs(m0), 1.0),
      'mass %.10f -> %.10f  (drift %.3e)' % (m0, m1, m1 - m0))

# --- C. growth preserves existing tracers ----------------------------------
d = build()
d.add_tracer('a', beta=1.0)
d.set_tracer('a', 0.25)
ca = d.get_tracer('a').copy()
d.tracer_explicit_update[0, :] = 7.0     # a distinctive non-zero row
d.add_tracer('b')
check('C1. growing to ns=2 keeps tracer 0 concentration intact',
      np.array_equal(d.get_tracer('a'), ca))
check('C2. and keeps its other arrays intact',
      np.all(d.tracer_explicit_update[0] == 7.0))
check('C3. the new row is zeroed, not aliased',
      np.all(d.tracer_explicit_update[1] == 0.0)
      and d.number_of_tracers == 2 and d.get_tracer_index('b') == 1)

# --- D. seeding ------------------------------------------------------------
d = build()
d.add_tracer('s')
h = np.maximum(d.quantities['stage'].centroid_values
               - d.quantities['elevation'].centroid_values, 0.0)
field = np.linspace(0.0, 1.0, d.number_of_elements)
d.set_tracer('s', field)
check('D1. set_tracer sets c', np.allclose(d.get_tracer('s'), field))
check('D2. and seeds the conserved m = h*c consistently',
      np.allclose(d.tracer_conserved_values[0], h * field),
      'max|m - h*c| = %.3e' % np.abs(d.tracer_conserved_values[0] - h * field).max())
d.set_tracer('s', 0.5)
check('D3. a scalar fills the domain', np.all(d.get_tracer('s') == 0.5))

# --- E. guards -------------------------------------------------------------
d = build()
d.add_tracer('x', beta=1.0)


def raises(fn, exc=ValueError):
    try:
        fn()
    except exc:
        return True
    except Exception:
        return False
    return False


check('E1. duplicate names are rejected', raises(lambda: d.add_tracer('x')))
check('E2. a wrong-length field is rejected',
      raises(lambda: d.set_tracer('x', np.zeros(3))))
check('E3. an unknown tracer name is rejected',
      raises(lambda: d.get_tracer('nope')))
check('E4. a conflicting beta is rejected, not silently applied',
      raises(lambda: d.add_tracer('y', beta=0.0)),
      'beta_tracer is one scalar shared by every tracer')
check('E5. a matching beta is accepted', d.add_tracer('y', beta=1.0) == 1)

n = 5 + 4 + 3 + 3 + 5
print('\n  %d/%d passed' % (n - _fail[0], n))
raise SystemExit(1 if _fail[0] else 0)
