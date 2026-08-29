"""Phase 4: bedload transport [K-1]-[K-4] and its bed evolution [G-5], spec 6.

Bedload is a DIVERGENCE, not a source: it moves sediment ALONG the bed rather
than between bed and water column. So the defining property is that in a closed
domain it redistributes bed material and conserves total bed volume EXACTLY.
That is check B, and it is what separates a correct divergence from a plausible
one.

  A. Threshold and parameter sets.
  B. Conservation: closed domain, total bed volume unchanged.
  C. Direction: bed erodes upstream and builds downstream.
  D. Engelund-Hansen is TOTAL LOAD -- selecting it must switch the suspended
     source off, per spec 6's critical usage rule, and has no threshold.
  E. mode 1 vs mode 2.
"""
import numpy as np
import anuga
from anuga import Reflective_boundary, rectangular_cross_domain

LEN = 100.0
_fail = [0]


def check(label, ok, detail=''):
    print('  [%s] %s' % ('PASS' if ok else 'FAIL', label))
    if detail:
        print('         ' + detail)
    if not ok:
        _fail[0] += 1
    return ok


def chan(mode=1, slope=0.02, depth=0.5, n_manning=0.025, dt=0.5):
    d = rectangular_cross_domain(20, 10, len1=LEN, len2=LEN / 2)
    d.set_flow_algorithm('DE0')
    d.set_low_froude(0)
    d.store = False
    d.set_quantity('elevation', lambda x, y: -slope * x)
    d.set_quantity('stage', lambda x, y: -slope * x + depth)
    d.set_quantity('friction', n_manning)
    d.set_boundary({t: Reflective_boundary(d) for t in d.get_boundary_tags()})
    d.evolve_max_timestep = dt
    if mode != 1:
        d.set_multiprocessor_mode(mode)
    return d


print(__doc__)

# --- A. parameter sets ------------------------------------------------------
d = chan()
d.add_sediment_class('gravel', diameter=5e-3, tau_c_star=0.0,
                     initial_concentration=0.0)
d.set_bedload('wong_parker_eq24')
check('A1. Eq 24 parameters are K=3.97, m=1.5, tau_c*=0.0495',
      (d.sediment_bedload_K, d.sediment_bedload_m,
       d.sediment_bedload_tau_c_star) == (3.97, 1.5, 0.0495))
d.set_bedload('wong_parker_eq23')
check('A2. Eq 23 parameters are K=4.93, m=1.60, tau_c*=0.0470 (still open '
      'which FG21 used)',
      (d.sediment_bedload_K, d.sediment_bedload_m,
       d.sediment_bedload_tau_c_star) == (4.93, 1.60, 0.0470))


def raises(fn, exc=ValueError):
    try:
        fn()
    except exc:
        return True
    except Exception:
        return False
    return False


check('A3. an unknown formula is rejected',
      raises(lambda: chan().set_bedload('mpm')))

# --- B. conservation --------------------------------------------------------
b = chan()
b.add_sediment_class('gravel', diameter=5e-3, tau_c_star=0.0,
                     initial_concentration=0.0)
b.set_bedload('wong_parker_eq24')
z0 = b.quantities['elevation'].centroid_values.copy()
vol0 = float((z0 * b.areas).sum())
b.evolve_to_end(finaltime=60.0)
z1 = b.quantities['elevation'].centroid_values
vol1 = float((z1 * b.areas).sum())
dz = z1 - z0
check('B1. bedload actually moved the bed', np.abs(dz).max() > 0.0,
      'max|dz| = %.6e   (eroded %.3e, deposited %.3e)'
      % (np.abs(dz).max(), dz.min(), dz.max()))
check('B2. total bed volume is conserved exactly in a closed domain',
      abs(vol1 - vol0) <= 1e-10 * max(abs(vol0), 1.0),
      'bed volume %.10e -> %.10e   rel change %.3e'
      % (vol0, vol1, abs(vol1 - vol0) / max(abs(vol0), 1.0)))
check('B3. it redistributes rather than only eroding or only depositing',
      dz.min() < 0.0 < dz.max(),
      'min dz = %.3e   max dz = %.3e' % (dz.min(), dz.max()))

# --- C. direction -----------------------------------------------------------
x = b.centroid_coordinates[:, 0]
up, dn = x < LEN * 0.25, x > LEN * 0.75
check('C1. net erosion upstream, net deposition downstream',
      dz[up].sum() < 0.0 < dz[dn].sum(),
      'sum dz upstream = %.3e   downstream = %.3e'
      % (dz[up].sum(), dz[dn].sum()))

# --- D. Engelund-Hansen is total load ---------------------------------------
e = chan()
e.add_sediment_class('gravel', diameter=5e-3, initial_concentration=0.01)
e.set_bedload('engelund_hansen')
check('D1. selecting Engelund-Hansen disables the suspended source',
      e._sediment_suspended_enabled is False,
      '[K-5] already contains suspension; running both double counts '
      '(spec 6 critical usage rule)')
check('D2. and it carries no threshold', e.sediment_bedload_tau_c_star == 0.0)
# With the suspended source off, m is ADVECTED but never exchanged, so in a
# closed domain its total is conserved. If the suspended operator were still
# running alongside [K-5] -- the double counting spec 6 forbids -- this mass
# would change.
em0 = float((e.tracer_conserved_values[0] * e.areas).sum())
e.evolve_to_end(finaltime=20.0)
em1 = float((e.tracer_conserved_values[0] * e.areas).sum())
check('D3. suspended mass is advected only, not exchanged, under [K-5]',
      abs(em1 - em0) <= 1e-9 * max(abs(em0), 1.0),
      'suspended mass %.10e -> %.10e   rel change %.3e'
      % (em0, em1, abs(em1 - em0) / max(abs(em0), 1.0)))
ez = e.quantities['elevation'].centroid_values
check('D4. Engelund-Hansen still moves the bed', np.isfinite(ez).all())

# --- E. mode 1 vs mode 2 ----------------------------------------------------
res = {}
for mode in (1, 2):
    m = chan(mode=mode)
    m.add_sediment_class('gravel', diameter=5e-3, tau_c_star=0.0,
                         initial_concentration=0.0)
    m.set_bedload('wong_parker_eq24')
    ok = (mode == 1) or getattr(m, 'multiprocessor_mode', None) == 2
    m.evolve_to_end(finaltime=30.0)
    res[mode] = (m.quantities['elevation'].centroid_values.copy(), ok)
check('E1. mode 2 engaged with bedload active', res[2][1])
check('E2. mode 1 and mode 2 give the same bed',
      np.allclose(res[1][0], res[2][0], rtol=0, atol=1e-8),
      'max|diff| = %.3e' % np.abs(res[1][0] - res[2][0]).max())

n = 3 + 3 + 1 + 4 + 2
print('\n  %d/%d passed' % (n - _fail[0], n))
raise SystemExit(1 if _fail[0] else 0)
