"""Bed shear closures [T-1] and [T-7] -- spec 3.1, 3.4, divergence D1.

  [T-1]  tau_b = rho f_c |v|^2     quadratic drag (default)
  [T-7]  tau_b = rho g h S         depth-slope, aSM16 Eqs 6-7, hence anugaSed

[T-7] is the steady uniform flow approximation. Spec 3.4 keeps it only for
reproducing published anugaSed results and recommends [T-1] otherwise, because
normal-flow equilibrium fails in exactly the floods this work targets.

  A. Selection and default.
  B. The slope reconstruction is right: a known bed slope must come back.
  C. Substituting the ENERGY slope into [T-7] recovers [T-1] identically --
     the spec's own argument for why [T-1] supersedes it.
  D. The two closures give materially different erosion, as D1 implies.
  E. mode 1 vs mode 2.
"""
import numpy as np
import anuga
from anuga import Reflective_boundary, rectangular_cross_domain

LEN = 200.0
_fail = [0]


def check(label, ok, detail=''):
    print('  [%s] %s' % ('PASS' if ok else 'FAIL', label))
    if detail:
        print('         ' + detail)
    if not ok:
        _fail[0] += 1
    return ok


def chan(mode=1, slope=0.01, depth=1.0, n_manning=0.03, dt=1.0):
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

# --- A. selection -----------------------------------------------------------
d = chan()
check('A1. quadratic drag [T-1] is the default', d.sediment_shear_closure == 0)
d.set_shear_closure('depth_slope')
check('A2. depth_slope selects [T-7]', d.sediment_shear_closure == 1)


def raises(fn, exc=ValueError):
    try:
        fn()
    except exc:
        return True
    except Exception:
        return False
    return False


check('A3. an unknown closure is rejected',
      raises(lambda: chan().set_shear_closure('depth-slope')))

# --- B. the slope reconstruction --------------------------------------------
# grad z from the divergence theorem over a cell's own edges must return the
# bed slope actually imposed. Mirror the kernel here and check against truth.
for imposed in (0.01, 0.05):
    dd = chan(slope=imposed)
    bed_ev = dd.quantities['elevation'].edge_values
    nrm = dd.normals
    el = dd.edgelengths
    ar = dd.areas
    gx = (bed_ev * nrm[:, 0::2] * el).sum(axis=1) / ar
    gy = (bed_ev * nrm[:, 1::2] * el).sum(axis=1) / ar
    S = np.sqrt(gx**2 + gy**2)
    check('B%d. reconstructed bed slope matches the imposed %.2f'
          % (1 if imposed == 0.01 else 2, imposed),
          np.allclose(S, imposed, rtol=1e-9, atol=1e-12),
          'S: min %.6f  max %.6f  imposed %.6f' % (S.min(), S.max(), imposed))

# --- C. the energy slope recovers [T-1] -------------------------------------
# Spec 3.4 point 2: S_f = f_c |v|^2 / (g h). Substituting into tau_b = rho g h S
# gives rho f_c |v|^2, which is [T-1] exactly. Verified numerically here because
# it is the whole argument for preferring [T-1].
f_c, vel2, h, g = 0.00883, 4.0, 1.5, 9.81
S_f = f_c * vel2 / (g * h)
check('C1. energy slope substituted into [T-7] reproduces [T-1] exactly',
      abs(g * h * S_f - f_c * vel2) < 1e-15,
      'g h S_f = %.12e   f_c |v|^2 = %.12e' % (g * h * S_f, f_c * vel2))

# --- D. the closures differ -------------------------------------------------
res = {}
for name, closure in (('quadratic_drag', 'quadratic_drag'),
                      ('depth_slope', 'depth_slope')):
    r = chan()
    r.set_shear_closure(closure)
    r.add_sediment_class('sand', diameter=1e-4, initial_concentration=0.0)
    r.evolve_to_end(finaltime=30.0)
    res[name] = r.get_tracer('sand').mean()
check('D1. both closures run and entrain',
      all(np.isfinite(v) and v >= 0.0 for v in res.values()),
      '  '.join('%s %.4e' % kv for kv in res.items()))
check('D2. they give materially different answers -- this is divergence D1',
      abs(res['quadratic_drag'] - res['depth_slope'])
      > 0.01 * max(res.values()),
      'ratio depth_slope/quadratic_drag = %.3f'
      % (res['depth_slope'] / res['quadratic_drag']
         if res['quadratic_drag'] else float('inf')))

# --- E. mode 1 vs mode 2 ----------------------------------------------------
mm = {}
for m in (1, 2):
    r = chan(mode=m)
    r.set_shear_closure('depth_slope')
    r.add_sediment_class('sand', diameter=1e-4, initial_concentration=0.0)
    ok = (m == 1) or getattr(r, 'multiprocessor_mode', None) == 2
    r.evolve_to_end(finaltime=20.0)
    mm[m] = (r.get_tracer('sand').copy(), ok)
check('E1. mode 2 engaged with [T-7]', mm[2][1])
check('E2. mode 1 and mode 2 agree under [T-7]',
      np.allclose(mm[1][0], mm[2][0], rtol=0, atol=1e-8),
      'max|diff| = %.3e' % np.abs(mm[1][0] - mm[2][0]).max())

n = 3 + 2 + 1 + 2 + 2
print('\n  %d/%d passed' % (n - _fail[0], n))
raise SystemExit(1 if _fail[0] else 0)
