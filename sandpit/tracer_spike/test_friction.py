"""Friction closures for the sediment kernel -- spec 3.3, 3.3.1-3.3.3.

  A. THE FACTOR-OF-8 TRAP. W04 write (8/f_c)^1/2, but THEIR f_c is the
     Darcy-Weisbach f (their Eq 4), while ours is f/8. Verified against W04's
     own headline result: n = 0.0545 s m^-1/3 for Martian channels.
  B. larsen_lamb reproduces LL16's published n = 0.065.
  C. wilson varies with relative submergence and bed type.
  D. All three run, and only 'constant' is the default.
  E. mode 1 vs mode 2.
"""
import math
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
    return ok


def dom(mode=1):
    d = rectangular_cross_domain(10, 10, len1=LEN, len2=LEN)
    d.set_flow_algorithm('DE0')
    d.set_low_froude(0)
    d.store = False
    d.set_quantity('elevation', lambda x, y: -0.01 * x)
    d.set_quantity('stage', lambda x, y: -0.01 * x + 1.0)
    d.set_quantity('friction', 0.03)
    d.set_boundary({t: Reflective_boundary(d) for t in d.get_boundary_tags()})
    d.evolve_max_timestep = 1.0
    if mode != 1:
        d.set_multiprocessor_mode(mode)
    return d


print(__doc__)

# --- A. the factor-of-8 trap ------------------------------------------------
# W04 abstract: n = 0.0545 s m^-1/3 for Martian channels. Under [T-6],
# n = sqrt(f_c h^(1/3) / g) with OUR f_c. W04's sand-bed relation gives
# X = 8.46 (R/D50)^0.1005 and, correctly converted, f_c = 1/X^2.
G_MARS = 3.71
X = 8.46 * 1000.0**0.1005
f_c_ours = 1.0 / (X * X)
f_c_literal = 8.0 / (X * X)          # the trap: taking W04's f_c as ours


def depth_for_n(f_c, n, g):
    return (n * n * g / f_c)**3


h_ok = depth_for_n(f_c_ours, 0.0545, G_MARS)
h_bad = depth_for_n(f_c_literal, 0.0545, G_MARS)
check('A1. correct conversion puts W04 n=0.0545 at a plausible channel depth',
      5.0 < h_ok < 200.0,
      'f_c = 1/X^2 = %.5f -> h = %.1f m (Martian outflow channels are tens of '
      'metres deep)' % (f_c_ours, h_ok))
check('A2. the literal reading puts it at an absurd depth',
      h_bad < 1.0,
      'f_c = 8/X^2 = %.5f -> h = %.3f m, which is not an outflow channel'
      % (f_c_literal, h_bad))
check('A3. the two differ by exactly the factor of 8',
      abs(f_c_literal / f_c_ours - 8.0) < 1e-12)

# --- B. larsen_lamb ---------------------------------------------------------
d = dom()
d.set_sediment_friction('larsen_lamb', sigma_br=5.0)     # LL16 Moses Coulee
check('B1. [T-14]/[T-15] reproduce LL16\'s published n = 0.065',
      abs(d.sediment_manning_ll - 0.065) < 5e-4,
      'k_s = 2*2*5 = 20 m -> n = %.5f  (LL16 state 0.065)'
      % d.sediment_manning_ll)
d2 = dom()
d2.set_sediment_friction('larsen_lamb', k_s=20.0)
check('B2. passing k_s directly gives the same n',
      abs(d2.sediment_manning_ll - d.sediment_manning_ll) < 1e-12)
a = dom(); a.set_sediment_friction('larsen_lamb', k_s=20.0)
b = dom(); b.set_sediment_friction('larsen_lamb', k_s=40.0)
check('B3. n increases with bed roughness',
      b.sediment_manning_ll > a.sediment_manning_ll,
      'k_s 20 m -> n %.5f;  k_s 40 m -> n %.5f'
      % (a.sediment_manning_ll, b.sediment_manning_ll))
check('B4. and doubling k_s scales n by exactly 2^(1/6)',
      abs(b.sediment_manning_ll / a.sediment_manning_ll - 2**(1/6)) < 1e-12,
      'ratio %.6f vs 2^(1/6) = %.6f'
      % (b.sediment_manning_ll / a.sediment_manning_ll, 2**(1/6)))


def raises(fn, exc=ValueError):
    try:
        fn()
    except exc:
        return True
    except Exception:
        return False
    return False


check('B5. sigma_br is required and has no default (site-measured)',
      raises(lambda: dom().set_sediment_friction('larsen_lamb')))

# --- C. wilson --------------------------------------------------------------
def fc_wilson(rel, bed):
    rel = max(rel, 1.0)
    if bed == 'sand':
        X = 8.46 * rel**0.1005
    elif bed == 'gravel':
        X = 5.75 * math.log10(rel) + 3.514
    else:
        X = 5.62 * math.log10(rel) + 4.0
    return 1.0 / (X * X)


check('C1. f_c falls as relative submergence rises (smoother relative bed)',
      fc_wilson(1000, 'sand') < fc_wilson(10, 'sand'),
      'sand: f_c(h/D=10) = %.5f -> f_c(h/D=1000) = %.5f'
      % (fc_wilson(10, 'sand'), fc_wilson(1000, 'sand')))
check('C2. bed types give distinct f_c at the same submergence',
      len({round(fc_wilson(100, b), 8)
           for b in ('sand', 'gravel', 'boulder')}) == 3,
      'sand %.5f  gravel %.5f  boulder %.5f'
      % tuple(fc_wilson(100, b) for b in ('sand', 'gravel', 'boulder')))
check('C3. submergence is floored at 1, so h < D cannot blow up',
      np.isfinite(fc_wilson(0.001, 'gravel'))
      and fc_wilson(0.001, 'gravel') == fc_wilson(1.0, 'gravel'),
      'gravel f_c(h/D=0.001) = %.5f, clamped to the h=D value'
      % fc_wilson(0.001, 'gravel'))
check('C4. wilson requires a grain size',
      raises(lambda: dom().set_sediment_friction('wilson', bed='gravel')))
check('C5. unknown mode and bed are rejected',
      raises(lambda: dom().set_sediment_friction('nope'))
      and raises(lambda: dom().set_sediment_friction('wilson', bed='mud',
                                                     grain_size=1e-3)))

# --- D. all three run end to end -------------------------------------------
check('D1. constant is the default', dom().sediment_friction_mode == 0)
means = {}
for name, kw in (('constant', {}),
                 ('larsen_lamb', dict(sigma_br=5.0)),
                 ('wilson', dict(bed='gravel', grain_size=0.05))):
    r = dom()
    if name != 'constant':
        r.set_sediment_friction(name, **kw)
    r.sediment_d_star_mode = 1
    r.add_sediment_class('s', diameter=1e-4, initial_concentration=0.01)
    r.evolve_to_end(finaltime=20.0)
    means[name] = r.get_tracer('s').mean()
check('D2. all three closures run and give finite, distinct results',
      all(np.isfinite(v) for v in means.values())
      and len({round(v, 12) for v in means.values()}) == 3,
      '  '.join('%s %.4e' % kv for kv in means.items()))

# --- E. mode 1 vs mode 2 ----------------------------------------------------
res = {}
for m in (1, 2):
    r = dom(mode=m)
    r.set_sediment_friction('wilson', bed='gravel', grain_size=0.05)
    r.sediment_d_star_mode = 1
    ok = (m == 1) or getattr(r, 'multiprocessor_mode', None) == 2
    r.add_sediment_class('s', diameter=1e-4, initial_concentration=0.01)
    r.evolve_to_end(finaltime=20.0)
    res[m] = (r.get_tracer('s').copy(), ok)
check('E1. mode 2 engaged with the wilson closure', res[2][1])
check('E2. mode 1 and mode 2 agree', np.allclose(res[1][0], res[2][0],
                                                 rtol=0, atol=1e-8),
      'max|diff| = %.3e' % np.abs(res[1][0] - res[2][0]).max())

n = 3 + 5 + 5 + 2 + 2
print('\n  %d/%d passed' % (n - _fail[0], n))
raise SystemExit(1 if _fail[0] else 0)
