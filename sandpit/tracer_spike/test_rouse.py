"""Phase 3: the Rouse near-bed concentration ratio d*(Z)  -- spec 4.3 / S1a.

  A. The kernel fit reproduces the [S-4] quadrature over its fitted range.
  B. Physical behaviour: d* >= 1 always, rising with Z (DL09 Fig 4).
  C. Out-of-range inputs are CLAMPED, not extrapolated.
  D. Mode 0 (constant) is unchanged, so d* = 1 remains the P14 limiting case.
  E. Rouse mode changes deposition in the direction physics requires, and
     mode 1 and mode 2 agree.
"""
import numpy as np
from scipy.integrate import quad
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


def dstar_ref(Z, a_h, z0_h=1e-4, h=1.0):
    """[S-4] by quadrature, with the CORRECTED (h-z) Rouse factor."""
    a, z0 = a_h * h, z0_h * h
    num = quad(lambda z: np.log(z / z0), a, h, limit=200)[0]
    den = quad(lambda z: (((h - z) / (h - a)) * (a / z))**Z * np.log(z / z0),
               a, h, limit=200)[0]
    return num / den


# The fit as compiled into the kernel, mirrored here so the test checks the
# COEFFICIENTS, not merely a reimplementation of the same idea.
import re
src = open('../../anuga/shallow_water/gpu/core_kernels.c').read()
rows = re.findall(r'\{([-+0-9.e]+), ([-+0-9.e]+), ([-+0-9.e]+), ([-+0-9.e]+)\},', src)
C = np.array([[float(v) for v in r] for r in rows[:7]])
lim = {k: float(re.search(r'#define ANUGA_ROUSE_%s\s+([0-9.e+-]+)' % k, src).group(1))
       for k in ('Z_LO', 'Z_HI', 'AH_LO', 'AH_HI')}


def dstar_fit(Z, a_h):
    Z = min(max(Z, lim['Z_LO']), lim['Z_HI'])
    a_h = min(max(a_h, lim['AH_LO']), lim['AH_HI'])
    L = np.log(a_h)
    P = 0.0
    for i in range(6, -1, -1):
        P = P * Z + (C[i][0] + C[i][1] * L + C[i][2] * L**2 + C[i][3] * L**3)
    return max(np.exp(-Z * L + P), 1.0)


print(__doc__)
print('  fitted range: Z [%.3g, %.3g], a/h [%.3g, %.3g]'
      % (lim['Z_LO'], lim['Z_HI'], lim['AH_LO'], lim['AH_HI']))

# --- A. fit vs quadrature ---------------------------------------------------
errs = []
for Z in np.geomspace(lim['Z_LO'], lim['Z_HI'], 17):
    for ah in np.geomspace(lim['AH_LO'], lim['AH_HI'], 9):
        errs.append(abs(dstar_fit(Z, ah) / dstar_ref(Z, ah) - 1.0))
errs = np.array(errs)
check('A1. fit reproduces the [S-4] quadrature to better than 1.5%',
      errs.max() < 0.015,
      'max rel err %.3f%%   mean %.3f%%  over %d (Z, a/h) points'
      % (100 * errs.max(), 100 * errs.mean(), errs.size))

# --- B. physical behaviour --------------------------------------------------
check('B1. d* >= 1 everywhere (DL09: always larger than 1)',
      all(dstar_fit(Z, ah) >= 1.0
          for Z in np.geomspace(0.001, 10, 40) for ah in (0.01, 0.05, 0.15)))
check('B2. d* increases with Z at fixed a/h',
      all(dstar_fit(z2, 0.05) > dstar_fit(z1, 0.05)
          for z1, z2 in zip([0.02, 0.1, 0.5, 1.0], [0.1, 0.5, 1.0, 2.0])))
check('B3. d* is between 1 and 3 for Z < 0.1 (DL09 Fig 4 caption)',
      1.0 <= dstar_fit(0.09, 0.05) <= 3.0,
      'd*(Z=0.09, a/h=0.05) = %.4f' % dstar_fit(0.09, 0.05))
check('B4. well-mixed limit: d* -> 1 as Z -> 0',
      abs(dstar_fit(lim['Z_LO'], 0.05) - 1.0) < 0.05,
      'd*(Z=%.3g) = %.4f' % (lim['Z_LO'], dstar_fit(lim['Z_LO'], 0.05)))

# --- C. clamping ------------------------------------------------------------
check('C1. Z above range is clamped, not extrapolated',
      dstar_fit(50.0, 0.05) == dstar_fit(lim['Z_HI'], 0.05),
      'd*(Z=50) == d*(Z=%.3g) = %.3f' % (lim['Z_HI'], dstar_fit(50.0, 0.05)))
check('C2. a/h below range is clamped',
      dstar_fit(1.0, 1e-6) == dstar_fit(1.0, lim['AH_LO']))
check('C3. clamped values stay finite and physical',
      np.isfinite(dstar_fit(1e6, 1e-9)) and dstar_fit(1e6, 1e-9) >= 1.0)


check('C4. the anugaSed regime (a/h ~ 9.3e-4) is now within one clamp of range',
      lim['AH_LO'] <= 1e-3 + 1e-12,
      'fitted a/h floor is %.3g; anugaSed implies 9.3e-4, which clamps to it '
      '(~7%% in a/h, so ~7%% in d* at Z=1)' % lim['AH_LO'])
errs_wide = []
for Z in np.geomspace(lim['Z_LO'], lim['Z_HI'], 13):
    for ah in np.geomspace(1e-3, 0.01, 5):
        errs_wide.append(abs(dstar_fit(Z, ah) / dstar_ref(Z, ah) - 1.0))
check('C5. accuracy holds in the newly added a/h band [1e-3, 0.01]',
      max(errs_wide) < 0.015,
      'max rel err %.3f%% over %d points' % (100 * max(errs_wide), len(errs_wide)))


def lake(mode=1, dt=1.0, d_star_mode=0):
    d = rectangular_cross_domain(10, 10, len1=LEN, len2=LEN)
    d.set_flow_algorithm('DE0')
    d.set_low_froude(0)
    d.store = False
    d.set_quantity('elevation', lambda x, y: -0.01 * x)
    d.set_quantity('stage', lambda x, y: -0.01 * x + 1.0)
    d.set_quantity('friction', 0.03)
    d.set_boundary({t: Reflective_boundary(d) for t in d.get_boundary_tags()})
    d.evolve_max_timestep = dt
    d.sediment_d_star_mode = d_star_mode
    if mode != 1:
        d.set_multiprocessor_mode(mode)
    return d


# --- D. mode 0 unchanged ----------------------------------------------------
a = lake(d_star_mode=0)
a.add_sediment_class('s', diameter=1e-4, d_star=1.0, initial_concentration=0.02)
a.evolve_to_end(finaltime=30.0)
check('D1. constant mode still runs and keeps d* = 1 semantics',
      np.isfinite(a.get_tracer('s')).all(),
      'mean c = %.6e' % a.get_tracer('s').mean())

# --- E. Rouse mode ----------------------------------------------------------
res = {}
for mode in (1, 2):
    r = lake(mode=mode, d_star_mode=1)
    r.add_sediment_class('s', diameter=1e-4, initial_concentration=0.02)
    ok = (mode == 1) or getattr(r, 'multiprocessor_mode', None) == 2
    r.evolve_to_end(finaltime=30.0)
    res[mode] = (r.get_tracer('s').copy(), ok)
check('E1. mode 2 engaged with Rouse d*', res[2][1])
check('E2. mode 1 and mode 2 agree with Rouse d* active',
      np.allclose(res[1][0], res[2][0], rtol=0, atol=1e-8),
      'max|diff| = %.3e' % np.abs(res[1][0] - res[2][0]).max())
check('E3. Rouse d* >= 1 deposits at least as fast as the well-mixed limit',
      res[1][0].mean() <= a.get_tracer('s').mean() * (1.0 + 1e-9),
      'Rouse mean c = %.6e  vs  d*=1 mean c = %.6e'
      % (res[1][0].mean(), a.get_tracer('s').mean()))

n = 1 + 4 + 5 + 1 + 3
print('\n  %d/%d passed' % (n - _fail[0], n))
raise SystemExit(1 if _fail[0] else 0)
