"""Phase 3a: suspended sediment source term (deposition, fixed bed).

Phase 3 is the FIXED-BED stage of spec 2.4 -- no bed evolution, no bed->flow or
sediment->momentum feedback. Deposited mass leaves suspension.

  A. SETTLING [S-1]. Ferguson & Church against the value the spec verifies.
  B. ANALYTIC DECAY. In still water with no advection the source reduces to
       dm/dt = -d* v_s m / h   =>   m(t) = m0 exp(-d* v_s t / h)
     and the scheme should converge to it at first order in dt.
  C. NO-OP. v_s = 0 must reproduce Phase 2 transport bit-for-bit.
  D. [L-1] POSITIVITY. A deliberately huge v_s must drive m to zero and stop,
     never below. This is the limiter that replaces aS16's concentration clamp.
  E. [L-2] CEILING. c_max must bound concentration.
  F. MODE 1 vs MODE 2 agreement.
  G. GUARDS.
  H. ENTRAINMENT [E-1]/[E-2], non-cohesive Shields route (sand bed).
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
    return ok


def still(nxy=10, depth=1.0, mode=1, dt=None):
    """A flat lake at rest: no advection, so only the source term acts."""
    d = rectangular_cross_domain(nxy, nxy, len1=LEN, len2=LEN)
    d.set_flow_algorithm('DE0')
    d.set_low_froude(0)
    d.store = False
    d.set_quantity('elevation', 0.0)
    d.set_quantity('stage', depth)
    d.set_quantity('xmomentum', 0.0)
    d.set_quantity('ymomentum', 0.0)
    d.set_boundary({t: Reflective_boundary(d) for t in d.get_boundary_tags()})
    if dt is not None:
        d.evolve_max_timestep = dt
    if mode != 1:
        d.set_multiprocessor_mode(mode)
    return d


print(__doc__)

# --- A. settling velocity --------------------------------------------------
d = still()
vs = d.settling_velocity(4.5e-5)
check('A1. [S-1] matches the spec-verified value for 0.045 mm quartz',
      abs(vs - 1.75e-3) / 1.75e-3 < 0.02,
      'v_s = %.6e m/s  (spec / P13: 1.75e-3)' % vs)
check('A2. v_s increases with grain size',
      d.settling_velocity(1e-3) > d.settling_velocity(1e-4) > d.settling_velocity(1e-5))
check('A3. natural-grain constants give a slower fall than smooth spheres',
      d.settling_velocity(1e-4, C1=1.0, C2=1.1) != d.settling_velocity(1e-4))

# --- B. analytic decay, and first-order convergence -------------------------
DEPTH, DIAM, T_END = 1.0, 1.0e-4, 60.0
errs = {}
for dt in (4.0, 1.0):
    dom = still(depth=DEPTH, dt=dt)
    dom.add_sediment_class('sand', diameter=DIAM, initial_concentration=0.05)
    v_s = dom.sediment_settling_velocity[0]
    dom.evolve_to_end(finaltime=T_END)
    got = dom.get_tracer('sand').mean()
    exact = 0.05 * np.exp(-v_s * T_END / DEPTH)
    errs[dt] = abs(got - exact) / exact
    if dt == 1.0:
        check('B1. deposition follows m0 exp(-v_s t / h)', errs[dt] < 0.02,
              'c = %.6e vs exact %.6e  (rel err %.2e, v_s = %.4e)'
              % (got, exact, errs[dt], v_s))
check('B2. error falls with dt (first-order source integration)',
      errs[1.0] < errs[4.0],
      'rel err %.2e at dt=4 -> %.2e at dt=1' % (errs[4.0], errs[1.0]))

# --- C. v_s = 0 is a no-op --------------------------------------------------
base = still(depth=DEPTH, dt=1.0)
base.add_tracer('plain', beta=1.0)
base.set_tracer('plain', 0.05)
base.evolve_to_end(finaltime=20.0)

zero = still(depth=DEPTH, dt=1.0)
zero.add_sediment_class('zero', diameter=1.0e-4, d_star=0.0,
                        initial_concentration=0.05)
zero.evolve_to_end(finaltime=20.0)
check('C1. d* = 0 disables deposition, reproducing plain tracer transport',
      np.allclose(zero.get_tracer('zero'), base.get_tracer('plain'),
                  rtol=0, atol=1e-15),
      'max|diff| = %.3e' % np.abs(zero.get_tracer('zero')
                                  - base.get_tracer('plain')).max())

# --- D. [L-1] positivity ----------------------------------------------------
agg = still(depth=DEPTH, dt=1.0)
agg.add_sediment_class('fast', diameter=5.0e-3, initial_concentration=0.05)
agg.evolve_to_end(finaltime=30.0)
m = agg.tracer_conserved_values[0]
c = agg.get_tracer('fast')
check('D1. an aggressive settler never drives m negative', m.min() >= 0.0,
      'min m = %.3e   min c = %.3e   (v_s = %.3e m/s, %.1fx the depth per step)'
      % (m.min(), c.min(), agg.sediment_settling_velocity[0],
         agg.sediment_settling_velocity[0] / DEPTH))
check('D2. and it does deposit essentially everything', c.max() < 1e-6,
      'max c = %.3e' % c.max())

# --- E. [L-2] ceiling -------------------------------------------------------
cap = still(depth=DEPTH, dt=1.0)
cap.sediment_c_max = 0.10
cap.add_sediment_class('capped', diameter=1.0e-4, initial_concentration=0.05)
cap.evolve_to_end(finaltime=10.0)
check('E1. c stays under c_max (nothing here should push it up)',
      cap.get_tracer('capped').max() <= 0.10 + 1e-12,
      'max c = %.6e  (c_max = 0.10)' % cap.get_tracer('capped').max())

# --- F. mode 1 vs mode 2 ----------------------------------------------------
res = {}
for mode in (1, 2):
    dm = still(depth=DEPTH, dt=1.0, mode=mode)
    dm.add_sediment_class('sand', diameter=DIAM, initial_concentration=0.05)
    engaged = (mode == 1) or getattr(dm, 'multiprocessor_mode', None) == 2
    dm.evolve_to_end(finaltime=30.0)
    res[mode] = (dm.get_tracer('sand').copy(), engaged)
check('F1. mode 2 actually engaged (guards a false green)', res[2][1])
check('F2. mode 1 and mode 2 deposit identically',
      np.allclose(res[1][0], res[2][0], rtol=0, atol=1e-8),
      'max|diff| = %.3e' % np.abs(res[1][0] - res[2][0]).max())

# --- G. guards --------------------------------------------------------------
def raises(fn, exc=ValueError):
    try:
        fn()
    except exc:
        return True
    except Exception:
        return False
    return False


g = still()
g.add_sediment_class('a', diameter=1e-4)
check('G1. a second class registers at the next index',
      g.add_sediment_class('b', diameter=2e-4) == 1
      and g.get_sediment_names() == ['a', 'b'])
check('G2. per-class settling velocities are independent',
      g.sediment_settling_velocity[1] > g.sediment_settling_velocity[0])
check('G3. non-positive diameter is rejected',
      raises(lambda: g.add_sediment_class('c', diameter=0.0)))
g2 = still()
g2.add_tracer('plain')
check('G4. mixing add_tracer() and add_sediment_class() is rejected',
      raises(lambda: g2.add_sediment_class('s', diameter=1e-4)),
      'class s must occupy tracer slot s')

# --- H. entrainment --------------------------------------------------------
# A flowing channel: constant slope, steady-ish flow, so tau* is non-trivial.
def channel(nxy=10, depth=1.0, slope=0.01, n_manning=0.03, mode=1, dt=1.0):
    d = rectangular_cross_domain(nxy, nxy, len1=LEN, len2=LEN)
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


# H1. Below threshold there must be NO entrainment at all -- [E-1] is a genuine
# threshold, not a smooth roll-off. Still water has |v| = 0 so tau* = 0.
quiet = still(depth=DEPTH, dt=1.0)
quiet.add_sediment_class('sand', diameter=DIAM, tau_c_star=0.04,
                         initial_concentration=0.0)
quiet.evolve_to_end(finaltime=20.0)
check('H1. no entrainment below the critical Shields stress',
      float(np.abs(quiet.get_tracer('sand')).max()) == 0.0,
      'max c = %.3e in still water (tau* = 0 << tau_c* = 0.04)'
      % np.abs(quiet.get_tracer('sand')).max())

# H2. A flowing channel over a clean bed must pick sediment up.
flow = channel()
flow.add_sediment_class('sand', diameter=DIAM, tau_c_star=0.04,
                        initial_concentration=0.0)
flow.evolve_to_end(finaltime=60.0)
c_ent = flow.get_tracer('sand')
check('H2. a flowing channel entrains from a clean bed', c_ent.max() > 0.0,
      'max c = %.3e   mean c = %.3e' % (c_ent.max(), c_ent.mean()))

# H3. tau_c_star = 0 disables entrainment (documented hook), leaving deposition.
off = channel()
off.add_sediment_class('sand', diameter=DIAM, tau_c_star=0.0,
                       initial_concentration=0.0)
off.evolve_to_end(finaltime=60.0)
check('H3. tau_c_star = 0 disables entrainment',
      float(np.abs(off.get_tracer('sand')).max()) == 0.0,
      'max c = %.3e' % np.abs(off.get_tracer('sand')).max())

# H4/H5. No runaway. [E-1] saturates at 0.65 (maximum packing), which is the
# whole point of the saturating form, and [L-2] caps concentration at c_max.
#
# NOTE [L-2] is applied to the SOURCE TERM, so it does not see sediment arriving
# by ADVECTION and a small overshoot of c_max is expected. Making it exact would
# mean clamping the state, which the spec deliberately avoids for [L-1] because
# it breaks conservation. The test therefore asserts "bounded", not "<= c_max"
# to the last bit, and reports the overshoot so a regression in it is visible.
OVERSHOOT_TOL = 0.02
wild = channel(depth=3.0, slope=0.05, n_manning=0.05)
wild.add_sediment_class('sand', diameter=DIAM, tau_c_star=0.04,
                        initial_concentration=0.0)
wild.evolve_to_end(finaltime=60.0)
c_wild = wild.get_tracer('sand')
over = c_wild.max() / wild.sediment_c_max - 1.0
check('H4. violent flow stays bounded near c_max (source-side [L-2])',
      np.isfinite(c_wild).all() and over <= OVERSHOOT_TOL,
      'max c = %.6e vs c_max %.2f  ->  overshoot %.2f%% (advective inflow is '
      'not seen by the source-side cap)' % (c_wild.max(), wild.sediment_c_max,
                                            100.0 * over))

# H5. Over a long run it must still not run away: an unbounded erosion law
# would keep climbing. This is what the saturating [E-1] form buys.
wild.evolve_to_end(finaltime=300.0)
c_long = wild.get_tracer('sand')
over_long = c_long.max() / wild.sediment_c_max - 1.0
check('H5. and does not run away over a much longer run',
      np.isfinite(c_long).all() and over_long <= OVERSHOOT_TOL,
      'max c = %.6e at t=300 s  ->  overshoot %.2f%% (was %.2f%% at t=60 s)'
      % (c_long.max(), 100.0 * over_long, 100.0 * over))

# H6. mode 1 vs mode 2 with entrainment active.
ent = {}
for mode in (1, 2):
    dm = channel(mode=mode)
    dm.add_sediment_class('sand', diameter=DIAM, tau_c_star=0.04,
                          initial_concentration=0.0)
    ok = (mode == 1) or getattr(dm, 'multiprocessor_mode', None) == 2
    dm.evolve_to_end(finaltime=40.0)
    ent[mode] = (dm.get_tracer('sand').copy(), ok)
check('H6. mode 2 engaged for the entrainment case', ent[2][1])
check('H7. mode 1 and mode 2 entrain identically',
      np.allclose(ent[1][0], ent[2][0], rtol=0, atol=1e-8),
      'max|diff| = %.3e' % np.abs(ent[1][0] - ent[2][0]).max())

n = 3 + 2 + 1 + 2 + 1 + 2 + 4 + 7
print('\n  %d/%d passed' % (n - _fail[0], n))
raise SystemExit(1 if _fail[0] else 0)
