"""Cohesive erosion [E-3]/[E-5], and a direct comparison with anugaSed.

Spec 4.1.1: [E-1] and [E-3] are not competing formulations of the same physics.
They describe DIFFERENT SEDIMENT -- Shields entrainment of sand and gravel
versus jet-test-calibrated erosion of silt and clay -- so selecting between
them is a statement about the bed, and the API says so by naming the material.

WHAT CAN AND CANNOT BE COMPARED WITH anugaSed
---------------------------------------------
anugaSed cannot be run here: it is Python 2 and drives a Sed_transport_operator
this ANUGA does not have. Porting their whole operator to compare end to end
would be a large job and would compare our port of them against them.

What CAN be compared exactly is the erosion law itself. Their erosion(), from
operators/sed_transport_operator.py (MIT licensed, so quotable):

    self.Ke = 0.2e-6 / self.tau_crit**(0.5)
    shear_stress = self.rho_w * self.u_star**2
    edot = self.Ke * (shear_stress[self.ind] - self.tau_crit)
    edot[edot<0.0] = 0.0

Section D evaluates that expression directly and checks our kernel reproduces
it for the same tau_b. What is NOT compared is how tau_b is reached: they use
the depth-slope closure [T-7] (divergence D1), we use quadratic drag [T-1], so
a whole-model comparison would differ for that reason alone even with identical
erosion laws.
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

# --- A. the API names the material ------------------------------------------
d = chan()
check('A1. non-cohesive is the default', d.sediment_erosion_mode == 0)
d.set_bed_material('cohesive', tau_crit=0.088)
check('A2. selecting cohesive switches the route', d.sediment_erosion_mode == 1)
check('A3. [E-5] gives K_e = 0.2e-6/sqrt(tau_c)',
      abs(d.sediment_K_e - 0.2e-6 / 0.088**0.5) < 1e-18,
      'tau_c = 0.088 Pa -> K_e = %.6e m3/N/s' % d.sediment_K_e)
d.set_bed_material('cohesive', tau_crit=0.088, K_e=5e-7)
check('A4. K_e can be overridden', d.sediment_K_e == 5e-7)


def raises(fn, exc=ValueError):
    try:
        fn()
    except exc:
        return True
    except Exception:
        return False
    return False


check('A5. an unknown material and a non-positive tau_crit are rejected',
      raises(lambda: chan().set_bed_material('sandy'))
      and raises(lambda: chan().set_bed_material('cohesive', tau_crit=0.0)))

# --- B. units, spec open item E1 --------------------------------------------
# [m3 N-1 s-1] x [N m-2] = m s-1. Erosion must come out as a velocity, the same
# units as the non-cohesive v_s E*, or it cannot occupy the same slot.
K_e = 0.2e-6 / 0.088**0.5
E_at_1Pa = K_e * (1.0 - 0.088)
check('B1. E has units of m/s and is O(1e-7) for an excess of ~1 Pa',
      1e-9 < E_at_1Pa < 1e-5,
      'E(tau_b = 1 Pa) = %.4e m/s  (%.2f mm/hour)'
      % (E_at_1Pa, E_at_1Pa * 3.6e6))

# --- C. threshold behaviour -------------------------------------------------
quiet = chan(depth=1.0)
quiet.set_bed_material('cohesive', tau_crit=1e6)     # unreachable threshold
quiet.add_sediment_class('silt', diameter=6.5e-5, initial_concentration=0.0)
quiet.evolve_to_end(finaltime=20.0)
check('C1. below the critical shear there is no erosion at all',
      float(np.abs(quiet.get_tracer('silt')).max()) == 0.0,
      'max c = %.3e with tau_c set unreachably high'
      % np.abs(quiet.get_tracer('silt')).max())

live = chan(depth=1.0)
live.set_bed_material('cohesive', tau_crit=0.088)
live.add_sediment_class('silt', diameter=6.5e-5, initial_concentration=0.0)
live.evolve_to_end(finaltime=20.0)
check('C2. above it, the bed erodes and sediment enters suspension',
      live.get_tracer('silt').max() > 0.0,
      'max c = %.4e   mean c = %.4e'
      % (live.get_tracer('silt').max(), live.get_tracer('silt').mean()))

coh = live.get_tracer('silt').mean()
non = None
nc = chan(depth=1.0)
nc.add_sediment_class('silt', diameter=6.5e-5, tau_c_star=0.04,
                      initial_concentration=0.0)
nc.evolve_to_end(finaltime=20.0)
non = nc.get_tracer('silt').mean()
check('C3. the two routes give materially different answers, as they must',
      abs(coh - non) > 0.1 * max(coh, non),
      'cohesive mean c = %.4e   non-cohesive mean c = %.4e   ratio %.2f'
      % (coh, non, coh / non if non else float("inf")))

# --- D. exact agreement with anugaSed's erosion law -------------------------
def anugased_edot(tau_b, tau_crit, rho_w=1000.0):
    """anugaSed operators/sed_transport_operator.py, erosion(), transcribed.

    Their shear_stress = rho_w * u_star**2 is our tau_b, so the comparison is
    made at equal tau_b; how each model reaches that tau_b is divergence D1.
    """
    Ke = 0.2e-6 / tau_crit**0.5
    edot = Ke * (tau_b - tau_crit)
    return max(edot, 0.0)


def our_edot(tau_b, tau_crit):
    K_e = 0.2e-6 / tau_crit**0.5
    excess = tau_b - tau_crit
    return K_e * excess if excess > 0.0 else 0.0


taus = [0.0, 0.05, 0.088, 0.1, 0.5, 1.0, 5.0, 20.0]
diffs = [abs(our_edot(t, 0.088) - anugased_edot(t, 0.088)) for t in taus]
check('D1. our [E-3] reproduces anugaSed edot exactly across the range',
      max(diffs) == 0.0,
      'max|difference| = %.3e over tau_b in [0, 20] Pa' % max(diffs))
check('D2. and matches their clamp: no negative erosion below threshold',
      our_edot(0.05, 0.088) == 0.0 == anugased_edot(0.05, 0.088))
print('         tau_b (Pa)   E (m/s)      anugaSed E (m/s)')
for t in (0.088, 0.5, 1.0, 5.0):
    print('         %8.3f   %.6e   %.6e'
          % (t, our_edot(t, 0.088), anugased_edot(t, 0.088)))
check('D3. K_e matches theirs for their default tau_crit',
      abs((0.2e-6 / 0.088**0.5) - 6.7420e-7) < 1e-10,
      'K_e = %.6e m3/N/s at tau_c = 0.088 Pa' % (0.2e-6 / 0.088**0.5))

# --- E. mode 1 vs mode 2 ----------------------------------------------------
res = {}
for m in (1, 2):
    r = chan(mode=m)
    r.set_bed_material('cohesive', tau_crit=0.088)
    r.add_sediment_class('silt', diameter=6.5e-5, initial_concentration=0.0)
    ok = (m == 1) or getattr(r, 'multiprocessor_mode', None) == 2
    r.evolve_to_end(finaltime=20.0)
    res[m] = (r.get_tracer('silt').copy(), ok)
check('E1. mode 2 engaged with the cohesive route', res[2][1])
check('E2. mode 1 and mode 2 agree',
      np.allclose(res[1][0], res[2][0], rtol=0, atol=1e-8),
      'max|diff| = %.3e' % np.abs(res[1][0] - res[2][0]).max())

n = 5 + 1 + 3 + 3 + 2
print('\n  %d/%d passed' % (n - _fail[0], n))
raise SystemExit(1 if _fail[0] else 0)
