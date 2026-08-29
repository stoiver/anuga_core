"""Method of Manufactured Solutions for the sediment source terms (RDy26 3.3).

MMS is the rigorous way to verify an implementation: prescribe a smooth exact
solution, derive the source term that makes it exact, then check the code
recovers it at the design order of accuracy. Unlike every other test here, a
failure cannot be argued away -- the answer is known.

SCOPE, AND WHY IT IS THIS SCOPE
-------------------------------
RDy26's own MMS takes its manufactured hydrodynamics (h, u, v, z, n) from a
different paper (Bisht et al. 2025), which is not to hand, so their case cannot
be reproduced exactly. Their manufactured SEDIMENT field is given in full
(their B2-B5) and is used here:

    c_s(x,y,t) = C [1 + sin(Kx) sin(Ky)] exp(alpha_s t/T)
    K = pi/L,  L = 5 m,  T = 20 s,  C = 0.5,  alpha_1 = +1, alpha_2 = -1

This test runs it in STILL WATER, so u = v = 0 and the advective term vanishes.
That is deliberate, not a dodge:

  * advection is already verified to MACHINE PRECISION by the RDy26 dam-break
    contact test (test_rdycore_benchmarks.py B4/B5: SSC flat to 3.6e-15 across
    a rarefaction and a shock);
  * what NOTHING else verifies is the SOURCE-TERM coupling and its order in
    time, which is exactly what RDy26 say their MMS adds over the passive
    cases.

So the two together cover both halves of [G-3]. With u = 0 the manufactured
source reduces to

    m = h c,   dm/dt = h dc/dt = h C [1+sin sin] (alpha/T) exp(alpha t/T)
    E = 0 (no shear, so no entrainment)
    D = d* c v_s
    => S_ms = h dc/dt + d* c v_s

and the expected convergence is FIRST ORDER in dt: the source is applied as a
fractional step with the full timestep.
"""
import numpy as np
import anuga
from anuga import Reflective_boundary, rectangular_cross_domain

C_SCALE, L_MMS, T_MMS = 0.5, 5.0, 20.0
K = np.pi / L_MMS
DEPTH = 1.0
D50 = 1.0e-4
T_END = 4.0
_fail = [0]


def check(label, ok, detail=''):
    print('  [%s] %s' % ('PASS' if ok else 'FAIL', label))
    if detail:
        print('         ' + detail)
    if not ok:
        _fail[0] += 1
    return ok


def c_exact(x, y, t, alpha):
    return C_SCALE * (1.0 + np.sin(K * x) * np.sin(K * y)) * np.exp(alpha * t / T_MMS)


def dcdt_exact(x, y, t, alpha):
    return c_exact(x, y, t, alpha) * (alpha / T_MMS)


def run(nxy, dt, alpha=1.0):
    """One MMS run. Returns the L1, L2 and Linf errors in the conserved m."""
    d = rectangular_cross_domain(nxy, nxy, len1=L_MMS, len2=L_MMS)
    d.set_flow_algorithm('DE0')
    d.set_low_froude(0)
    d.store = False
    d.set_quantity('elevation', 0.0)
    d.set_quantity('stage', DEPTH)
    d.set_quantity('friction', 0.0)
    d.set_boundary({t: Reflective_boundary(d) for t in d.get_boundary_tags()})
    d.evolve_max_timestep = dt
    d.sediment_bed_evolution = False        # fixed bed: h stays at DEPTH
    d.sediment_c_max = 10.0                 # do not let [L-2] clip the solution
    d.sediment_c_pack = 10.0
    x, y = d.centroid_coordinates[:, 0], d.centroid_coordinates[:, 1]

    d.add_sediment_class('mms', diameter=D50, tau_c_star=0.0,
                         initial_concentration=c_exact(x, y, 0.0, alpha))
    v_s = d.sediment_settling_velocity[0]

    # The operator applies the source every step with the CURRENT time, so the
    # manufactured source has to be refreshed each step. A tiny operator does
    # that, registered after the sediment operator so it is seen in the same
    # fractional step.
    class MMS_source(anuga.operators.base_operator.Operator):
        def __call__(self):
            t = self.domain.get_time()
            S = DEPTH * dcdt_exact(x, y, t, alpha) + c_exact(x, y, t, alpha) * v_s
            self.domain.set_tracer_source('mms', S)

        def parallel_safe(self):
            return True

        def statistics(self):
            return 'MMS source'

        def timestepping_statistics(self):
            return ''

    MMS_source(d)
    d.evolve_to_end(finaltime=T_END)
    achieved_dt = d.get_timestep()

    m_num = d.tracer_conserved_values[0]
    m_ex = DEPTH * c_exact(x, y, T_END, alpha)
    e = np.abs(m_num - m_ex)
    a = d.areas
    return (float((e * a).sum() / a.sum()),
            float(np.sqrt((e**2 * a).sum() / a.sum())),
            float(e.max()), achieved_dt)


print(__doc__)

# --- A. the manufactured solution is recovered ------------------------------
L1, L2, Li, _ = run(20, 0.005)
check('A1. the manufactured solution is recovered to better than 1%',
      L1 / (DEPTH * C_SCALE) < 0.01,
      'L1 = %.4e   L2 = %.4e   Linf = %.4e   (scale h*C = %.2f)'
      % (L1, L2, Li, DEPTH * C_SCALE))

# --- B. convergence in time -------------------------------------------------
# The source is a fractional step applied with the full dt, so first order.
#
# THE REQUESTED dt MUST ACTUALLY CONTROL. evolve_max_timestep is a CAP: if it
# is above the CFL limit the solver uses the CFL value instead, every run gets
# the same timestep, and the study measures nothing. A first attempt used
# dt = 0.2 down to 0.025, all of which were clipped to ~0.01 by CFL, and duly
# reported an observed order of 0.037 -- which looked like a broken source term
# and was a broken experiment. The achieved dt is printed below so the reader
# can see it is the requested one.
print('\n      requested  achieved      L1            L2            Linf       rate(L1)')
prev = None
rates = []
for dt in (0.008, 0.004, 0.002, 0.001):
    L1, L2, Li, adt = run(20, dt)
    rate = (np.log(prev / L1) / np.log(2.0)) if prev else float('nan')
    if prev:
        rates.append(rate)
    print('       %7.4f   %7.4f   %.6e  %.6e  %.6e   %s'
          % (dt, adt, L1, L2, Li, ('%.3f' % rate) if prev else '   -'))
    prev = L1

check('B1. the error falls monotonically as dt is halved',
      all(r > 0 for r in rates),
      'observed rates: ' + ', '.join('%.3f' % r for r in rates))
check('B2. observed order is first order, as the fractional step implies',
      0.8 < np.mean(rates) < 1.3,
      'mean observed order %.3f (expected 1.0)' % np.mean(rates))

# --- C. the decaying class --------------------------------------------------
# alpha = -1 is RDy26's classes 2 and 3: the manufactured field decays rather
# than grows, so the source must be able to take sediment OUT as well.
L1d, _, _, _ = run(20, 0.005, alpha=-1.0)
check('C1. the decaying manufactured class is recovered too',
      L1d / (DEPTH * C_SCALE) < 0.01,
      'alpha = -1: L1 = %.4e' % L1d)

# --- D. the external source is not clipped by the limiters ------------------
# S_ms is applied after [L-1]/[L-2] by design; if it were clipped the
# manufactured solution could not be imposed and A1 would fail. Verify the
# path directly rather than inferring it.
d = rectangular_cross_domain(4, 4, len1=L_MMS, len2=L_MMS)
d.set_flow_algorithm('DE0')
d.store = False
d.set_quantity('elevation', 0.0)
d.set_quantity('stage', DEPTH)
d.set_boundary({t: Reflective_boundary(d) for t in d.get_boundary_tags()})
d.sediment_c_max = 1e-6                    # absurdly tight [L-2]
d.add_sediment_class('s', diameter=D50, tau_c_star=0.0, d_star=0.0,
                     initial_concentration=0.0)
d.set_tracer_source('s', 1.0e-3)           # steady external supply
m0 = float((d.tracer_conserved_values[0] * d.areas).sum())
d.evolve_to_end(finaltime=2.0)
m1 = float((d.tracer_conserved_values[0] * d.areas).sum())
check('D1. an external source is delivered even against a tight c_max',
      m1 > m0,
      'suspended mass %.4e -> %.4e with c_max = 1e-6; [L-2] bounds bed '
      'exchange, not external supply' % (m0, m1))

n = 1 + 2 + 1 + 1
print('\n  %d/%d passed' % (n - _fail[0], n))
raise SystemExit(1 if _fail[0] else 0)
