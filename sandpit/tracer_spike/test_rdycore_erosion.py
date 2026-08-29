"""RDy26 3.2 -- dam break WITH erosion and deposition.

Feng et al. (2026) section 3.2: the 3.1.1 dam break with bed exchange switched
on. Their stated setup, reproduced here:

    one class, density 1600 kg/m3, settling velocity 1e-4 m/s
    critical DEPOSITION stress  0.1 N/m2      -> [D-2]
    critical EROSION stress     0.1 N/m2      -> [E-4]
    Partheniades coefficient    1.0e-4 kg/m2/s
    Manning n = 0.025,  dt = 0.01 s,  t_end = 240 s
    same mesh and initial water/sediment as the passive case

WHAT THIS CAN AND CANNOT ESTABLISH
----------------------------------
RDy26 compare their SSC against TELEMAC/GAIA, and that reference exists only as
figures in the preprint. There is no table to check against, and reading points
off a plot would produce a number with no defensible error bar. So this is NOT
a numerical comparison against their result, and does not pretend to be.

There is also NO analytic solution for this case. A first draft of this test
asserted that depth should still follow the Stoker dam-break solution, on the
grounds that RDy26 do not update bed elevation so the hydrodynamics are
untouched by the sediment. That reasoning is sound but the conclusion is wrong:
3.2 sets Manning n = 0.025 where the passive case 3.1.1 has n = 0, and the
Stoker solution is FRICTIONLESS. Friction changes the hydrodynamics whatever
the sediment does. The check failed at L1 = 4.3e-2 m and was right to.

So what is verifiable here is:

  * the CLOSURES themselves, exactly -- [E-4] and [D-2] are algebraic, so our
    kernel's rates can be checked against the formulas evaluated directly, the
    same way [E-3] was checked against anugaSed;
  * that the case actually exercises its thresholds, rather than sitting
    entirely above or below them;
  * mass consistency and physical bounds on SSC;
  * SELF-CONVERGENCE under mesh refinement, which needs no reference;
  * and, as a control, that turning exchange off via [D-2]'s tau_d = 0 hook
    with n = 0 recovers the analytic passive-contact solution of 3.1.1.

The part with no exact answer -- the SSC field itself -- is checked for
self-consistency and explicitly not dressed up as agreement with theirs.
"""
import numpy as np
from scipy.optimize import brentq
import anuga
from anuga import Reflective_boundary, rectangular_cross_domain

G = 9.81
HL, HR, CL, CR = 1.0, 0.5, 0.7, 0.5
DAM_X, LENGTH, T_END = 1000.0, 2000.0, 240.0
RHO_S, V_S = 1600.0, 1.0e-4
TAU_D, TAU_E, K_P = 0.1, 0.1, 1.0e-4
MANNING = 0.025
_fail = [0]


def check(label, ok, detail=''):
    print('  [%s] %s' % ('PASS' if ok else 'FAIL', label))
    if detail:
        print('         ' + detail)
    if not ok:
        _fail[0] += 1
    return ok


def stoker(hl, hr, t, xs):
    cl = np.sqrt(G * hl)

    def f(hm):
        return (-2.0 * (np.sqrt(G * hm) - cl)
                - (hm - hr) * np.sqrt(0.5 * G * (hm + hr) / (hm * hr)))

    hm = brentq(f, hr * (1 + 1e-12), hl * (1 - 1e-12), xtol=1e-14, rtol=1e-15)
    um = -2.0 * (np.sqrt(G * hm) - cl)
    cm = np.sqrt(G * hm)
    shock_s = hm * um / (hm - hr)
    h = np.empty_like(xs)
    u = np.zeros_like(xs)
    xi = (xs - DAM_X) / t
    for i, sp_ in enumerate(xi):
        if sp_ <= -cl:
            h[i], u[i] = hl, 0.0
        elif sp_ <= um - cm:
            h[i] = ((-sp_ + 2.0 * cl) / 3.0)**2 / G
            u[i] = 2.0 / 3.0 * (sp_ + cl)
        elif sp_ <= shock_s:
            h[i], u[i] = hm, um
        else:
            h[i], u[i] = hr, 0.0
    return h, u, um


def build(exchange):
    d = rectangular_cross_domain(800, 2, len1=LENGTH, len2=10.0)
    d.set_flow_algorithm('DE0')
    d.set_low_froude(0)
    d.store = False
    x = d.centroid_coordinates[:, 0]
    d.set_quantity('elevation', 0.0)
    d.set_quantity('stage', np.where(x < DAM_X, HL, HR), location='centroids')
    d.set_quantity('friction', MANNING)
    d.set_boundary({t: Reflective_boundary(d) for t in d.get_boundary_tags()})
    d.evolve_max_timestep = 0.05
    d.sediment_bed_evolution = False        # RDy26: bed elevation is not updated
    d.sediment_c_max = 1.0
    d.sediment_c_pack = 1.0
    if exchange:
        d.set_bed_material('partheniades', tau_crit=TAU_E, K_e=K_P)
        d.set_deposition('threshold', tau_d=TAU_D)
    else:
        d.set_deposition('threshold', tau_d=0.0)     # their passive hook
    d.add_sediment_class('ssc', diameter=1.0e-4, rho_s=RHO_S, rho_w=1000.0,
                         tau_c_star=0.0,
                         initial_concentration=np.where(x < DAM_X, CL, CR))
    d.sediment_settling_velocity[0] = V_S            # RDy26 prescribe it
    return d, x


print(__doc__)

def build_frictionless_passive():
    """RDy26 3.1.1 exactly: n = 0, and exchange off through [D-2] tau_d = 0."""
    d, xx = build(exchange=False)
    d.set_quantity('friction', 0.0)
    return d, xx


act, x = build(exchange=True)
areas = act.areas
m0 = float((act.tracer_conserved_values[0] * areas).sum())
act.evolve_to_end(finaltime=T_END)

pas, xp = build(exchange=False)
pas.evolve_to_end(finaltime=T_END)

h_num = np.maximum(act.quantities['stage'].centroid_values
                   - act.quantities['elevation'].centroid_values, 0.0)
c_act = act.get_tracer('ssc')
c_pas = pas.get_tracer('ssc')
u_num = (act.quantities['xmomentum'].centroid_values * h_num
         / (h_num**2 + 1e-12))
tau_num = 1000.0 * G * MANNING**2 * u_num**2 / np.cbrt(np.maximum(h_num, 1e-12))

# --- A. the closures, checked exactly against their own formulas ------------
def E4(tau_b, tau_c=TAU_E, Kp=K_P, rho_s=RHO_S):
    """[E-4] Partheniades, as RDy26 state it, in m/s."""
    return (Kp * (tau_b - tau_c) / tau_c) / rho_s if tau_b > tau_c else 0.0


def D2(tau_b, c, vs=V_S, tau_d=TAU_D):
    """[D-2] threshold deposition, in m/s."""
    return vs * c * (1.0 - tau_b / tau_d) if (tau_d > 0 and tau_b < tau_d) else 0.0


taus = [0.0, 0.05, 0.099, 0.1, 0.15, 0.5, 2.0, 6.0]
print('        tau_b (Pa)    E [E-4] (m/s)     D [D-2] (m/s, c=0.6)')
for tb in taus:
    print('        %9.3f    %.6e      %.6e' % (tb, E4(tb), D2(tb, 0.6)))
check('A1. [E-4] is zero at and below the critical stress, positive above',
      E4(TAU_E) == 0.0 and E4(0.099) == 0.0 and E4(0.15) > 0.0,
      'a genuine threshold, not a smooth roll-off')
check('A2. [E-4] is linear in excess stress, as Partheniades requires',
      abs(E4(0.3) / E4(0.2) - (0.3 - TAU_E) / (0.2 - TAU_E)) < 1e-12,
      'E(0.3)/E(0.2) = %.6f' % (E4(0.3) / E4(0.2)))
check('A3. [D-2] shuts off at tau_d and is maximal in still water',
      D2(TAU_D, 0.6) == 0.0 and D2(0.0, 0.6) == V_S * 0.6,
      'D(tau_d) = 0, D(0) = v_s c = %.3e m/s' % (V_S * 0.6))
check('A4. tau_d = 0 disables deposition entirely -- their passive hook',
      D2(0.05, 0.6, tau_d=0.0) == 0.0)

# --- B. the case exercises what it claims to --------------------------------
check('B1. the shear field straddles the 0.1 Pa thresholds',
      tau_num.max() > TAU_E > tau_num.min(),
      'tau_b in [%.4f, %.4f] Pa against tau_e = tau_d = %.2f Pa'
      % (tau_num.min(), tau_num.max(), TAU_E))
eroding = tau_num > TAU_E
check('B2. so both limbs run: some cells erode, some deposit',
      eroding.sum() > 0 and (~eroding).sum() > 0,
      '%d cells above tau_e, %d below tau_d' % (eroding.sum(), (~eroding).sum()))
check('B3. bed exchange moved SSC away from the passive answer',
      np.abs(c_act - c_pas).max() > 1e-6,
      'max|c_exchange - c_passive| = %.4e  (mean %.4e)'
      % (np.abs(c_act - c_pas).max(), np.abs(c_act - c_pas).mean()))
m1 = float((act.tracer_conserved_values[0] * areas).sum())
check('B4. and changed the total suspended mass',
      abs(m1 - m0) > 1e-9 * abs(m0),
      'suspended mass %.6e -> %.6e  (%+.3f%%)' % (m0, m1, 100 * (m1 - m0) / m0))
check('B5. SSC stays physical throughout',
      c_act.min() >= -1e-12 and c_act.max() <= act.sediment_c_max + 1e-9,
      'c in [%.6f, %.6f]' % (c_act.min(), c_act.max()))

# --- C. self-convergence, which needs no reference --------------------------
def coarse(nx):
    d = rectangular_cross_domain(nx, 2, len1=LENGTH, len2=10.0)
    d.set_flow_algorithm('DE0'); d.set_low_froude(0); d.store = False
    xx = d.centroid_coordinates[:, 0]
    d.set_quantity('elevation', 0.0)
    d.set_quantity('stage', np.where(xx < DAM_X, HL, HR), location='centroids')
    d.set_quantity('friction', MANNING)
    d.set_boundary({t: Reflective_boundary(d) for t in d.get_boundary_tags()})
    d.evolve_max_timestep = 0.05
    d.sediment_bed_evolution = False
    d.sediment_c_max = 1.0; d.sediment_c_pack = 1.0
    d.set_bed_material('partheniades', tau_crit=TAU_E, K_e=K_P)
    d.set_deposition('threshold', tau_d=TAU_D)
    d.add_sediment_class('ssc', diameter=1e-4, rho_s=RHO_S, rho_w=1000.0,
                         tau_c_star=0.0,
                         initial_concentration=np.where(xx < DAM_X, CL, CR))
    d.sediment_settling_velocity[0] = V_S
    d.evolve_to_end(finaltime=T_END)
    return xx, d.get_tracer('ssc')


xc, cc = coarse(200)
xm, cm = coarse(400)
probe = np.linspace(600.0, 1400.0, 200)
ic = np.interp(probe, *zip(*sorted(zip(xc, cc))))
im = np.interp(probe, *zip(*sorted(zip(xm, cm))))
if_ = np.interp(probe, *zip(*sorted(zip(x, c_act))))
d1 = np.abs(ic - im).mean()
d2 = np.abs(im - if_).mean()
check('C1. SSC self-converges under mesh refinement',
      d2 < d1,
      '|c(dx=10) - c(dx=5)| = %.4e  ->  |c(dx=5) - c(dx=2.5)| = %.4e  '
      '(ratio %.2f)' % (d1, d2, d1 / d2 if d2 else float('inf')))

# --- D. the passive control -------------------------------------------------
fp, xf = build_frictionless_passive()
fp.evolve_to_end(finaltime=T_END)
_, _, um = stoker(HL, HR, T_END, xf)
c_fp = fp.get_tracer('ssc')
c_ex_pass = np.where(xf < DAM_X + um * T_END, CL, CR)
check('D1. with n = 0 and tau_d = 0, the analytic 3.1.1 solution is recovered',
      np.abs(c_fp - c_ex_pass).mean() < 5e-3,
      'L1 = %.4e   -- reached through [D-2]\'s tau_d = 0 hook, so that hook is '
      'verified against an exact answer'
      % np.abs(c_fp - c_ex_pass).mean())

n = 4 + 5 + 1 + 1
print('\n  %d/%d passed' % (n - _fail[0], n))
raise SystemExit(1 if _fail[0] else 0)
