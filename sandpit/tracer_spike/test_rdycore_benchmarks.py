"""RDycore-sediment (RDy26) passive-transport benchmarks, spec 3.1.

Feng et al. (2026), EGUsphere 2026-4859, section 3.1. These reproduce the
convective-step tests of Le et al. (2015), originally from Audusse & Bristeau
(2003) for passive pollutant transport coupled to shallow water flow.

RDycore itself is not built here -- it is C/PETSc and that is a large lift --
but neither case needs it: both compare against ANALYTICAL solutions, which is
what makes them worth having. RDycore is BSD-2-clause, so reading the paper's
setup and reproducing it is unencumbered.

Bed exchange is disabled, as they do it: RDycore sets the critical deposition
shear to zero and the Partheniades coefficient to zero. Ours is the same
statement in our API -- tau_c_star = 0 disables entrainment, d_star = 0
disables deposition -- so the sediment equation reduces to conservative
advection by the hydrodynamic field.

  A. LAKE AT REST (3.1.2). Nonflat bed, h + z = 1, u = 0, a discontinuous SSC
     field. Nothing should move: not the free surface, not the sediment.
  B. DAM-BREAK PASSIVE TRANSPORT (3.1.1). The physically interesting one. The
     analytical solution has a rarefaction, a shock and a contact
     discontinuity, and the SSC must change ONLY across the contact, remaining
     flat across both nonlinear water waves.
"""
import numpy as np
from scipy.optimize import brentq
import anuga
from anuga import Reflective_boundary, rectangular_cross_domain

G = 9.81
_fail = [0]


def check(label, ok, detail=''):
    print('  [%s] %s' % ('PASS' if ok else 'FAIL', label))
    if detail:
        print('         ' + detail)
    if not ok:
        _fail[0] += 1
    return ok


print(__doc__)

# ===========================================================================
# A. Lake at rest (RDy26 3.1.2)
# ===========================================================================
# 20 m domain, dx = 0.1 m, h + z = 1, u = 0, c = 1 g/L over 8 < x < 12.
# The bump shape is not given in the paper (it comes from Le et al. 2015); any
# nonflat bed exercises the same property, so a smooth bump is used here and
# the choice is noted rather than hidden.
LAKE_T = 100.0
d = rectangular_cross_domain(200, 2, len1=20.0, len2=2.0)
d.set_flow_algorithm('DE0')
d.set_low_froude(0)
d.store = False
x = d.centroid_coordinates[:, 0]
bump = 0.2 * np.exp(-((x - 10.0) / 1.0)**2)
d.set_quantity('elevation', bump, location='centroids')
d.set_quantity('stage', 1.0)
d.set_quantity('friction', 0.0)
d.set_boundary({t: Reflective_boundary(d) for t in d.get_boundary_tags()})
c0 = np.where((x > 8.0) & (x < 12.0), 1.0e-3, 0.0)     # 1 g/L = 1e-3 by volume
d.add_sediment_class('ssc', diameter=1e-4, tau_c_star=0.0, d_star=0.0,
                     initial_concentration=c0)
w0 = d.quantities['stage'].centroid_values.copy()
z0 = d.quantities['elevation'].centroid_values.copy()
d.evolve_to_end(finaltime=LAKE_T)
w1 = d.quantities['stage'].centroid_values
c1 = d.get_tracer('ssc')
u1 = d.quantities['xmomentum'].centroid_values

check('A1. free surface is preserved over the bump (h + z = 1)',
      np.allclose(w1, w0, rtol=0, atol=1e-12),
      'max|stage - 1| = %.3e after %.0f s' % (np.abs(w1 - w0).max(), LAKE_T))
check('A2. no spurious momentum is generated',
      np.abs(u1).max() < 1e-12,
      'max|xmomentum| = %.3e' % np.abs(u1).max())
check('A3. the discontinuous SSC field does not move at all',
      np.allclose(c1, c0, rtol=0, atol=1e-14),
      'max|c - c0| = %.3e   (u = 0, no diffusion, so transport must be zero)'
      % np.abs(c1 - c0).max())
check('A4. and the bed did not move (exchange disabled)',
      np.allclose(d.quantities['elevation'].centroid_values, z0,
                  rtol=0, atol=1e-14))

# ===========================================================================
# B. Dam-break passive transport (RDy26 3.1.1)
# ===========================================================================
HL, HR, CL, CR = 1.0, 0.5, 0.7, 0.5
DAM_X, DAM_T, LENGTH = 1000.0, 240.0, 2000.0


def stoker(hl, hr, t, xs):
    """Classical wet-bed dam-break (Stoker). Returns h(x) and the contact speed.

    Left rarefaction, right shock, contact between them. Solved for the star
    state by matching the rarefaction and shock relations.
    """
    cl, cr = np.sqrt(G * hl), np.sqrt(G * hr)

    def f(hm):
        # left: rarefaction;  right: shock
        u_raref = -2.0 * (np.sqrt(G * hm) - cl)
        u_shock = (hm - hr) * np.sqrt(0.5 * G * (hm + hr) / (hm * hr))
        return u_raref - u_shock

    hm = brentq(f, hr * (1 + 1e-12), hl * (1 - 1e-12), xtol=1e-14, rtol=1e-15)
    um = -2.0 * (np.sqrt(G * hm) - cl)
    cm = np.sqrt(G * hm)
    shock_s = hm * um / (hm - hr)          # Rankine-Hugoniot mass jump

    h = np.empty_like(xs)
    xi = (xs - DAM_X) / t
    for i, s in enumerate(xi):
        if s <= -cl:
            h[i] = hl
        elif s <= um - cm:
            h[i] = ((-s + 2.0 * cl) / 3.0)**2 / G      # rarefaction fan
        elif s <= shock_s:
            h[i] = hm
        else:
            h[i] = hr
    return h, um, hm, shock_s


def run_dambreak(nx):
    """One dam-break run at nx cells in x. Returns x, h, c and the mesh size."""
    dd = rectangular_cross_domain(nx, 2, len1=LENGTH, len2=10.0)
    dd.set_flow_algorithm('DE0')
    dd.set_low_froude(0)
    dd.store = False
    xd = dd.centroid_coordinates[:, 0]
    dd.set_quantity('elevation', 0.0)
    dd.set_quantity('stage', np.where(xd < DAM_X, HL, HR), location='centroids')
    dd.set_quantity('friction', 0.0)
    dd.set_boundary({t: Reflective_boundary(dd) for t in dd.get_boundary_tags()})
    dd.add_sediment_class('ssc', diameter=1e-4, tau_c_star=0.0, d_star=0.0,
                          initial_concentration=np.where(xd < DAM_X, CL, CR))
    dd.evolve_to_end(finaltime=DAM_T)
    h = np.maximum(dd.quantities['stage'].centroid_values
                   - dd.quantities['elevation'].centroid_values, 0.0)
    return xd, h, dd.get_tracer('ssc'), LENGTH / nx


# The solution is DISCONTINUOUS -- a shock and a contact -- so the max norm is
# the wrong acceptance metric: any finite-volume scheme smears a jump over a
# few cells, giving a max error of O(jump) at every resolution. L1 is the right
# norm, and the decisive statement is that it CONVERGES under refinement.
res = {}
for nx in (400, 800):
    xd, h_num, c_num, dx = run_dambreak(nx)
    h_ex, um, hm, shock_s = stoker(HL, HR, DAM_T, xd)
    c_ex = np.where(xd < DAM_X + um * DAM_T, CL, CR)
    res[nx] = dict(dx=dx, x=xd, h=h_num, c=c_num, h_ex=h_ex, c_ex=c_ex,
                   um=um, hm=hm, shock_s=shock_s,
                   L1h=np.abs(h_num - h_ex).mean(),
                   L1c=np.abs(c_num - c_ex).mean())

r = res[800]
print('         star state: h* = %.6f m   u* = %.6f m/s   shock speed %.6f m/s'
      % (r['hm'], r['um'], r['shock_s']))
print('         contact at x = %.1f m at t = %.0f s'
      % (DAM_X + r['um'] * DAM_T, DAM_T))
print('         dx = %.2f m   L1(h) = %.4e   L1(c) = %.4e'
      % (res[400]['dx'], res[400]['L1h'], res[400]['L1c']))
print('         dx = %.2f m   L1(h) = %.4e   L1(c) = %.4e'
      % (res[800]['dx'], res[800]['L1h'], res[800]['L1c']))

check('B1. water depth agrees with the analytical Stoker solution in L1',
      res[800]['L1h'] < 5e-3,
      'L1 = %.4e m at dx = %.2f m' % (res[800]['L1h'], res[800]['dx']))
check('B2. SSC agrees with the passive-contact solution in L1',
      res[800]['L1c'] < 5e-3,
      'L1 = %.4e at dx = %.2f m' % (res[800]['L1c'], res[800]['dx']))
check('B3. and both CONVERGE under refinement',
      res[800]['L1h'] < res[400]['L1h'] and res[800]['L1c'] < res[400]['L1c'],
      'h: %.4e -> %.4e  (%.2fx)   c: %.4e -> %.4e  (%.2fx)'
      % (res[400]['L1h'], res[800]['L1h'], res[400]['L1h'] / res[800]['L1h'],
         res[400]['L1c'], res[800]['L1c'], res[400]['L1c'] / res[800]['L1c']))

# The property this case exists to test: SSC changes ONLY at the contact. It
# must be flat across both nonlinear water waves.
xd, c_num, um, hm, shock_s = r['x'], r['c'], r['um'], r['hm'], r['shock_s']
raref = (xd > DAM_X - np.sqrt(G * HL) * DAM_T * 0.9) & \
        (xd < DAM_X + (um - np.sqrt(G * hm)) * DAM_T * 0.9)
shock_zone = (xd > DAM_X + um * DAM_T * 1.1) & \
             (xd < DAM_X + shock_s * DAM_T * 0.9)
check('B4. SSC is flat across the RAREFACTION (a nonlinear water wave)',
      raref.sum() > 10 and np.ptp(c_num[raref]) < 1e-3,
      'over %d cells in the fan, c varies by %.3e -- sediment rides the flow, '
      'not the water waves' % (raref.sum(), np.ptp(c_num[raref])))
check('B5. SSC is flat between the contact and the SHOCK',
      shock_zone.sum() > 10 and np.ptp(c_num[shock_zone]) < 1e-3,
      'over %d cells, c varies by %.3e'
      % (shock_zone.sum(), np.ptp(c_num[shock_zone])))
check('B6. SSC develops no new extrema',
      c_num.min() >= CR - 1e-9 and c_num.max() <= CL + 1e-9,
      'c in [%.6f, %.6f], initial states were %.1f and %.1f'
      % (c_num.min(), c_num.max(), CR, CL))

n = 4 + 6
print('\n  %d/%d passed' % (n - _fail[0], n))
raise SystemExit(1 if _fail[0] else 0)
