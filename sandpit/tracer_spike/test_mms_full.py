"""FULL Method of Manufactured Solutions -- RDycore's own, hydrodynamics included.

RDy26 section 3.3 takes its manufactured hydrodynamics from Bisht et al. (2025),
which is paywalled -- but the specification is in the RDycore repository itself
(BSD 2-clause), in `docs/common/mms.md` and
`driver/tests/sediment/sediment_mms_conv_study.yaml`. Those are reproduced in
References/rdycore_mms/ and used here verbatim:

    h  = H (1 + sin(Kx) sin(Ky)) exp(t/T)      H = 0.005
    u  = U cos(Kx) sin(Ky) exp(t/T)            U = V = 0.025
    v  = V sin(Kx) cos(Ky) exp(t/T)            Z = 0.0025
    z  = Z sin(Kx) sin(Ky)                     C = 0.5
    c0 = C (1 + sin(Kx) sin(Ky)) exp(+t/T)     T = 20 s
    c1 = C (1 + sin(Kx) sin(Ky)) exp(-t/T)     K = pi/5

This is the full test rather than the still-water one of test_mms.py: the
hydrodynamics EVOLVE, forced by manufactured mass and momentum sources applied
through an operator, so the sediment is advected by a spatially and temporally
varying flow. It therefore measures SPATIAL convergence of the sediment
equation, which nothing else here does.

Source terms are derived symbolically with sympy at import, not by hand. As a
guard against mis-transcribing the manufactured solution, every derivative
sympy computes is checked against the one RDycore states explicitly; a mismatch
fails the test rather than silently poisoning the source terms.

Two deliberate departures, both stated rather than buried:

  * FRICTIONLESS. RDycore manufacture a Manning field n(x,y) too. ANUGA applies
    friction semi-implicitly, so an exactly cancelling manufactured source is
    not straightforward, and n = 0 keeps the manufactured solution exact. The
    convergence RATE, which is what is being compared, does not depend on it.
  * FIXED BED. z is time-independent in their manufactured solution, so bed
    evolution is off, as their MMS also has it.

RDycore's own achieved rates, from their YAML, are the target:

    h  L1 0.94    hu/hv L1 0.91    c0 L1 0.94    c1 L1 0.93
"""
import numpy as np
import sympy as sp
import anuga
from anuga import rectangular_cross_domain
from anuga.abstract_2d_finite_volumes.generic_boundary_conditions import Boundary

_fail = [0]

# ---------------------------------------------------------------- constants
Hs, Ts, Us, Vs, Zs, Cs = 0.005, 20.0, 0.025, 0.025, 0.0025, 0.5
Ks = float(np.pi / 5.0)
GRAV = 9.81
V_S = None            # settling velocity, filled once the domain exists
STOP, DT = 5.0, 0.01


def check(label, ok, detail=''):
    print('  [%s] %s' % ('PASS' if ok else 'FAIL', label))
    if detail:
        print('         ' + detail)
    if not ok:
        _fail[0] += 1
    return ok


# ------------------------------------------------- symbolic derivation ------
_x, _y, _t = sp.symbols('x y t', real=True)
_H, _T, _U, _V, _Z, _C, _K, _al, _vs, _g = sp.symbols(
    'H T U V Z C K alpha v_s g', real=True)
_E = sp.exp(_t / _T)
_s = sp.sin(_K * _x) * sp.sin(_K * _y)
_h = _H * (1 + _s) * _E
_u = _U * sp.cos(_K * _x) * sp.sin(_K * _y) * _E
_v = _V * sp.sin(_K * _x) * sp.cos(_K * _y) * _E
_z = _Z * _s
_c = _C * (1 + _s) * sp.exp(_al * _t / _T)

# Guard: sympy's derivatives must equal the ones RDycore writes down.
_stated = [
    (sp.diff(_h, _x), _H * _K * sp.sin(_K * _y) * sp.cos(_K * _x) * _E),
    (sp.diff(_h, _y), _H * _K * sp.sin(_K * _x) * sp.cos(_K * _y) * _E),
    (sp.diff(_h, _t), _H / _T * (1 + _s) * _E),
    (sp.diff(_u, _x), -_U * _K * sp.sin(_K * _x) * sp.sin(_K * _y) * _E),
    (sp.diff(_u, _y), _U * _K * sp.cos(_K * _x) * sp.cos(_K * _y) * _E),
    (sp.diff(_v, _x), _K * _V * sp.cos(_K * _x) * sp.cos(_K * _y) * _E),
    (sp.diff(_v, _y), -_K * _V * sp.sin(_K * _x) * sp.sin(_K * _y) * _E),
    (sp.diff(_z, _x), _Z * _K * sp.cos(_K * _x) * sp.sin(_K * _y)),
    (sp.diff(_z, _y), _Z * _K * sp.sin(_K * _x) * sp.cos(_K * _y)),
]
_DERIVS_OK = all(sp.simplify(a - b) == 0 for a, b in _stated)

_S_h = sp.diff(_h, _t) + sp.diff(_h * _u, _x) + sp.diff(_h * _v, _y)
_S_hu = (sp.diff(_h * _u, _t) + sp.diff(_h * _u * _u + _g * _h * _h / 2, _x)
         + sp.diff(_h * _u * _v, _y) + _g * _h * sp.diff(_z, _x))
_S_hv = (sp.diff(_h * _v, _t) + sp.diff(_h * _u * _v, _x)
         + sp.diff(_h * _v * _v + _g * _h * _h / 2, _y) + _g * _h * sp.diff(_z, _y))
_m = _h * _c
_S_ms = sp.diff(_m, _t) + sp.diff(_u * _m, _x) + sp.diff(_v * _m, _y) + _c * _vs

_sub = {_H: Hs, _T: Ts, _U: Us, _V: Vs, _Z: Zs, _C: Cs, _K: Ks, _g: GRAV}
_L = lambda e: sp.lambdify((_x, _y, _t), e.subs(_sub), 'numpy')
h_ex, u_ex, v_ex = _L(_h), _L(_u), _L(_v)
z_ex = sp.lambdify((_x, _y), _z.subs(_sub), 'numpy')
S_h, S_hu, S_hv = _L(_S_h), _L(_S_hu), _L(_S_hv)
c_ex = {a: _L(_c.subs(_al, a)) for a in (1, -1)}


def make_S_ms(alpha, vs):
    return _L(_S_ms.subs(_al, alpha).subs(_vs, vs))


print(__doc__)
check('A1. sympy derivatives match every derivative RDycore states',
      _DERIVS_OK,
      'guards against mis-transcribing the manufactured solution')


# ------------------------------------------------------------- boundary ----
class MMS_boundary(Boundary):
    """Dirichlet on the exact solution -- what MMS requires at the boundary."""

    def __init__(self, domain):
        Boundary.__init__(self)
        self.domain = domain

    def evaluate_segment(self, domain, segment_edges):
        if segment_edges is None or domain is None:
            return
        ids = segment_edges
        vol = domain.boundary_cells[ids]
        edg = domain.boundary_edges[ids]
        em = domain.get_edge_midpoint_coordinates()
        xb, yb = em[3 * vol + edg, 0], em[3 * vol + edg, 1]
        t = domain.get_time()
        hb, ub, vb = h_ex(xb, yb, t), u_ex(xb, yb, t), v_ex(xb, yb, t)
        domain.quantities['stage'].boundary_values[ids] = z_ex(xb, yb) + hb
        domain.quantities['xmomentum'].boundary_values[ids] = hb * ub
        domain.quantities['ymomentum'].boundary_values[ids] = hb * vb
        for s, a in enumerate(ALPHAS):
            domain.tracer_boundary_values[s][ids] = c_ex[a](xb, yb, t)


ALPHAS = (1, -1)


class MMS_source_operator(anuga.operators.base_operator.Operator):
    """Applies the manufactured mass, momentum and sediment sources."""

    def __init__(self, domain, sms):
        anuga.operators.base_operator.Operator.__init__(self, domain)
        self.x = domain.centroid_coordinates[:, 0]
        self.y = domain.centroid_coordinates[:, 1]
        self.sms = sms

    def __call__(self):
        d = self.domain
        dt = d.get_timestep()
        if dt <= 0.0:
            return
        t = d.get_time()
        x, y = self.x, self.y
        d.quantities['stage'].centroid_values[:] += dt * S_h(x, y, t)
        d.quantities['xmomentum'].centroid_values[:] += dt * S_hu(x, y, t)
        d.quantities['ymomentum'].centroid_values[:] += dt * S_hv(x, y, t)
        for name, f in self.sms:
            d.set_tracer_source(name, f(x, y, t))

    def parallel_safe(self):
        return True

    def statistics(self):
        return 'MMS sources'

    def timestepping_statistics(self):
        return ''


def run(nxy, dt=None):
    d = rectangular_cross_domain(nxy, nxy, len1=5.0, len2=5.0)
    d.set_flow_algorithm('DE0')
    d.set_low_froude(0)
    d.store = False
    x, y = d.centroid_coordinates[:, 0], d.centroid_coordinates[:, 1]
    d.set_quantity('elevation', z_ex(x, y), location='centroids')
    d.set_quantity('stage', z_ex(x, y) + h_ex(x, y, 0.0), location='centroids')
    d.set_quantity('xmomentum', h_ex(x, y, 0.0) * u_ex(x, y, 0.0),
                   location='centroids')
    d.set_quantity('ymomentum', h_ex(x, y, 0.0) * v_ex(x, y, 0.0),
                   location='centroids')
    d.set_quantity('friction', 0.0)                # see the docstring
    d.evolve_max_timestep = DT if dt is None else dt
    d.sediment_bed_evolution = False               # z is time-independent
    d.sediment_c_max = 100.0                       # do not let limiters clip
    d.sediment_c_pack = 100.0

    names = []
    for i, a in enumerate(ALPHAS):
        nm = 'c%d' % i
        d.add_sediment_class(nm, diameter=1.0e-4, tau_c_star=0.0,
                             auto_operator=False,
                             initial_concentration=c_ex[a](x, y, 0.0))
        names.append(nm)
    vs = d.sediment_settling_velocity[0]

    d.set_boundary({tg: MMS_boundary(d) for tg in d.get_boundary_tags()})

    # order matters: sources must be set before the sediment operator consumes
    # them, and operators run in registration order.
    MMS_source_operator(d, [(nm, make_S_ms(a, vs))
                            for nm, a in zip(names, ALPHAS)])
    from anuga.operators.sediment_operator import Sediment_operator
    Sediment_operator(d)

    d.evolve_to_end(finaltime=STOP)

    a_ = d.areas
    out = {}
    hn = np.maximum(d.quantities['stage'].centroid_values
                    - d.quantities['elevation'].centroid_values, 0.0)
    fields = {'h': (hn, h_ex(x, y, STOP)),
              'hu': (d.quantities['xmomentum'].centroid_values,
                     h_ex(x, y, STOP) * u_ex(x, y, STOP)),
              'hv': (d.quantities['ymomentum'].centroid_values,
                     h_ex(x, y, STOP) * v_ex(x, y, STOP))}
    for i, a in enumerate(ALPHAS):
        fields['c%d' % i] = (d.get_tracer('c%d' % i), c_ex[a](x, y, STOP))
    for k, (num, exact) in fields.items():
        e = np.abs(num - exact)
        out[k] = (float((e * a_).sum() / a_.sum()),
                  float(np.sqrt((e**2 * a_).sum() / a_.sum())),
                  float(e.max()))
    return out


# Refine SPACE AND TIME TOGETHER. This is the whole ballgame and the first
# attempt got it wrong: refining dx three times while holding dt = 0.01 fixed
# leaves a CONSTANT temporal error, which comes to dominate as the spatial
# error shrinks, and the apparent rate decays -- 0.802 then 0.509. The error
# field gives it away, being spatially UNIFORM (boundary band 4.46e-4 against
# interior 4.90e-4 at the finest mesh), so it is not a boundary artefact and
# not a spatial one. Halving dt with dx recovers first order.
#
# RDycore's own YAML holds dt fixed across refinements and still reports 0.94,
# so their spatial error presumably still dominates at their resolutions. Ours
# does not, so the refinement path has to be diagonal.
GRIDS = ((10, 0.02), (20, 0.01), (40, 0.005))    # dx = 0.5, 0.25, 0.125

res = {n: run(n, dt=dt) for n, dt in GRIDS}
MESHES = [n for n, _ in GRIDS]

RD = {'h': 0.94, 'hu': 0.91, 'hv': 0.91, 'c0': 0.94, 'c1': 0.93}
print('\n     field      dx=0.500      dx=0.250      dx=0.125     rate   RDycore')
rates = {}
for f in ('h', 'hu', 'hv', 'c0', 'c1'):
    L1 = [res[n][f][0] for n in MESHES]
    r = [np.log(L1[i] / L1[i + 1]) / np.log(2.0) for i in range(len(L1) - 1)]
    rates[f] = float(np.mean(r))
    print('     %-6s  %.4e  %.4e  %.4e   %.3f    %.2f'
          % (f, L1[0], L1[1], L1[2], rates[f], RD[f]))

check('B1. every field converges under refinement',
      all(rates[f] > 0.5 for f in rates),
      'observed L1 rates: ' + ', '.join('%s %.3f' % (f, rates[f])
                                        for f in rates))
check('B2. sediment converges at first order, as RDycore report',
      all(0.8 < rates[f] < 1.3 for f in ('c0', 'c1')),
      'c0 %.3f (RDycore 0.94)   c1 %.3f (RDycore 0.93)'
      % (rates['c0'], rates['c1']))
check('B3. sediment converges no worse than the hydrodynamics carrying it',
      min(rates['c0'], rates['c1']) > 0.8 * min(rates['h'], rates['hu']),
      'min sediment %.3f vs min hydrodynamic %.3f'
      % (min(rates['c0'], rates['c1']), min(rates['h'], rates['hu'])))
check('B4. absolute error is small at the finest mesh',
      all(res[MESHES[-1]][f][0] < 0.05 * max(abs(Cs), Hs)
          for f in ('c0', 'c1')),
      'c0 L1 %.3e   c1 L1 %.3e' % (res[MESHES[-1]]['c0'][0],
                                   res[MESHES[-1]]['c1'][0]))

# Kept as a standing warning, not a check: holding dt fixed misreports the rate.
print('\n     diagnostic -- refining SPACE ONLY, dt fixed at 0.01:')
prev = None
for nxy in MESHES:
    L1 = run(nxy, dt=0.01)['c0'][0]
    print('       nxy=%2d  L1(c0) %.4e   %s'
          % (nxy, L1, '' if prev is None else 'rate %.3f'
             % (np.log(prev / L1) / np.log(2.0))))
    prev = L1
print('     -> the rate decays because the temporal error is held constant.')

n = 1 + 4
print('\n  %d/%d passed' % (n - _fail[0], n))
raise SystemExit(1 if _fail[0] else 0)
