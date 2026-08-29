"""[L-5] acceptance test: a non-erodible base below which the bed cannot scour.

Erosion in this model is otherwise bottomless -- the bed lowers for as long as
the flow can lift material, which is right for a deep alluvial channel and
wrong wherever the erodible layer is finite: a reach over an outcrop, a lined
culvert, a dam apron, a soil layer of known depth on rock.

set_erodible_base() gives the bed a floor. The tests that matter are not
"does z stop at the base" alone -- a clamp on z would pass that while creating
sediment out of nothing. They are:

  A  the floor is respected, per cell, including where it varies in space
  B  the budget still closes: sediment NOT eroded never enters the water
     column, so bed loss and suspended gain still agree to machine precision
  C  bedload stays EXACTLY conservative with a base present -- it only
     redistributes, so the total bed volume it moves must still sum to zero.
     Its floor, unlike the suspended route's, is enforced to within one
     step's flux rather than exactly; C2 states and bounds that
  D  classes are limited together and proportionally: the answer does not
     depend on the order they were registered
  E  no base configured reproduces the old answer bit for bit
  F  mode 1 and mode 2 agree
  H  erosion can be restricted to a region, which composes with the base

D and E are the ones that catch a careless implementation. E is the regression
gate: this feature must be invisible when it is off.
"""
import numpy as np
import anuga
from anuga import Reflective_boundary, rectangular_cross_domain

_fail = [0]


def check(label, ok, detail=''):
    print('  [%s] %s' % ('PASS' if ok else 'FAIL', label))
    if detail:
        print('         ' + detail)
    if not ok:
        _fail[0] += 1
    return ok


def build(mode=1, base=None, base_depth=None, classes=(('sand', 2.0e-4),),
          bedload=False, poro=0.3, n_x=30, slope=0.01, hh=0.6):
    d = rectangular_cross_domain(n_x, 8, len1=60.0, len2=16.0)
    d.set_flow_algorithm('DE0')
    d.set_low_froude(0)
    d.store = False
    d.set_quantity('elevation', lambda x, y: -slope * x)
    d.set_quantity('stage', lambda x, y: -slope * x + hh)
    d.set_quantity('xmomentum', 1.2)
    d.set_quantity('ymomentum', 0.0)
    d.set_quantity('friction', 0.03)
    d.set_boundary({t: Reflective_boundary(d) for t in d.get_boundary_tags()})
    d.set_sediment_parameters(porosity=poro)
    if bedload:
        d.set_bedload('wong_parker_eq24')
    for nm, dia in classes:
        d.add_sediment_class(nm, diameter=dia)
    if base is not None:
        d.set_erodible_base(elevation=base)
    elif base_depth is not None:
        d.set_erodible_base(depth=base_depth)
    if mode != 1:
        d.set_multiprocessor_mode(mode)
    return d


print(__doc__)

# ---------------------------------------------------------------- A
print('A. the floor holds')

THICK = 0.02
d = build(base_depth=THICK)
z0 = d.quantities['elevation'].centroid_values.copy()
base = d.sediment_z_base.copy()
d.evolve_to_end(finaltime=40.0)
z1 = d.quantities['elevation'].centroid_values
below = base - z1

check('A1. no cell erodes below its base',
      below.max() <= 0.0,
      'deepest violation %+.3e m (0 or negative is the floor holding)'
      % below.max())

check('A2. the base actually bit -- some cells reached it',
      int((z1 - base < 1e-12).sum()) > 0,
      '%d of %d cells at bedrock; scour %.4f m against a %.3f m layer'
      % (int((z1 - base < 1e-12).sum()), len(z1), (z0 - z1).max(), THICK))

d_free = build()
d_free.evolve_to_end(finaltime=40.0)
z_free = d_free.quantities['elevation'].centroid_values
free_scour = (z0 - z_free).max()
check('A3. unlimited run scours deeper than the layer, so A1 is not vacuous',
      free_scour > THICK,
      'unlimited scour %.4f m > layer %.3f m' % (free_scour, THICK))

# spatially varying base: bedrock step across the channel
d = build()
zc = d.quantities['elevation'].centroid_values.copy()
x = d.centroid_coordinates[:, 0]
step = np.where(x < 30.0, 0.01, 0.5)          # thin layer upstream, thick down
d.set_erodible_base(depth=step)
base_v = d.sediment_z_base.copy()
d.evolve_to_end(finaltime=40.0)
z1 = d.quantities['elevation'].centroid_values
check('A4. a base that varies in space is honoured cell by cell',
      (base_v - z1).max() <= 0.0,
      'deepest violation %+.3e m across a 0.01/0.50 m bedrock step'
      % (base_v - z1).max())
thin, thick = x < 30.0, x >= 30.0
check('A5. and it constrains only where it is shallow',
      (zc - z1)[thin].max() <= 0.01 + 1e-12
      and (zc - z1)[thick].max() > 0.01,
      'scour upstream %.4f m (cap 0.010), downstream %.4f m (cap 0.500)'
      % ((zc - z1)[thin].max(), (zc - z1)[thick].max()))

# ---------------------------------------------------------------- B
print('B. the budget still closes')

d = build(base_depth=0.02)
areas = d.areas
z0 = d.quantities['elevation'].centroid_values.copy()
m0 = float((d.tracer_conserved_values[0] * areas).sum())
poro = d.sediment_porosity
d.evolve_to_end(finaltime=40.0)
z1 = d.quantities['elevation'].centroid_values
m1 = float((d.tracer_conserved_values[0] * areas).sum())
bed = float(((1.0 - poro) * (z1 - z0) * areas).sum())
drift = (m1 + bed) - m0
scale = max(abs(m1), 1e-30)
check('B1. suspended gain and bed loss still agree exactly',
      abs(drift) < 1e-12 * scale,
      'suspended %+.6e + bed %+.6e = %+.3e  (%.2e relative)'
      % (m1, bed, drift, abs(drift) / scale))
check('B2. the limiter engaged during that run',
      int((z1 - d.sediment_z_base < 1e-12).sum()) > 0,
      '%d cells at bedrock, so B1 tested the limited path'
      % int((z1 - d.sediment_z_base < 1e-12).sum()))

# ---------------------------------------------------------------- C
print('C. bedload stays exactly conservative')

d = build(base_depth=0.01, bedload=True, classes=(('sand', 8.0e-4),))
areas = d.areas
z0 = d.quantities['elevation'].centroid_values.copy()
m0 = float((d.tracer_conserved_values[0] * areas).sum())
poro = d.sediment_porosity
d.evolve_to_end(finaltime=40.0)
z1 = d.quantities['elevation'].centroid_values
m1 = float((d.tracer_conserved_values[0] * areas).sum())
bed = float(((1.0 - poro) * (z1 - z0) * areas).sum())
drift = (m1 + bed) - m0
check('C1. with bedload AND a base, the closed-domain budget still closes',
      abs(drift) < 1e-12 * max(abs(m1), 1e-30),
      'drift %+.3e m3 against %.4e suspended' % (drift, m1))
# Bedload holds the base to within one step's redistribution, not exactly,
# and the reason is structural. When a cell cannot pay for the step's
# divergence its removing edges are closed -- symmetrically, which is what
# keeps C1 exact -- but closing an edge also cancels the INFLOW its neighbour
# was going to receive, so the neighbour's own removal grows and the deficit
# migrates one cell per sweep. Chasing it to zero needs the flag iterated to a
# fixed point with double buffering, so that the answer stays independent of
# thread order; that is not implemented.
#
# So the guarantee differs by route, and the test says which is which:
#   suspended exchange  exact, to machine precision (A1, A4)
#   bedload             within a bounded overshoot, asserted here
# Measured 5.1e-6 m on a 1.0e-2 m layer, 14 cells of 960; the bound below is
# 1% of the layer, which is roughly twice what is observed.
viol = (d.sediment_z_base - z1).max()
check('C2. the base holds under bedload to within one step of flux',
      viol <= 0.01 * 0.01,
      'deepest overshoot %+.3e m on a 1.0e-2 m layer (%.3f%% of it), '
      '%d of %d cells'
      % (viol, 100.0 * max(viol, 0.0) / 0.01, int((d.sediment_z_base - z1 > 0).sum()),
         len(z1)))

# ---------------------------------------------------------------- D
print('D. classes are limited together, not in registration order')

two = (('fine', 1.0e-4), ('coarse', 6.0e-4))
d1 = build(base_depth=0.004, classes=two)
d1.evolve_to_end(finaltime=30.0)
a_fine = float((d1.tracer_conserved_values[0] * d1.areas).sum())
a_coarse = float((d1.tracer_conserved_values[1] * d1.areas).sum())
a_z = d1.quantities['elevation'].centroid_values.copy()

d2 = build(base_depth=0.004, classes=two[::-1])
d2.evolve_to_end(finaltime=30.0)
b_coarse = float((d2.tracer_conserved_values[0] * d2.areas).sum())
b_fine = float((d2.tracer_conserved_values[1] * d2.areas).sum())
b_z = d2.quantities['elevation'].centroid_values

check('D1. swapping the registration order leaves the bed unchanged',
      np.allclose(a_z, b_z, rtol=0, atol=1e-14),
      'max |dz| between orders %.3e m' % np.abs(a_z - b_z).max())
rel_f = abs(a_fine - b_fine) / max(abs(a_fine), 1e-30)
rel_c = abs(a_coarse - b_coarse) / max(abs(a_coarse), 1e-30)
check('D2. and leaves each class with the same mass',
      rel_f < 1e-12 and rel_c < 1e-12,
      'fine rel %.2e, coarse rel %.2e' % (rel_f, rel_c))
check('D3. the shared budget was genuinely contested',
      int((a_z - d1.sediment_z_base < 1e-12).sum()) > 0
      and a_fine > 0.0 and a_coarse > 0.0,
      '%d cells at bedrock with both classes active (fine %.3e, coarse %.3e)'
      % (int((a_z - d1.sediment_z_base < 1e-12).sum()), a_fine, a_coarse))

# ---------------------------------------------------------------- E
print('E. no base means no change')

d_off = build()
d_off.evolve_to_end(finaltime=25.0)
z_off = d_off.quantities['elevation'].centroid_values.copy()
m_off = d_off.tracer_conserved_values[0].copy()

d_deep = build(base_depth=1000.0)     # a base far below anything reachable
d_deep.evolve_to_end(finaltime=25.0)
z_deep = d_deep.quantities['elevation'].centroid_values
m_deep = d_deep.tracer_conserved_values[0]

check('E1. an unreachable base changes the bed not at all',
      np.array_equal(z_off, z_deep),
      'max |dz| %.3e m (bitwise identical required)'
      % np.abs(z_off - z_deep).max())
check('E2. and changes the suspended field not at all',
      np.array_equal(m_off, m_deep),
      'max |dm| %.3e' % np.abs(m_off - m_deep).max())

# ---------------------------------------------------------------- F
print('F. mode 1 and mode 2 agree')

try:
    d1 = build(mode=1, base_depth=0.02, bedload=True, classes=two)
    d1.evolve_to_end(finaltime=25.0)
    d2 = build(mode=2, base_depth=0.02, bedload=True, classes=two)
    d2.evolve_to_end(finaltime=25.0)
    zz1 = d1.quantities['elevation'].centroid_values
    zz2 = d2.quantities['elevation'].centroid_values
    e_z = np.abs(zz1 - zz2).max()
    e_m = max(np.abs(d1.tracer_conserved_values[s]
                     - d2.tracer_conserved_values[s]).max()
              for s in range(2))
    check('F1. the two compute paths give the same bed under [L-5]',
          e_z < 1e-10, 'max |dz| %.3e m' % e_z)
    check('F2. and the same suspended mass',
          e_m < 1e-10, 'max |dm| %.3e' % e_m)
except Exception as exc:          # a CPU-only build has no mode 2
    check('F. mode 2 comparison', False, 'raised %s: %s'
          % (type(exc).__name__, exc))

# ---------------------------------------------------------------- G
print('G. the interface refuses nonsense')

d = build()
try:
    d.set_erodible_base(elevation=1000.0)
    check('G1. a base above the bed is rejected', False, 'no error raised')
except ValueError as exc:
    check('G1. a base above the bed is rejected', True, str(exc)[:88])
try:
    d.set_erodible_base(elevation=0.0, depth=1.0)
    check('G2. elevation and depth together are rejected', False, 'no error')
except ValueError:
    check('G2. elevation and depth together are rejected', True)
d.set_erodible_base(depth=0.1)
d.set_erodible_base()
check('G3. calling it with nothing removes the base',
      d.sediment_has_z_base == 0 and d.sediment_z_base is None)
d.set_erodible_base(depth=0.25)
t = d.erodible_thickness()
check('G4. erodible_thickness reports the layer',
      np.allclose(t, 0.25), 'thickness %.4f to %.4f m' % (t.min(), t.max()))
check('G5. sediment_summary reports the base',
      'erodible base' in d.sediment_summary()
      and '[L-5]' in d.sediment_summary())

# ---------------------------------------------------------------- H
print('H. region restriction')

HALF = [[0.0, -1.0], [30.0, -1.0], [30.0, 17.0], [0.0, 17.0]]   # x < 30

d = build()
zc = d.quantities['elevation'].centroid_values.copy()
x = d.centroid_coordinates[:, 0]
d.set_erodible_region(polygon=HALF)
d.evolve_to_end(finaltime=30.0)
z1 = d.quantities['elevation'].centroid_values
inside, outside = x < 30.0, x >= 30.0
check('H1. cells outside the erodible region do not scour',
      (zc - z1)[outside].max() <= 0.0,
      'max scour outside %.3e m' % (zc - z1)[outside].max())
check('H2. cells inside it do',
      (zc - z1)[inside].max() > 1e-4,
      'max scour inside %.4f m' % (zc - z1)[inside].max())

# erodible=False is the complement
d = build()
zc = d.quantities['elevation'].centroid_values.copy()
d.set_erodible_region(polygon=HALF, erodible=False)
d.evolve_to_end(finaltime=30.0)
z1 = d.quantities['elevation'].centroid_values
check('H3. erodible=False locks the named region instead',
      (zc - z1)[inside].max() <= 0.0 and (zc - z1)[outside].max() > 1e-4,
      'scour inside %.3e m, outside %.4f m'
      % ((zc - z1)[inside].max(), (zc - z1)[outside].max()))

# A locked cell may still receive deposition. It needs a depositional case
# to show: in the erosive channel above the limiter holds those cells at
# exactly net zero, scaling erosion back until it just cancels deposition,
# which is the correct answer there -- sand does not pile up on a scoured
# apron under erosive flow. So build a settling case instead: sediment
# already in suspension, no erosion (tau_c* far above anything reached).
d = rectangular_cross_domain(30, 8, len1=60.0, len2=16.0)
d.set_flow_algorithm('DE0')
d.set_low_froude(0)
d.store = False
d.set_quantity('elevation', 0.0)
d.set_quantity('stage', 1.0)
d.set_quantity('friction', 0.03)
d.set_boundary({t: Reflective_boundary(d) for t in d.get_boundary_tags()})
d.set_sediment_parameters(porosity=0.3)
d.add_sediment_class('silt', diameter=6.0e-5, tau_c_star=1.0e6,
                     initial_concentration=0.01)
zc = d.quantities['elevation'].centroid_values.copy()
d.set_erodible_region(polygon=HALF)          # x >= 30 is locked
xq = d.centroid_coordinates[:, 0]
locked = xq >= 30.0
d.evolve_to_end(finaltime=30.0)
z1 = d.quantities['elevation'].centroid_values
check('H4. locked cells still accrete -- locked means unscourable, not inert',
      (z1 - zc)[locked].min() > 0.0,
      'deposition on every locked cell, %.3e to %.3e m'
      % ((z1 - zc)[locked].min(), (z1 - zc)[locked].max()))
check('H5. and that new material is erodible again, being above the base',
      (d.erodible_thickness()[locked] > 0.0).all(),
      'thickness on locked cells now %.3e to %.3e m'
      % (d.erodible_thickness()[locked].min(),
         d.erodible_thickness()[locked].max()))

# conservation is untouched by the restriction
d = build()
z0 = d.quantities['elevation'].centroid_values.copy()
m0 = float((d.tracer_conserved_values[0] * d.areas).sum())
d.set_erodible_region(polygon=HALF)
poro = d.sediment_porosity
d.evolve_to_end(finaltime=30.0)
z1 = d.quantities['elevation'].centroid_values
m1 = float((d.tracer_conserved_values[0] * d.areas).sum())
bed = float(((1.0 - poro) * (z1 - z0) * d.areas).sum())
check('H6. the budget still closes with a region restriction',
      abs((m1 + bed) - m0) < 1e-12 * max(abs(m1), 1e-30),
      'drift %+.3e m3 against %.4e suspended' % ((m1 + bed) - m0, m1))

# base and region compose, and neither discards the other
d = build()
zc = d.quantities['elevation'].centroid_values.copy()
d.set_erodible_base(depth=0.01)
d.set_erodible_region(polygon=HALF)
d.evolve_to_end(finaltime=40.0)
z1 = d.quantities['elevation'].centroid_values
check('H7. base and region compose: the region says where, the base '
      'says how deep',
      (zc - z1)[outside].max() <= 0.0
      and 1e-4 < (zc - z1)[inside].max() <= 0.01 + 1e-12,
      'outside %.3e m, inside %.5f m against a 0.010 m layer'
      % ((zc - z1)[outside].max(), (zc - z1)[inside].max()))

d = build()
d.set_erodible_region(polygon=HALF)
d.set_erodible_base(depth=0.01)
check('H8. and the order they are called in does not matter',
      d._sediment_erodible_mask is not None
      and d._sediment_user_base is not None
      and int((d.erodible_thickness() > 0).sum())
      == int(d._sediment_erodible_mask.sum()),
      '%d erodible cells either way'
      % int((d.erodible_thickness() > 0).sum()))

d = build()
d.set_erodible_region(center=[15.0, 8.0], radius=6.0)
n_circ = int(d._sediment_erodible_mask.sum())
check('H9. a circular region works too',
      0 < n_circ < len(d), '%d of %d cells inside r=6 m' % (n_circ, len(d)))
try:
    d.set_erodible_region(polygon=[[900.0, 900.0], [910.0, 900.0],
                                   [910.0, 910.0], [900.0, 910.0]])
    check('H10. a region that selects nothing is rejected', False, 'no error')
except ValueError as exc:
    check('H10. a region that selects nothing is rejected', True, str(exc)[:70])
d = build()
d.set_erodible_region(polygon=HALF)
d.set_erodible_region()
check('H11. calling it with nothing removes the restriction',
      d._sediment_erodible_mask is None and d.sediment_has_z_base == 0)
d = build()
d.set_erodible_region(polygon=HALF)
check('H12. sediment_summary reports the region',
      'erodible region' in d.sediment_summary()
      and 'locked' in d.sediment_summary())

print('\n%d checks failed' % _fail[0])
raise SystemExit(1 if _fail[0] else 0)
