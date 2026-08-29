"""Spec 7 acceptance test: angle-of-repose relaxation.

FG21 §2.2.4. Where the centroid-to-centroid bed slope exceeds a critical angle,
bed material is diffused downslope until it does not. FG21 are explicit that
this is a NUMERICAL HEURISTIC, not physics -- real slope failures are advective
-- and that it suppresses knickpoint retreat that may be real. It exists to
stop the rest of the model breaking on over-steep slopes.

The spec sets four implementation requirements, and this tests all four:

  A  it relaxes: an over-steep bed ends at or below the critical angle
  B  MASS IS CONSERVED -- material removed from an over-steep cell is deposited
     on its neighbours, never discarded. This is the requirement that separates
     it from the older sanddune operator, which drops the material.
  C  the sweep cap is hard, and hitting it is REPORTED rather than swallowed
  D  it composes with [L-5]: a cell may not slump away material it is not
     allowed to lose

plus the standing obligations: off by default and then bitwise inert (E),
mode 1 and mode 2 agree (F), and the interface refuses nonsense (G).
"""
import math
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


def cone(x, y, height=6.0, radius=8.0, cx=30.0, cy=8.0):
    """A cone far steeper than any repose angle -- about 37 degrees."""
    r = np.sqrt((x - cx)**2 + (y - cy)**2)
    return np.where(r < radius, height * (1.0 - r / radius), 0.0)


def build(mode=1, angle=None, relax=1.0, max_sweeps=50, dry=True,
          base_depth=None, n_x=40):
    d = rectangular_cross_domain(n_x, 12, len1=60.0, len2=16.0)
    d.set_flow_algorithm('DE0')
    d.set_low_froude(0)
    d.store = False
    x, y = d.centroid_coordinates[:, 0], d.centroid_coordinates[:, 1]
    d.set_quantity('elevation', cone(x, y), location='centroids')
    # A dry cone: repose is a bed process and this isolates it from any
    # hydrodynamic response.
    d.set_quantity('stage', -1.0 if dry else 8.0)
    d.set_quantity('friction', 0.03)
    d.set_boundary({t: Reflective_boundary(d) for t in d.get_boundary_tags()})
    d.set_sediment_parameters(porosity=0.3)
    d.add_sediment_class('sand', diameter=2.0e-4)
    if angle is not None:
        d.set_angle_of_repose(angle, relax=relax, max_sweeps=max_sweeps)
    if base_depth is not None:
        d.set_erodible_base(depth=base_depth)
    if mode != 1:
        d.set_multiprocessor_mode(mode)
    return d


def max_slope(d):
    """Steepest centroid-to-centroid bed slope, as an angle in degrees."""
    z = d.quantities['elevation'].centroid_values
    cc = d.centroid_coordinates
    nb = d.neighbours
    worst = 0.0
    for i in range(3):
        j = nb[:, i]
        ok = j >= 0
        dz = np.abs(z[ok] - z[j[ok]])
        dist = np.sqrt(((cc[ok] - cc[j[ok]])**2).sum(axis=1))
        good = dist > 0
        worst = max(worst, float((dz[good] / dist[good]).max()))
    return math.degrees(math.atan(worst))


print(__doc__)

ANGLE = 30.0

# ---------------------------------------------------------------- A
print('A. it relaxes an over-steep bed')

d0 = build()
before = max_slope(d0)
# A cold start from a badly over-steep bed is the slow case: this is an
# explicit diffusion solve and it took 793 sweeps to converge here. A running
# model needs a handful per step; see the note in set_angle_of_repose.
d = build(angle=ANGLE, max_sweeps=2000)
d.evolve_to_end(finaltime=1.0)
after = max_slope(d)
check('A1. the cone starts steeper than the critical angle',
      before > ANGLE + 1.0,
      'initial max slope %.2f degrees against a %.1f degree limit'
      % (before, ANGLE))
# The kernel's convergence criterion is a 1e-3 relative tolerance on the
# threshold SLOPE, not on the angle, because convergence is asymptotic and a
# strict test never terminates. At 30 degrees that is at most 30.04.
TOL_DEG = math.degrees(math.atan(math.tan(math.radians(ANGLE)) * 1.001))
check('A2. and ends at or below it, to the kernel tolerance',
      after <= TOL_DEG + 1e-3,
      'relaxed to %.4f degrees, tolerance allows %.4f' % (after, TOL_DEG))

z0 = d0.quantities['elevation'].centroid_values
z1 = d.quantities['elevation'].centroid_values
check('A3. by lowering the peak and raising the apron, not by flattening '
      'everything',
      z1.max() < z0.max() and z1.max() > 0.5 * z0.max(),
      'peak %.3f -> %.3f m, footprint spread from %d to %d cells above 1 cm'
      % (z0.max(), z1.max(), int((z0 > 0.01).sum()), int((z1 > 0.01).sum())))

# ---------------------------------------------------------------- B
print('B. mass is conserved')

d = build(angle=ANGLE, max_sweeps=2000)
areas = d.areas
v0 = float((d.quantities['elevation'].centroid_values * areas).sum())
d.evolve_to_end(finaltime=1.0)
v1 = float((d.quantities['elevation'].centroid_values * areas).sum())
check('B1. bed volume is unchanged: what leaves an over-steep cell arrives '
      'on its neighbours',
      abs(v1 - v0) < 1e-11 * max(abs(v0), 1.0),
      'volume %.10e -> %.10e, drift %+.3e m3 (%.2e relative)'
      % (v0, v1, v1 - v0, abs(v1 - v0) / max(abs(v0), 1.0)))

# The comparison that gives B1 its meaning: the older operator does NOT do this.
check('B2. and this is the difference from sanddune_erosion_operator, which '
      'lowers the cell and discards the material',
      True,
      'stated, not measured here; see its update_quantities(), which sets '
      'elev_c without crediting any neighbour')

# ---------------------------------------------------------------- C
print('C. the sweep cap is hard and reported')

d = build(angle=ANGLE, max_sweeps=3)
op = [o for o in d.fractional_step_operators
      if type(o).__name__ == 'Sediment_operator'][0]
d.evolve_to_end(finaltime=0.2)
check('C1. a cap too small to converge is hit',
      op.repose_cap_hits > 0,
      'cap hit on %d timestep(s), %d sweeps last step'
      % (op.repose_cap_hits, op.repose_sweeps))
check('C2. and the bed is then still over-steep -- the cap is not silently '
      'papering over it',
      max_slope(d) > ANGLE,
      'still %.2f degrees against the %.1f degree limit'
      % (max_slope(d), ANGLE))
check('C3. sweeps never exceed the cap',
      op.repose_sweeps <= 3, 'last sweep count %d, cap 3' % op.repose_sweeps)

d = build(angle=ANGLE, max_sweeps=2000)
op = [o for o in d.fractional_step_operators
      if type(o).__name__ == 'Sediment_operator'][0]
d.evolve_to_end(finaltime=1.0)
check('C4. with a workable cap it converges and stops early',
      op.repose_cap_hits == 0 and op.repose_sweeps < 2000,
      'no cap hits; last step needed %d sweeps of 2000' % op.repose_sweeps)

# ---------------------------------------------------------------- D
print('D. it composes with [L-5]')

# The layer has to be thinner than the relaxation actually wants, or the
# test proves nothing: unconstrained, this cone lowers its peak only 0.3425 m,
# so a 0.5 m layer never binds and a "constrained" run reproduces the free one
# exactly. 0.1 m does bind.
LAYER = 0.1
d_free = build(angle=ANGLE, max_sweeps=2000)
zc = d_free.quantities['elevation'].centroid_values.copy()
d_free.evolve_to_end(finaltime=1.0)
drop_free = (zc - d_free.quantities['elevation'].centroid_values).max()

d = build(angle=ANGLE, max_sweeps=2000, base_depth=LAYER)
base = d.sediment_z_base.copy()
d.evolve_to_end(finaltime=1.0)
z1 = d.quantities['elevation'].centroid_values
check('D1. relaxation cannot slump a cell below its erodible base',
      (base - z1).max() <= 0.0,
      'deepest violation %+.3e m' % (base - z1).max())
check('D2. and the constraint bit: a thin layer holds back a slump the '
      'unconstrained run makes',
      (zc - z1).max() < drop_free - 1e-6
      and int((d.erodible_thickness() <= 1e-9).sum()) > 0,
      'peak lowered %.4f m against %.4f m unconstrained; %d cells at base'
      % ((zc - z1).max(), drop_free,
         int((d.erodible_thickness() <= 1e-9).sum())))

d = build(angle=ANGLE, max_sweeps=2000)
d.set_erodible_region(center=[30.0, 8.0], radius=3.0, erodible=False)
zc = d.quantities['elevation'].centroid_values.copy()
locked = ~d._sediment_erodible_mask
d.evolve_to_end(finaltime=1.0)
z1 = d.quantities['elevation'].centroid_values
check('D3. a locked cell does not slump away, though it may receive',
      (zc - z1)[locked].max() <= 1e-14,
      '%d locked cells, max loss %+.3e m' % (int(locked.sum()),
                                             (zc - z1)[locked].max()))

# ---------------------------------------------------------------- E
print('E. off by default')

d_off = build()
d_off.evolve_to_end(finaltime=1.0)
d_none = build(angle=None)
d_none.evolve_to_end(finaltime=1.0)
check('E1. the default changes nothing, bitwise',
      np.array_equal(d_off.quantities['elevation'].centroid_values,
                     d_none.quantities['elevation'].centroid_values),
      'identical elevation fields required')
check('E2. and the over-steep cone survives untouched',
      max_slope(d_off) > ANGLE,
      'still %.2f degrees, as it should be with relaxation off'
      % max_slope(d_off))

# ---------------------------------------------------------------- F
print('F. mode 1 and mode 2 agree')

d1 = build(mode=1, angle=ANGLE, max_sweeps=2000)
d1.evolve_to_end(finaltime=1.0)
d2 = build(mode=2, angle=ANGLE, max_sweeps=2000)
d2.evolve_to_end(finaltime=1.0)
e = np.abs(d1.quantities['elevation'].centroid_values
           - d2.quantities['elevation'].centroid_values).max()
check('F1. the Jacobi sweep gives the same bed on both compute paths',
      e < 1e-10, 'max |dz| %.3e m' % e)

# ---------------------------------------------------------------- G
print('G. the interface')

d = build()
for bad, why in ((dict(angle=0.0), 'zero'), (dict(angle=90.0), '90'),
                 (dict(angle=-5.0), 'negative')):
    try:
        d.set_angle_of_repose(**bad)
        ok = (bad['angle'] == 0.0 and d.sediment_repose_tan == 0.0)
        check('G1. angle=%s handled' % why, ok,
              'zero disables; out-of-range must raise')
    except ValueError:
        check('G1. angle=%s rejected' % why, bad['angle'] != 0.0)
try:
    d.set_angle_of_repose(30.0, relax=0.0)
    check('G2. relax outside (0, 1] is rejected', False, 'no error')
except ValueError:
    check('G2. relax outside (0, 1] is rejected', True)
try:
    d.set_angle_of_repose(30.0, max_sweeps=0)
    check('G3. max_sweeps < 1 is rejected', False, 'no error')
except ValueError:
    check('G3. max_sweeps < 1 is rejected', True)
d.set_angle_of_repose(35.0)
check('G4. the angle round-trips through tan',
      abs(math.degrees(math.atan(d.sediment_repose_tan)) - 35.0) < 1e-9,
      'stored tan %.6f = %.6f degrees'
      % (d.sediment_repose_tan,
         math.degrees(math.atan(d.sediment_repose_tan))))
check('G5. sediment_summary reports it',
      'angle of repose' in d.sediment_summary()
      and '35.0 degrees' in d.sediment_summary())
d.set_angle_of_repose(None)
check('G6. and it can be switched off again',
      d.sediment_repose_tan == 0.0
      and 'off' in [ln.split(':')[1].strip().split()[0]
                    for ln in d.sediment_summary().splitlines()
                    if 'angle of repose' in ln][0])

print('\n%d checks failed' % _fail[0])
raise SystemExit(1 if _fail[0] else 0)
