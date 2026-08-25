#!/usr/bin/env python3
"""Phase 1: tracer edge reconstruction tests.

Properties a reconstruction must have, in the order they matter:

  1. FIRST ORDER      beta_tracer = 0  =>  edge value == centroid value exactly.
  2. CONSTANT EXACT   c uniform => every edge value equals it, at any beta.
                      (a reconstruction that fails this is broken outright)
  3. MONOTONE         c a step function => no edge value outside the global
                      [min,max] of the centroids. This is what protects
                      positivity of concentration.
  4. POSITIVITY       c >= 0 everywhere => every edge value >= 0.
  5. SECOND ORDER     a smooth linear field is reproduced exactly (a linear
                      function is the fixed point of a 2nd-order reconstruction),
                      and a smooth nonlinear field converges at ~O(h^2) while
                      first order converges at ~O(h).
"""
import numpy as np
import anuga
from anuga import rectangular_cross_domain, Reflective_boundary
from anuga.shallow_water.sw_domain_openmp_ext import extrapolate_second_order_sw

LEN = 1000.0


def build(nxy, beta_tracer, wet=True):
    d = rectangular_cross_domain(nxy, nxy, len1=LEN, len2=LEN)
    d.set_flow_algorithm('DE0')
    d.set_low_froude(0)
    d.store = False
    d.set_quantity('elevation', 0.0)
    # uniformly wet and deep so the wet/dry hfactor does not throttle beta
    d.set_quantity('stage', 5.0 if wet else 0.0)
    d.set_quantity('xmomentum', 0.0)
    d.set_quantity('ymomentum', 0.0)
    d.set_boundary({t: Reflective_boundary(d) for t in d.get_boundary_tags()})
    n = d.number_of_elements
    d.number_of_tracers = 1
    d.beta_tracer = beta_tracer
    d.tracer_centroid_values = np.zeros((1, n))
    d.tracer_edge_values = np.zeros((1, 3 * n))
    d.tracer_boundary_values = np.zeros((1, d.boundary_length))
    d.tracer_explicit_update = np.zeros((1, n))
    d.tracer_conserved_values = np.zeros((1, n))
    d.tracer_backup_values = np.zeros((1, n))
    d._Domain_C_struct = None
    return d


def reconstruct(d, cfield):
    # c is derived from the conserved m each substep, so seed m = h*c.
    h = (d.quantities['stage'].centroid_values
         - d.quantities['elevation'].centroid_values)
    d.tracer_conserved_values[0, :] = np.maximum(h, 0.0) * cfield
    d.tracer_centroid_values[0, :] = cfield
    d.tracer_edge_values[0, :] = np.nan          # poison, so we detect non-writes
    extrapolate_second_order_sw(d, update_domain_c_struct=True)
    ev = d.tracer_edge_values[0].copy()
    assert not np.isnan(ev).any(), "some edge values were never written"
    return ev


results = []
def check(name, ok, detail):
    results.append(ok)
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}\n         {detail}")


# --- 1. first order -------------------------------------------------------
d = build(12, beta_tracer=0.0)
xc = d.centroid_coordinates[:, 0]
c = np.sin(2 * np.pi * xc / LEN) * 0.4 + 0.5
ev = reconstruct(d, c)
exact = np.repeat(c, 3)
check("1. beta=0 gives first order (edge == centroid to round-off)",
      np.max(np.abs(ev - exact)) < 1e-15,
      f"max|diff| = {np.max(np.abs(ev-exact)):.3e}  (c is derived as m/h, so the "
      f"round-trip costs ~1 ulp; bit-exactness no longer applies)")

# --- 2. constant reproduced exactly --------------------------------------
d = build(12, beta_tracer=1.0)
ev = reconstruct(d, np.full(d.number_of_elements, 0.7))
check("2. uniform c reproduced exactly at beta=1",
      np.allclose(ev, 0.7, rtol=0, atol=1e-15), f"max|diff from 0.7| = {np.max(np.abs(ev-0.7)):.3e}")

# --- 3/4. monotone + positive on a step ----------------------------------
d = build(24, beta_tracer=1.0)
xc = d.centroid_coordinates[:, 0]
c = np.where(xc < LEN / 2, 1.0, 0.0)
ev = reconstruct(d, c)
lo, hi = c.min(), c.max()
over = float(max(ev.max() - hi, 0.0))
under = float(max(lo - ev.min(), 0.0))
check("3. step function creates no new extrema (monotone)",
      over <= 1e-14 and under <= 1e-14,
      f"edge range [{ev.min():.6f}, {ev.max():.6f}] vs centroid [{lo}, {hi}]  "
      f"overshoot={over:.2e} undershoot={under:.2e}")
check("4. positivity: no negative edge concentration",
      ev.min() >= 0.0, f"min edge value = {ev.min():.3e}")

# --- 5a. linear field reproduced exactly ---------------------------------
# Only meaningful on cells with no boundary edge. Where a cell touches the
# boundary, surrogate_neighbours substitutes the cell itself, degrading the
# gradient stencil -- this is ANUGA's existing behaviour and stage shows the
# identical error on the identical cells (verified separately).
d = build(16, beta_tracer=1.0)
cc = d.centroid_coordinates
c = 0.5 + 0.3 * (cc[:, 0] / LEN) + 0.2 * (cc[:, 1] / LEN)
ev = reconstruct(d, c)
ec = d.get_edge_midpoint_coordinates()
exact = 0.5 + 0.3 * (ec[:, 0] / LEN) + 0.2 * (ec[:, 1] / LEN)
err = np.abs(ev - exact)
nb_edge = np.repeat(d.number_of_boundaries, 3)
interior = nb_edge == 0
err_int = float(err[interior].max())
err_bdy = float(err[~interior].max())
check("5a. linear field reconstructed exactly on interior cells",
      err_int < 1e-12,
      f"interior max|error| = {err_int:.3e} over {int(interior.sum())} edges; "
      f"boundary-touching cells {err_bdy:.3e} (degraded stencil, matches stage)")

# --- 5b. order of accuracy on smooth fields -----------------------------
# Two fields on purpose. A limiter cannot be both unconditionally monotone and
# cleanly 2nd order at a smooth extremum (Godunov); it clips there and drops
# locally to 1st order. So the honest order test uses a monotone field, and the
# extremal field is reported to document the trade-off tests 3/4 depend on.
def converge(f, beta, sizes=(8, 16, 32, 64)):
    es = []
    for nxy in sizes:
        d = build(nxy, beta_tracer=beta)
        cc = d.centroid_coordinates
        ev = reconstruct(d, f(cc[:, 0], cc[:, 1]))
        ec = d.get_edge_midpoint_coordinates()
        m = np.repeat(d.number_of_boundaries, 3) == 0
        es.append(float(np.sqrt(np.mean((ev[m] - f(ec[m, 0], ec[m, 1]))**2))))
    return es, [np.log2(es[i]/es[i+1]) for i in range(len(es)-1)]

f_mono = lambda X, Y: 0.5 + 0.4 * (X / LEN)**2
f_extr = lambda X, Y: 0.5 + 0.4 * np.sin(2*np.pi*X/LEN) * np.cos(2*np.pi*Y/LEN)

e1, o1 = converge(f_mono, 0.0)
e2, o2 = converge(f_mono, 1.0)
e3, o3 = converge(f_extr, 1.0)

print("\n  5b. convergence, interior cells, RMS edge error:")
print(f"      {'cells/side':>10} {'beta=0':>12} {'beta=1':>12}   (smooth monotone c)")
for i, nxy in enumerate((8, 16, 32, 64)):
    print(f"      {nxy:>10} {e1[i]:>12.3e} {e2[i]:>12.3e}")
print(f"      {'order':>10} {np.mean(o1):>12.2f} {np.mean(o2):>12.2f}")

check("5b. beta=0 is 1st order, beta=1 is 2nd order on a smooth monotone field",
      1.3 > np.mean(o1) > 0.7 and np.mean(o2) > 1.95,
      f"observed order: first={np.mean(o1):.2f}  second={np.mean(o2):.2f} "
      f"(per level {[f'{x:.2f}' for x in o2]})")

print(f"\n      For reference, a field WITH interior extrema at beta=1 gives order "
      f"{np.mean(o3):.2f}\n      (per level {[f'{x:.2f}' for x in o3]}) — the limiter clips at the crests.")
print( "      That is the price of tests 3 and 4, and it is the correct trade-off:")
print( "      an unlimited scheme would overshoot and drive c negative.\n")

print()
print(f"  {sum(results)}/{len(results)} passed")
raise SystemExit(0 if all(results) else 1)
