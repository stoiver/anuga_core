#!/usr/bin/env python3
"""Ns=1 correctness tests for the tracer flux-kernel spike.

Calls the flux kernel directly rather than going through evolve(), because the
time-integration half of the tracer (applying explicit_update to h*c, and
reconstructing edge values) is Phase-1 work and not part of this spike. What is
implemented -- the flux itself -- can still be tested decisively.

Three properties, in increasing subtlety:

  A. CONSISTENCY  c == 1 everywhere  =>  tracer_eu == stage_eu bit-for-bit.
     m = h*c = h, so dm/dt must equal dh/dt. Exercises the whole flux path.

  B. LINEARITY    c == K everywhere  =>  tracer_eu == K * stage_eu.

  C. UPWINDING    c = 1 upstream, 0 downstream. Cells with c=0 that receive
     inflow from a c=1 neighbour must GAIN tracer; cells with c=0 whose
     neighbours are all c=0 must have exactly zero tendency. If the donor were
     picked from the wrong side, the first set would be 0.

  D. CONSERVATION sum(tracer_eu * area) == 0 to round-off on a closed domain:
     the same edge flux is debited from one cell and credited to its neighbour.
"""
import numpy as np
import anuga
from anuga import rectangular_cross_domain, Reflective_boundary
from anuga.shallow_water.sw_domain_openmp_ext import compute_fluxes_ext_central

NX = NY = 20          # 1600 triangles
LEN = 1000.0


def build(ns=1):
    d = rectangular_cross_domain(NX, NY, len1=LEN, len2=LEN)
    d.set_flow_algorithm('DE0')
    d.set_low_froude(0)
    d.store = False
    d.set_quantity('elevation', 0.0)
    d.set_quantity('stage', lambda x, y: np.where(x < LEN / 2, 2.0, 0.5))
    d.set_quantity('xmomentum', 0.0)
    d.set_quantity('ymomentum', 0.0)
    d.set_boundary({t: Reflective_boundary(d) for t in d.get_boundary_tags()})

    n = d.number_of_elements
    bl = d.boundary_length
    d.number_of_tracers = ns
    d.tracer_centroid_values = np.zeros((ns, n), dtype=float)
    d.tracer_edge_values = np.zeros((ns, 3 * n), dtype=float)
    d.tracer_boundary_values = np.zeros((ns, bl), dtype=float)
    d.tracer_explicit_update = np.zeros((ns, n), dtype=float)
    d.tracer_conserved_values = np.zeros((ns, n), dtype=float)
    d.tracer_backup_values = np.zeros((ns, n), dtype=float)
    d.beta_tracer = 0.0          # first order: isolates the flux from the limiter
    d._Domain_C_struct = None
    return d


def prime(d):
    """Advance far enough that there is real flow through interior edges."""
    for _ in d.evolve(yieldstep=2.0, finaltime=2.0):
        pass
    return d


def fluxes(d):
    """Run one flux evaluation with the current tracer state."""
    d.tracer_explicit_update[:] = 0.0
    # distribute_to_vertices_and_edges has already run inside evolve; recompute
    # so stage/xmom/ymom edge values match the current centroids.
    d.distribute_to_vertices_and_edges()
    d.update_boundary()
    compute_fluxes_ext_central(d, d.evolve_max_timestep, update_domain_c_struct=True)
    return (d.quantities['stage'].explicit_update.copy(),
            d.tracer_explicit_update[0].copy())


def _seed(d, cfield):
    """Seed the conserved m = h*c; edge values are derived, not assigned."""
    h = (d.quantities['stage'].centroid_values
         - d.quantities['elevation'].centroid_values)
    d.tracer_conserved_values[0, :] = np.maximum(h, 0.0) * cfield
    d.tracer_centroid_values[0, :] = cfield


def set_uniform(d, val):
    _seed(d, np.full(d.number_of_elements, float(val)))
    d.tracer_boundary_values[0, :] = val


results = []


def check(name, ok, detail):
    results.append((name, ok, detail))
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}\n         {detail}")


# ---------------------------------------------------------------- A ---
d = prime(build())
set_uniform(d, 1.0)
stage_eu, tr_eu = fluxes(d)
maxdiff = float(np.max(np.abs(stage_eu - tr_eu)))
scale = max(float(np.max(np.abs(stage_eu))), 1e-300)
check("A. consistency  c=1 => tracer_eu == stage_eu (to round-off)",
      maxdiff / scale < 1e-14,
      f"max|diff| = {maxdiff:.3e}  rel = {maxdiff/scale:.3e}   "
      f"nonzero cells = {int(np.count_nonzero(stage_eu))}")

# ---------------------------------------------------------------- B ---
K = 0.375
set_uniform(d, K)
stage_eu, tr_eu = fluxes(d)
err = np.max(np.abs(tr_eu - K * stage_eu))
scale = max(np.max(np.abs(K * stage_eu)), 1e-300)
check(f"B. linearity    c={K} => tracer_eu == c*stage_eu",
      err / scale < 1e-14, f"max relative error = {err/scale:.3e}")

# ---------------------------------------------------------------- C ---
xc = d.centroid_coordinates[:, 0]
upstream = xc < LEN / 2
_seed(d, np.where(upstream, 1.0, 0.0))
d.tracer_boundary_values[0, :] = 0.0
stage_eu, tr_eu = fluxes(d)

neigh = d.neighbours
c = d.tracer_centroid_values[0]
has_hot_neighbour = np.zeros(len(c), dtype=bool)
for i in range(3):
    nb = neigh[:, i]
    ok = nb >= 0
    has_hot_neighbour[ok] |= (c[nb[ok]] > 0.5)

cold_isolated = (c < 0.5) & ~has_hot_neighbour
cold_adjacent = (c < 0.5) & has_hot_neighbour

isolated_zero = np.all(tr_eu[cold_isolated] == 0.0)
gained = int(np.count_nonzero(tr_eu[cold_adjacent] > 0.0))
check("C. upwinding    cold cells far from the front have zero tendency",
      isolated_zero,
      f"{int(cold_isolated.sum())} isolated cold cells, "
      f"max|tendency| = {float(np.max(np.abs(tr_eu[cold_isolated]))) if cold_isolated.any() else 0.0:.3e}")
check("C. upwinding    cold cells at the front gain tracer from upstream",
      gained > 0,
      f"{gained} of {int(cold_adjacent.sum())} front cells have tracer_eu > 0")

# ---------------------------------------------------------------- D ---
rng = np.random.default_rng(0)
cv = rng.random(d.number_of_elements)
_seed(d, cv)
d.tracer_boundary_values[0, :] = 0.0
stage_eu, tr_eu = fluxes(d)
areas = d.areas
net = float(np.sum(tr_eu * areas))
gross = float(np.sum(np.abs(tr_eu) * areas))
rel = abs(net) / max(gross, 1e-300)
check("D. conservation sum(tracer_eu * area) == 0 on a closed domain",
      rel < 1e-13, f"net = {net:.6e}   gross = {gross:.6e}   ratio = {rel:.3e}")

# ---------------------------------------------------------------------
print()
npass = sum(1 for _, ok, _ in results if ok)
print(f"  {npass}/{len(results)} passed")
raise SystemExit(0 if npass == len(results) else 1)
