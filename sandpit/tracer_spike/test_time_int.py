#!/usr/bin/env python3
"""Phase 1: tracer time integration, driven through a real evolve().

The tracer's conserved variable is m = h*c. It is integrated by
update/backup/saxpy exactly like stage; c is derived from it each substep,
exactly as height is derived from stage.

Properties, strongest first:

  A. FREE-STREAM PRESERVATION.  A uniform concentration must stay *exactly*
     uniform for all time, whatever the flow does. Any error in the m<->c
     conversion, the reconstruction, the upwinding or the RK combination shows
     up here immediately. This is the single sharpest test of the whole chain.

  B. MASS CONSERVATION.  sum(m*area) constant over the whole run on a closed
     domain, to round-off.

  C. POSITIVITY.  c >= 0 throughout. Not enforced by clamping (that would break
     B); it should emerge from upwinding under CFL. Measured, not assumed.

  D. BOUNDEDNESS.  c never exceeds its initial maximum -- no new extrema.

  E. TRANSPORT.  The tracer actually moves with the water rather than sitting
     still: a blob released upstream must show up downstream.
"""
import numpy as np
import anuga
from anuga import rectangular_cross_domain, Reflective_boundary
from anuga.shallow_water.sw_domain_openmp_ext import update_Domain_C_struct

LEN = 1000.0


def build(nxy=30, beta_tracer=1.0, mode=1):
    d = rectangular_cross_domain(nxy, nxy, len1=LEN, len2=LEN)
    d.set_flow_algorithm('DE0')
    d.set_low_froude(0)
    d.store = False
    d.set_quantity('elevation', 0.0)
    d.set_quantity('stage', lambda x, y: np.where(x < LEN / 2, 2.0, 0.5))
    d.set_quantity('xmomentum', 0.0)
    d.set_quantity('ymomentum', 0.0)
    d.set_boundary({t: Reflective_boundary(d) for t in d.get_boundary_tags()})
    n = d.number_of_elements
    d.number_of_tracers = 1
    d.beta_tracer = beta_tracer
    for name, shape in (('tracer_centroid_values', (1, n)),
                        ('tracer_edge_values', (1, 3 * n)),
                        ('tracer_boundary_values', (1, d.boundary_length)),
                        ('tracer_explicit_update', (1, n)),
                        ('tracer_conserved_values', (1, n)),
                        ('tracer_backup_values', (1, n))):
        setattr(d, name, np.zeros(shape))
    if mode >= 1:
        d.set_multiprocessor_mode(mode)
    # The cached C struct is built once and evolve() never passes
    # update_domain_c_struct=True, so a tracer registered afterwards would be
    # invisible to the kernels. Invalidate it: the next call rebuilds it with the
    # tracer arrays wired. (Phase-1 API note: domain.add_tracer() must do this.)
    d._Domain_C_struct = None
    return d


def set_c(d, cfield):
    """Seed the conserved m = h*c from a concentration field."""
    h = d.quantities['stage'].centroid_values - d.quantities['elevation'].centroid_values
    d.tracer_conserved_values[0, :] = np.maximum(h, 0.0) * cfield
    d.tracer_centroid_values[0, :] = cfield
    d.tracer_boundary_values[0, :] = 0.0


def mass(d):
    return float((d.tracer_conserved_values[0] * d.areas).sum())


def conc(d):
    h = d.quantities['stage'].centroid_values - d.quantities['elevation'].centroid_values
    wet = h > 1e-3
    c = np.zeros_like(h)
    c[wet] = d.tracer_conserved_values[0][wet] / h[wet]
    return c, wet


results = []
def check(name, ok, detail):
    results.append(ok)
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}\n         {detail}")


# ---------------------------------------------------------------- A ---
# Uniform c must stay exactly uniform. Reflective boundaries: no inflow, so the
# boundary concentration never enters; the interior must hold c0 by itself.
C0 = 0.35
d = build()
set_c(d, np.full(d.number_of_elements, C0))
d.tracer_boundary_values[0, :] = C0
m0 = mass(d)
for _ in d.evolve(yieldstep=10.0, finaltime=30.0):
    pass
c, wet = conc(d)
dev = float(np.max(np.abs(c[wet] - C0)))
check("A. free-stream: uniform c stays uniform through a dam break",
      dev < 1e-12,
      f"max|c - {C0}| = {dev:.3e} over {int(wet.sum())} wet cells, {d.number_of_steps} steps")

# ---------------------------------------------------------------- B ---
m1 = mass(d)
rel = abs(m1 - m0) / abs(m0)
check("B. mass conservation over the whole run",
      rel < 1e-12, f"m0 = {m0:.10e}  m1 = {m1:.10e}  relative drift = {rel:.3e}")

# ------------------------------------------------------------- C/D/E ---
# A blob on the deep upstream side, carried by the dam break.
d = build()
xc = d.centroid_coordinates[:, 0]
c_init = np.where(xc < LEN / 2, 1.0, 0.0)   # the whole upstream reservoir
set_c(d, c_init)
m0 = mass(d)
cmax_init = float(c_init.max())
_m_area0 = d.tracer_conserved_values[0] * d.areas
com_init = float((_m_area0 * xc).sum() / _m_area0.sum())

cmin_seen, cmax_seen = 0.0, 0.0
for _ in d.evolve(yieldstep=10.0, finaltime=120.0):
    c, wet = conc(d)
    if wet.any():
        cmin_seen = min(cmin_seen, float(c[wet].min()))
        cmax_seen = max(cmax_seen, float(c[wet].max()))

m1 = mass(d)
rel = abs(m1 - m0) / abs(m0)
check("B2. mass conservation while the tracer is transported",
      rel < 1e-12, f"relative drift = {rel:.3e} over {d.number_of_steps} steps")
check("C. positivity: c never goes negative (not clamped, emergent)",
      cmin_seen >= -1e-14, f"min c seen over the whole run = {cmin_seen:.3e}")
check("D. boundedness: c never exceeds its initial maximum",
      cmax_seen <= cmax_init + 1e-12,
      f"max c seen = {cmax_seen:.6f} vs initial max {cmax_init:.6f}")

# Centre of mass is a far better transport probe than a threshold at an
# arbitrary station: it is defined everywhere, needs no tuning, and moves
# monotonically with the flow.
m_area = d.tracer_conserved_values[0] * d.areas
com_final = float((m_area * xc).sum() / m_area.sum())
check("E. transport: the tracer centre of mass moves downstream with the flow",
      com_final > com_init + 1.0,
      f"x_com {com_init:.2f} m -> {com_final:.2f} m  (moved {com_final-com_init:+.2f} m "
      f"in {d.number_of_steps} steps)")

print()
print(f"  {sum(results)}/{len(results)} passed")
raise SystemExit(0 if all(results) else 1)
