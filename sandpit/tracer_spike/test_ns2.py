"""Ns=2: verify the tracer-major stride s*n / s*3n is indexed correctly."""
import numpy as np, anuga
from anuga import rectangular_cross_domain, Reflective_boundary
from anuga.shallow_water.sw_domain_openmp_ext import compute_fluxes_ext_central
LEN=1000.0; NX=NY=20
d = rectangular_cross_domain(NX,NY,len1=LEN,len2=LEN)
d.set_flow_algorithm('DE0'); d.set_low_froude(0); d.store=False
d.set_quantity('elevation',0.0)
d.set_quantity('stage', lambda x,y: np.where(x<LEN/2,2.0,0.5))
d.set_quantity('xmomentum',0.0); d.set_quantity('ymomentum',0.0)
d.set_boundary({t: Reflective_boundary(d) for t in d.get_boundary_tags()})
n=d.number_of_elements; bl=d.boundary_length; ns=2
d.number_of_tracers=ns
d.tracer_centroid_values=np.zeros((ns,n)); d.tracer_edge_values=np.zeros((ns,3*n))
d.tracer_boundary_values=np.zeros((ns,bl)); d.tracer_explicit_update=np.zeros((ns,n))
d.tracer_conserved_values=np.zeros((ns,n)); d.tracer_backup_values=np.zeros((ns,n))
d.beta_tracer=0.0; d._Domain_C_struct=None
for _ in d.evolve(yieldstep=2.0, finaltime=2.0): pass

K0, K1 = 1.0, 0.375
h=d.quantities['stage'].centroid_values-d.quantities['elevation'].centroid_values
for s,K in ((0,K0),(1,K1)):
    d.tracer_conserved_values[s,:]=np.maximum(h,0.0)*K
    d.tracer_centroid_values[s,:]=K; d.tracer_boundary_values[s,:]=K
d.tracer_explicit_update[:]=0.0
d.distribute_to_vertices_and_edges(); d.update_boundary()
compute_fluxes_ext_central(d, d.evolve_max_timestep, update_domain_c_struct=True)
seu=d.quantities['stage'].explicit_update
e0=np.max(np.abs(d.tracer_explicit_update[0]-K0*seu))
e1=np.max(np.abs(d.tracer_explicit_update[1]-K1*seu))
sc=max(np.max(np.abs(seu)),1e-300)
ok0 = e0/sc < 1e-14
ok1 = e1/sc < 1e-14
print(f"  [{'PASS' if ok0 else 'FAIL'}] tracer 0 (c=1.0)   vs stage_eu: rel err {e0/sc:.3e}")
print(f"  [{'PASS' if ok1 else 'FAIL'}] tracer 1 (c=0.375) rel err vs c*stage_eu: {e1/sc:.3e}")
print(f"  [{'PASS' if not np.array_equal(d.tracer_explicit_update[0],d.tracer_explicit_update[1]) else 'FAIL'}] the two tracer slots are independent (no aliasing)")
raise SystemExit(0 if (ok0 and ok1) else 1)
