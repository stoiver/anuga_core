"""Entrainment of bed sediment into a tank of still water.

The bed is flat on one half of the tank and slopes gently on the other, and
is held fixed. With the depth-slope shear closure [T-7] the bed shear stress
is a function of depth and bed slope alone, so the flat half sits below the
Shields threshold (nothing may happen there) while the sloped half entrains
sand until deposition balances it. There is no flow, so every cell is a
closed system and relaxes to its equilibrium concentration along the
closed-form curve in analytical_sediment_erosion.py. The run checks the
entrainment law [E-1]/[E-2] and its threshold, the depth-slope closure, and
the balance with deposition [D-1] in the conserved sediment mass [G-3]. See
the analytical module for why the bed is fixed.
"""
import json
import numpy as np
import anuga
from anuga import Domain, myid, finalize, distribute

output_dir = '.'
output_file = 'sediment_erosion'

L, W = 20.0, 4.0          # tank size, m
dx = 1.0
L_flat = 10.0             # x < L_flat: flat bed (control); beyond: sloped
S_bed = 1.0e-3            # bed slope of the eroding half
h0 = 1.0                  # water depth over the flat half, m
diameter = 5.0e-4         # 0.5 mm sand: v_s ~ 0.1 m/s, so the relaxation
                          # time h0/v_s is ~10 s and a short run resolves it;
                          # tau* = h S/(R d) ~ 1.2 on the slope, ~30 tau_c*
d_star = 1.0              # well-mixed near-bed profile factor: D = v_s c
porosity = 0.30
finaltime = 120.0
yieldstep = 2.0


def bed(x, y):
    return -S_bed * np.maximum(x - L_flat, 0.0)


args = anuga.get_args()
alg = args.alg
verbose = args.verbose

if myid == 0:
    # The kink in the bed lies on a mesh line (L_flat is a multiple of dx),
    # so no cell straddles it: every cell is exactly flat or exactly sloped.
    points, vertices, boundary = anuga.rectangular_cross(int(L / dx), int(W / dx), L, W)
    domain = Domain(points, vertices, boundary)
    domain.set_name(output_file)
    domain.set_datadir(output_dir)
    domain.set_flow_algorithm(alg)
    domain.set_quantity('elevation', bed)
    domain.set_quantity('friction', 0.0)
    domain.set_quantity('stage', h0)
else:
    domain = None

domain = distribute(domain)

Br = anuga.Reflective_boundary(domain)
domain.set_boundary({'left': Br, 'right': Br, 'top': Br, 'bottom': Br})

# The sediment: one fraction, no initial load, the Shields (non-cohesive)
# entrainment law with its default critical stress, the depth-slope shear
# closure so that still water carries a bed stress, and a fixed bed.
domain.initialize_sediment_operator(porosity=porosity, bed_evolution=False)
domain.set_shear_closure('depth_slope')
domain.set_bed_material('noncohesive')
domain.set_deposition(law='d_star', near_bed='constant')
domain.add_sediment_fraction('sand', diameter=diameter, d_star=d_star,
                             initial_concentration=0.0)

def cell_slopes(domain):
    """The per-cell bed slope the kernel sees, [T-7]: the gradient of the
    bed edge values by the divergence theorem over the cell's own edges."""
    ze = domain.quantities['elevation'].edge_values
    nrm = domain.normals
    el = domain.edgelengths
    gx = (ze * nrm[:, 0::2] * el).sum(axis=1) / domain.areas
    gy = (ze * nrm[:, 1::2] * el).sum(axis=1) / domain.areas
    return np.hypot(gx, gy)


if myid == 0:
    v_s = float(domain.sediment_settling_velocity[0])
    if verbose:
        print(domain.sediment_summary())
        print('settling velocity %.4g m/s, relaxation time %.1f s'
              % (v_s, h0 / (d_star * v_s)))

from anuga.validation_utilities import save_parameters_tex
save_parameters_tex(domain)

S_cell = None
for t in domain.evolve(yieldstep=yieldstep, finaltime=finaltime):
    if myid == 0 and verbose:
        print(domain.timestepping_statistics())
    if S_cell is None:
        # Recorded after the first step, once the solver has reconstructed
        # the bed edge values (stage - height) it works with. In the
        # interior this is exactly the constructed slope; along the far wall
        # and the kink the limiter reduces it, and the validation checks
        # those cells against the slope they were actually given.
        S_cell = cell_slopes(domain)
        h_cell = (domain.quantities['stage'].centroid_values
                  - domain.quantities['elevation'].centroid_values)

if myid == 0:
    prm = {'h0': h0, 'S_bed': S_bed, 'L_flat': L_flat, 'diameter': diameter,
           'v_s': v_s, 'R': float(domain.sediment_R[0]),
           'tau_c_star': float(domain.sediment_tau_c_star[0]),
           'gamma0': float(domain.sediment_gamma0),
           'd_star': d_star, 'porosity': porosity,
           'finaltime': finaltime, 'yieldstep': yieldstep,
           'S_cell': S_cell.tolist(), 'h_cell': h_cell.tolist(),
           'x_cell': domain.centroid_coordinates[:, 0].tolist(),
           'y_cell': domain.centroid_coordinates[:, 1].tolist()}
    with open('erosion_parameters.json', 'w') as f:
        json.dump(prm, f, indent=1)

domain.sww_merge(delete_old=True)
finalize()
