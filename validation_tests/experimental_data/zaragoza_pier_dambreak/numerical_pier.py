"""Dam-break over an erodible sand bed around a pier: the Zaragoza Kinect
experiments of Segovia-Burillo et al. (2026), J. Hydraulic Research 64(2),
data DOI 10.5281/zenodo.17777387.

A 6 m x 0.24 m PVC flume (Manning n = 0.010) fed by a pneumatic gate from an
81 cm x 157 cm reservoir holding 8 cm of water; the flume bed is dry. From
1.0 to 2.5 m past the gate the bed is a 5 cm layer of sand (0.7 and 1.3 mm,
50/50, rho_s 2650, porosity 0.34) levelled with the PVC; a pier stands on
the centreline 1.6 m from the gate, a 3 cm cylinder (P1) or a 3 x 6.5 cm
rounded rectangle (P2). The flat reach ends at 3.0 m and the flume then
drops at 4 % to the sediment traps. The 8 cm dam-break is released three
times over the evolving bed; a Kinect sensor records the bed after each and
the water surface during each.

Selected by environment: ZARAGOZA_CASE (P1 default, or P2) and
ZARAGOZA_EVENTS (1 default, up to 3). Each event runs T_EVENT seconds from
the gate opening; between events the reservoir is refilled to 8 cm, the
flume dried, and the bed kept.
"""
import json
import os

import numpy as np

import anuga
from anuga import Reflective_boundary, Transmissive_boundary, myid, finalize

args = anuga.get_args()
alg = args.alg
verbose = args.verbose

CASE = os.environ.get('ZARAGOZA_CASE', 'P1')
EVENTS = int(os.environ.get('ZARAGOZA_EVENTS', '1'))
T_EVENT = float(os.environ.get('ZARAGOZA_T_EVENT', '5.0'))     # s per dam-break
output_file = 'pier_%s' % CASE

# --- geometry (m); x from the gate along the flume, y from the right wall ---
W = 0.24
X_RES, W_RES = -1.57, 0.81            # reservoir, centred on the flume
Y_RES0 = 0.5 * (W - W_RES)
X_SAND0, X_SAND1 = 1.0, 2.5           # erodible reach
SAND_DEPTH = 0.05
X_FLAT, X_END, S_DOWN = 3.0, 3.5, 0.04
X_PIER, Y_PIER = 1.6, 0.5 * W
H0 = 0.08                             # reservoir depth
N_PVC, N_SAND = 0.010, 0.015          # Manning; sand from the paper's Table 1
POROSITY, D_SAND = 0.34, 1.0e-3       # one fraction at the mix's geometric mean
REPOSE = 32.0                         # degrees

# mesh resolution (m^2 per triangle)
A_COARSE, A_SAND, A_PIER = 1.0e-3, 1.0e-4, 2.0e-5


def pier_polygon(case, n=16):
    r = 0.015
    if case == 'P1':
        th = np.linspace(0.0, 2 * np.pi, 2 * n, endpoint=False)
        return [(X_PIER + r * np.cos(a), Y_PIER + r * np.sin(a)) for a in th]
    # P2: a stadium 6.5 cm long along the flow, 3 cm wide
    half = 0.5 * (0.065 - 0.03)
    pts = []
    for a in np.linspace(-0.5 * np.pi, 0.5 * np.pi, n):          # downstream cap
        pts.append((X_PIER + half + r * np.cos(a), Y_PIER + r * np.sin(a)))
    for a in np.linspace(0.5 * np.pi, 1.5 * np.pi, n):           # upstream cap
        pts.append((X_PIER - half + r * np.cos(a), Y_PIER + r * np.sin(a)))
    return pts


bounding = [(X_RES, Y_RES0), (0.0, Y_RES0), (0.0, 0.0), (X_END, 0.0),
            (X_END, W), (0.0, W), (0.0, Y_RES0 + W_RES), (X_RES, Y_RES0 + W_RES)]
boundary_tags = {'wall': [0, 1, 2, 4, 5, 6, 7], 'outflow': [3]}
hole = pier_polygon(CASE)
sand_box = [(X_SAND0 - 0.05, 0.005), (X_SAND1 + 0.05, 0.005),
            (X_SAND1 + 0.05, W - 0.005), (X_SAND0 - 0.05, W - 0.005)]
pier_box = [(X_PIER - 0.12, 0.03), (X_PIER + 0.15, 0.03),
            (X_PIER + 0.15, W - 0.03), (X_PIER - 0.12, W - 0.03)]

if myid == 0:
    domain = anuga.create_domain_from_regions(
        bounding, boundary_tags=boundary_tags, maximum_triangle_area=A_COARSE,
        interior_regions=[(sand_box, A_SAND), (pier_box, A_PIER)],
        interior_holes=[hole], hole_tags=[{'wall': list(range(len(hole)))}],
        mesh_geo_reference=anuga.Geo_reference(xllcorner=0.0, yllcorner=0.0),
        use_cache=False, verbose=verbose)
    domain.set_name(output_file)
    domain.set_datadir('.')
    domain.set_flow_algorithm(alg)
    domain.set_store_vertices_uniquely()

    def elevation(x, y):
        return np.where(x > X_FLAT, -S_DOWN * (x - X_FLAT), 0.0)

    def friction(x, y):
        return np.where((x >= X_SAND0) & (x <= X_SAND1), N_SAND, N_PVC)

    domain.set_quantity('elevation', elevation)
    domain.set_quantity('friction', friction)
    domain.set_quantity('stage', expression='elevation')

    # --- sediment: bedload only, over a 5 cm erodible layer in the sand reach ---
    domain.initialize_sediment_operator(porosity=POROSITY, bed_evolution=True)
    domain.set_deposition(law='threshold', tau_d=0.0)              # no settling exchange
    domain.add_sediment_fraction('sand', diameter=D_SAND, tau_c_star=1.0e9,   # no entrainment
                                 initial_concentration=0.0)
    domain.set_bedload('wong_parker_eq24', open_boundaries=['outflow'], min_depth=2.0 * D_SAND)
    xc = domain.centroid_coordinates[:, 0]
    z = domain.quantities['elevation'].centroid_values
    base = np.where((xc >= X_SAND0) & (xc <= X_SAND1), z - SAND_DEPTH, z)
    domain.set_erodible_base(elevation=base)
    domain.set_angle_of_repose(REPOSE)
    if verbose:
        print(domain.sediment_summary())
        print('mesh: %d triangles' % domain.number_of_elements)
else:
    domain = None

domain = anuga.distribute(domain)

Br = Reflective_boundary(domain)
Bt = Transmissive_boundary(domain)
domain.set_boundary({'wall': Br, 'outflow': Bt})


def release_reservoir():
    x = domain.centroid_coordinates[:, 0]
    z = domain.quantities['elevation'].centroid_values
    stage = np.where(x < 0.0, H0, z)
    domain.set_quantity('stage', stage, location='centroids')
    domain.set_quantity('xmomentum', 0.0)
    domain.set_quantity('ymomentum', 0.0)


z0 = domain.quantities['elevation'].centroid_values.copy()
beds = []
for event in range(EVENTS):
    release_reservoir()
    for t in domain.evolve(yieldstep=0.1, finaltime=(event + 1) * T_EVENT):
        if myid == 0 and verbose:
            print(domain.timestepping_statistics())
    beds.append(domain.quantities['elevation'].centroid_values.copy())
    if myid == 0 and verbose:
        dz = beds[-1] - z0
        print('event %d: bed change min %.1f mm max %.1f mm' % (event + 1, 1000 * dz.min(), 1000 * dz.max()))

domain.sww_merge(delete_old=True)
if myid == 0:
    np.savez('pier_%s_bed.npz' % CASE, x=domain.centroid_coordinates[:, 0],
             y=domain.centroid_coordinates[:, 1], z0=z0, beds=np.array(beds),
             areas=domain.areas)
    with open('pier_%s_parameters.json' % CASE, 'w') as f:
        json.dump({'case': CASE, 'events': EVENTS, 't_event': T_EVENT, 'alg': alg,
                   'porosity': POROSITY, 'd_sand': D_SAND, 'n_pvc': N_PVC, 'n_sand': N_SAND,
                   'repose': REPOSE, 'triangles': int(domain.number_of_elements)}, f, indent=1)
finalize()
