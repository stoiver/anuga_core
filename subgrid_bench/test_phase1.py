"""Phase 1 go/no-go checks for the subgrid storage prototype.

    OMP_NUM_THREADS=1 python test_phase1.py

1. tables: depth(eta(h)) == h, and storage against level matches the DTM truth
   as the static-storage study did
2. lake at rest: a flat level that leaves channel cells partly wet, reflective
   walls, no forcing -- velocities stay at round-off, stage stays flat, mass is
   conserved to round-off
3. mass conservation under flow: a sloping release, total volume constant
"""
import sys

import numpy as np

import anuga
import terrain as T
from static_storage import Cells, make_mesh, storage_truth
from subgrid_tables import SubgridTables, Subgrid_storage_operator

N = 8
FAILS = []


def check(name, ok, detail):
    print(f'  [{"PASS" if ok else "FAIL"}] {name}: {detail}')
    if not ok:
        FAILS.append(name)


def build(max_area=2000.0):
    nodes, tris = make_mesh(max_area)
    px, py, pz = T.dtm()
    cells = Cells(nodes, tris, px, py)
    from run_phase0 import friction
    tables = SubgridTables.from_cells(cells, N, px, py, pz, manning=friction)
    return nodes, tris, cells, tables, (px, py, pz)


def make_domain(nodes, tris, tables, eta0):
    domain = anuga.Domain(nodes, tris)
    domain.set_compute_mode('legacy')
    domain.set_store(False)
    domain.set_quantity('friction', 0.035, location='centroids')
    br = anuga.Reflective_boundary(domain)
    domain.set_boundary({tag: br for tag in domain.get_boundary_tags()})
    op = Subgrid_storage_operator(domain, tables)
    op.set_initial_level(eta0)
    return domain, op


def volume(domain):
    w = domain.quantities['stage'].centroid_values
    d = domain.quantities['elevation'].centroid_values
    return np.sum(domain.areas * (w - d))


def main():
    print(f'building coarse2000 mesh and n={N} tables')
    nodes, tris, cells, tables, (px, py, pz) = build()

    print('1. tables')
    h = np.random.default_rng(1).uniform(0, 3, len(tables.area))
    h[::7] = 0.0
    back = tables.depth(tables.eta(h))
    check('depth(eta(h)) == h', np.max(np.abs(back - h)) < 1e-12,
          f'max error {np.max(np.abs(back - h)):.1e}')
    for level in (0.5, 2.0, 2.5):
        eta = T.water_surface(cells.centroid[:, 0], level)
        v = np.sum(tables.volume(eta))
        vt = storage_truth(px, pz, level)[0]
        print(f'     depth {level} m above thalweg: volume error {100 * (v - vt) / vt:+.1f}% '
              f'(static study n=8: -16.7 / +0.3 / +1.0%)')

    print('2. lake at rest, partly wet cells')
    level = 7.0            # thalweg is 8.0 at x=0, 6.1 at x=1900: channel partly wet downstream
    domain, op = make_domain(nodes, tris, tables, np.full(len(tris), level))
    h0 = tables.depth(np.full(len(tris), level))
    partly = np.sum((h0 > 0) & (tables.wet_area(np.full(len(tris), level)) < tables.area))
    v0 = volume(domain)
    for t in domain.evolve(yieldstep=300.0, finaltime=1800.0):
        pass
    w = domain.quantities['stage'].centroid_values
    wet = (w - domain.quantities['elevation'].centroid_values) > 1e-9   # ignore round-off films
    umax = np.max(np.abs(domain.quantities['xmomentum'].centroid_values)) + \
        np.max(np.abs(domain.quantities['ymomentum'].centroid_values))
    dv = volume(domain) - v0
    print(f'     {np.sum(h0 > 0)} wet cells, {partly} of them partly wet; volume {v0:.1f} m3')
    check('momentum at round-off', umax < 1e-10, f'max |uh|+|vh| = {umax:.1e}')
    check('stage flat on wet cells', np.ptp(w[wet]) < 1e-10, f'range {np.ptp(w[wet]):.1e} m')
    check('mass conserved', abs(dv) < 1e-9 * max(v0, 1.0), f'change {dv:.1e} m3')

    print('3. mass conservation under flow')
    eta0 = np.where(cells.centroid[:, 0] < 400, 10.5, 0.0)   # water above bank-full upstream
    domain, op = make_domain(nodes, tris, tables, eta0)
    v0 = volume(domain)
    vols = []
    for t in domain.evolve(yieldstep=300.0, finaltime=3600.0):
        vols.append(volume(domain))
    drift = np.max(np.abs(np.array(vols) - v0)) / v0
    front = cells.centroid[(domain.quantities['stage'].centroid_values -
                            domain.quantities['elevation'].centroid_values) > 1e-3, 0].max()
    print(f'     released {v0:.0f} m3; wet front reached x = {front:.0f} m after 1 h')
    check('mass conserved', drift < 1e-12, f'max relative drift {drift:.1e}')

    print('\nALL PASS' if not FAILS else f'\nFAILED: {FAILS}')
    return 1 if FAILS else 0


if __name__ == '__main__':
    sys.exit(main())
