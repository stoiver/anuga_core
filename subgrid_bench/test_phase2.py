"""Phase 2 go/no-go checks for the subgrid edge flux kernel.

    OMP_NUM_THREADS=1 python test_phase2.py

1. flat bed: with every subface at the cell bed, the subgrid kernel matches the
   standard central kernel on a dam break (first order: to round-off)
2. lake at rest with partly wet cells and the channel crossing edges: momentum
   and stage stay at round-off, mass conserved
3. mass conservation under flow: a sloping release, total volume constant
Tests 1-3 run with the plain closure (uniform side velocity) and with the
conveyance-weighted subface velocity (split); on a flat bed the split must
reduce to the plain closure.
"""
import sys

import numpy as np

import anuga
import terrain as T
import test_phase1 as P1
from run_phase0 import friction
from anuga.operators.base_operator import Operator
from subgrid_tables import SubgridTables, Subgrid_storage_operator, edge_subface_beds

NSUB = 16
EDGE_CAP_CFL = 0.75
FAILS = []
check = P1.check


class _FlatKc(Operator):
    """Flat-bed cells: n_eff = n, every subface 1/n (split must change nothing)."""
    N = 0.03

    def __init__(self, domain):
        Operator.__init__(self, domain)
        domain.subgrid_edge_invn = np.full_like(domain.subgrid_edge_z, 1.0 / self.N)
        domain.subgrid_cell_neff = np.full(len(domain), self.N)

    def __call__(self):
        pass

    def statistics(self):
        return ''

    def timestepping_statistics(self):
        return ''


def dam_break(subgrid, order, split=False):
    domain = anuga.rectangular_cross_domain(20, 4, len1=100.0, len2=20.0)
    domain.set_compute_mode('legacy')
    domain.set_store(False)
    domain.set_default_order(order)
    domain.set_quantity('elevation', 0.0)
    domain.set_quantity('friction', 0.0)
    domain.set_quantity('stage', lambda x, y: np.where(x < 50.0, 2.0, 0.5))
    br = anuga.Reflective_boundary(domain)
    domain.set_boundary({'left': br, 'right': br, 'top': br, 'bottom': br})
    if subgrid:
        domain.subgrid_edge_z = np.zeros((3 * len(domain), 4))
        if split:
            _FlatKc(domain)
    for t in domain.evolve(yieldstep=2.0, finaltime=6.0):
        pass
    return domain.quantities['stage'].centroid_values.copy(), domain.get_water_volume()


def make_domain(nodes, tris, tables, eta0, nsub=NSUB, split=False):
    domain = anuga.Domain(nodes, tris)
    domain.set_compute_mode('legacy')
    domain.set_store(False)
    domain.set_quantity('friction', 0.035, location='centroids')
    br = anuga.Reflective_boundary(domain)
    domain.set_boundary({tag: br for tag in domain.get_boundary_tags()})
    ein = None
    if split:
        ez, ein = edge_subface_beds(domain, tables, nsub, T.bed, manning=friction)
    else:
        ez = edge_subface_beds(domain, tables, nsub, T.bed)
    # full wet-fraction cap: with subface depths at the edges, nearly dry cells
    # exchange water through the true flow area, and the f >= 0.05 cut-off that
    # was safe with the central kernel lets lake at rest grow ~100x per 300 s
    # and cfl=0.75: the cap lags the kernel's subface wave speed by one step;
    # at 1.0 cells overdraw and the clamp adds ~1e-9 relative mass
    # With the draining-time limiter (drain_limiter=True) depths stay >= 0 for any
    # step, so the cap drops the wet fraction (cap_exponent=0): the step is then
    # ~30x larger, and 0.75 x the subface CFL keeps lake at rest at round-off.
    op = Subgrid_storage_operator(domain, tables, cfl=EDGE_CAP_CFL, edge_z=ez,
                                  cap_exponent=0.0, drain_limiter=True, edge_invn=ein)
    op.set_initial_level(eta0)
    return domain, op


def main():
    print('1. flat bed: subgrid kernel vs central kernel (dam break)')
    for order in (1, 2):
        w0, v0 = dam_break(False, order)
        for split in (False, True):
            w1, v1 = dam_break(True, order, split)
            d = np.max(np.abs(w1 - w0))
            tol = 1e-12 if order == 1 else 5e-2
            tag = f'order {order}{" split" if split else ""}'
            check(f'{tag} stage agrees', d < tol, f'max |difference| {d:.1e} m (tol {tol:g})')
            check(f'{tag} mass', abs(v1 - v0) < 1e-9 * v0, f'volume difference {v1 - v0:.1e} m3')

    print(f'building coarse2000 mesh, n={P1.N} cell tables, {NSUB} subfaces per edge')
    nodes, tris, cells, tables, _ = P1.build()

    for split in (False, True):
        tag = ' (conveyance split)' if split else ''
        print(f'2. lake at rest, partly wet cells, channel through the edges{tag}')
        level = 7.0
        domain, op = make_domain(nodes, tris, tables, np.full(len(tris), level), split=split)
        v0 = P1.volume(domain)
        for t in domain.evolve(yieldstep=300.0, finaltime=1800.0):
            pass
        w = domain.quantities['stage'].centroid_values
        wet = (w - domain.quantities['elevation'].centroid_values) > 1e-9
        umax = np.max(np.abs(domain.quantities['xmomentum'].centroid_values)) + \
            np.max(np.abs(domain.quantities['ymomentum'].centroid_values))
        dv = P1.volume(domain) - v0
        check('momentum at round-off', umax < 1e-10, f'max |uh|+|vh| = {umax:.1e}')
        check('stage flat on wet cells', np.ptp(w[wet]) < 1e-10, f'range {np.ptp(w[wet]):.1e} m')
        check('mass conserved', abs(dv) < 1e-9 * max(v0, 1.0), f'change {dv:.1e} m3')

        print(f'3. mass conservation under flow{tag}')
        eta0 = np.where(cells.centroid[:, 0] < 400, 10.5, 0.0)
        domain, op = make_domain(nodes, tris, tables, eta0, split=split)
        v0 = P1.volume(domain)
        vols = []
        for t in domain.evolve(yieldstep=300.0, finaltime=3600.0):
            vols.append(P1.volume(domain))
        drift = np.max(np.abs(np.array(vols) - v0)) / v0
        hmin = np.min(domain.quantities['stage'].centroid_values - domain.quantities['elevation'].centroid_values)
        front = cells.centroid[(domain.quantities['stage'].centroid_values -
                                domain.quantities['elevation'].centroid_values) > 1e-3, 0].max()
        print(f'     released {v0:.0f} m3; wet front reached x = {front:.0f} m after 1 h')
        check('mass conserved', drift < 1e-12, f'max relative drift {drift:.1e}')
        check('no negative depth', hmin > -1e-12, f'min depth {hmin:.1e} m')
        print(f'     {domain.subgrid_limited_cell_steps} cell-steps limited by the draining-time limiter')

    fails = P1.FAILS
    print('\nALL PASS' if not fails else f'\nFAILED: {fails}')
    return 1 if fails else 0


if __name__ == '__main__':
    sys.exit(main())
