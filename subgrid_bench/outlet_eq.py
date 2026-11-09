"""Compare coarse subgrid peak levels with the reference's volume-equivalent level per coarse cell."""
import sys
import numpy as np
from matplotlib.tri import Triangulation
sys.path.insert(0, '.')
import terrain as T
from static_storage import Cells
from subgrid_tables import SubgridTables

ref = np.load('results/band025_diag_tu.npz')
px, py, pz = T.dtm()
keep = px > 1300
px, py, pz = px[keep], py[keep], pz[keep]
rf = Triangulation(ref['nodes'][:, 0], ref['nodes'][:, 1], ref['triangles']).get_trifinder()
rstage = ref['peak_stage'][rf(px, py)]
depth_ref = np.maximum(rstage - pz, 0)
X = np.arange(1400, 2000.1, 25.0)
Y = T.centreline(X)
for name in ['coarse500_sub8EKLC_diag', 'coarse2000_sub8EKLC_diag', 'coarse500_sub8EKL_diag']:
    d = np.load(f'results/{name}.npz')
    nodes, tri = d['nodes'], d['triangles']
    # all pixels (full DTM) are needed by Cells; build tables for cells touching x > 1300
    fx, fy, fz = T.dtm()
    cells = Cells(nodes, tri, fx, fy)
    tab = SubgridTables.from_cells(cells, 8, fx, fy, fz)
    own = cells.owner[fx > 1300]
    vol = np.bincount(own, weights=depth_ref * 0.25, minlength=len(tri))
    h_eq = vol / cells.area
    eta_eq = tab.eta(h_eq)
    finder = Triangulation(nodes[:, 0], nodes[:, 1], tri).get_trifinder()
    c = finder(X, Y)
    pt_ref = ref['peak_stage'][rf(X, Y)]
    print(f'\n{name}: x, cell centroid x, coarse - ref(point), coarse - ref(volume-equivalent level)')
    for i, x in enumerate(X):
        print(f'  {x:6.0f} {cells.centroid[c[i], 0]:7.0f}  {d["peak_stage"][c[i]] - pt_ref[i]:+7.3f}  {d["peak_stage"][c[i]] - eta_eq[c[i]]:+7.3f}')
