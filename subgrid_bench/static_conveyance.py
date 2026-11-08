"""Phase 2 static check: section conveyance from the coarse cells vs the DTM.

    python static_conveyance.py

At sections x = 250 ... 1750 m and a level water surface at several depths above
the thalweg, the conveyance K = sum over the section of h^(5/3)/n * dy is computed
  truth       from the DTM pixels along the section (0.5 m spacing)
  centroid    each pixel takes its coarse cell's centroid bed and centroid n
  storage     each pixel takes its cell's subgrid mean depth h = V(eta)/A and centroid n
              (what the Phase 1 storage-only runs use in the friction term)
  conveyance  each pixel takes its cell's Kc(eta) from the subgrid conveyance table
and reported as a percentage error against the truth.
"""
import numpy as np
from matplotlib.tri import Triangulation

import terrain as T
from run_phase0 import friction
from static_storage import Cells, make_mesh
from subgrid_tables import SubgridTables

SECTIONS = [250.0, 500.0, 1000.0, 1500.0, 1750.0]
DEPTHS = [0.5, 1.0, 2.0, 2.5, 3.0]     # level above the thalweg (bank-full = 2.0)
DY = 0.5


def main():
    px, py, pz = T.dtm()
    y = np.arange(0.25, T.WIDTH, DY)
    for max_area in (500.0, 2000.0):
        nodes, tris = make_mesh(max_area)
        cells = Cells(nodes, tris, px, py)
        finder = Triangulation(nodes[:, 0], nodes[:, 1], tris).get_trifinder()
        cx, cy = cells.centroid[:, 0], cells.centroid[:, 1]
        zc, nc = T.bed(cx, cy), friction(cx, cy)
        tabs = {n: SubgridTables.from_cells(cells, n, px, py, pz, manning=friction) for n in (8, 16)}
        print(f'\ncoarse{max_area:.0f}: {len(tris)} triangles. Conveyance error against the DTM (%)')
        print(f'{"x":>6s}{"depth":>6s}{"K truth":>10s}{"centroid":>10s}'
              + ''.join(f'{"storage" + str(n):>10s}{"conv" + str(n):>9s}' for n in tabs))
        errs = {k: [] for k in ['centroid'] + [f'{a}{n}' for n in tabs for a in ('storage', 'conv')]}
        for xs in SECTIONS:
            x = np.full_like(y, xs)
            zp, npx = T.bed(x, y), friction(x, y)
            own = finder(x, y)
            for dep in DEPTHS:
                eta = T.water_surface(np.array([xs]), dep)[0]
                kt = np.sum(np.maximum(eta - zp, 0) ** (5 / 3) / npx) * DY
                row = {'centroid': np.sum(np.maximum(eta - zc[own], 0) ** (5 / 3) / nc[own]) * DY}
                e = np.full(len(tris), eta)
                for n, t in tabs.items():
                    row[f'storage{n}'] = np.sum(t.depth(e)[own] ** (5 / 3) / nc[own]) * DY
                    row[f'conv{n}'] = np.sum(t.conveyance(e)[own]) * DY
                line = f'{xs:6.0f}{dep:6.1f}{kt:10.0f}'
                for k, v in row.items():
                    err = 100 * (v - kt) / kt
                    errs[k].append(err)
                    line += f'{err:+10.1f}' if not k.startswith('conv') else f'{err:+9.1f}'
                print(line)
        print(f'{"mean |error| %":>22s}' + ''.join(
            f'{np.mean(np.abs(v)):10.1f}' if not k.startswith('conv') else f'{np.mean(np.abs(v)):9.1f}'
            for k, v in errs.items()))


if __name__ == '__main__':
    main()
