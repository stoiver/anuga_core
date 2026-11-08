"""Phase 2 static check: what the coarse edges along the channel can carry.

    python static_edges.py

For every coarse edge the channel centreline crosses, with a level water
surface at a given depth above the thalweg, compare against the DTM truth
along the edge (sampled every 0.1 m):

  flow area   A = integral of max(eta - z, 0) ds
  conveyance  K = integral of max(eta - z, 0)^(5/3) / n ds

The coarse scheme sees each side of the edge as a uniform depth over the whole
edge length (first-order picture):
  centroid    h = eta - z(centroid)                (plain coarse mesh)
  storage     h = mean depth V(eta)/A from the subgrid tables (Phase 1)
  cellK       conveyance only: L * Kc(eta), the cell conveyance (Phase 2 start)
taking the mean of the two sides. Flow along the channel crosses these edges in
series, so the 'series' column, (sum 1/K^2)^-1/2 relative to the truth, is the
one that sets the base-flow water level; a single dry edge makes it 0. Edge tables with n subfaces (each subface's
bed the mean of the DTM along it) are what Phase 2 would add; they are checked
against the truth the same way.
"""
import numpy as np
from matplotlib.tri import Triangulation

import terrain as T
from run_phase0 import friction
from static_storage import Cells, make_mesh
from subgrid_tables import SubgridTables

DEPTHS = [0.72, 2.0, 2.5]     # band025 base-flow depth, bank-full, overbank
DS = 0.1
N_TABLE = 8                   # sub-triangles per side, as in the runs
SUBFACES = (8, 16)


def crossed_edges(nodes, tris):
    """Interior edges (a, b, left cell, right cell) the centreline crosses, in order."""
    xs = np.clip(np.linspace(0, T.LENGTH, 400001), 1e-6, T.LENGTH - 1e-6)
    own = Triangulation(nodes[:, 0], nodes[:, 1], tris).get_trifinder()(xs, T.centreline(xs))
    keep = own >= 0
    own = own[keep]
    change = np.nonzero(np.diff(own))[0]
    out, skipped = [], 0
    for k in change:
        c0, c1 = own[k], own[k + 1]
        shared = set(tris[c0]) & set(tris[c1])
        if len(shared) != 2:          # passed through a vertex
            skipped += 1
            continue
        a, b = sorted(shared)
        out.append((a, b, c0, c1))
    return out, skipped


def edge_samples(pa, pb):
    L = np.hypot(*(pb - pa))
    m = max(int(np.ceil(L / DS)), 1)
    s = (np.arange(m) + 0.5) / m
    p = pa[None, :] + s[:, None] * (pb - pa)[None, :]
    return p[:, 0], p[:, 1], L / m, L


def main():
    px, py, pz = T.dtm()
    for max_area in (500.0, 2000.0):
        nodes, tris = make_mesh(max_area)
        cells = Cells(nodes, tris, px, py)
        tab = SubgridTables.from_cells(cells, N_TABLE, px, py, pz, manning=friction)
        cx, cy = cells.centroid[:, 0], cells.centroid[:, 1]
        zc, nc = T.bed(cx, cy), friction(cx, cy)
        edges, skipped = crossed_edges(nodes, tris)
        print(f'\ncoarse{max_area:.0f}: {len(tris)} triangles, {len(edges)} edges crossed by the '
              f'channel ({skipped} vertex crossings skipped)')
        for dep in DEPTHS:
            rows = []
            for a, b, c0, c1 in edges:
                ex, ey, ds, L = edge_samples(nodes[a], nodes[b])
                xm = 0.5 * (nodes[a, 0] + nodes[b, 0])
                eta = T.water_surface(np.array([xm]), dep)[0]
                z, n = T.bed(ex, ey), friction(ex, ey)
                h = np.maximum(eta - z, 0)
                A_t, K_t = h.sum() * ds, (h ** (5 / 3) / n).sum() * ds
                e2 = np.full(2, eta)
                cc = np.array([c0, c1])
                h_c = np.maximum(eta - zc[cc], 0)
                h_s = np.array([tab.depth(np.full(len(tris), eta))[c] for c in cc])
                kc = tab.conveyance(np.full(len(tris), eta))[cc]
                r = dict(A_t=A_t, K_t=K_t,
                         A_centroid=L * h_c.mean(), K_centroid=L * np.mean(h_c ** (5 / 3) / nc[cc]),
                         A_storage=L * h_s.mean(), K_storage=L * np.mean(h_s ** (5 / 3) / nc[cc]),
                         K_cellK=L * kc.mean())
                for nf in SUBFACES:      # edge table: nf subfaces, mean bed on each
                    zf = z[: (len(z) // nf) * nf].reshape(nf, -1).mean(axis=1) if len(z) >= nf else z
                    nfv = n[: (len(n) // nf) * nf].reshape(nf, -1).mean(axis=1) if len(n) >= nf else n
                    hf = np.maximum(eta - zf, 0)
                    r[f'A_edge{nf}'] = hf.sum() * L / len(zf)
                    r[f'K_edge{nf}'] = (hf ** (5 / 3) / nfv).sum() * L / len(zf)
                rows.append(r)
            A_t = np.array([r['A_t'] for r in rows])
            K_t = np.array([r['K_t'] for r in rows])
            print(f'  depth {dep:.2f} m above thalweg: ratio coarse / truth over the crossed edges')
            print(f'    {"":12s}{"median":>8s}{"p10":>8s}{"p90":>8s}{"sum":>8s}{"series":>8s}'
                  '   (sum: total / total truth; series: K of the edges in a chain, (sum 1/K^2)^-1/2, / truth)')
            for key in ['A_centroid', 'A_storage', 'A_edge8', 'A_edge16',
                        'K_centroid', 'K_storage', 'K_cellK', 'K_edge8', 'K_edge16']:
                v = np.array([r[key] for r in rows])
                tr = A_t if key[0] == 'A' else K_t
                q = v / tr
                with np.errstate(divide='ignore'):
                    ser = np.sqrt(np.sum(1 / tr ** 2) / np.sum(1 / v ** 2)) if key[0] == 'K' else np.nan
                print(f'    {key:12s}{np.median(q):8.2f}{np.percentile(q, 10):8.2f}'
                      f'{np.percentile(q, 90):8.2f}{v.sum() / tr.sum():8.2f}{ser:8.2f}')


if __name__ == '__main__':
    main()
