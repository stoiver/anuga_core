"""Phase 0, static part: stored volume and wet area against water level.

No simulation. For a water surface parallel to the valley at a given depth
above the thalweg, compare the volume and wet area stored by

  truth      the 0.5 m DTM itself
  resolved   a resolved ANUGA mesh (channel 4 m2, floodplain 25 m2)
  coarse     coarse meshes, with the bed per cell set three ways:
               centroid   bed sampled at the cell centroid (set_quantity default
                          with location='centroids')
               cellmean   mean of the DTM pixels in the cell
               trench     centroid bed, with the cells crossed by the
                          centreline lowered to the thalweg (a conditioned
                          "equivalent channel", as in #453)
             and with subgrid tables: each cell split uniformly into n^2
             sub-triangles holding the mean of their DTM pixels, water level
             taken as level within the cell (n = 2, 4, 8, 16).

The subgrid code here is the prototype table builder for Phase 1.
"""
import sys
import time

import numpy as np
from matplotlib.tri import Triangulation

import anuga
import terrain as T

LEVELS = np.array([0.25, 0.5, 1.0, 1.5, 2.0, 2.1, 2.25, 2.5, 2.75, 3.0, 3.5])
SUB_N = [2, 4, 8, 16]


def make_mesh(max_area, band_area=None):
    regions = None
    if band_area is not None:
        regions = [[T.channel_band_polygon(), band_area]]
    domain = anuga.create_domain_from_regions(
        T.BOUNDING_POLYGON, T.BOUNDARY_TAGS, maximum_triangle_area=max_area,
        interior_regions=regions)
    return domain.nodes.copy(), domain.triangles.copy()


class Cells:
    """A triangle mesh plus the DTM pixels located in each triangle."""

    def __init__(self, nodes, triangles, px, py):
        self.nodes = nodes
        self.tri = triangles
        p = nodes[triangles]                       # (ntri, 3, 2)
        self.p0 = p[:, 0]
        self.e1 = p[:, 1] - p[:, 0]
        self.e2 = p[:, 2] - p[:, 0]
        self.area = 0.5 * np.abs(self.e1[:, 0] * self.e2[:, 1]
                                 - self.e1[:, 1] * self.e2[:, 0])
        self.centroid = p.mean(axis=1)
        finder = Triangulation(nodes[:, 0], nodes[:, 1], triangles).get_trifinder()
        self.owner = finder(px, py)
        if np.any(self.owner < 0):
            raise RuntimeError(f'{np.sum(self.owner < 0)} pixels outside the mesh')

    @property
    def ntri(self):
        return len(self.tri)

    def barycentric(self, px, py):
        """Coordinates (l1, l2) of each pixel along e1, e2 of its owner."""
        t = self.owner
        dx = px - self.p0[t, 0]
        dy = py - self.p0[t, 1]
        a, c = self.e1[t, 0], self.e1[t, 1]
        b, d = self.e2[t, 0], self.e2[t, 1]
        det = a * d - b * c
        return (d * dx - b * dy) / det, (-c * dx + a * dy) / det

    def subgrid(self, n, px, py, pz, return_centroids=False):
        """Sub-triangle centroids, areas and mean beds for an n^2 split.

        Returns arrays shaped (ntri, 2*n*n) with NaN in unused slots
        (slot = 2*(i*n + j) + down).
        With return_centroids, also returns the sub-centroid coordinates sx, sy.
        """
        nslot = 2 * n * n
        i_ = np.arange(n)
        I, J = np.meshgrid(i_, i_, indexing='ij')
        I = I.ravel()
        J = J.ravel()
        used = np.zeros(nslot, bool)
        cu = np.full((nslot, 2), np.nan)           # (i, j) offsets of sub-centroids
        for down, off in ((0, 1.0 / 3), (1, 2.0 / 3)):
            ok = I + J <= n - 1 - down
            slot = 2 * (I * n + J) + down
            used[slot[ok]] = True
            cu[slot[ok], 0] = (I[ok] + off) / n
            cu[slot[ok], 1] = (J[ok] + off) / n
        assert used.sum() == n * n

        # sub-centroid coordinates and analytic bed there (fallback for empty subs)
        sx = (self.p0[:, None, 0] + cu[None, :, 0] * self.e1[:, None, 0]
              + cu[None, :, 1] * self.e2[:, None, 0])
        sy = (self.p0[:, None, 1] + cu[None, :, 0] * self.e1[:, None, 1]
              + cu[None, :, 1] * self.e2[:, None, 1])

        l1, l2 = self.barycentric(px, py)
        u = np.clip(l1, 0, 1) * n
        v = np.clip(l2, 0, 1) * n
        i = np.clip(np.floor(u), 0, n - 1).astype(np.int64)
        j = np.clip(np.floor(v), 0, n - 1).astype(np.int64)
        down = ((u - i) + (v - j) >= 1.0).astype(np.int64)
        # points rounding just outside the triangle: push back to a valid slot
        over = i + j + down > n - 1
        down[over] = 0
        over = i + j > n - 1
        j[over] = n - 1 - i[over]
        key = self.owner.astype(np.int64) * nslot + 2 * (i * n + j) + down

        count = np.bincount(key, minlength=self.ntri * nslot).reshape(self.ntri, nslot)
        zsum = np.bincount(key, weights=pz, minlength=self.ntri * nslot).reshape(self.ntri, nslot)
        with np.errstate(invalid='ignore', divide='ignore'):
            zsub = np.where(count > 0, zsum / count, T.bed(sx, sy))
        zsub[:, ~used] = np.nan
        empty = np.sum((count == 0) & used[None, :])
        asub = np.where(used[None, :], self.area[:, None] / (n * n), 0.0)
        if return_centroids:
            return zsub, asub, empty, sx, sy
        return zsub, asub, empty

    def cell_mean(self, pz):
        cnt = np.bincount(self.owner, minlength=self.ntri)
        s = np.bincount(self.owner, weights=pz, minlength=self.ntri)
        z = T.bed(self.centroid[:, 0], self.centroid[:, 1])
        return np.where(cnt > 0, s / np.maximum(cnt, 1), z)


def storage_cells(cells, zc, level):
    eta = T.water_surface(cells.centroid[:, 0], level)
    h = np.maximum(eta - zc, 0.0)
    return np.sum(cells.area * h), np.sum(cells.area * (h > 0))


def storage_subgrid(cells, zsub, asub, level):
    eta = T.water_surface(cells.centroid[:, 0], level)[:, None]   # level in each cell
    h = np.where(np.isnan(zsub), 0.0, np.maximum(eta - np.nan_to_num(zsub, nan=np.inf), 0.0))
    return np.sum(asub * h), np.sum(asub * (h > 0))


def storage_truth(px, pz, level):
    h = np.maximum(T.water_surface(px, level) - pz, 0.0)
    a = T.PIXEL ** 2
    return np.sum(h) * a, np.sum(h > 0) * a


def trench_bed(cells):
    """Centroid bed with the cells crossed by the centreline lowered to the thalweg."""
    xs = np.linspace(0, T.LENGTH, 200001)
    xs = np.clip(xs, 1e-6, T.LENGTH - 1e-6)
    finder = Triangulation(cells.nodes[:, 0], cells.nodes[:, 1], cells.tri).get_trifinder()
    crossed = np.unique(finder(xs, T.centreline(xs)))
    crossed = crossed[crossed >= 0]
    z = T.bed(cells.centroid[:, 0], cells.centroid[:, 1])
    z[crossed] = T.channel_bed(cells.centroid[crossed, 0])
    return z, len(crossed)


def main():
    t0 = time.time()
    px, py, pz = T.dtm()
    print(f'DTM: {len(pz)} pixels of {T.PIXEL} m  ({time.time() - t0:.1f} s)')

    truth = np.array([storage_truth(px, pz, L) for L in LEVELS])
    rows = {}        # name -> (V, A) arrays over LEVELS
    info = {}

    meshes = [('coarse2000', 2000.0, None), ('coarse500', 500.0, None),
              ('resolved', 25.0, 4.0)]
    for name, max_area, band in meshes:
        t1 = time.time()
        nodes, tris = make_mesh(max_area, band)
        cells = Cells(nodes, tris, px, py)
        edge = np.sqrt(4 * np.median(cells.area) / np.sqrt(3))
        info[name] = (cells.ntri, np.median(cells.area), edge)
        print(f'{name}: {cells.ntri} triangles, median area {np.median(cells.area):.1f} m2, '
              f'equilateral edge ~{edge:.1f} m  ({time.time() - t1:.1f} s)')

        zc = T.bed(cells.centroid[:, 0], cells.centroid[:, 1])
        rows[f'{name} centroid'] = np.array([storage_cells(cells, zc, L) for L in LEVELS])
        zm = cells.cell_mean(pz)
        rows[f'{name} cellmean'] = np.array([storage_cells(cells, zm, L) for L in LEVELS])
        if name.startswith('coarse'):
            zt, ncross = trench_bed(cells)
            rows[f'{name} trench'] = np.array([storage_cells(cells, zt, L) for L in LEVELS])
            info[f'{name} trench cells'] = ncross
            for n in SUB_N:
                zsub, asub, empty = cells.subgrid(n, px, py, pz)
                rows[f'{name} subgrid n={n}'] = np.array(
                    [storage_subgrid(cells, zsub, asub, L) for L in LEVELS])
                if empty:
                    print(f'   n={n}: {empty} empty sub-triangles (analytic fallback)')

    np.savez('static_storage.npz', levels=LEVELS, truth=truth,
             **{k.replace(' ', '_'): v for k, v in rows.items()})

    def table(col, label, unit_scale, unit):
        print(f'\n{label}: truth ({unit}) and relative error (%) by depth above thalweg (m)')
        print(f'{"":26s}' + ''.join(f'{L:8.2f}' for L in LEVELS))
        print(f'{"truth":26s}' + ''.join(f'{v / unit_scale:8.1f}' for v in truth[:, col]))
        for k, v in rows.items():
            err = 100 * (v[:, col] - truth[:, col]) / truth[:, col]
            print(f'{k:26s}' + ''.join(f'{e:8.1f}' for e in err))

    table(0, 'Volume', 1e3, '1000 m3')
    table(1, 'Wet area', 1e3, '1000 m2')
    print(f'\nchannel top width {T.TOP_WIDTH} m; bank-full at depth {T.DEPTH} m')
    print(f'total {time.time() - t0:.0f} s')


if __name__ == '__main__':
    sys.exit(main())
