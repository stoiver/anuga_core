"""crest_lines.py: find narrow ridge crests in a DTM raster and return them as
polylines, to be used as mesh breaklines so that coarse subgrid cells get an edge
along each crest (see SUBGRID_STATUS.md, "Ridges: put mesh edges on the crests").

Method (scipy and shapely only):

1. White top-hat: z minus its grey opening over a window `width` wide. That keeps
   features narrower than the window (levees, banks, road crowns) and removes
   broad relief.
2. Crest pixels: top-hat above `prominence`, and a local maximum of z along at
   least one of the four pixel directions (so the mask is a thin line).
3. Each connected group of crest pixels gives one main polyline: its longest
   pixel path (two shortest-path passes). The mask is first dilated by one
   pixel so small sideways steps do not break a crest. Side branches longer
   than `min_branch` are kept as further polylines that start on a vertex of
   the main one.
4. Lines shorter than `min_length` are dropped, the rest simplified to `tol`.
5. Ends within `snap` of the boundary polygon are moved onto it, so no sliver
   of cell is left between the end of a crest and the wall.

    python crest_lines.py ridged|rough|mixed|flat      # flat-plain bench terrains
"""
import numpy as np
from scipy import ndimage
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import dijkstra


def crest_mask(z, dx, width=10.0, prominence=0.05):
    """Bool raster of crest pixels (steps 1 and 2)."""
    w = max(int(round(width / dx)) | 1, 3)
    top = z - ndimage.grey_opening(z, size=(w, w), mode='nearest')
    p = np.pad(z, 1, mode='edge')
    c = p[1:-1, 1:-1]
    ismax = np.zeros(z.shape, bool)
    for (a, b) in (((1, 0), (1, 2)), ((0, 1), (2, 1)), ((0, 0), (2, 2)), ((0, 2), (2, 0))):
        n1 = p[a[0]:a[0] + z.shape[0], a[1]:a[1] + z.shape[1]]
        n2 = p[b[0]:b[0] + z.shape[0], b[1]:b[1] + z.shape[1]]
        ismax |= (c >= n1) & (c >= n2) & ((c > n1) | (c > n2))
    ismax[[0, -1], :] = False          # the raster's rim is not a crest
    ismax[:, [0, -1]] = False
    return (top > prominence) & ismax


def _longest_path(iy, ix, cost=None):
    """Pixels (in order) of the longest cheapest-path through one 8-connected
    group. cost: per pixel (default 1); a step costs its length times the mean
    cost of its two pixels, so the path keeps to the cheap (true crest) pixels."""
    n = len(iy)
    if n < 2:
        return np.arange(n)
    cost = np.ones(n) if cost is None else cost
    key = {(int(a), int(b)): k for k, (a, b) in enumerate(zip(iy, ix))}
    rows, cols, wts = [], [], []
    for k, (a, b) in enumerate(zip(iy, ix)):
        for da, db in ((0, 1), (1, -1), (1, 0), (1, 1)):
            j = key.get((int(a) + da, int(b) + db))
            if j is not None:
                rows += [k, j]
                cols += [j, k]
                wts += [0.5 * (cost[k] + cost[j]) * np.hypot(da, db)] * 2
    g = coo_matrix((wts, (rows, cols)), shape=(n, n)).tocsr()
    far = int(np.argmax(np.where(np.isfinite(d := dijkstra(g, indices=0)), d, -1)))
    d, pred = dijkstra(g, indices=far, return_predecessors=True)
    path = [int(np.argmax(np.where(np.isfinite(d), d, -1)))]
    while pred[path[-1]] >= 0:
        path.append(int(pred[path[-1]]))
    return np.array(path)


def crest_lines(z, dx, width=10.0, prominence=0.05, min_length=50.0, tol=3.0,
                boundary=None, snap=None, origin=(0.0, 0.0), min_branch=None):
    """Crest polylines of raster z (ny, nx), pixel size dx, pixel (i, j) centred at
    origin + ((j + 0.5) dx, (i + 0.5) dx). boundary: optional list of (x, y), the
    domain polygon; line ends within snap (default width / 2) of it are moved onto
    it. min_branch (default min_length / 4): shortest side branch kept.
    Returns a list of (n, 2) arrays; a branch starts on a vertex of its parent."""
    from shapely.geometry import LineString, Point, Polygon
    core = crest_mask(z, dx, width, prominence)
    # one pixel of dilation joins a crest whose line of maxima steps sideways;
    # paths still keep to the undilated crest pixels where they can
    mask = ndimage.binary_dilation(core, structure=np.ones((3, 3)))
    lab, nlab = ndimage.label(mask, structure=np.ones((3, 3)))
    ring = Polygon(boundary).exterior if boundary is not None else None
    snap = width / 2 if snap is None else snap
    min_branch = min_length / 4 if min_branch is None else min_branch
    s8 = np.ones((3, 3))

    def world(iy, ix, sl):
        return np.column_stack([origin[0] + (ix + sl[1].start + 0.5) * dx,
                                origin[1] + (iy + sl[0].start + 0.5) * dx])

    def snap_end(pts, e, inner):
        p = Point(pts[e])
        if ring is None or ring.distance(p) > snap:
            return
        # extend along the line's end direction if that meets the boundary nearby,
        # else move to the nearest boundary point
        d = pts[e] - pts[inner]
        d = d / max(np.hypot(*d), 1e-12)
        hit = LineString([pts[e], pts[e] + 3 * snap * d]).intersection(ring)
        q = ring.interpolate(ring.project(p)) if hit.is_empty else min(
            getattr(hit, 'geoms', [hit]), key=lambda g: g.distance(p))
        q = q.representative_point() if q.geom_type != 'Point' else q
        pts[e] = (q.x, q.y)

    lines = []
    for sl, k in zip(ndimage.find_objects(lab), range(1, nlab + 1)):
        comp = lab[sl] == k
        if (comp.sum() + 1) * dx * 1.5 < min_length:        # cannot be long enough
            continue
        iy, ix = np.nonzero(comp)
        path = _longest_path(iy, ix, np.where(core[sl][iy, ix], 1.0, 4.0))
        main = LineString(world(iy[path], ix[path], sl))
        if main.length < min_length:
            continue
        mpts = np.array(main.simplify(tol).coords)
        snap_end(mpts, 0, 1)
        snap_end(mpts, -1, -2)
        # side branches: what is left of the group more than 2 pixels from the main path
        used = np.zeros(comp.shape, bool)
        used[iy[path], ix[path]] = True
        rest = comp & ~ndimage.binary_dilation(used, structure=s8, iterations=2)
        blab, nb = ndimage.label(rest, structure=s8)
        branches = []
        for b in range(1, nb + 1):
            by, bx = np.nonzero(blab == b)
            bp = _longest_path(by, bx, np.where(core[sl][by, bx], 1.0, 4.0))
            bxy = world(by[bp], bx[bp], sl)
            if len(bxy) < 2 or LineString(bxy).length < min_branch:
                continue
            ml = LineString(mpts)
            if ml.distance(Point(bxy[-1])) < ml.distance(Point(bxy[0])):
                bxy = bxy[::-1]
            if ml.distance(Point(bxy[0])) > 5 * dx + tol:
                continue                                    # not attached to the main line
            if ml.distance(Point(bxy[-1])) < max(width / 2, 4 * dx):
                continue                                    # runs alongside the main line: the crest band's own width
            # join: a new vertex on the main line where the branch meets it
            t = ml.project(Point(bxy[0]))
            j = np.array(ml.interpolate(t).coords[0])
            seg = np.cumsum(np.r_[0.0, np.hypot(*np.diff(mpts, axis=0).T)])
            near = np.argmin(np.abs(seg - t))
            if abs(seg[near] - t) < 2 * tol:
                j = mpts[near].copy()                       # reuse a nearby vertex
            else:
                mpts = np.insert(mpts, np.searchsorted(seg, t), j, axis=0)
            bpts = np.array(LineString(np.vstack([j, bxy])).simplify(tol).coords)
            snap_end(bpts, -1, -2)
            branches.append(bpts)
        lines.append(mpts)
        lines += branches
    return lines


if __name__ == '__main__':
    import sys
    import flat_plain as F
    terr = sys.argv[1] if len(sys.argv) > 1 else 'mixed'
    px, py = np.meshgrid(np.arange(0.5, F.L), np.arange(0.5, F.W))
    z = F.ground(terr)(px, py)
    lines = crest_lines(z, 1.0, min_length=float(sys.argv[2]) if len(sys.argv) > 2 else 50.0, boundary=[[0, 0], [F.L, 0], [F.L, F.W], [0, F.W]])
    print(f'{terr}: {len(lines)} crest lines')
    for ln in lines:
        length = np.hypot(*np.diff(ln, axis=0).T).sum()
        print(f'  {len(ln):3d} points, length {length:6.1f} m, x {ln[:, 0].min():7.1f}-{ln[:, 0].max():7.1f}, '
              f'y {ln[:, 1].min():6.1f}-{ln[:, 1].max():6.1f}')
