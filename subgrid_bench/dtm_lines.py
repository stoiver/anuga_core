"""dtm_lines.py: derive the mesh controls for a subgrid flood model straight from a
high-resolution DTM and a channel line.

A coarse subgrid cell has one water level, so a cell that straddles a barrier (a
bank top, a levee, the sill of a floodplain compartment) lets water through below
the barrier's crest. The cure is a mesh edge along every such barrier. This module
finds them:

  bank_lines     the top of each river bank, on transects of the channel line. The
                 ground between the two lines is the river region.
  sill_lines     the divides between the river and the floodplain compartments (ground
                 below the level at which it first connects to the river), and between
                 one compartment and the next. Found by a watershed, so they are
                 continuous and follow broad banks as well as narrow levees.
  crest_lines    narrow ridges (crest_lines.py), which slow shallow flow even where
                 they do not close a compartment.
  suggest_areas  maximum triangle areas for the river region and the floodplain, from
                 the width between the bank lines.

`derive` runs them all. Rasters are (ny, nx) with row 0 at the SOUTH; pixel (i, j) is
centred at origin + ((j + 0.5) dx, (i + 0.5) dx). NaN marks ground outside the domain.
Needs numpy, scipy and shapely.
"""
import heapq

import numpy as np
from scipy import ndimage
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import dijkstra

from crest_lines import _longest_path, crest_lines

_NBR8 = ((-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1))


def _sample(z, dx, origin, x, y):
    """Bilinear z at points (NaN outside the raster or next to missing ground)."""
    c = (np.asarray(x, float) - origin[0]) / dx - 0.5
    r = (np.asarray(y, float) - origin[1]) / dx - 0.5
    bad = np.isnan(z)
    v = ndimage.map_coordinates(np.where(bad, 0.0, z), [r, c], order=1, mode='nearest')
    w = ndimage.map_coordinates(bad.astype(float), [r, c], order=1, mode='nearest')
    out = (r < -0.5) | (c < -0.5) | (r > z.shape[0] - 0.5) | (c > z.shape[1] - 0.5)
    return np.where((w > 0.0) | out, np.nan, v)


def extend_to_boundary(z, dx, origin, line, reach=600.0, penalty=20.0):
    """Carry both ends of a channel line on to the edge of the DTM (the first missing
    or outermost pixel) along the cheapest low path, so the bank lines reach the
    domain boundary. line: (n, 2) array, returned extended. An end already within
    2 dx of the edge is left alone."""
    line = np.asarray(line, float)
    ny, nx = z.shape
    valid = ~np.isnan(z)
    edge = valid & ~ndimage.binary_erosion(valid, structure=np.ones((3, 3)), border_value=0)

    def one(p, back):
        j0, i0 = int((p[0] - origin[0]) / dx), int((p[1] - origin[1]) / dx)
        h = int(reach / dx)
        a, b = max(i0 - h, 0), min(i0 + h + 1, ny)
        c, d = max(j0 - h, 0), min(j0 + h + 1, nx)
        zz, vv, ee = z[a:b, c:d], valid[a:b, c:d], edge[a:b, c:d]
        ii, jj = np.indices(zz.shape)
        si, sj = i0 - a, j0 - c
        if not vv[si, sj]:
            return []
        if ee[max(si - 2, 0):si + 3, max(sj - 2, 0):sj + 3].any():
            return []
        u = (p - back) / max(np.hypot(*(p - back)), 1e-9)
        ahead = (jj - sj) * u[0] + (ii - si) * u[1]
        zref = np.nanmin(zz[max(si - 3, 0):si + 4, max(sj - 3, 0):sj + 4])
        cost = np.where(vv, 1.0 + penalty * np.maximum(zz - zref - 0.3, 0.0), np.inf)
        idx = np.full(zz.shape, -1)
        idx[vv] = np.arange(vv.sum())
        rows, cols, wts = [], [], []
        for di, dj in ((0, 1), (1, -1), (1, 0), (1, 1)):
            s1 = (slice(0, zz.shape[0] - di), slice(max(-dj, 0), zz.shape[1] - max(dj, 0)))
            s2 = (slice(di, zz.shape[0]), slice(max(dj, 0), zz.shape[1] - max(-dj, 0)))
            ok = vv[s1] & vv[s2]
            w = 0.5 * (cost[s1] + cost[s2])[ok] * np.hypot(di, dj)
            rows += [idx[s1][ok], idx[s2][ok]]
            cols += [idx[s2][ok], idx[s1][ok]]
            wts += [w, w]
        n = int(vv.sum())
        g = coo_matrix((np.concatenate(wts), (np.concatenate(rows), np.concatenate(cols))), shape=(n, n)).tocsr()
        dist, pred = dijkstra(g, indices=idx[si, sj], return_predecessors=True)
        tgt = ee & (ahead > 3)
        if not tgt.any():
            return []
        t = idx[tgt][np.argmin(dist[idx[tgt]])]
        if not np.isfinite(dist[t]):
            return []
        path = [t]
        while pred[path[-1]] >= 0:
            path.append(pred[path[-1]])
        fi, fj = ii[vv], jj[vv]
        pts = np.c_[origin[0] + (fj[path] + c + 0.5) * dx, origin[1] + (fi[path] + a + 0.5) * dx][::-1]
        from shapely.geometry import LineString
        return list(np.array(LineString(pts).simplify(2 * dx).coords)[1:])

    k = min(3, len(line) - 1)
    head = one(line[0], line[k])
    tail = one(line[-1], line[-1 - k])
    return np.array([list(q) for q in head[::-1]] + [list(q) for q in line] + [list(q) for q in tail])


def bank_lines(z, dx, origin, channel, step=10.0, drop=0.3, behind=80.0, reach=150.0,
               min_height=1.0, smooth=9, outlier=1.8, outlier_window=41, break_slope=0.08, near_break=30.0, min_dist=10.0):
    """Bank-top line on each side of a channel line.

    On transects every `step` m, going outward from the channel line, the bank top
    is the first crest that stands at least `min_height` above the channel bed and
    has ground at least `drop` lower behind it within `behind` m. A transect with no
    such crest (the ground just keeps rising) takes the first point as high above
    the bed as its neighbours' bank tops, or failing that their distance. The break of slope is
    where the steep bank face ends (slope under `break_slope` over 4 m, at least
    `min_height` above the bed); a crest more than `near_break` m beyond it is a
    feature of the terrace, not the bank, and the break is used instead. A bank
    line is never closer than `min_dist` to the channel line. A top more
    than `outlier` times as far out as the median of its neighbours (over
    `outlier_window` transects) is treated the same way.

    Returns dict: left, right ((n, 2) arrays, in the direction of the channel line),
    left_found, right_found (bool), left_d, right_d (distance from the channel
    line), s (chainage of the transects), bed (channel bed level).
    """
    import shapely
    from shapely.geometry import LineString
    line = LineString(np.asarray(channel, float))
    s = np.arange(0.0, line.length + 1e-6, step)
    P = shapely.get_coordinates(shapely.line_interpolate_point(line, s))
    A = shapely.get_coordinates(shapely.line_interpolate_point(line, np.maximum(s - 15, 0)))
    B = shapely.get_coordinates(shapely.line_interpolate_point(line, np.minimum(s + 15, line.length)))
    t = (B - A) / np.hypot(*(B - A).T)[:, None]
    d = np.arange(0.0, reach + behind, dx)
    out = {'s': s}
    prof = {}
    for nm, sg in (('left', 1.0), ('right', -1.0)):
        n = np.c_[-t[:, 1], t[:, 0]] * sg
        zz = _sample(z, dx, origin, P[:, 0, None] + n[:, 0, None] * d, P[:, 1, None] + n[:, 1, None] * d)
        prof[nm] = (n, zz)
    w = max(int(10.0 / dx), 1)                      # channel bed: lowest ground within 10 m of the line
    bed = np.nanmin(np.c_[prof['left'][1][:, :w], prof['right'][1][:, :w]], axis=1)
    bed = np.where(np.isnan(bed), np.nanmedian(bed), bed)
    out['bed'] = bed
    nb, nr = int(behind / dx), int(reach / dx)
    for nm in ('left', 'right'):
        n, zz = prof[nm]
        zz = ndimage.uniform_filter1d(np.where(np.isnan(zz), -9999.0, zz), 3, axis=1)
        zz[zz < -9000] = np.nan
        dist = np.full(len(s), np.nan)
        top = np.full(len(s), np.nan)
        brk = np.full(len(s), np.nan)
        for i in range(len(s)):
            p = zz[i]
            if np.isnan(p).any():
                p = p[:int(np.argmax(np.isnan(p)))]
            if len(p) < 5:
                continue
            M = np.maximum.accumulate(p)
            jm = np.maximum.accumulate(np.where(p >= M, np.arange(len(p)), 0))     # index of the running maximum
            k = np.where((p < M - drop) & (M - bed[i] >= min_height) & (np.arange(len(p)) - jm <= nb) & (jm <= nr))[0]
            if len(k):
                dist[i] = jm[k[0]] * dx
                top[i] = M[k[0]]
            # break of slope: where the steep bank face ends
            m4 = max(int(4.0 / dx), 1)
            flat = [] if break_slope is None else np.where(
                (p[:-m4] - bed[i] >= min_height) & ((p[m4:] - p[:-m4]) / (m4 * dx) < break_slope))[0]
            if len(flat) and flat[0] <= nr:
                brk[i] = flat[0] * dx
                if np.isnan(dist[i]) or dist[i] - brk[i] > near_break:
                    dist[i] = np.nan            # no crest, or one far back on the terrace: use the break
        found = ~np.isnan(dist)
        for _ in range(2):          # a top far beyond its neighbours (a point bar, a side valley) is not the bank
            if found.sum() < 2:
                break
            near = ndimage.median_filter(np.interp(s, s[found], dist[found]), outlier_window, mode='nearest')
            found &= ~(dist > np.maximum(outlier * near, near + 15.0))
        if found.sum() < 2:
            raise ValueError('no bank top found on the %s bank: check the channel line and min_height' % nm)
        dist = np.where(found, dist, np.nan)
        f = np.where(found, dist, np.interp(s, s[found], dist[found]))
        h = np.interp(s, s[found], (top - bed)[found])          # bank height where none was found
        for i in np.where(~found)[0]:
            p = zz[i, :nr]
            k = np.where(p >= bed[i] + h[i])[0]
            if len(k):
                f[i] = min(f[i], k[0] * dx)
            if not np.isnan(brk[i]):
                f[i] = min(f[i], brk[i])
        f = np.maximum(ndimage.median_filter(f, smooth, mode='nearest'), min_dist)
        out[nm] = np.c_[P[:, 0] + n[:, 0] * f, P[:, 1] + n[:, 1] * f]
        out[nm + '_found'] = found
        out[nm + '_d'] = f
    return out


def snap_to_crest(z, dx, origin, line, reach=6.0, window=2.0, smooth=3):
    """Move each vertex of a bank line across the line to the highest ground within reach.

    A bank line found section by section can cut across a low point beside the crest; a wall built on it
    then has a notch. For each vertex the ground (highest within window, so a narrow crest is seen) is
    sampled along the normal out to reach on both sides and the vertex goes to the highest sample; ties
    keep the vertex where it is. The offsets get a running median over smooth vertices so the line
    does not zigzag. Returns (line, rise): the moved line and the gain in crest level at each vertex."""
    from scipy.ndimage import maximum_filter, median_filter
    line = np.asarray(line, float)
    k = max(int(round(window / dx)), 0)
    zc = maximum_filter(np.where(np.isnan(z), -np.inf, z), size=2 * k + 1) if k else z
    t = np.gradient(line, axis=0)
    t /= np.maximum(np.hypot(t[:, 0], t[:, 1]), 1e-9)[:, None]
    nrm = np.c_[-t[:, 1], t[:, 0]]
    offs = np.arange(-reach, reach + 0.5 * dx, dx)
    offs = offs[np.argsort(np.abs(offs), kind='stable')]            # nearest first: ties stay put
    def level(o):
        x, y = line[:, 0] + o * nrm[:, 0], line[:, 1] + o * nrm[:, 1]
        j = np.clip(((x - origin[0]) / dx).astype(int), 0, z.shape[1] - 1)
        i = np.clip(((y - origin[1]) / dx).astype(int), 0, z.shape[0] - 1)
        v = zc[i, j]
        return np.where(np.isfinite(v), v, -np.inf)
    v = np.array([level(o) for o in offs])
    best = offs[np.argmax(v > v.max(0) - 0.02, axis=0)]
    if smooth > 1:
        best = median_filter(best, size=smooth, mode='nearest')
    out = line + best[:, None] * nrm
    j = np.clip(((out[:, 0] - origin[0]) / dx).astype(int), 0, z.shape[1] - 1)
    i = np.clip(((out[:, 1] - origin[1]) / dx).astype(int), 0, z.shape[0] - 1)
    return out, zc[i, j] - v[0]


def river_polygon(banks):
    """Polygon between the two bank lines (largest piece, small notches closed)."""
    from shapely.geometry import Polygon
    u = Polygon(np.vstack([banks['left'], banks['right'][::-1]])).buffer(0)
    if u.geom_type != 'Polygon':
        u = max(u.geoms, key=lambda p: p.area)
    return Polygon(u.buffer(3.0).buffer(-3.0).exterior)


def raster_mask(z, dx, origin, polygon):
    """Bool raster: pixel centre inside a shapely polygon."""
    import shapely
    jj, ii = np.meshgrid(np.arange(z.shape[1]), np.arange(z.shape[0]))
    return shapely.contains_xy(polygon, origin[0] + (jj + 0.5) * dx, origin[1] + (ii + 0.5) * dx)


def spill_level(z, seed, level=None):
    """Lowest water level at which each pixel connects to the seed pixels (minimax of
    z over 8-connected paths), and the ground level of the seed pixel it connects
    to (or of `level` there, if given). NaN ground is never crossed.
    Returns (spill, source_level)."""
    ny, nx = z.shape
    Z = np.where(np.isnan(z), np.inf, z)
    sp = np.full(z.shape, np.inf)
    src = np.full(z.shape, np.nan)
    sp[seed] = Z[seed]
    src[seed] = Z[seed] if level is None else level[seed]
    done = np.zeros(z.shape, bool)
    h = [(sp[i, j], i, j) for i, j in zip(*np.nonzero(seed & (Z < np.inf)))]
    heapq.heapify(h)
    while h:
        v0, i, j = heapq.heappop(h)
        if done[i, j]:
            continue
        done[i, j] = True
        for di, dj in _NBR8:
            a, b = i + di, j + dj
            if 0 <= a < ny and 0 <= b < nx and not done[a, b] and Z[a, b] < np.inf:
                v = v0 if v0 > Z[a, b] else Z[a, b]
                if v < sp[a, b]:
                    sp[a, b] = v
                    src[a, b] = src[i, j]
                    heapq.heappush(h, (v, a, b))
    return sp, src


def compartments(z, dx, river, min_depth=0.3, min_area=2000.0, max_height=3.0, bank_window=80.0):
    """Label raster of the river (1) and the floodplain compartments (2, 3, ...), and
    the spill raster. A compartment is a connected patch of ground more than
    `min_depth` below its spill level from the river, at least `min_area` m2. Every
    other pixel less than `max_height` above the bank level (the highest river-region
    ground within `bank_window`) of the river reach it connects to is given
    to the river or compartment whose water reaches it first (watershed); higher
    ground is 0."""
    k = max(int(bank_window / dx), 1) | 1         # bank level: highest river-region ground nearby
    bank = ndimage.maximum_filter(np.where(river & ~np.isnan(z), z, -np.inf), size=k, mode='nearest')
    spill, src = spill_level(z, river, level=bank)
    pond = (spill - z > min_depth) & ~river & np.isfinite(spill)
    lab, n = ndimage.label(pond, structure=np.ones((3, 3)))
    if n:
        size = ndimage.sum(pond, lab, index=np.arange(1, n + 1)) * dx * dx
        keep = np.r_[False, size >= min_area]
        lab = np.where(keep[lab], lab, 0)
    ids = np.unique(lab[lab > 0])
    markers = np.zeros(z.shape, np.int32)
    for new, old in enumerate(ids, start=2):
        markers[lab == old] = new
    markers[river] = 1
    zone = np.isfinite(spill) & ((z - src < max_height) | (markers > 0))
    zmin = np.nanmin(z[zone])
    q = np.clip(np.round((np.where(zone, z, zmin) - zmin) * 100.0), 0, 65000).astype(np.uint16)
    q[~zone] = 65535
    out = ndimage.watershed_ift(q, markers, structure=np.ones((3, 3)))
    out[~zone] = 0
    return out, spill


def sill_lines(z, dx, origin, labels, min_length=60.0, tol=3.0, min_rise=0.15):
    """Polylines along the divides between neighbouring labels (see `compartments`).

    One line per connected stretch of divide between each pair of labels, at least
    `min_length` long. A divide is kept only if its crest stands at least `min_rise`
    above the lowest ground of the lower of the two sides within 30 m, so the flat
    contact between two patches of the same slope is not a line.
    Returns a list of (n, 2) arrays and a list of (label a, label b) pairs.
    """
    from shapely.geometry import LineString
    ny, nx = labels.shape
    pair = {}
    for di, dj in ((0, 1), (1, 0)):
        a = labels[:ny - di, :nx - dj]
        b = labels[di:, dj:]
        ok = (a != b) & (a > 0) & (b > 0)
        ii, jj = np.nonzero(ok)
        za, zb = z[:ny - di, :nx - dj][ok], z[di:, dj:][ok]
        hi = zb > za                                   # the divide pixel is the higher of the two
        pi, pj = ii + di * hi, jj + dj * hi
        la, lb = np.minimum(a[ok], b[ok]), np.maximum(a[ok], b[ok])
        for key in set(zip(la.tolist(), lb.tolist())):
            sel = (la == key[0]) & (lb == key[1])
            pair.setdefault(key, []).append((pi[sel], pj[sel]))
    zf = np.where(np.isnan(z), np.inf, z)
    w = max(int(30.0 / dx), 1) * 2 + 1
    lines, pairs = [], []
    for key, parts in pair.items():
        m = np.zeros(labels.shape, bool)
        for pi, pj in parts:
            m[pi, pj] = True
        m = ndimage.binary_dilation(m, structure=np.ones((3, 3)))
        lab, n = ndimage.label(m, structure=np.ones((3, 3)))
        for sl, k in zip(ndimage.find_objects(lab), range(1, n + 1)):
            comp = lab[sl] == k
            if comp.sum() * dx < min_length:
                continue
            iy, ix = np.nonzero(comp)
            path = _longest_path(iy, ix, 1.0 + 5.0 * (zf[sl][iy, ix].max() - zf[sl][iy, ix]))
            pi, pj = iy[path] + sl[0].start, ix[path] + sl[1].start
            xy = np.c_[origin[0] + (pj + 0.5) * dx, origin[1] + (pi + 0.5) * dx]
            if len(xy) < 2 or LineString(xy).length < min_length:
                continue
            # rise of the crest above the low ground on each side
            big = (slice(max(sl[0].start - w, 0), sl[0].stop + w), slice(max(sl[1].start - w, 0), sl[1].stop + w))
            near = ndimage.binary_dilation(lab[big] == k, structure=np.ones((3, 3)), iterations=w // 2)
            lo = [np.min(zf[big][near & (labels[big] == c)], initial=np.inf) for c in key]
            crest = np.median(z[pi, pj])
            if not np.isfinite(max(lo)) or crest - max(lo) < min_rise:
                continue
            lines.append(np.array(LineString(xy).simplify(tol).coords))
            pairs.append(key)
    return lines, pairs


def rim_lines(z, dx, origin, mask=None, reach=6.0, min_drop=0.5, top_tol=0.3, min_length=100.0, tol=3.0,
              smooth=2.0, bridge=6.0):
    """Polylines along the top edge of scarps: where nearly level ground drops at least
    min_drop within reach (the rim of a terrace or bench above lower ground).

    A cell that lies across such an edge holds one water level, so water arriving on the
    upper ground is at once on the lower ground: the scarp needs a mesh edge even though
    it is neither a crest nor a sill. A pixel is on a rim if the lowest ground within
    reach is min_drop or more below it and no ground within reach is more than top_tol
    above it (so the middle of a long slope is not a rim). Each connected band of such
    pixels, at least min_length long, gives one line, kept to the pixels nearest the
    drop. The ground is first smoothed over smooth (m) so that roughness does not break
    the band, and breaks shorter than bridge (m) are joined. mask: bool raster of the
    pixels allowed to carry a line.
    """
    from shapely.geometry import LineString
    k = max(int(round(reach / dx)), 1)
    bad = np.isnan(z)
    if smooth and smooth > dx:                       # normalised smoothing: missing ground does not spread
        wt = ndimage.gaussian_filter((~bad).astype(float), smooth / dx / 2.0)
        z = ndimage.gaussian_filter(np.where(bad, 0.0, z), smooth / dx / 2.0) / np.maximum(wt, 1e-6)
    zf = np.where(bad, np.inf, z)
    lo = ndimage.minimum_filter(zf, size=2 * k + 1)
    hi = ndimage.maximum_filter(np.where(bad, -np.inf, z), size=2 * k + 1)
    near = ndimage.minimum_filter(zf, size=3)                      # lowest ground next to the pixel
    with np.errstate(invalid='ignore'):
        band = ~bad & (zf - lo >= min_drop) & (hi - zf <= top_tol)
    if bridge and bridge > dx:
        b = max(int(round(bridge / dx / 2.0)), 1)
        band = ndimage.binary_closing(band, structure=np.ones((3, 3)), iterations=b) & ~bad
    if mask is not None:
        band &= mask
    lab, n = ndimage.label(band, structure=np.ones((3, 3)))
    lines = []
    for sl, c in zip(ndimage.find_objects(lab), range(1, n + 1)):
        comp = lab[sl] == c
        if comp.sum() * dx < min_length:
            continue
        iy, ix = np.nonzero(comp)
        step = (zf[sl] - near[sl])[iy, ix]                           # large on the pixels at the very edge
        path = _longest_path(iy, ix, 1.0 + 5.0 * (step.max() - step))
        xy = np.c_[origin[0] + (ix[path] + sl[1].start + 0.5) * dx, origin[1] + (iy[path] + sl[0].start + 0.5) * dx]
        if len(xy) >= 2 and LineString(xy).length >= min_length:
            lines.append(np.array(LineString(xy).simplify(tol).coords))
    return lines


def _near_lines(shape, dx, origin, lines, dist):
    """Bool raster: pixel within dist of any of the polylines."""
    m = np.zeros(shape, bool)
    for ln in lines:
        ln = np.asarray(ln, float)
        seg = np.hypot(*np.diff(ln, axis=0).T)
        t = np.r_[0.0, np.cumsum(seg)]
        s = np.arange(0.0, t[-1] + 0.5 * dx, 0.5 * dx)
        x, y = np.interp(s, t, ln[:, 0]), np.interp(s, t, ln[:, 1])
        j = ((x - origin[0]) / dx).astype(int)
        i = ((y - origin[1]) / dx).astype(int)
        ok = (i >= 0) & (i < shape[0]) & (j >= 0) & (j < shape[1])
        m[i[ok], j[ok]] = True
    r = max(int(round(dist / dx)), 1)
    return ndimage.binary_dilation(m, structure=np.ones((3, 3)), iterations=r) if m.any() else m


def d8(z):
    """Flow directions and accumulation of raster z by a priority flood from the
    edge of the valid ground (the lowest edge pixels are the outlets). Each pixel
    drains to the neighbour the flood reached it from, so pits and flats are routed
    out over their sills. Returns (parent, acc): parent is the flat index of the
    pixel each pixel drains to (-1 at an outlet or missing ground), acc the number
    of pixels draining through it (itself included)."""
    ny, nx = z.shape
    valid = ~np.isnan(z)
    edge = valid & ~ndimage.binary_erosion(valid, structure=np.ones((3, 3)), border_value=0)
    Z = np.where(valid, z, np.inf)
    parent = np.full(ny * nx, -1, np.int64)
    done = ~valid
    done = done.copy()
    queued = edge.copy()
    h = [(Z[i, j], i, j) for i, j in zip(*np.nonzero(edge))]
    heapq.heapify(h)
    order = []
    while h:
        v0, i, j = heapq.heappop(h)
        if done[i, j]:
            continue
        done[i, j] = True
        order.append(i * nx + j)
        for di, dj in _NBR8:
            a, b = i + di, j + dj
            if 0 <= a < ny and 0 <= b < nx and not done[a, b] and not queued[a, b]:
                queued[a, b] = True
                parent[a * nx + b] = i * nx + j
                heapq.heappush(h, (v0 if v0 > Z[a, b] else Z[a, b], a, b))
    acc = np.zeros(ny * nx, np.int64)
    acc[order] = 1
    for k in order[::-1]:
        q = parent[k]
        if q >= 0:
            acc[q] += acc[k]
    return parent.reshape(ny, nx), acc.reshape(ny, nx)


def flow_path(parent, start):
    """Pixels (i, j) from pixel `start` (i, j) down the D8 directions to an outlet."""
    ny, nx = parent.shape
    k = start[0] * nx + start[1]
    out = []
    flat = parent.ravel()
    while k >= 0:
        out.append(k)
        k = flat[k]
    out = np.array(out)
    return out // nx, out % nx


def channel_from_inlet(z, dx, origin, parent, inlet, snap=40.0, tol=6.0):
    """Channel line from the inflow point down the D8 directions to the edge of the
    DTM. The inlet is first moved to the lowest ground within `snap` m. Returns an
    (n, 2) array, upstream first."""
    from shapely.geometry import LineString
    j0, i0 = int((inlet[0] - origin[0]) / dx), int((inlet[1] - origin[1]) / dx)
    h = int(snap / dx)
    a, c = max(i0 - h, 0), max(j0 - h, 0)
    win = np.where(np.isnan(z[a:i0 + h + 1, c:j0 + h + 1]), np.inf, z[a:i0 + h + 1, c:j0 + h + 1])
    wi, wj = np.unravel_index(np.argmin(win), win.shape)
    pi, pj = flow_path(parent, (a + wi, c + wj))
    xy = np.c_[origin[0] + (pj + 0.5) * dx, origin[1] + (pi + 0.5) * dx]
    return np.array(LineString(xy).simplify(tol).coords)


def centre_channel(channel, banks, tol=6.0):
    """Channel line moved to midway between its two bank lines."""
    from shapely.geometry import LineString
    mid = 0.5 * (banks['left'] + banks['right'])
    return np.array(LineString(mid).simplify(tol).coords)


def drainage_lines(z, dx, origin, parent, acc, min_catchment=10000.0, min_length=80.0, min_incision=0.15,
                   window=20.0, tol=3.0, mask=None):
    """Polylines along the drainage lines (ditches, side channels): pixels draining
    at least `min_catchment` m2, traced down the D8 directions from each head to the
    next line or the outlet. A line is kept only where it is a real thalweg, at
    least `min_incision` below the median ground within `window` m, and at least
    `min_length` long. mask: bool raster, lines are cut to it (for example the
    floodable ground outside the river region). Returns a list of (n, 2) arrays."""
    from shapely.geometry import LineString
    ny, nx = z.shape
    stream = (acc * dx * dx >= min_catchment) & ~np.isnan(z)
    w = max(int(window / dx), 1) | 1
    zf = np.where(np.isnan(z), np.nanmax(z), z)
    incised = ndimage.median_filter(zf, size=w, mode='nearest') - zf >= min_incision
    ok = stream & incised if mask is None else stream & incised & mask
    flat = parent.ravel()
    haschild = np.zeros(ny * nx, bool)
    idx = np.nonzero(stream.ravel())[0]
    up = flat[idx]
    haschild[up[(up >= 0)]] = True
    heads = idx[~haschild[idx]]
    heads = heads[np.argsort(-acc.ravel()[heads])]
    seen = np.zeros(ny * nx, bool)
    okf = ok.ravel()
    lines = []
    for hd in heads:
        k = hd
        run = []
        runs = []
        while k >= 0 and not seen[k]:
            seen[k] = True
            if okf[k]:
                run.append(k)
            elif run:
                runs.append(run)
                run = []
            k = flat[k]
        if run:
            if k >= 0:
                run.append(k)              # end on the line it joins
            runs.append(run)
        for r in runs:
            r = np.array(r)
            xy = np.c_[origin[0] + (r % nx + 0.5) * dx, origin[1] + (r // nx + 0.5) * dx]
            if len(xy) > 1 and LineString(xy).length >= min_length:
                lines.append(np.array(LineString(xy).simplify(tol).coords))
    return lines


def pour_points(z, dx, origin, labels, spill):
    """For each compartment (label >= 2): its sill level, area, and the lowest point
    of its rim (the pour point), found as the lowest pixel just outside the ponded
    ground that is at the sill level. Returns a list of dicts."""
    out = []
    for lab in range(2, int(labels.max()) + 1):
        pond = (labels == lab) & (spill - z > 0.05)
        if not pond.any():
            continue
        level = float(np.median(spill[pond]))
        rim = ndimage.binary_dilation(pond, structure=np.ones((3, 3)), iterations=2) & ~pond & ~np.isnan(z)
        cand = np.where(rim & (np.abs(z - level) < 0.05))
        if len(cand[0]) == 0:
            cand = np.where(rim)
        k = int(np.argmin(np.abs(z[cand] - level)))
        out.append(dict(label=lab, level=level, area=float(pond.sum() * dx * dx),
                        x=float(origin[0] + (cand[1][k] + 0.5) * dx), y=float(origin[1] + (cand[0][k] + 0.5) * dx)))
    return out



def suggest_areas(banks, fp_ratio=12.0):
    """Maximum triangle areas (river, floodplain): the river cells are equilateral
    triangles as wide as the median distance between the bank lines, rounded to one
    figure; the floodplain limit is fp_ratio times that."""
    width = float(np.median(banks['left_d'] + banks['right_d']))
    a = 0.433 * width ** 2
    p = 10.0 ** np.floor(np.log10(a))
    a = float(np.round(a / p) * p)
    return a, fp_ratio * a, width


def derive(z, dx, origin, channel=None, inlet=None, drainage=True, min_catchment=10000.0, step=10.0, drop=0.3, min_height=1.0, break_slope=0.08, min_depth=0.3,
           min_area=2000.0, max_height=3.0, crest_width=30.0, crest_prominence=0.3,
           min_length=100.0, crest_band=250.0, coarse=2, verbose=True,
           snap=6.0, rims=True, rim_drop=0.5, rim_reach=6.0):
    """All the mesh controls from a DTM and a channel line.

    z is used at full resolution for the bank lines and every `coarse`-th pixel
    (block minimum is NOT taken: plain subsampling, as the crest must survive) for
    the compartments, sills and crests. Returns a dict with channel (extended to the
    DTM edge), banks, river (shapely polygon), sills, sill_pairs, crests, labels,
    spill, river_area, fp_area, width.
    """
    from shapely.geometry import LineString
    zc = z[::coarse, ::coarse]
    dc = dx * coarse
    oc = (origin[0] + 0.5 * dx - 0.5 * dc, origin[1] + 0.5 * dx - 0.5 * dc)   # pixel (0, 0) keeps its centre
    parent = acc = None
    if channel is None or drainage:
        parent, acc = d8(zc)
    if channel is None:
        if inlet is None:
            raise ValueError('give a channel line or the inflow point')
        # the D8 path keeps to the channel; carry its upstream end on to the DTM edge
        ch = extend_to_boundary(z, dx, origin, channel_from_inlet(zc, dc, oc, parent, inlet))
    else:
        ch = extend_to_boundary(z, dx, origin, channel)
    banks = bank_lines(z, dx, origin, ch, step=step, drop=drop, min_height=min_height, break_slope=break_slope)
    if snap:                                   # keep the bank lines on the crest: no notch in the wall
        for side in ('left', 'right'):
            banks[side] = snap_to_crest(z, dx, origin, banks[side], reach=snap)[0]
    river = river_polygon(banks)
    rmask = raster_mask(zc, dc, oc, river) & ~np.isnan(zc)
    labels, spill = compartments(zc, dc, rmask, min_depth=min_depth, min_area=min_area, max_height=max_height)
    sills, pairs = sill_lines(zc, dc, oc, labels, min_length=min_length * 0.6)
    lo = np.nanmin(zc)
    cl = crest_lines(np.where(np.isnan(zc), lo - 5.0, zc), dc, width=crest_width, prominence=crest_prominence,
                     min_length=min_length, tol=4.0, origin=oc)
    zone = LineString(ch).buffer(crest_band)
    crests = []
    for ln in cl:
        g = LineString(ln).intersection(zone)
        for p in getattr(g, 'geoms', [g]):
            if p.geom_type == 'LineString' and p.length >= min_length:
                crests.append(np.array(p.coords))
    rim = []
    if rims:      # scarp tops in the flood zone, away from the river outline and from the lines already found
        ok = (labels > 0) & ~raster_mask(zc, dc, oc, river.buffer(10.0)) & raster_mask(zc, dc, oc, zone)
        ok &= ~_near_lines(zc.shape, dc, oc, list(crests) + list(sills), 10.0)
        rim = rim_lines(zc, dc, oc, mask=ok, reach=rim_reach, min_drop=rim_drop, min_length=min_length)
    drains = []
    if drainage:
        drains = drainage_lines(zc, dc, oc, parent, acc, min_catchment=min_catchment,
                                mask=(labels > 0) & ~raster_mask(zc, dc, oc, river.buffer(10.0)))
    pours = pour_points(zc, dc, oc, labels, spill)
    ra, fa, width = suggest_areas(banks)
    if verbose:
        print('%d drainage lines, %.1f km' % (len(drains), sum(LineString(d).length for d in drains) / 1e3))
        for nm in ('left', 'right'):
            print('%s bank: top found on %.0f%% of %d transects, median %.0f m from the channel line'
                  % (nm, 100 * banks[nm + '_found'].mean(), len(banks['s']), np.median(banks[nm + '_d'])))
        print('river region %.1f ha, median width %.0f m -> river area %g m2, floodplain area %g m2'
              % (river.area / 1e4, width, ra, fa))
        print('%d compartments; %d sill lines, %.1f km; %d crest lines, %.1f km'
              % (labels.max() - 1 if labels.max() > 1 else 0, len(sills), sum(LineString(s).length for s in sills) / 1e3,
                 len(crests), sum(LineString(c).length for c in crests) / 1e3))
    if verbose:
        print('%d rim lines, %.1f km' % (len(rim), sum(LineString(q).length for q in rim) / 1e3))
    return dict(channel=ch, banks=banks, river=river, sills=sills, sill_pairs=pairs, crests=crests, drains=drains, rims=rim,
                pours=pours, parent=parent, acc=acc, labels=labels,
                spill=spill, river_area=ra, fp_area=fa, width=width, dx_coarse=dc, origin_coarse=oc)


def _synthetic(nx=600, ny=300, dx=2.0):
    """Test terrain: a sloping valley with a meandering channel between levees, a
    floodplain compartment closed by a broad embankment on the left bank, and a
    narrow ridge on the right."""
    x = (np.arange(nx) + 0.5) * dx
    y = (np.arange(ny) + 0.5) * dx
    X, Y = np.meshgrid(x, y)
    yc = 300.0 + 60.0 * np.sin(2 * np.pi * X / 600.0)
    d = Y - yc
    z = 10.0 - 0.001 * X + 0.004 * np.abs(d)                               # valley floor
    z -= 3.0 * np.exp(-(d / 12.0) ** 2)                                    # channel
    z += 0.8 * np.exp(-((np.abs(d) - 28.0) / 5.0) ** 2)                    # levees
    z += 1.0 * np.exp(-((X - 500.0) / 15.0) ** 2) * (d > 28)               # broad embankments across the left floodplain
    z += 1.0 * np.exp(-((X - 800.0) / 15.0) ** 2) * (d > 28)
    z += 0.6 * np.exp(-((Y - 120.0 - 0.1 * X) / 3.0) ** 2) * (X > 300) * (X < 900) * (d < -60)   # narrow ridge, right
    z += 0.02 * Y * (d > 0) * 0 + 1.5 * (Y > 560)                          # valley side on the left
    chan = np.c_[x[::10], 300.0 + 60.0 * np.sin(2 * np.pi * x[::10] / 600.0)]
    return z, dx, (0.0, 0.0), chan[(chan[:, 0] > 150) & (chan[:, 0] < 1050)]


if __name__ == '__main__':
    z, dx, origin, chan = _synthetic()
    r = derive(z, dx, origin, chan, coarse=1, min_area=1000.0)
    r2 = derive(z, dx, origin, inlet=(160.0, 300.0 + 60.0 * np.sin(2 * np.pi * 160.0 / 600.0)), coarse=1, min_area=1000.0, verbose=False)
    from shapely.geometry import LineString as _L
    import shapely as _s
    dd = _s.distance(_s.points(r2['channel']), _L(chan))
    print('channel from the inlet by D8: %.0f m long, from x %.0f to %.0f, within %.1f m (median) / %.1f m (max) of the true line'
          % (_L(r2['channel']).length, r2['channel'][:, 0].min(), r2['channel'][:, 0].max(), np.median(dd), dd.max()))
    print('pour points:', [(q['label'], round(q['level'], 2), round(q['x']), round(q['y'])) for q in r['pours']])
    b = r['banks']
    print('bank distance left/right median %.0f / %.0f m (levee crests are 28 m out)' % (np.median(b['left_d']), np.median(b['right_d'])))
    print('channel line from x %.0f to %.0f (raster 0 to %.0f)' % (r['channel'][:, 0].min(), r['channel'][:, 0].max(), z.shape[1] * dx))
    for s, p in zip(r['sills'], r['sill_pairs']):
        print('  sill between labels %s: %d points, x %.0f-%.0f, y %.0f-%.0f' % (p, len(s), s[:, 0].min(), s[:, 0].max(), s[:, 1].min(), s[:, 1].max()))
    # rim lines: a terrace 0.8 m above lower ground along the line y = 100 + 0.3 x, on a gentle slope
    xs = (np.arange(300) + 0.5) * 2.0
    Xs, Ys = np.meshgrid(xs, (np.arange(200) + 0.5) * 2.0)
    edge = Ys - (100.0 + 0.3 * Xs)
    zt = 5.0 + 0.002 * Ys + 0.8 / (1.0 + np.exp(-edge / 0.8))
    rl = rim_lines(zt, 2.0, (0.0, 0.0), min_drop=0.5)
    off = np.concatenate([q[:, 1] - (100.0 + 0.3 * q[:, 0]) for q in rl]) / np.hypot(1.0, 0.3)
    print('rim lines on a 0.8 m terrace edge: %d line(s), %.0f m (edge is %.0f m long), %.1f to %.1f m on the upper side of the edge'
          % (len(rl), sum(_L(q).length for q in rl), 600.0 * np.hypot(1.0, 0.3), off.min(), off.max()))
    assert len(rl) == 1 and 0.0 < off.min() and off.max() < 8.0
