"""Flood wave down a meandering channel and its floodplain: a public test of the subgrid scheme.

Terrain: examples/topo.asc of anugaSed (M. Perignon, MIT licence, https://github.com/mperignon/anugaSed), a
4 m DEM, 1.1 km square, of a channel about 10 m wide and 2 m deep crossing a floodplain between valley walls.
The mesh outline is that example's outline.csv. No sediment transport here.

A discharge hydrograph enters the channel near its upstream end (spread over the cells within --inflow-radius
of the lowest ground there) and leaves over a free outfall on the two downstream sides of the outline; the
other sides are walls. The bed is the DEM interpolated bilinearly, the same surface for every run.

    --reference AREA    standard ANUGA, one bed per cell (its centroid), triangles of at most AREA m2
    --area AREA         subgrid cells of at most AREA m2, tables from the DEM resampled to --pixel metres
        --tensor RING   friction from the conveyance tensor
        --mode unified  the unified kernels (the GPU on a GPU build)

Each run writes <out>/<name>/: the .sww, series.csv (volume held, inflow, boundary flux, wet area, levels at
gauges down the channel) and maxima.npz (maximum depth on the DEM pixels). Compare runs with
meander_floodplain_compare.py.

    python meander_floodplain.py --reference 8
    python meander_floodplain.py --area 400 --tensor 2
"""
import argparse
import csv
import sys
import time
from pathlib import Path

import numpy as np
from matplotlib.path import Path as MplPath
from matplotlib.tri import Triangulation
from scipy.interpolate import RegularGridInterpolator
from scipy.ndimage import distance_transform_edt

HERE = Path(__file__).resolve().parent
EXAMPLE = Path.home() / "Projects" / "Sediment_Transport" / "anugaSed" / "examples"
TAGS = {"bottom": [0], "side1": [1], "side2": [2], "top": [3], "side3": [4], "side4": [5]}
OPEN = ("bottom", "side4")                                    # the channel leaves over side4, the plain over bottom


def read_asc(path):
    """ESRI ASCII grid: (z with row 0 at the top and nan for nodata, xll, yll, cellsize)."""
    hdr = {}
    with open(path) as f:
        for _ in range(6):
            k, v = f.readline().split()
            hdr[k.lower()] = float(v)
    z = np.loadtxt(path, skiprows=6)
    z[z == hdr["nodata_value"]] = np.nan
    return z, hdr["xllcorner"], hdr["yllcorner"], hdr["cellsize"]


def subcell_beds(nodes, tri, n, zg, x0, ytop, pix):
    """Mean bed of each of the n*n sub-triangles of every cell from the pixels of zg (row 0 at the top, top-left
    corner x0, ytop): (ncell, n*n), nan where a sub-triangle holds no pixel; and the sub-centroids."""
    ntri = len(tri)
    p = nodes[tri]
    p0, e1, e2 = p[:, 0], p[:, 1] - p[:, 0], p[:, 2] - p[:, 0]
    det = e1[:, 0] * e2[:, 1] - e2[:, 0] * e1[:, 1]
    finder = Triangulation(nodes[:, 0], nodes[:, 1], tri).get_trifinder()
    nslot = 2 * n * n
    X, Y = np.meshgrid(x0 + (np.arange(zg.shape[1]) + 0.5) * pix, ytop - (np.arange(zg.shape[0]) + 0.5) * pix)
    px, py, pz = X.ravel(), Y.ravel(), zg.ravel().astype(float)
    t = finder(px, py)
    k = t >= 0
    px, py, pz, t = px[k], py[k], pz[k], t[k]
    dx, dy = px - p0[t, 0], py - p0[t, 1]
    u = np.clip((e2[t, 1] * dx - e2[t, 0] * dy) / det[t], 0, 1) * n
    v = np.clip((-e1[t, 1] * dx + e1[t, 0] * dy) / det[t], 0, 1) * n
    i = np.clip(np.floor(u), 0, n - 1).astype(np.int64)
    j = np.clip(np.floor(v), 0, n - 1).astype(np.int64)
    down = ((u - i) + (v - j) >= 1.0).astype(np.int64)
    down[i + j + down > n - 1] = 0
    over = i + j > n - 1
    j[over] = n - 1 - i[over]
    key = t.astype(np.int64) * nslot + 2 * (i * n + j) + down
    cnt = np.bincount(key, minlength=ntri * nslot)
    zsum = np.bincount(key, weights=pz, minlength=ntri * nslot)
    I, J = (a.ravel() for a in np.meshgrid(np.arange(n), np.arange(n), indexing="ij"))
    used = np.zeros(nslot, bool)
    cu = np.full((nslot, 2), np.nan)
    for down, off in ((0, 1.0 / 3), (1, 2.0 / 3)):
        ok = I + J <= n - 1 - down
        slot = 2 * (I * n + J) + down
        used[slot[ok]] = True
        cu[slot[ok], 0] = (I[ok] + off) / n
        cu[slot[ok], 1] = (J[ok] + off) / n
    cnt = cnt.reshape(ntri, nslot)[:, used]
    zsum = zsum.reshape(ntri, nslot)[:, used]
    sx = p0[:, None, 0] + cu[None, used, 0] * e1[:, None, 0] + cu[None, used, 1] * e2[:, None, 0]
    sy = p0[:, None, 1] + cu[None, used, 0] * e1[:, None, 1] + cu[None, used, 1] * e2[:, None, 1]
    with np.errstate(invalid="ignore", divide="ignore"):
        zsub = np.where(cnt > 0, zsum / cnt, np.nan)
    return zsub, sx, sy


def hydrograph(args):
    """Q(t): base flow, a linear rise to the peak, a hold, a linear fall back to the base flow."""
    tr, th, tf = 60 * args.rise, 60 * args.hold, 60 * args.fall
    ts = np.array([0.0, tr, tr + th, tr + th + tf, 1e12])
    qs = np.array([args.base, args.peak, args.peak, args.base, args.base])
    return (lambda t: float(np.interp(t, ts, qs))), ts, qs


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dem", type=Path, default=EXAMPLE / "topo.asc")
    ap.add_argument("--outline", type=Path, default=EXAMPLE / "outline.csv")
    ap.add_argument("--out", type=Path, default=HERE / "results_meander")
    ap.add_argument("--name")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--reference", type=float, metavar="AREA")
    g.add_argument("--area", type=float, metavar="AREA")
    ap.add_argument("--friction", type=float, default=0.035)
    ap.add_argument("--peak", type=float, default=60.0, help="m3/s")
    ap.add_argument("--base", type=float, default=3.0, help="m3/s")
    ap.add_argument("--rise", type=float, default=15.0, help="minutes")
    ap.add_argument("--hold", type=float, default=15.0, help="minutes")
    ap.add_argument("--fall", type=float, default=30.0, help="minutes")
    ap.add_argument("--minutes", type=float, default=120.0)
    ap.add_argument("--yieldstep", type=float, default=60.0)
    ap.add_argument("--outputstep", type=float, default=300.0)
    ap.add_argument("--inflow-radius", type=float, default=15.0)
    ap.add_argument("--pixel", type=float, default=1.0, help="subgrid: resample the DEM to this for the tables (m)")
    ap.add_argument("--nsub", type=int, default=8)
    ap.add_argument("--edge-nsub", type=int, default=16)
    ap.add_argument("--tensor", type=int, metavar="RING")
    ap.add_argument("--ponds", action="store_true",
                    help="ground below its spill level to the outfall is a second store that fills and drains over a weir; "
                         "tables, edges and tensor see the bed raised to the spill level")
    ap.add_argument("--controls", type=Path, metavar="JSON",
                    help="mesh controls from meander_controls.py: the river region (between the bank tops) is meshed at "
                         "--river-area, and the sills, crests and rims become breaklines, so mesh edges lie on them")
    ap.add_argument("--river-area", type=float, help="with --controls: maximum triangle area in the river region (default: the suggested one)")
    ap.add_argument("--clip", type=float, default=8.0, help="with --controls: the river region and breaklines stop this far inside the outline (m)")
    ap.add_argument("--walls", type=float, default=0.0, metavar="R",
                    help="with --controls: edges on the river outline and the breaklines take the highest DEM within R m as "
                         "their bed, so a line a little off the crest still stands as high as the crest")
    ap.add_argument("--wall-reach", type=float, default=0.0, metavar="W",
                    help="with --walls: and at least the highest DEM on a transect normal to the edge, W m either side")
    ap.add_argument("--tensor-form", choices=["gradient", "flow"], default="gradient")
    ap.add_argument("--mode", choices=["legacy", "unified"], default="legacy")
    ap.add_argument("--ngauge", type=int, default=8)
    ap.add_argument("--open", nargs="+", default=list(OPEN), choices=list(TAGS), help="sides with a free outfall")
    ap.add_argument("--low-froude", type=int, default=None, choices=[0, 1, 5],
                    help="flux form: 0 plain central-upwind, 1 low-Froude momentum diffusion, 5 the same with upwind "
                         "tangential momentum. Default: 1 for subgrid runs, 0 (ANUGA's default) for standard runs")
    args = ap.parse_args()

    import anuga
    sub = args.reference is None
    max_area = args.area if sub else args.reference
    name = args.name or (f"sub{max_area:g}" + (f"_t{args.tensor}" if args.tensor is not None else "") + ("_p" if args.ponds else "")
                        + ("_c" if args.controls else "") + ("w" if args.walls else "") if sub else f"ref{max_area:g}")
    outdir = args.out / name
    outdir.mkdir(parents=True, exist_ok=True)

    # the DEM, nodata filled from the nearest pixel, as one bilinear surface in coordinates relative to its corner
    zraw, xll, yll, pix0 = read_asc(args.dem)
    valid = np.isfinite(zraw)
    zf = zraw[tuple(distance_transform_edt(~valid, return_distances=False, return_indices=True))]
    ny0, nx0 = zf.shape
    xc0 = (np.arange(nx0) + 0.5) * pix0
    yc0 = (ny0 - np.arange(ny0) - 0.5) * pix0
    bed_i = RegularGridInterpolator((yc0[::-1], xc0), zf[::-1], bounds_error=False, fill_value=None)

    def bed(x, y):
        x, y = np.asarray(x, float), np.asarray(y, float)
        return bed_i(np.stack([y.ravel(), x.ravel()], -1)).reshape(x.shape)

    def manning(x, y):
        return np.full(np.shape(x), args.friction)

    poly = np.loadtxt(args.outline, delimiter=",") - [xll, yll]
    t0 = time.time()
    interior, brk, wall_lines = None, None, []
    if args.controls:
        import json
        from shapely.geometry import LineString, Polygon
        ctl = json.load(open(args.controls))
        keep = Polygon(poly).buffer(-args.clip)
        river = Polygon(ctl["river"]).buffer(0).intersection(keep)
        river = max(getattr(river, "geoms", [river]), key=lambda g: g.area).simplify(1.0)
        # crests and rims found along the edge of the data are artefacts: keep lines that stay 15 m inside it
        data = Polygon(poly).buffer(-15.0)
        lines = [ln for ln in ctl["sills"]] + [ln for ln in ctl["crests"] + ctl["rims"] if LineString(ln).within(data)]
        ra = args.river_area or ctl["river_area"]
        brk, interior = anuga.clean_breaklines(lines, [[[list(v) for v in list(river.exterior.coords)[:-1]], ra]],
                                               bounding_polygon=poly.tolist(), gap=10.0)
        wall_lines = [interior[0][0] + [interior[0][0][0]]] + [list(b) for b in brk]
        print(f"==> Controls: river region {river.area/1e4:.1f} ha at {ra:g} m2; {len(ctl['sills'])} sill lines and "
              f"{len(lines) - len(ctl['sills'])} of {len(ctl['crests']) + len(ctl['rims'])} crest and rim lines kept, "
              f"{len(brk)} breakline pieces, {sum(LineString(b).length for b in brk)/1e3:.2f} km", flush=True)
    domain = anuga.create_domain_from_regions(poly.tolist(), boundary_tags=TAGS, maximum_triangle_area=max_area,
                                              interior_regions=interior, breaklines=brk,
                                              mesh_filename=str(outdir / "mesh.msh"), verbose=False)
    if args.controls:                                         # gate: slivers from near-touching lines collapse the time step
        amin = 0.005 * min(max_area, ra)
        if domain.areas.min() < amin:
            sys.exit(f"mesh has {int((domain.areas < amin).sum())} cells under {amin:g} m2 (smallest {domain.areas.min():.3g} m2)")
    domain.set_name(name)
    domain.set_datadir(str(outdir))
    domain.set_store_vertices_smoothly(False)
    domain.set_quantities_to_be_stored({"elevation": 1, "stage": 2, "xmomentum": 2, "ymomentum": 2})
    domain.set_compute_mode(args.mode)
    n, area = len(domain), domain.areas
    nodes, tri = domain.nodes, domain.triangles
    cen = domain.centroid_coordinates
    print(f"==> {name}: {n:,} triangles, area 5/50/95% {np.percentile(area, [5, 50, 95]).round(1)} m2, "
          f"min radius {domain.radii.min():.2f} m, {time.time()-t0:.0f}s", flush=True)

    # pixels of the DEM that are real data inside the outline: where results are compared
    Xp, Yp = np.meshgrid(xc0, yc0)
    inside = valid & MplPath(poly).contains_points(np.c_[Xp.ravel(), Yp.ravel()]).reshape(valid.shape)
    # inflow: the lowest ground 40-70 m from the upstream corner of the outline; gauges: lowest ground in
    # bands of distance from it, as far as the outfall
    d3 = np.hypot(Xp - poly[3, 0], Yp - poly[3, 1])
    cand = inside & (d3 > 40) & (d3 < 70)
    k = np.argmin(np.where(cand, zf, np.inf))
    xin, yin = Xp.ravel()[k], Yp.ravel()[k]
    din = np.hypot(Xp - xin, Yp - yin)
    bands = np.linspace(60.0, din[inside].max() - 30.0, args.ngauge + 1)
    gxy = []
    for a, b in zip(bands[:-1], bands[1:]):
        k = np.argmin(np.where(inside & (din >= a) & (din < b), zf, np.inf))
        gxy.append((Xp.ravel()[k], Yp.ravel()[k]))
    gxy = np.array(gxy)
    finder = Triangulation(nodes[:, 0], nodes[:, 1], tri).get_trifinder()
    gid = finder(gxy[:, 0], gxy[:, 1])
    assert (gid >= 0).all()

    if sub:
        sys.path.insert(0, str(HERE))
        from subgrid_tables import SubgridTables, Subgrid_storage_operator, edge_subface_beds
        domain.set_flow_algorithm("DE0")
        domain.set_low_froude(1 if args.low_froude is None else args.low_froude)
        domain.set_betas(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        pix = args.pixel
        nyf, nxf = int(round(ny0 * pix0 / pix)), int(round(nx0 * pix0 / pix))
        xs = (np.arange(nxf) + 0.5) * pix
        ys = (nyf - np.arange(nyf) - 0.5) * pix
        zg = bed(*np.meshgrid(xs, ys))                        # row 0 at the top
        zsub, sx, sy = subcell_beds(nodes, tri, args.nsub, zg, 0.0, nyf * pix, pix)
        hole = np.isnan(zsub)
        zsub[hole] = bed(sx[hole], sy[hole])
        zsub_real, bed_edge, zraise = None, bed, None
        if args.ponds:
            from pond_storage import spill_levels, PondStore, Pond_exchange_operator
            ownp = finder(*(a.ravel() for a in np.meshgrid(xs, ys))).reshape(nyf, nxf)      # row 0 at the top, as zg
            out_cells = sorted({v for (v, _), tag in domain.boundary.items() if tag in args.open})
            spill = spill_levels(np.where(ownp >= 0, zg, np.nan), np.isin(ownp, out_cells))
            ok = np.isfinite(spill) & (ownp >= 0)
            zs_g = np.where(ok, np.maximum(spill, zg), zg)
            deep = (zs_g - zg)[ownp >= 0]
            print(f"==> Ponds: {100*(deep > 0.01).mean():.1f}% of the pixels lie below their spill level, holding "
                  f"{deep.sum()*pix*pix:,.0f} m3; deepest {deep.max():.2f} m", flush=True)
            raised_i = RegularGridInterpolator((ys[::-1], xs), zs_g[::-1], bounds_error=False, fill_value=None)

            def bed_edge(x, y):
                x, y = np.asarray(x, float), np.asarray(y, float)
                return raised_i(np.stack([y.ravel(), x.ravel()], -1)).reshape(x.shape)
            zsub_real = zsub
            zsub, _, _ = subcell_beds(nodes, tri, args.nsub, zs_g, 0.0, nyf * pix, pix)
            zsub[hole] = bed_edge(sx[hole], sy[hole])
            zsub = np.maximum(zsub, zsub_real)
            zg = zs_g                                         # the tensor sees the raised bed
            zraise = raised_i(np.stack([Yp[inside], Xp[inside]], -1)) - zf[inside]      # pond depth under each DEM pixel
        tables = SubgridTables(zsub, area, np.full(zsub.shape, args.friction))
        if args.tensor is not None:
            own = finder(*(a.ravel() for a in np.meshgrid(xs, ys[::-1]))).reshape(nyf, nxf).astype(np.int32)
            tables.add_tensor(own, own.shape, zg[::-1], args.friction, dx=pix, nlev=40, ring=args.tensor, workers=8)
        ez, ein = edge_subface_beds(domain, tables, args.edge_nsub, bed_edge, ds=0.5 * pix, manning=manning)
        if args.walls and wall_lines:
            import shapely
            from scipy import ndimage
            r = int(round(args.walls / pix))
            yy, xx = np.ogrid[-r:r + 1, -r:r + 1]
            zmx = ndimage.maximum_filter(zg, footprint=(xx * xx + yy * yy) <= r * r, mode="nearest")
            wall_i = RegularGridInterpolator((ys[::-1], xs), zmx[::-1], bounds_error=False, fill_value=None)

            def wall_bed(x, y):                               # rows are the sample points along one edge
                x, y = np.asarray(x, float), np.asarray(y, float)
                w = np.maximum(wall_i(np.stack([y.ravel(), x.ravel()], -1)).reshape(x.shape), bed_edge(x, y))
                if args.wall_reach:
                    ux, uy = x[:, -1] - x[:, 0], y[:, -1] - y[:, 0]
                    ln = np.maximum(np.hypot(ux, uy), 1e-9)
                    nx_, ny_ = (-uy / ln)[:, None], (ux / ln)[:, None]
                    for o in np.arange(-args.wall_reach, args.wall_reach + 0.01, 0.5):
                        w = np.maximum(w, bed_edge(x + o * nx_, y + o * ny_))
                return w
            ml = shapely.MultiLineString(wall_lines)
            pa = nodes[tri[:, [1, 2, 0]]].reshape(-1, 2)
            pb = nodes[tri[:, [2, 0, 1]]].reshape(-1, 2)
            on = np.ones(len(pa), bool)
            for q in (pa, pb, 0.5 * (pa + pb)):
                on &= shapely.distance(ml, shapely.points(q[:, 0], q[:, 1])) < 0.3
            wz, win = edge_subface_beds(domain, tables, args.edge_nsub, wall_bed, ds=0.5 * pix, manning=manning)
            print(f"==> Walls: {int(on.sum())} edge rows of {len(on)} on the river outline and breaklines "
                  f"({domain.edgelengths.reshape(-1)[on].sum()/2000:.2f} km), raised by median "
                  f"{np.median((wz - ez)[on].mean(axis=1)):.2f} m, 90% {np.percentile((wz - ez)[on].mean(axis=1), 90):.2f} m", flush=True)
            ez[on], ein[on] = wz[on], win[on]
        relief = tables.z[:, -1] - tables.z[:, 0]
        print(f"==> Tables from the DEM at {pix:g} m: {100*hole.mean():.1f}% of sub-triangles without a pixel; "
              f"relief inside a cell 5/50/95% {np.percentile(relief, [5, 50, 95]).round(2)} m", flush=True)
        zlow = tables.zmin
    else:
        if args.low_froude is not None:
            domain.set_low_froude(args.low_froude)
        domain.set_quantity("elevation", bed, location="centroids")
        domain.set_quantity("stage", expression="elevation")
        zlow = domain.quantities["elevation"].centroid_values
    domain.set_quantity("friction", args.friction)
    outfall = float(zf[inside].min()) - 10.0
    Br, Bo = anuga.Reflective_boundary(domain), anuga.Dirichlet_boundary([outfall, 0.0, 0.0])
    domain.set_boundary({tag: (Bo if tag in args.open else Br) for tag in TAGS})

    Q, ts, qs = hydrograph(args)
    cin = np.flatnonzero(np.hypot(cen[:, 0] - xin, cen[:, 1] - yin) <= args.inflow_radius)
    if len(cin) == 0:
        cin = np.array([int(finder(xin, yin))])
    ain = float(area[cin].sum())
    rop = anuga.Rate_operator(domain, rate=lambda t: Q(t) / ain, indices=cin.tolist())
    domain.set_evolve_max_timestep(1.0)                      # the rate is taken at the start of a step
    print(f"==> Inflow at ({xin:.0f}, {yin:.0f}), bed {bed(xin, yin):.2f} m, over {len(cin)} cells ({ain:.0f} m2): base {args.base:g}, "
          f"peak {args.peak:g} m3/s from {args.rise:g} to {args.rise+args.hold:g} min, base again at {args.rise+args.hold+args.fall:g} min; "
          f"n = {args.friction:g}; outfall on {args.open}; {domain.get_flow_algorithm()}, low_froude {domain.low_froude}, compute mode {args.mode}", flush=True)

    pond = pop = None
    if sub and zsub_real is not None:
        pond = PondStore(zsub_real, zsub, area, args.nsub)
        pop = Pond_exchange_operator(domain, tables, pond)   # before the storage operator: runs first
        print(f"==> Ponds: {len(pond.pc):,} in {len(pond.cells):,} cells, capacity {pond.vmax.sum():,.0f} m3", flush=True)
    if sub:
        op = Subgrid_storage_operator(domain, tables, cap_min_fraction=0.0, cfl=0.5, conveyance_friction=True,
                                      edge_z=ez, cap_exponent=0.0, drain_limiter=True, edge_invn=ein,
                                      tensor=args.tensor is not None, tensor_form=args.tensor_form)
        op.set_initial_level(tables.zmin)

    stage = domain.quantities["stage"].centroid_values
    elev = domain.quantities["elevation"].centroid_values
    # maximum depth on the DEM pixels: the level of the pixel's cell over the pixel's own bed (subgrid), or
    # the cell's depth (reference)
    pown = finder(Xp[inside], Yp[inside])
    pok = pown >= 0
    pown = np.maximum(pown, 0)
    pz = zf[inside]
    dmax = np.zeros(pz.shape, np.float32)
    tmax = np.zeros(pz.shape, np.float32)                     # time of first wetting (s), nan if never
    tmax[:] = np.nan
    rows, wall0, nsteps = [], time.time(), 0
    for t in domain.evolve(yieldstep=args.yieldstep, outputstep=args.outputstep, finaltime=args.minutes * 60.0):
        h = np.maximum(stage - elev, 0.0)
        if pop is not None:
            pop.sync()
        if sub and pond is not None:
            # depth over the raised bed where the cell is wet, plus the pond water under it, with every pond of
            # the cell taken to be as full as the cell's ponds are on the whole
            fill = np.bincount(pond.pc, weights=pond.V, minlength=n) / np.maximum(np.bincount(pond.pc, weights=pond.vmax, minlength=n), 1e-12)
            d = np.where(h[pown] > 1e-4, np.maximum(stage[pown] - (pz + zraise), 0.0), 0.0) + fill[pown] * zraise
            wet = float(tables.wet_area(stage)[h > 0.0].sum())
        elif sub:
            d = np.where(h[pown] > 1e-4, np.maximum(stage[pown] - pz, 0.0), 0.0)
            wet = float(tables.wet_area(stage)[h > 0.0].sum())
        else:
            d = h[pown]
            wet = float(area[h > 0.01].sum())
        d = np.where(pok, d, 0.0)
        np.maximum(dmax, d, out=dmax)
        first = np.isnan(tmax) & (d > 0.05)
        tmax[first] = t
        nsteps += domain.number_of_steps
        gs = [float(stage[k]) if stage[k] - zlow[k] > 0.01 else float("nan") for k in gid]
        rows.append([t, domain.number_of_steps, float((h * area).sum()) + (float(pond.V.sum()) if pond is not None else 0.0), float(rop.cumulative_influx), float(domain.get_boundary_flux_integral()),
                     wet, float((d > 0.05).sum()) * pix0 * pix0, time.time() - wall0] + gs)
        if abs(t / 600.0 - round(t / 600.0)) < 1e-9:
            r = rows[-1]
            print(f"   {t/60:5.0f} min  steps {nsteps:8d}  volume {r[2]:9.0f}  in {r[3]:9.0f}  boundary {r[4]:10.0f}  "
                  f"balance {r[2]-r[3]-r[4]:8.2f} m3  wet {r[6]/1e4:6.2f} ha  wall {r[7]:6.0f} s", flush=True)
    with open(outdir / "series.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["t_s", "steps", "volume_m3", "inflow_m3", "boundary_m3", "wet_m2", "wet_pixels_m2", "wall_s"]
                   + [f"gauge{i}" for i in range(len(gid))])
        w.writerows(rows)
    full = np.full(valid.shape, np.nan, np.float32)
    full[inside] = dmax
    tfull = np.full(valid.shape, np.nan, np.float32)
    tfull[inside] = tmax
    np.savez_compressed(outdir / "maxima.npz", depth=full, first_wet_s=tfull, x=xc0, y=yc0, bed=np.where(inside, zf, np.nan),
                        gauges=gxy, inflow=np.array([xin, yin]), ncell=n, steps=nsteps, wall=time.time() - wall0)
    print(f"==> Done: {nsteps:,} steps in {time.time()-wall0:,.0f} s; {outdir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
