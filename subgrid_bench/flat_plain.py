"""flat_plain.py: how fast does a shallow flood spread across a floodplain?

A level floodplain, 1500 m x 150 m, Manning n = 0.05, closed on three sides,
fed at x = 0 by a constant inflow Q (default 5 m3/s, 0.033 m2/s per metre
of width). Ground:

  flat     z = 0
  ridged   z = 0 plus a 0.15 m ridge (Gaussian, about 4 m wide) every 100 m
           across the full width: the sheet flow must pond and spill over each
  rough    random micro-relief, 0.1 m standard deviation, about 15 m correlation
           length (fixed seed)
  mixed    the ridges plus the random micro-relief

Meshes: plain ANUGA at --area (fine: the reference) or the subgrid EKLC
operator (8x8 sub-triangle tables, 16 subfaces per edge, conveyance friction,
draining limiter, conveyance split) on coarse cells.

Each minute it records the wet fraction of 1 m ground pixels in 10 m bands
along x (a pixel is wet when its cell holds water and the cell's level is
above the pixel by 0.01 m). The front is the furthest band more than half wet.

    python flat_plain.py --terrain flat --area 4               # plain reference
    python flat_plain.py --terrain flat --area 1000 --subgrid  # subgrid
Writes results/flat_<terrain>_<area>_<plain|sub>.npz.
"""
import argparse
import os
import time
import types

import numpy as np

import anuga

L, W, N = 1500.0, 150.0, float(os.environ.get("FLAT_N", 0.05))
BAND = 10.0


def ground(terrain):
    if terrain == 'flat':
        return lambda x, y: np.zeros(np.shape(x))
    if terrain == 'ridged':
        def z(x, y):
            d = np.mod(np.asarray(x, float), 100.0) - 50.0
            z = 0.15 * np.exp(-(d / 2.0) ** 2)
            # clip the tails: beds of 1e-26 m and below make the subgrid flux kernel give NaN
            return np.where(z > 1e-6, z, 0.0)
        return z
    if terrain == 'rough':
        from scipy.interpolate import RegularGridInterpolator
        from scipy.ndimage import gaussian_filter
        rng = np.random.default_rng(453)
        g = gaussian_filter(rng.standard_normal((int(W) + 41, int(L) + 41)), 6.0, mode='wrap')
        g *= 0.1 / g.std()
        f = RegularGridInterpolator((np.arange(g.shape[0]) - 20.0, np.arange(g.shape[1]) - 20.0), g,
                                    bounds_error=False, fill_value=None)   # unused sub-slots pass NaN

        def z(x, y):
            x, y = np.asarray(x, float), np.asarray(y, float)
            return f(np.stack([y.ravel(), x.ravel()], -1)).reshape(x.shape)
        return z
    if terrain == 'mixed':
        zr, zo = ground('ridged'), ground('rough')
        return lambda x, y: zr(x, y) + zo(x, y)
    raise ValueError(terrain)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--terrain', default='flat', choices=['flat', 'ridged', 'rough', 'mixed'])
    ap.add_argument('--area', type=float, default=1000.0)
    ap.add_argument('--subgrid', action='store_true')
    ap.add_argument('--no-conv', action='store_true', help='subgrid without conveyance friction or split (Manning on the mean depth)')
    ap.add_argument('--conn', action='store_true', help='subgrid with the edge-connectivity rule')
    ap.add_argument('--dir', action='store_true', help='subgrid with direction-dependent friction')
    ap.add_argument('--tensor', action='store_true', help='subgrid with the cell-problem conveyance tensor in the friction')
    ap.add_argument('--tring', type=int, default=0, help='oversampling ring for the cell problem, pixels')
    ap.add_argument('--tfloor', type=float, default=0.01, help='lowest eigenvalue of K*/Kc')
    ap.add_argument('--breaklines', action='store_true', help='mesh edges along the ridge crests (ridged ground)')
    ap.add_argument('--auto-breaklines', type=float, default=0.0, metavar='MIN_LENGTH',
                    help='mesh edges along crests found in the 1 m DTM (crest_lines.py), at least this long (m)')
    ap.add_argument('--crest-tol', type=float, default=3.0, help='simplification tolerance of the detected crest lines (m)')
    ap.add_argument('--crest-branch', type=float, default=None, help='shortest side branch kept (m; default a quarter of MIN_LENGTH)')
    ap.add_argument('--pond', action='store_true', help='subgrid with two-compartment storage (ponds behind sills fill through a weir)')
    ap.add_argument('--nsub', type=int, default=8, help='sub-triangle split n (n^2 per cell)')
    ap.add_argument('--q', type=float, default=5.0)
    ap.add_argument('--hours', type=float, default=4.0)
    args = ap.parse_args()
    bed = ground(args.terrain)
    tag = f'flat_{args.terrain}_{args.area:g}_{"sub" if args.subgrid else "plain"}' + ('nc' if args.no_conv else '') + ('C' if args.conn else '') + ('D' if args.dir else '') + ('T' if args.tensor else '') + (f'r{args.tring}' if args.tensor and args.tring else '') + (f'f{args.tfloor:g}' if args.tensor and args.tfloor != 0.01 else '') + ('P' if args.pond else '') + ('B' if args.breaklines else '') + (f'A{args.auto_breaklines:g}' if args.auto_breaklines else '') + (f't{args.crest_tol:g}' if args.crest_tol != 3.0 else '') + (f'b{args.crest_branch:g}' if args.crest_branch is not None else '') + (f'n{args.nsub}' if args.nsub != 8 else '') + {'parallel': '', 'geometric': 'G', 'series': 'S'}[os.environ.get('SUBGRID_KMEAN', 'parallel')]

    brk = [[[xr, 0.0], [xr, W]] for xr in np.arange(50.0, L, 100.0)] if args.breaklines else None
    if args.auto_breaklines:
        from crest_lines import crest_lines
        gx, gy = np.meshgrid(np.arange(0.5, L), np.arange(0.5, W))
        brk = [ln.tolist() for ln in crest_lines(bed(gx, gy), 1.0, min_length=args.auto_breaklines, tol=args.crest_tol, min_branch=args.crest_branch,
                                                 boundary=[[0, 0], [L, 0], [L, W], [0, W]])]
        print(f'crest lines: {len(brk)}, total length {sum(np.hypot(*np.diff(np.array(b), axis=0).T).sum() for b in brk):.0f} m', flush=True)
    domain = anuga.create_domain_from_regions([[0, 0], [L, 0], [L, W], [0, W]],
                                              {'bottom': [0], 'right': [1], 'top': [2], 'left': [3]},
                                              maximum_triangle_area=args.area,
                                              breaklines=brk or None)
    domain.set_name(tag)
    domain.set_store(False)
    domain.set_compute_mode('legacy' if args.subgrid else 'unified')
    x, y = domain.centroid_coordinates.T
    domain.set_quantity('friction', N, location='centroids')

    # 1 m ground pixels and their owning cells (for the wet fraction)
    px, py = [a.ravel() for a in np.meshgrid(np.arange(0.5, L), np.arange(0.5, W))]
    pz = bed(px, py)
    from static_storage import Cells
    cells = Cells(domain.nodes, domain.triangles, px, py)
    own = cells.owner
    band = (px // BAND).astype(int)
    nb = int(L // BAND)
    per_band = np.bincount(band, minlength=nb)

    op = None
    pond = None
    spix = pz
    ebed = bed
    if args.pond:
        from pond_storage import spill_levels, PondStore, Pond_exchange_operator
        nx, ny = int(L), int(W)
        spix = spill_levels(pz.reshape(ny, nx), (px < 2.0).reshape(ny, nx)).ravel()
        print(f'spill: {np.mean(spix > pz + 0.01) * 100:.0f}% of pixels lie below their spill level, '
              f'pond volume {np.sum(spix - pz):.0f} m3', flush=True)

        def ebed(x, y):
            x, y = np.asarray(x, float), np.asarray(y, float)
            ix = np.clip(np.nan_to_num(x).astype(int), 0, nx - 1)
            iy = np.clip(np.nan_to_num(y).astype(int), 0, ny - 1)
            return np.maximum(bed(x, y), spix[iy * nx + ix])
    if args.subgrid:
        import static_storage
        from subgrid_tables import SubgridTables, Subgrid_storage_operator, edge_subface_beds
        static_storage.T = types.SimpleNamespace(bed=bed)
        man = lambda a, b: np.full(np.shape(a), N)  # noqa: E731
        tables = SubgridTables.from_cells(cells, args.nsub, px, py, spix, manning=man)
        if args.pond:
            z_, _, _ = cells.subgrid(args.nsub, px, py, pz)
            s_, _, _ = cells.subgrid(args.nsub, px, py, spix)
            u_ = ~np.isnan(z_[0])
            pond = PondStore(z_[:, u_], s_[:, u_], cells.area, args.nsub)
            print(f'ponds in {len(pond.cells)} of {len(domain)} cells, capacity {pond.vmax.sum():.0f} m3, '
                  f'{len(pond.pc)} ponds, weir width median {np.median(pond.w) if len(pond.pc) else 0:.0f} m', flush=True)
        if args.dir:
            zs, _, _, sx, sy = cells.subgrid(args.nsub, px, py, pz, return_centroids=True)
            u = ~np.isnan(zs[0])
            tables.add_directional(sx[:, u], sy[:, u])
        if args.tensor:
            t1 = time.time()
            tables.add_tensor(own, (int(W), int(L)), pz, np.full(pz.shape, N), dx=1.0, floor=args.tfloor, ring=args.tring, workers=8)
            print(f'tensor tables: {time.time() - t1:.0f} s; R_xx at the top inner level, quartiles '
                  f'{np.round(np.percentile(tables.tR[:, 11, 0], [25, 50, 75]), 2)}, R_yy '
                  f'{np.round(np.percentile(tables.tR[:, 11, 2], [25, 50, 75]), 2)}', flush=True)
        if args.conn:
            zs, _, _ = cells.subgrid(args.nsub, px, py, pz)
            tables.add_connectivity(zs[:, ~np.isnan(zs[0])], args.nsub, domain)
        domain.set_quantity('elevation', tables.zmin, location='centroids')
        domain.set_quantity('stage', tables.zmin, location='centroids')
    else:
        domain.set_quantity('elevation', bed(x, y), location='centroids')
        domain.set_quantity('stage', bed(x, y), location='centroids')

    br = anuga.Reflective_boundary(domain)
    domain.set_boundary({'left': br, 'right': br, 'top': br, 'bottom': br})
    anuga.Inlet_operator(domain, [[1.0, 1.0], [1.0, W - 1.0]], Q=args.q)
    if args.subgrid:
        if args.no_conv:
            ez, ein = edge_subface_beds(domain, tables, 16, ebed, ds=0.5), None
        else:
            ez, ein = edge_subface_beds(domain, tables, 16, ebed, ds=0.5, manning=man)
        if pond is not None:
            pop = Pond_exchange_operator(domain, tables, pond)    # before the storage operator: runs first
        op = Subgrid_storage_operator(domain, tables, cap_min_fraction=0.0, cfl=0.75 if args.no_conv else 0.5,
                                      conveyance_friction=not args.no_conv, edge_z=ez, cap_exponent=0.0,
                                      drain_limiter=True, edge_invn=ein,
                                      connectivity=args.conn, directional=args.dir, tensor=args.tensor)
        op.set_initial_level(tables.zmin)

    stage = domain.quantities['stage']
    elev = domain.quantities['elevation']
    times, frac, vol, dprof = [], [], [], []
    t0 = time.time()
    steps = 0
    print(f'{tag}: {len(domain)} triangles, median area {np.median(domain.areas):.1f} m2', flush=True)
    for t in domain.evolve(yieldstep=60.0, finaltime=args.hours * 3600.0):
        steps += domain.number_of_steps
        s = stage.centroid_values
        h = s - elev.centroid_values
        wet = (h[own] > 1e-3) & (s[own] > pz + 0.01)
        # pixel depth: the cell level over the pixel (subgrid), the cell depth (plain)
        pd = np.maximum(s[own] - pz, 0.0) if args.subgrid else np.maximum(h[own], 0.0)
        pd = np.where(h[own] > 1e-6, pd, 0.0)
        if pond is not None:
            ep = pond.pixel_level(own, spix)
            pd = np.where(h[own] > 1e-6, np.maximum(s[own] - spix, 0.0), 0.0) + np.maximum(ep - pz, 0.0)
            wet = pd > 0.01
        dprof.append(np.bincount(band, weights=pd, minlength=nb) / per_band)
        times.append(t)
        frac.append(np.bincount(band[wet], minlength=nb) / per_band)
        vol.append(np.sum(np.maximum(h, 0) * domain.areas) + (pond.V.sum() if pond is not None else 0.0))
        if t % 1800 == 0:
            f = frac[-1]
            front = BAND * (np.nonzero(f > 0.5)[0].max() + 1) if (f > 0.5).any() else 0.0
            print(f't {t / 3600:4.2f} h  front {front:6.0f} m  volume {vol[-1]:9.0f} (in {args.q * t:9.0f})  '
                  f'steps {steps}  wall {time.time() - t0:5.0f} s', flush=True)
    os.makedirs('results', exist_ok=True)
    np.savez_compressed(f'results/{tag}.npz', t=np.array(times), frac=np.array(frac), vol=np.array(vol), dprof=np.array(dprof),
                        q=args.q, ntri=len(domain), steps=steps, wall=time.time() - t0)
    if op is not None and args.conn:
        print(f'connectivity: {op.reach_events} cell-steps reached new ground', flush=True)
    print('saved', f'results/{tag}.npz', flush=True)


if __name__ == '__main__':
    main()
