"""Figures for the subgrid status report (claude/SUBGRID_STATUS.md).

    python report_figures.py [--out ../claude/figures/subgrid] [--page-data DIR]

Reads results/*.npz (run_phase0.py --diag outputs, see the report for the list),
writes PNGs to --out, and with --page-data also the JSON the web version of the
report embeds (figs.json, hydro.json). Synthetic data only.
"""
import argparse
import json
import os
import sys

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
from matplotlib.path import Path
from matplotlib.tri import Triangulation

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import terrain as T                      # noqa: E402
from static_storage import Cells         # noqa: E402
from subgrid_tables import SubgridTables  # noqa: E402

N_CH, N_FP = 0.035, 0.05

def friction(x, y):
    return np.where(T.distance_to_centreline(x, y) < T.TOP_WIDTH / 2, N_CH, N_FP)


def load(name):
    return dict(np.load(f'results/{name}.npz'))


def q64(a, lo, hi):
    """Quantize to 64 levels as a string of chars (offset 48)."""
    k = np.clip(np.round((a - lo) / (hi - lo) * 63), 0, 63).astype(int)
    return ''.join(chr(48 + v) for v in k.ravel())



def extract():
    """Figure data: terrain, meshes, an example cell and edge, extent maps, profiles."""
    out = {}

    # 1. terrain, detrended (bed minus valley level), 4 m grid
    dx = 4.0
    xs = np.arange(dx / 2, T.LENGTH, dx)
    ys = np.arange(dx / 2, T.WIDTH, dx)
    X, Y = np.meshgrid(xs, ys)
    rel = T.bed(X, Y) - T.valley(X)
    out['terrain'] = dict(nx=len(xs), ny=len(ys), dx=dx, lo=-2.0, hi=1.0, q=q64(rel, -2.0, 1.0))
    cx = np.linspace(0, T.LENGTH, 401)
    out['centreline'] = np.round(np.column_stack([cx, T.centreline(cx)]), 1).tolist()
    band = np.array(T.channel_band_polygon())
    out['band_poly'] = np.round(band, 1).tolist()

    # 2. mesh windows
    W = (925.0, 175.0, 1075.0, 325.0)
    meshes = {}
    for key, name in (('band1', 'band1_diag_lf4'), ('c500', 'coarse500_sub8EKLC_diag'),
                      ('c2000', 'coarse2000_sub8EKLC_diag')):
        d = np.load(f'results/{name}.npz')
        nodes, tri = d['nodes'], d['triangles']
        p = nodes[tri]
        lo, hi = p.min(axis=1), p.max(axis=1)
        keep = (hi[:, 0] > W[0]) & (lo[:, 0] < W[2]) & (hi[:, 1] > W[1]) & (lo[:, 1] < W[3])
        pts = p[keep] - np.array([W[0], W[1]])
        meshes[key] = dict(ntri=int(len(tri)), path=' '.join(
            'M{:.1f} {:.1f}L{:.1f} {:.1f}L{:.1f} {:.1f}Z'.format(
                t[0, 0], 150 - t[0, 1], t[1, 0], 150 - t[1, 1], t[2, 0], 150 - t[2, 1]) for t in pts))
    wx = np.linspace(W[0], W[2], 151)
    out['mesh_window'] = dict(x0=W[0], y0=W[1], size=150.0,
                              cl=np.round(np.column_stack([wx - W[0], 150 - (T.centreline(wx) - W[1])]), 2).tolist(),
                              half_top=T.TOP_WIDTH / 2)
    out['meshes'] = meshes

    # 3. one coarse2000 cell crossed by the channel
    d2 = load('coarse2000_sub8EKLC_diag')
    nodes, tri = d2['nodes'], d2['triangles']
    finder = Triangulation(nodes[:, 0], nodes[:, 1], tri).get_trifinder()
    best = None
    for x0 in np.arange(900.0, 1100.0, 5.0):
        c = int(finder(x0, T.centreline(x0)))
        pc = nodes[tri[c]]
        zc = T.bed(*pc.mean(axis=0))
        # want the centroid on the floodplain (the coarse scheme misses the channel)
        dcen = T.distance_to_centreline(*pc.mean(axis=0))
        if best is None or dcen > best[1]:
            best = (c, dcen)
    cell = best[0]
    pc = nodes[tri[cell]]
    xmin, ymin = pc.min(axis=0) - 1
    xmax, ymax = pc.max(axis=0) + 1
    pxs = np.arange(np.floor(xmin / T.PIXEL) * T.PIXEL + T.PIXEL / 2, xmax, T.PIXEL)
    pys = np.arange(np.floor(ymin / T.PIXEL) * T.PIXEL + T.PIXEL / 2, ymax, T.PIXEL)
    PX, PY = np.meshgrid(pxs, pys)
    inside = Path(pc).contains_points(np.column_stack([PX.ravel(), PY.ravel()])).reshape(PX.shape)
    PZ = T.bed(PX, PY)
    one = Cells(pc, np.array([[0, 1, 2]]), PX[inside], PY[inside])
    nsub = 8
    zsub, _, _, sx, sy = one.subgrid(nsub, PX[inside], PY[inside], PZ[inside], return_centroids=True)
    used = ~np.isnan(zsub[0])
    # sub-triangle polygons in slot order
    e1, e2, p0 = one.e1[0], one.e2[0], one.p0[0]
    polys = []
    for i in range(nsub):
        for j in range(nsub):
            for down in (0, 1):
                if i + j > nsub - 1 - down:
                    polys.append(None)
                    continue
                bc = [(i, j), (i + 1, j), (i, j + 1)] if down == 0 else [(i + 1, j), (i + 1, j + 1), (i, j + 1)]
                polys.append([(p0 + a / nsub * e1 + b / nsub * e2).tolist() for a, b in bc])
    subs = [dict(p=np.round(np.array(polys[k]) - [xmin, ymin], 2).tolist(), z=round(float(zsub[0, k]), 3))
            for k in range(len(polys)) if used[k]]
    tables = SubgridTables(zsub[:, used], one.area, friction(sx[:, used], sy[:, used]))
    zc = float(T.bed(*pc.mean(axis=0)))
    z_lo = float(tables.zmin[0])
    etas = np.linspace(z_lo, zc + 1.5, 121)
    h = np.array([tables.depth(np.array([e]))[0] for e in etas])
    f = np.array([tables.wet_area(np.array([e]))[0] for e in etas]) / one.area[0]
    kc = np.array([tables.conveyance(np.array([e]))[0] for e in etas])
    nflat = float(friction(*pc.mean(axis=0)))
    peak_eta = float(d2['peak_stage'][cell])
    ref = load('band025_diag_tu')
    pix_ref = Triangulation(ref['nodes'][:, 0], ref['nodes'][:, 1], ref['triangles']).get_trifinder()
    out['cell'] = dict(
        tri=np.round(pc - [xmin, ymin], 2).tolist(), bw=float(xmax - xmin), bh=float(ymax - ymin),
        area=float(one.area[0]), zc=zc, zmin=z_lo, zmax=float(np.nanmax(zsub)), nflat=nflat,
        pixels=dict(nx=len(pxs), ny=len(pys), q=q64(np.where(inside, PZ, z_lo), z_lo, float(np.nanmax(PZ[inside]))),
                    mask=''.join('1' if v else '0' for v in inside.ravel()),
                    lo=z_lo, hi=float(np.nanmax(PZ[inside]))),
        subs=subs, eta=np.round(etas, 3).tolist(), h=np.round(h, 4).tolist(),
        f=np.round(f, 4).tolist(), kc=np.round(kc, 4).tolist(),
        h_flat=np.round(np.maximum(etas - zc, 0), 4).tolist(),
        kc_flat=np.round(np.maximum(etas - zc, 0) ** (5 / 3) / nflat, 4).tolist(),
        peak_eta=peak_eta, x0=float(xmin), y0=float(ymin),
        cl=np.round(np.column_stack([np.linspace(xmin, xmax, 60) - xmin,
                                      T.centreline(np.linspace(xmin, xmax, 60)) - ymin]), 2).tolist())

    # 4. the edge of that cell with the biggest bed range: 16 subfaces
    best_e = None
    for i in range(3):
        a, b = pc[(i + 1) % 3], pc[(i + 2) % 3]
        s = np.linspace(0, 1, 2001)
        zz = T.bed(a[0] + s * (b[0] - a[0]), a[1] + s * (b[1] - a[1]))
        if best_e is None or np.ptp(zz) > best_e[0]:
            best_e = (np.ptp(zz), a, b)
    _, a, b = best_e
    L = float(np.hypot(*(b - a)))
    s = np.linspace(0, 1, 1601)
    ex, ey = a[0] + s * (b[0] - a[0]), a[1] + s * (b[1] - a[1])
    prof = T.bed(ex, ey)
    m = 16
    per = 100
    sm = (np.arange(m * per) + 0.5) / (m * per)
    zs = T.bed(a[0] + sm * (b[0] - a[0]), a[1] + sm * (b[1] - a[1])).reshape(m, per).mean(axis=1)
    invn = (1 / friction(a[0] + sm * (b[0] - a[0]), a[1] + sm * (b[1] - a[1]))).reshape(m, per).mean(axis=1)
    eta = peak_eta
    hj = np.maximum(eta - zs, 0)
    hbar = float(tables.depth(np.array([eta]))[0])
    neff = hbar ** (5 / 3) / float(tables.conveyance(np.array([eta]))[0])
    q_uni = hj.copy()
    q_split = neff * invn * hj ** (5 / 3) / hbar ** (2 / 3)
    q_uni /= q_uni.sum()
    q_split /= q_split.sum()
    out['edge'] = dict(L=L, s=np.round(s * L, 2).tolist()[::4], z=np.round(prof, 3).tolist()[::4],
                       zs=np.round(zs, 3).tolist(), eta=eta, hbar=hbar, neff=neff,
                       n=np.round(1 / invn, 4).tolist(), q_uni=np.round(q_uni, 4).tolist(),
                       q_split=np.round(q_split, 4).tolist(),
                       u_uni=1.0, u_split=np.round(neff * invn * hj ** (2 / 3) / hbar ** (2 / 3), 3).tolist())

    # 5. flood-extent agreement maps at 2.5 m
    gx = np.arange(1.25, T.LENGTH, 2.5)
    gy = np.arange(1.25, T.WIDTH, 2.5)
    GX, GY = np.meshgrid(gx, gy)
    GX, GY = GX.ravel(), GY.ravel()
    GZ = T.bed(GX, GY)


    def extent(run):
        f = Triangulation(run['nodes'][:, 0], run['nodes'][:, 1], run['triangles']).get_trifinder()
        c = f(GX, GY)
        wet = run['peak_stage'] - run['bed'] > 0.01
        return wet[c] & (run['peak_stage'][c] > GZ + 0.01)


    ref_ext = extent(ref)
    maps = {}
    for key, name in (('centroid', 'coarse500_centroid'), ('ekl', 'coarse500_sub8EKL_diag'),
                      ('eklc', 'coarse500_sub8EKLC_diag'), ('eklc2000', 'coarse2000_sub8EKLC_diag')):
        e = extent(load(name))
        code = np.where(e & ref_ext, 1, np.where(ref_ext, 2, np.where(e, 3, 0)))
        csi = (e & ref_ext).sum() / (e | ref_ext).sum()
        maps[key] = dict(q=''.join(map(str, code)), csi=round(float(csi), 3))
    out['maps'] = dict(nx=len(gx), ny=len(gy), runs=maps)

    # 6. peak stage along the centreline
    PX_ = np.arange(25.0, 1975.0, 25.0)
    PY_ = T.centreline(PX_)
    prof = {}
    for key, name in (('ref', 'band025_diag_tu'), ('eklc', 'coarse500_sub8EKLC_diag'),
                      ('eklc2000', 'coarse2000_sub8EKLC_diag'), ('ekl', 'coarse500_sub8EKL_diag'),
                      ('centroid', 'coarse500_centroid'), ('band1', 'band1_diag')):
        r = ref if key == 'ref' else load(name)
        f = Triangulation(r['nodes'][:, 0], r['nodes'][:, 1], r['triangles']).get_trifinder()
        prof[key] = r['peak_stage'][f(PX_, PY_)]
    out['profile'] = dict(x=PX_.tolist(), **{k: np.round(v - prof['ref'], 3).tolist() for k, v in prof.items() if k != 'ref'})

    return out


def hydro():
    """Outlet discharge and near-floodplain storage every 3 min, 1.75-5.5 h."""
    out = {}
    tt = np.arange(1.75, 5.51, 0.05)
    for k, f in [('ref', 'band025_diag_tu'), ('band1', 'band1_diag'), ('e500', 'coarse500_sub8EKLC_diag'),
                 ('e2000', 'coarse2000_sub8EKLC_diag'), ('ekl500', 'coarse500_sub8EKL_diag')]:
        d = np.load(f'results/{f}.npz')
        out[k] = np.round(np.interp(tt, d['q_times'] / 3600, d['q_sections'][:, -1]), 2).tolist()
        st = d['diag_store'][:, :, 1].sum(axis=1) / 1000
        out[k + '_st'] = np.round(np.interp(tt, d['diag_t'] / 3600, st), 2).tolist()
    out['t'] = np.round(tt, 2).tolist()
    return out


# ---------------------------------------------------------------- plotting
INK, MUTED, GRID = '#17232b', '#56656f', '#e3e9ed'
C1, C2, C3, OLD = '#2a78d6', '#eb6834', '#1baf7a', '#8a97a0'
SEQ = matplotlib.colors.LinearSegmentedColormap.from_list('seq', ['#0b4f6a', '#eef3f5'])
plt.rcParams.update({'font.size': 9, 'axes.edgecolor': MUTED, 'axes.labelcolor': INK,
                     'xtick.color': MUTED, 'ytick.color': MUTED, 'axes.grid': True,
                     'grid.color': GRID, 'axes.spines.top': False, 'axes.spines.right': False,
                     'legend.frameon': False, 'savefig.dpi': 150, 'figure.dpi': 150})


def decode(q, shape):
    return (np.frombuffer(q.encode(), dtype=np.uint8).astype(float) - 48).reshape(shape) / 63


def fig_problem(F, out):
    fig = plt.figure(figsize=(9, 5.2))
    ax = fig.add_axes([0.06, 0.52, 0.88, 0.44])
    t = F['terrain']
    im = ax.imshow(decode(t['q'], (t['ny'], t['nx'])) * (t['hi'] - t['lo']) + t['lo'], origin='lower',
                   extent=[0, T.LENGTH, 0, T.WIDTH], cmap=SEQ, vmin=t['lo'], vmax=t['hi'], aspect='equal')
    ax.grid(False)
    w = F['mesh_window']
    ax.add_patch(plt.Rectangle((w['x0'], w['y0']), 150, 150, fill=False, ec=C2, lw=1.2))
    ax.set_xlabel('x (m)')
    ax.set_ylabel('y (m)')
    ax.set_title('Test valley: height above the floodplain level (m); orange square = patch below', fontsize=9, loc='left')
    fig.colorbar(im, ax=ax, fraction=0.02, pad=0.01)
    names = [('band1_diag_lf4', 'band1 · 188 198 triangles'), ('coarse500_sub8EKLC_diag', 'coarse500 · 3082'),
             ('coarse2000_sub8EKLC_diag', 'coarse2000 · 780')]
    xs = np.linspace(w['x0'], w['x0'] + 150, 200)
    yc = T.centreline(xs)
    for i, (name, lab) in enumerate(names):
        a = fig.add_axes([0.06 + i * 0.31, 0.03, 0.27, 0.40])
        d = np.load(f'results/{name}.npz')
        a.fill_between(xs, yc - T.TOP_WIDTH / 2, yc + T.TOP_WIDTH / 2, color='#0b4f6a', alpha=0.3, lw=0)
        a.triplot(Triangulation(d['nodes'][:, 0], d['nodes'][:, 1], d['triangles']), color=INK,
                  lw=0.08 if i == 0 else 0.6)
        a.set_xlim(w['x0'], w['x0'] + 150)
        a.set_ylim(w['y0'], w['y0'] + 150)
        a.set_aspect('equal')
        a.set_xticks([])
        a.set_yticks([])
        a.grid(False)
        a.set_title(lab, fontsize=9, loc='left')
    fig.savefig(f'{out}/problem.png')
    plt.close(fig)


def fig_cell(F, out):
    c = F['cell']
    P = c['pixels']
    fig = plt.figure(figsize=(9, 6.2))
    lo, hi = P['lo'], P['hi']
    a = fig.add_axes([0.03, 0.47, 0.42, 0.5])
    z = decode(P['q'], (P['ny'], P['nx'])) * (hi - lo) + lo
    mask = np.frombuffer(P['mask'].encode(), dtype=np.uint8).reshape(P['ny'], P['nx']) == ord('1')
    a.imshow(np.where(mask, z, np.nan), origin='lower', extent=[0, P['nx'] * 0.5, 0, P['ny'] * 0.5],
             cmap=SEQ, vmin=lo, vmax=hi)
    a.set_title('0.5 m terrain pixels in the cell', fontsize=9, loc='left')
    b = fig.add_axes([0.5, 0.47, 0.42, 0.5])
    polys = [s['p'] for s in c['subs']]
    pc = PolyCollection(polys, array=np.array([s['z'] for s in c['subs']]), cmap=SEQ, clim=(lo, hi),
                        edgecolors='white', linewidths=0.5)
    b.add_collection(pc)
    tri = np.array(c['tri'])
    b.add_patch(plt.Polygon(tri, fill=False, ec=INK, lw=1.2))
    cen = tri.mean(axis=0)
    b.plot(*cen, 'o', color=C2, ms=6, mec='white')
    b.annotate(f'centroid, {c["zc"]:.2f} m', cen, xytext=(6, 4), textcoords='offset points', fontsize=8)
    b.set_title(f'64 sub-triangles (lowest {c["zmin"]:.2f} m)', fontsize=9, loc='left')
    fig.colorbar(pc, ax=b, fraction=0.04, pad=0.02, label='bed (m)')
    for ax in (a, b):
        ax.set_xlim(0, c['bw'])
        ax.set_ylim(0, c['bh'])
        ax.set_aspect('equal')
        ax.set_xticks([])
        ax.set_yticks([])
        ax.grid(False)
    eta = np.array(c['eta'])
    panels = [('h', 'h_flat', 'mean depth h (m)'), ('f', None, 'wet fraction f'), ('kc', 'kc_flat', 'conveyance Kc (m²/s)')]
    for i, (k, kf, lab) in enumerate(panels):
        ax = fig.add_axes([0.07 + i * 0.32, 0.07, 0.25, 0.3])
        ax.plot(eta, c[k], color=C1, lw=1.8, label='subgrid tables')
        flat = c[kf] if kf else (eta > c['zc']).astype(float)
        ax.plot(eta, flat, color=OLD, lw=1.4, ls='--', label='flat bed at centroid')
        ax.axvline(c['peak_eta'], color=MUTED, ls=':', lw=1)
        ax.text(c['peak_eta'], ax.get_ylim()[1], ' peak', va='top', fontsize=7.5, color=MUTED)
        ax.set_xlabel('level η (m)')
        ax.set_title(lab, fontsize=9, loc='left')
        if i == 0:
            ax.legend(fontsize=7.5, loc='upper left')
    fig.savefig(f'{out}/cell.png')
    plt.close(fig)


def fig_edge(F, out):
    e = F['edge']
    fig, (a, b) = plt.subplots(2, 1, figsize=(9, 5.4), sharex=True, gridspec_kw=dict(height_ratios=[1.1, 1]))
    s, z = np.array(e['s']), np.array(e['z'])
    a.fill_between(s, np.minimum(z, e['eta']), e['eta'], color=C1, alpha=0.18, lw=0)
    a.axhline(e['eta'], color=C1, lw=1.2)
    a.text(s[-1], e['eta'] + 0.05, f'level at peak η = {e["eta"]:.2f} m', ha='right', va='bottom', fontsize=8)
    a.plot(s, z, color=INK, lw=1.4, label='terrain along the edge')
    ds = e['L'] / len(e['zs'])
    for j, zj in enumerate(e['zs']):
        a.plot([j * ds + 0.2, (j + 1) * ds - 0.2], [zj, zj], color=C2, lw=2.5, label='16 subface beds' if j == 0 else None)
    a.set_ylabel('elevation (m)')
    a.legend(fontsize=8, loc='lower left')
    xc = (np.arange(len(e['zs'])) + 0.5) * ds
    w = ds * 0.4
    b.bar(xc - w / 2, np.array(e['q_uni']) * 100, w, color=OLD, label='mean velocity on every subface')
    b.bar(xc + w / 2, np.array(e['q_split']) * 100, w, color=C1, label='conveyance split')
    b.set_ylabel('share of edge discharge (%)')
    b.set_xlabel('distance along the edge (m)')
    b.legend(fontsize=8, loc='upper left')
    fig.tight_layout()
    fig.savefig(f'{out}/edge.png')
    plt.close(fig)


def fig_steps(out):
    rows = [('centroid bed (no subgrid)', 23274), ('sub8 · storage only', 54948),
            ('sub8E · edge subfaces, f-scaled cap', 714590), ('sub8EK · + conveyance friction', 1170570),
            ('sub8EKL · + draining limiter', 36402), ('sub8EKLC · + conveyance split', 55178)][::-1]
    fig, ax = plt.subplots(figsize=(8, 2.8))
    ax.barh([r[0] for r in rows], [r[1] for r in rows], color=[C1, C1, OLD, OLD, OLD, OLD], height=0.6)
    for i, (_, v) in enumerate(rows):
        ax.text(v + 15000, i, f'{v:,}', va='center', fontsize=8)
    ax.set_xlim(0, 1.35e6)
    ax.set_xlabel('time steps for the 5.5 h flood on coarse500')
    ax.grid(axis='y', visible=False)
    fig.tight_layout()
    fig.savefig(f'{out}/steps.png')
    plt.close(fig)


def fig_extent(F, out):
    M = F['maps']
    cmap = matplotlib.colors.ListedColormap(['white', C1, C2, C3])
    names = [('centroid', 'coarse500 centroid bed'), ('ekl', 'coarse500_sub8EKL'),
             ('eklc', 'coarse500_sub8EKLC'), ('eklc2000', 'coarse2000_sub8EKLC')]
    fig, axs = plt.subplots(2, 2, figsize=(9, 3.4))
    for ax, (k, lab) in zip(axs.ravel(), names):
        q = np.frombuffer(M['runs'][k]['q'].encode(), dtype=np.uint8).reshape(M['ny'], M['nx']) - 48
        ax.imshow(q, origin='lower', cmap=cmap, vmin=-0.5, vmax=3.5, extent=[0, T.LENGTH, 0, T.WIDTH],
                  interpolation='nearest')
        ax.set_title(f'{lab}   CSI {M["runs"][k]["csi"]:.3f}', fontsize=8.5, loc='left')
        ax.set_xticks([])
        ax.set_yticks([])
        ax.grid(False)
        for sp in ax.spines.values():
            sp.set_visible(True)
            sp.set_color(GRID)
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in (C1, C2, C3)]
    fig.legend(handles, ['wet in both', 'wet only in the reference (missed)', 'wet only in the run (over-predicted)'],
               loc='lower center', ncol=3, fontsize=8)
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    fig.savefig(f'{out}/extent.png')
    plt.close(fig)


SER = [('ref', 'band025TU (reference)', INK, '-', 2.2), ('e500', 'EKLC 500', C1, '-', 1.6),
       ('e2000', 'EKLC 2000', C2, '-', 1.6), ('ekl500', 'EKL 500 (no split)', C3, '-', 1.4),
       ('band1', 'band1, old flux', OLD, '--', 1.4)]


def fig_hydro(Hd, out):
    fig, (a, b) = plt.subplots(1, 2, figsize=(9, 3.3))
    for k, lab, col, ls, lw in SER:
        a.plot(Hd['t'], Hd[k], color=col, ls=ls, lw=lw, label=lab)
        b.plot(Hd['t'], Hd[k + '_st'], color=col, ls=ls, lw=lw)
    a.set_title('outlet discharge (m³/s)', fontsize=9, loc='left')
    b.set_title('near-floodplain storage (thousand m³)', fontsize=9, loc='left')
    for ax in (a, b):
        ax.set_xlabel('time (h)')
        ax.set_ylim(bottom=0)
    a.legend(fontsize=7.5, loc='lower center')
    fig.tight_layout()
    fig.savefig(f'{out}/hydrographs.png')
    plt.close(fig)


def fig_profile(F, out):
    p = F['profile']
    fig, ax = plt.subplots(figsize=(9, 3))
    for k, lab, col, ls, lw in [('eklc', 'EKLC 500', C1, '-', 1.6), ('eklc2000', 'EKLC 2000', C2, '-', 1.6),
                                ('ekl', 'EKL 500 (no split)', C3, '-', 1.4), ('band1', 'band1, old flux', OLD, '--', 1.4)]:
        ax.plot(p['x'], p[k], color=col, ls=ls, lw=lw, label=lab)
    ax.axhline(0, color=MUTED, lw=0.8)
    ax.axvline(1900, color=MUTED, ls=':', lw=1)
    ax.text(1895, 0.33, 'steep outlet reach →', ha='right', fontsize=7.5, color=MUTED)
    ax.set_ylim(-0.4, 0.4)
    ax.set_xlabel('distance down the valley (m)')
    ax.set_ylabel('peak level − reference (m)')
    ax.legend(fontsize=7.5, ncol=4, loc='lower left')
    fig.tight_layout()
    fig.savefig(f'{out}/profile.png')
    plt.close(fig)


def fig_cost(out):
    pts = [('centroid bed 500', 3, 0.459, 0), ('trench 500', 3, 0.222, 0), ('storage only 500', 19, 0.367, 0),
           ('EKL 500', 61, 0.110, 0), ('EKLC 500', 99, 0.027, 1), ('EKLC 2000', 14, 0.037, 1),
           ('band1, old flux', 1063, 0.054, 2), ('band1LF4', 1032, 0.018, 2), ('band025, old flux', 5653, 0.020, 2)]
    off = {'EKLC 2000': (-6, 6, 'right'), 'band025, old flux': (0, 8, 'center'), 'band1LF4': (6, -10, 'left'),
           'band1, old flux': (6, 4, 'left')}
    fig, ax = plt.subplots(figsize=(7.5, 3.6))
    for lab, w, e, kind in pts:
        if kind == 2:
            ax.plot(w, e, 'o', mfc='white', mec=INK, ms=7, mew=1.5)
        else:
            ax.plot(w, e, 'o', color=C1 if kind else MUTED, ms=7)
        dx, dy, ha = off.get(lab, (6, -3, 'left'))
        ax.annotate(lab, (w, e), xytext=(dx, dy), textcoords='offset points', ha=ha, fontsize=8,
                    fontweight='bold' if kind == 1 else 'normal')
    ax.set_xscale('log')
    ax.set_xlim(1, 2e4)
    ax.set_ylim(0, 0.52)
    ax.set_xlabel('wall time (s, log scale): filled = 8 CPU threads, hollow = GPU')
    ax.set_ylabel('mean |peak stage error| (m)')
    fig.tight_layout()
    fig.savefig(f'{out}/cost.png')
    plt.close(fig)


def fig_bias(out):
    fig, ax = plt.subplots(figsize=(5, 3))
    x = np.arange(3)
    old, tu = [0.63, 0.83, 0.92], [0.79, 0.93, 0.97]
    ax.bar(x - 0.18, old, 0.34, color=OLD, label='old flux')
    ax.bar(x + 0.18, tu, 0.34, color=C1, label='LF1 + tangential upwind')
    for i in range(3):      # labels inside the bars: the tall ones would sit on the dashed line
        ax.text(x[i] - 0.18, old[i] - 0.1, f'{old[i]:.2f}', ha='center', fontsize=8, color='white')
        ax.text(x[i] + 0.18, tu[i] - 0.1, f'{tu[i]:.2f}', ha='center', fontsize=8, color='white')
    ax.axhline(1, color=MUTED, ls='--', lw=1)
    ax.set_xticks(x, ['4 m² cells', '1 m² cells', '0.25 m² cells'])
    ax.set_ylim(0, 1.25)
    ax.set_ylabel('share of true conveyance')
    ax.legend(fontsize=8, loc='upper left', ncol=2, frameon=False)
    ax.grid(axis='x', visible=False)
    fig.tight_layout()
    fig.savefig(f'{out}/bias.png')
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    here = os.path.dirname(os.path.abspath(__file__))
    ap.add_argument('--out', default=os.path.join(here, '..', 'claude', 'figures', 'subgrid'))
    ap.add_argument('--page-data', default=None)
    args = ap.parse_args()
    os.chdir(here)
    os.makedirs(args.out, exist_ok=True)
    F = extract()
    Hd = hydro()
    if args.page_data:
        json.dump(F, open(os.path.join(args.page_data, 'figs.json'), 'w'), separators=(',', ':'))
        json.dump(Hd, open(os.path.join(args.page_data, 'hydro.json'), 'w'))
    fig_problem(F, args.out)
    fig_cell(F, args.out)
    fig_edge(F, args.out)
    fig_steps(args.out)
    fig_extent(F, args.out)
    fig_hydro(Hd, args.out)
    fig_profile(F, args.out)
    fig_cost(args.out)
    fig_bias(args.out)
    print('figures in', os.path.abspath(args.out))


if __name__ == '__main__':
    main()
