"""compare_outlet.py: steady Q = 50 outlet-reach comparison (results/steady_*.npz).

Straight channel: levels on the centreline against the 1D standard-step profile
(gvf_1d.py) and band1; meander: against band1. A subgrid cell's stage is its flat
level eta, compared at the centreline point inside the cell.
Run with SUBGRID_AMPLITUDE unset (the meander centreline is computed here).

    python compare_outlet.py [--fig DIR] [--page-data DIR]

--fig writes DIR/outlet.png (the report figure); --page-data writes DIR/outlet.json
for the web version of the report. Synthetic data only.
"""
import argparse
import json
import os

import numpy as np
from matplotlib.tri import Triangulation

import terrain as T

Q = 50
XS = np.array([1000, 1200, 1400, 1500, 1600, 1650, 1700, 1750, 1800, 1825, 1850, 1875, 1890])
STRAIGHT = [('band1', 'band1_Q50_straight'), ('EKLC 500', 'coarse500_sub8EKLC_Q50_straight'),
            ('EKLC 2000', 'coarse2000_sub8EKLC_Q50_straight'), ('EKL 500', 'coarse500_sub8EKL_Q50_straight'),
            ('EKLC 500 + outlet 25 m2', 'coarse500_sub8EKLC_Q50_straight_out25')]
MEANDER = [('band1', 'band1_Q50_meander'), ('EKLC 500', 'coarse500_sub8EKLC_Q50_meander'),
           ('EKLC 500 + outlet 25 m2', 'coarse500_sub8EKLC_Q50_meander_out25')]


def levels(pairs, yc):
    cols = {}
    for lab, f in pairs:
        p = f'results/steady_{f}.npz'
        if not os.path.exists(p):
            continue
        r = np.load(p)
        tf = Triangulation(r['nodes'][:, 0], r['nodes'][:, 1], r['tri']).get_trifinder()
        cols[lab] = r['w'][tf(XS, yc)]
        print(f'   {lab}: t = {float(r["t"]):.0f} s, max |drift| last hour {np.abs(r["drift"]).max():.4f} m, '
              f'Q at 1000/1750/2000 m {np.round(r["q_sections"][-1][[2, 4, 6]], 1)}')
    return cols


def show(title, cols, ref):
    print(f'\n== {title}: level minus {ref} on the centreline (m)')
    if ref not in cols:
        print('   reference missing')
        return
    others = [k for k in cols if k != ref]
    print(f'  {"x":>5s} {"depth":>6s}  ' + ' '.join(f'{k:>24s}' for k in others))
    for i, x in enumerate(XS):
        print(f'  {x:5.0f} {cols[ref][i] - T.channel_bed(x):6.3f}  '
              + ' '.join(f'{cols[k][i] - cols[ref][i]:+24.3f}' for k in others))


XD = np.arange(1000.0, 1896.0, 5.0)   # dense centreline sampling for the figure; stops at the slope
# break, beyond which a flat cell level against a steep surface is a sampling artefact
FIG_RUNS = [('band1', 'band1'), ('EKLC 500', 'eklc'), ('EKLC 2000', 'eklc2000'),
            ('EKL 500', 'ekl'), ('EKLC 500 + outlet 25 m2', 'out25')]


def dense(pairs, yc):
    out = {}
    for lab, f in pairs:
        p = f'results/steady_{f}.npz'
        if os.path.exists(p):
            r = np.load(p)
            tf = Triangulation(r['nodes'][:, 0], r['nodes'][:, 1], r['tri']).get_trifinder()
            out[lab] = r['w'][tf(XD, yc)]
    return out


def figure_data():
    g = np.load(f'results/gvf_1d_Q{Q}.npz')
    gx = XD[XD <= g['x'].max()]
    one_d = np.interp(gx, g['x'][::-1], g['eta'][::-1])
    st = dense(STRAIGHT, np.full(len(XD), T.WIDTH / 2))
    me = dense(MEANDER, T.WIDTH / 2 + 40.0 * np.sin(2 * np.pi * XD / T.WAVELENGTH))
    r3 = lambda a: [None if not np.isfinite(v) else round(float(v), 3) for v in a]
    keys = dict(FIG_RUNS)
    return {'x': XD.tolist(), 'nx1d': len(gx),
            'straight': {keys[k]: r3(v[:len(gx)] - one_d) for k, v in st.items()},
            'meander': {keys[k]: r3(v - me['band1']) for k, v in me.items() if k != 'band1'}}


def make_figure(F, out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from report_figures import C1, C3, OLD, MUTED
    C4 = '#eda100'
    style = {'band1': ('band1 (2D reference)', OLD, '--', 1.4), 'eklc': ('EKLC 500', C1, '-', 1.6),
             'ekl': ('EKL 500 (no split)', C3, '-', 1.2),
             'out25': ('EKLC 500, outlet refined to 25 m²', C4, '-', 2.0)}
    x = np.array(F['x'])
    fig, axs = plt.subplots(1, 2, figsize=(10, 3.4), sharey=True)
    for ax, key, ttl in [(axs[0], 'straight', 'straight channel: level − 1D profile'),
                         (axs[1], 'meander', 'meandering channel: level − band1')]:
        for k in ['band1', 'ekl', 'eklc2000', 'eklc', 'out25']:
            if k in F[key] and k != 'eklc2000':      # EKLC 2000: 65 m cells, too coarse to sample here
                y = np.array(F[key][k], dtype=float)
                lab, col, ls, lw = style[k]
                ax.plot(x[:len(y)], y, color=col, ls=ls, lw=lw, label=lab)
        ax.axhline(0, color=MUTED, lw=0.8)
        ax.axvspan(1800, 1900, color=MUTED, alpha=0.08, lw=0)
        ax.text(1850, 0.19, 'refined', ha='center', fontsize=7.5, color=MUTED)
        ax.text(1897, -0.38, 'slope break →', ha='right', fontsize=7.5, color=MUTED)
        ax.set_title(ttl, fontsize=9)
        ax.set_xlabel('distance down the valley (m)')
        ax.set_xlim(1000, 1900)
    axs[0].set_ylim(-0.4, 0.25)
    axs[0].set_ylabel('steady level difference (m), Q = 50 m³/s')
    h, lab = axs[0].get_legend_handles_labels()
    fig.legend(h, lab, loc='lower center', ncol=4, fontsize=7.5, frameon=False)
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    fig.savefig(f'{out}/outlet.png', dpi=150)
    plt.close(fig)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--fig', default=None)
    ap.add_argument('--page-data', default=None)
    args = ap.parse_args()
    if args.fig or args.page_data:
        F = figure_data()
        if args.fig:
            make_figure(F, args.fig)
        if args.page_data:
            json.dump(F, open(os.path.join(args.page_data, 'outlet.json'), 'w'), separators=(',', ':'))
        raise SystemExit
    print('straight channel runs:')
    st = levels(STRAIGHT, np.full(len(XS), T.WIDTH / 2))
    g = np.load(f'results/gvf_1d_Q{Q}.npz')
    st = {'1D': np.interp(XS, g['x'][::-1], g['eta'][::-1]), **st}
    show('straight channel', st, '1D')
    show('straight channel', st, 'band1')
    print('\nmeandering channel runs:')
    me = levels(MEANDER, T.WIDTH / 2 + 40.0 * np.sin(2 * np.pi * XS / T.WAVELENGTH))
    show('meandering channel', me, 'band1')
