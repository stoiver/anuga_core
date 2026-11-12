"""grad_probe.py VARIANT Q [FINAL_TIME]: steady run, then print the limited stage
gradient (from the edge values the flux kernel reads) against the unlimited one
(plane through the three neighbour centroids) for the channel cells near the
slope break, and the level against the DE0 baseline run if that exists.

    SUBGRID_AMPLITUDE=0 python grad_probe.py coarse500_sub8EKLC 50         # DE0 baseline
    SUBGRID_AMPLITUDE=0 PROBE_ALG=DE_ader2 python grad_probe.py ...        # other algorithm
    SUBGRID_AMPLITUDE=0 SUBGRID_P1=1 [SUBGRID_P1_BETA=1] [PROBE_TAG=_b1] python grad_probe.py ...

Writes results/gradprobe_VARIANT_Q<Q><suffix>.npz."""
import os
import sys

import numpy as np

var, Q = sys.argv[1], float(sys.argv[2])
fin = sys.argv[3] if len(sys.argv) > 3 else '18000'
os.environ.setdefault('SUBGRID_TAG', '_gradprobe')
sys.argv = ['run_phase0.py', var, '--diag', '--final-time', fin]
import run_phase0 as R  # noqa: E402
import terrain as T  # noqa: E402

R.inflow = lambda t: Q
ALG = os.environ.get('PROBE_ALG', '')
SUF = ('_' + ALG if ALG else '') + ('_p1' if os.environ.get('SUBGRID_P1') == '1' else '') + os.environ.get('PROBE_TAG', '')
if ALG:
    _create = R.anuga.create_domain_from_regions

    def create(*a, **k):
        d = _create(*a, **k)
        d.set_flow_algorithm(ALG)
        return d
    R.anuga.create_domain_from_regions = create
last = {}
orig = R.Diagnostics.record


def record(self, t, stage, elev, xmom):
    d = self.domain
    q = d.quantities
    last.update(
        t=t, w=stage.copy(), z=elev.copy(),
        we=q['stage'].edge_values.copy(), ze=q['elevation'].edge_values.copy(),
        he=q['height'].edge_values.copy(),
        uh=xmom.copy(), c=d.centroid_coordinates.copy(),
        ec=d.edge_midpoint_coordinates.copy() if hasattr(d, 'edge_midpoint_coordinates')
        else d.get_edge_midpoint_coordinates().copy(),
        nb=d.neighbours.copy(), area=d.areas.copy(),
        beta_w=d.beta_w, beta_w_dry=d.beta_w_dry, cfl=d.CFL, tsm=d.get_timestepping_method(), alg=d.get_flow_algorithm(),
        zmin=getattr(d, 'subgrid_edge_z', np.zeros((1, 1))).min())
    return orig(self, t, stage, elev, xmom)


R.Diagnostics.record = record
_savez = np.savez_compressed
R.np.savez_compressed = lambda *a, **k: None
R.main()

w, c, nb = last['w'], last['c'], last['nb']
ec = last['ec'].reshape(-1, 3, 2)
we = last['we']
n = len(w)
print(f"\nflow algorithm {last['alg']}, beta_w {last['beta_w']}, beta_w_dry {last['beta_w_dry']}, t = {last['t']:.0f} s, cfl {last['cfl']}, timestepping {last['tsm']}")

# limited gradient: least squares on the three edge-midpoint values
dx = ec - c[:, None, :]                      # (n, 3, 2)
dw = we - w[:, None]
glim = np.empty((n, 2))
for k in range(n):
    glim[k] = np.linalg.lstsq(dx[k], dw[k], rcond=None)[0]
resid = np.abs(np.einsum('kej,kj->ke', dx, glim) - dw).max(axis=1)

# unlimited gradient: plane through the three neighbour centroids (interior cells)
gun = np.full((n, 2), np.nan)
for k in range(n):
    if (nb[k] >= 0).all():
        A = c[nb[k][1:]] - c[nb[k][0]]
        b = w[nb[k][1:]] - w[nb[k][0]]
        if abs(np.linalg.det(A)) > 1e-9:
            gun[k] = np.linalg.solve(A, b)

yc = T.centreline(c[:, 0])
size = np.sqrt(last['area'])
sel = np.nonzero((np.abs(c[:, 1] - yc) < 0.75 * size) & (c[:, 0] > 1600) & (c[:, 0] < 1980))[0]
sel = sel[np.argsort(c[sel, 0])]
h = w - last['z']
print('\nchannel cells, straight channel, Q = %g: stage gradient along the valley (x), m/m' % Q)
print('    x      y   size   eta_c  mean h |  limited  unlimited  ratio | edge eta min/max - eta_c | resid')
for k in sel:
    r = glim[k, 0] / gun[k, 0] if abs(gun[k, 0]) > 1e-12 else np.nan
    print(f'{c[k,0]:6.0f} {c[k,1]:6.0f} {size[k]:5.0f} {w[k]:7.3f} {h[k]:6.3f} | {glim[k,0]:9.5f} {gun[k,0]:9.5f} {r:6.2f} |'
          f' {dw[k].min():+7.3f} {dw[k].max():+7.3f} | {resid[k]:.1e}')
for lo, hi in ((1600, 1800), (1800, 1900), (1900, 1980)):
    m = sel[(c[sel, 0] >= lo) & (c[sel, 0] < hi)]
    if len(m):
        print(f'x {lo}-{hi}: {len(m)} cells, mean limited {glim[m,0].mean():.5f}, mean unlimited {np.nanmean(gun[m,0]):.5f},'
              f' ratio of means {glim[m,0].mean()/np.nanmean(gun[m,0]):.2f}')
_savez('results/gradprobe_%s_Q%g%s.npz' % (var, Q, SUF), glim=glim, gun=gun, **last)

h = w - last['z']
k = np.nonzero((c[:, 0] > 1000) & (c[:, 0] < 1900) & (h > 0.2) & (nb >= 0).all(axis=1))[0]
hn = h[nb[k]]
hmin = np.minimum(hn.min(1), h[k]); hmax = np.maximum(hn.max(1), h[k])
rr = np.minimum(hmin / h[k], h[k] / hmax)
g = np.abs(glim).max(1)[k]
print('\nwet cells x 1000-1900 with mean h > 0.2:', len(k))
print('min(hmin/hc, hc/hmax) quartiles 25/50/75/90:', np.round(np.percentile(rr, [25, 50, 75, 90]), 2))
print('share with nonzero limited gradient: %.2f' % (g > 1e-9).mean())
for lo, hi in ((0, 0.1), (0.1, 0.3), (0.3, 0.5), (0.5, 1.01)):
    m = (rr >= lo) & (rr < hi)
    print(f'depth ratio {lo}-{hi}: {m.sum()} cells, nonzero-gradient share {(g[m] > 1e-9).mean() if m.any() else float("nan"):.2f}')

ref = 'results/gradprobe_%s_Q%g.npz' % (var, Q)
if SUF and os.path.exists(ref):
    w0 = np.load(ref)['w']
    print('\nlevel on the channel cells, %s minus DE0 baseline (m):' % SUF)
    yc = T.centreline(c[:, 0])
    ch = np.abs(c[:, 1] - yc) < 5
    for lo in range(1000, 1900, 50):
        m = ch & (c[:, 0] >= lo) & (c[:, 0] < lo + 50)
        if m.any():
            print(f'  x {lo}-{lo+50}: {(w[m]-w0[m]).mean():+.3f}  (n={m.sum()}, range {(w[m]-w0[m]).min():+.3f} to {(w[m]-w0[m]).max():+.3f})')
