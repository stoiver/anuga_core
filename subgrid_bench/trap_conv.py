"""trap_conv.py MAX_AREA [--gpu] [--rect] [--vertex]: steady uniform flow in a straight
trapezoidal channel (in-bank), full ANUGA. Compares depth and implied conveyance in
the middle reach with the Manning strip sum (the exact SWE uniform-flow solution)."""
import sys, time, numpy as np, anuga
area = float(sys.argv[1]); gpu = '--gpu' in sys.argv; rect = '--rect' in sys.argv
vertex = '--vertex' in sys.argv; wide = '--wide' in sys.argv; brk = '--brk' in sys.argv; struct = '--struct' in sys.argv; smooth = '--smooth' in sys.argv
wide = wide or smooth
L, W, YC, S, Q = 600.0, 30.0, 15.0, 1.0e-3, 25.0
BASE, BANK, DEPTH = 10.0, 2.0, 2.0
TOP = BASE + 2 * BANK * DEPTH
def bed(x, y):
    d = np.abs(y - YC)
    cut = DEPTH if rect else np.clip(DEPTH - (d - BASE / 2) / BANK, 0.0, DEPTH)
    if rect:
        cut = np.where(d < TOP / 2, DEPTH, 0.0)
    if smooth:
        cut = 1.0 + np.cos(2 * np.pi * (y - YC) / W)     # fully wet, depth varies ~3x across
    elif wide:
        cut = DEPTH + 0 * d
    return 10.0 - S * x - cut
def nman(x, y):
    return np.where((np.abs(y - YC) < TOP / 2) | wide, 0.035, 0.05)
yy = np.arange(0.005, W, 0.01)
def K(eta, x=300.0):
    z = bed(np.full_like(yy, x), yy); return np.sum(np.maximum(eta - z, 0) ** (5 / 3) / nman(x, yy)) * 0.01
th = lambda x: 10.0 - S * x - DEPTH
lo, hi = 0.01, 3.0
for _ in range(60):
    m = 0.5 * (lo + hi); lo, hi = (m, hi) if K(th(300) + m) * np.sqrt(S) < Q * (W / TOP if wide else 1) else (lo, m)
dn = 0.5 * (lo + hi)
anuga.set_gpu_offload(gpu)
if struct:
    dxs = 2 * np.sqrt(area)
    dom = anuga.rectangular_cross_domain(int(round(L / dxs)), int(round(W / dxs)), L, W)
else:
  dom = anuga.create_domain_from_regions([[0, 0], [L, 0], [L, W], [0, W]],
                                       {'bottom': [0], 'right': [1], 'top': [2], 'left': [3]},
                                       maximum_triangle_area=area,
                                       breaklines=([[[1.0, YC + s * o], [L - 1.0, YC + s * o]]
                                                    for o in (BASE / 2, TOP / 2) for s in (-1, 1)]
                                                   if brk else None))
dom.set_store(False)
LF = int(dict(a.split('=') for a in sys.argv if '=' in a).get('LF', 0))
dom.set_low_froude(LF)   # e.g. LF=5: LF1 + tangential upwind (#510)
dom.set_compute_mode('unified' if gpu else 'legacy')
if vertex:
    dom.set_quantity('elevation', bed)
    dom.set_quantity('friction', nman)
else:
    x, y = dom.centroid_coordinates.T
    dom.set_quantity('elevation', bed(x, y), location='centroids')
    dom.set_quantity('friction', nman(x, y), location='centroids')
dom.set_quantity('stage', lambda x, y: np.maximum(th(x) + dn, bed(x, y)))
br = anuga.Reflective_boundary(dom)
bo = anuga.Transmissive_n_momentum_zero_t_momentum_set_stage_boundary(dom, lambda t: th(L) + dn)
dom.set_boundary({'left': br, 'right': bo, 'top': br, 'bottom': br})
anuga.Inlet_operator(dom, [[1.0, 0.0 if wide else YC - TOP / 2], [1.0, W if wide else YC + TOP / 2]], Q=Q * (W / TOP if wide else 1))
t0 = time.time(); n = 0
for t in dom.evolve(yieldstep=600.0, finaltime=float(dict(a.split('=') for a in sys.argv if '=' in a).get('T', 3000))):
    n += dom.number_of_steps
    xc, yc_ = dom.centroid_coordinates.T
    uhq = dom.quantities['xmomentum'].centroid_values
    qs = [np.sum((uhq * dom.areas)[np.abs(xc - xs) < 10]) / 20 for xs in (100, 300, 500)]
    wq = dom.quantities['stage'].centroid_values
    cl = np.abs(yc_ - YC) < 2
    ws = [np.mean(wq[cl & (np.abs(xc - xs) < 10)] - th(xs)) for xs in (100, 300, 500, 590)]
    print(f'  t={t:5.0f} Q@100/300/500 ' + ' '.join(f'{v:5.1f}' for v in qs) + '  depth@100/300/500/590 ' + ' '.join(f'{v:5.2f}' for v in ws), flush=True)
w = dom.quantities['stage'].centroid_values; z = dom.quantities['elevation'].centroid_values
uh = dom.quantities['xmomentum'].centroid_values
x, y = dom.centroid_coordinates.T
sel = (x > 200) & (x < 450) & (np.abs(y - YC) < TOP / 2) & (w - z > 0.01)
c = np.linalg.lstsq(np.column_stack([np.ones(sel.sum()), x[sel]]), w[sel], rcond=None)[0]
Sw = -c[1]; eta300 = c[0] + c[1] * 300
d = eta300 - th(300)
# section discharge from momentum at x=300 (cells within 1 m)
b = (np.abs(x - 300) < 2.0)
Kimpl = Q * (W / TOP if wide else 1) / np.sqrt(Sw)
# local balance in the flat-bottom centre: q vs Manning at its depth
cen = sel & (np.abs(y - YC) < BASE / 2 - 1)
h = w - z
r = np.median(uh[cen] / (h[cen] ** (5 / 3) / 0.035 * np.sqrt(Sw)))
print(f'TRAP area {area:g}{" rect" if rect else ""}{" vertex" if vertex else ""}{" wide" if wide else ""}{" brk" if brk else ""}{" struct" if struct else ""}{" smooth" if smooth else ""} LF{LF} ntri {len(dom)}: depth {d:.3f} vs normal {dn:.3f} '
      f'({d - dn:+.3f} m), S_w {Sw:.3e}, K_implied/K_true(d) {Kimpl / K(eta300):.3f}, '
      f'centre q/Manning {r:.3f}, steps {n}, wall {time.time() - t0:.0f} s')
