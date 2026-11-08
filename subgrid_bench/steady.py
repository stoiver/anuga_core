"""steady.py VARIANT Q : constant inflow Q for 5 h on the current terrain; compare
the steady level at x = 800-1200 with the Manning normal depth of the DTM section,
and check the cell momentum balance q / Kc against sqrt(S)."""
import sys, numpy as np
var, Q = sys.argv[1], float(sys.argv[2])
fin = sys.argv[3] if len(sys.argv) > 3 else '18000'
sys.argv = ['run_phase0.py', var, '--diag', '--final-time', fin] + (['--gpu'] if 'GPU' in sys.argv[4:] else [])
import run_phase0 as R, terrain as T
R.inflow = lambda t: Q
last = {}
orig = R.Diagnostics.record
def record(self, t, stage, elev, xmom):
    q = self.domain.quantities
    last.update(t=t, w=stage.copy(), z=elev.copy(), uh=xmom.copy(), vh=q['ymomentum'].centroid_values.copy(),
                neff=getattr(self.domain, 'subgrid_cell_neff', q['friction'].centroid_values).copy(),
                c=self.domain.centroid_coordinates.copy(), hist=last.get('hist', []) + [(t, stage.copy())])
    return orig(self, t, stage, elev, xmom)
R.Diagnostics.record = record
saved = {}
R.np.savez_compressed = lambda *a, **k: saved.update(k)
R.main()
w, z, c = last['w'], last['z'], last['c']
h = np.maximum(w - z, 0)
dc = T.distance_to_centreline(c[:, 0], c[:, 1])
sel = (c[:, 0] > 800) & (c[:, 0] < 1200) & (dc < T.TOP_WIDTH / 2) & (h > 0.05)
# water-surface slope from a fit along the reach, and departure from the bed line
A = np.column_stack([np.ones(sel.sum()), c[sel, 0]])
coef = np.linalg.lstsq(A, w[sel], rcond=None)[0]
S = -coef[1]
dep = np.mean(w[sel] - T.channel_bed(c[sel, 0]))      # level above thalweg
# steadiness: change of mean level over the last hour
H = last['hist']; t0 = [i for i, (t, _) in enumerate(H) if t >= H[-1][0] - 3600][0]
drift = np.mean(H[-1][1][sel] - H[t0][1][sel])
# truth: normal depth of the DTM section at x = 1000 for this Q and S = valley slope
y = np.arange(0.05, T.WIDTH, 0.1); x = np.full_like(y, 1000.0)
zb, nb = T.bed(x, y), R.friction(x, y)
def K(d):
    return np.sum(np.maximum(T.channel_bed(1000.0) + d - zb, 0) ** (5 / 3) / nb) * 0.1
lo, hi = 0.01, 5.0
for _ in range(60):
    mid = 0.5 * (lo + hi)
    lo, hi = (mid, hi) if K(mid) * np.sqrt(T.SLOPE) < Q else (lo, mid)
d_true = 0.5 * (lo + hi)
# momentum balance: q/Kc vs sqrt(S); Kc = h^(5/3)/n_eff
q = np.hypot(last['uh'], last['vh'])
kc = h ** (5 / 3) / last['neff']
ratio = np.median(q[sel] / kc[sel] / np.sqrt(S))
qb = saved['q_sections'][-1]
print(f'STEADY {var} Q={Q:g}: depth above thalweg model {dep:.3f} truth {d_true:.3f} '
      f'(diff {dep - d_true:+.3f} m), slope {S:.2e}, Q@1000 {qb[2]:.2f}, drift/h {drift:+.4f} m, '
      f'median (q/Kc)/sqrt(S) {ratio:.3f}')
