"""steady_outlet.py VARIANT Q [FINAL_TIME] [GPU]: constant inflow Q, save the final
state for the outlet-reach investigation to results/steady_VARIANT_Q<Q><SUBGRID_TAG>.npz
(stage, bed, momenta, centroids, mesh). Use SUBGRID_AMPLITUDE=0 for the straight channel.
"""
import os
import sys

import numpy as np

var, Q = sys.argv[1], float(sys.argv[2])
fin = sys.argv[3] if len(sys.argv) > 3 else '18000'
gpu = 'GPU' in sys.argv[4:]
tag = os.environ.get('SUBGRID_TAG', '')
sys.argv = ['run_phase0.py', var, '--diag', '--final-time', fin] + (['--gpu'] if gpu else [])
import run_phase0 as R  # noqa: E402

R.inflow = lambda t: Q
last = {}
orig = R.Diagnostics.record


def record(self, t, stage, elev, xmom):
    q = self.domain.quantities
    last.update(t=t, w=stage.copy(), z=elev.copy(), uh=xmom.copy(),
                vh=q['ymomentum'].centroid_values.copy(), c=self.domain.centroid_coordinates.copy(),
                nodes=self.domain.nodes.copy(), tri=self.domain.triangles.copy(),
                hist=last.get('hist', []) + [stage.copy()])
    return orig(self, t, stage, elev, xmom)


R.Diagnostics.record = record
saved = {}
_savez = np.savez_compressed          # keep: the bench's own save is captured below
R.np.savez_compressed = lambda *a, **k: saved.update(k)
R.main()
H = last.pop('hist')
drift = H[-1] - H[-61] if len(H) > 61 else H[-1] * 0      # change over the last hour
out = f'results/steady_{var}_Q{Q:g}{tag}.npz'
_savez(out, drift=drift, q_sections=saved['q_sections'], section_x=saved['section_x'], **last)
print(f'saved {out}: t = {last["t"]:.0f} s, max |drift| over the last hour {np.abs(drift).max():.4f} m, '
      f'Q at sections {np.round(saved["q_sections"][-1], 2)}')
