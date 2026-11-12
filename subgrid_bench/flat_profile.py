"""Flat-plain bench: mean depth by 100 m reach at 1 and 2 h, and where the water is (x of 50% / 90% of the volume)."""
import glob, sys
import numpy as np
pat = sys.argv[1] if len(sys.argv) > 1 else '*'
for f in sorted(glob.glob(f'results/flat_{pat}.npz')):
    d = np.load(f)
    if 'dprof' not in d:
        continue
    t = d['t'] / 3600
    print(f.split('/')[-1][5:-4])
    for h in (1, 2):
        i = int(np.argmin(abs(t - h))); p = d['dprof'][i]; c = np.cumsum(p) / p.sum()
        reach = p.reshape(-1, 10).mean(1)
        print(f'  {h} h: depth by 100 m ' + ' '.join(f'{x:.2f}' for x in reach[:12]) + f' | x50 {10 * np.searchsorted(c, 0.5):.0f}  x90 {10 * np.searchsorted(c, 0.9):.0f}  x99 {10 * np.searchsorted(c, 0.99):.0f}')
