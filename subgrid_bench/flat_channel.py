"""Flat-plain bench, channel grounds (meander, mrough): by time, how far the water has
got along the channel (furthest 10 m band whose centreline pixels are more than half
wet), over the plain (furthest band more than half wet) and where the volume is (x of
50% / 90% of it).

    python flat_channel.py [meander|mrough ...]
"""
import glob
import sys

import numpy as np

HRS = [0.25, 0.5, 1, 1.5, 2, 3]


def furthest(f):
    return 10 * (np.nonzero(f > 0.5)[0].max() + 1) if (f > 0.5).any() else 0


for terr in sys.argv[1:] or ('meander', 'mrough'):
    fs = sorted(glob.glob(f'results/flat_{terr}_*.npz'), key=lambda f: ('_sub' in f, -float(f.split('_')[2]) if '_plain' in f else float(f.split('_')[2]), f))
    for name, get in (('channel front', lambda d, i: furthest(d['cfrac'][i])),
                      ('plain front', lambda d, i: furthest(d['frac'][i])),
                      ('x50', lambda d, i: 10 * np.searchsorted(np.cumsum(d['dprof'][i]) / d['dprof'][i].sum(), 0.5)),
                      ('x90', lambda d, i: 10 * np.searchsorted(np.cumsum(d['dprof'][i]) / d['dprof'][i].sum(), 0.9))):
        print(f'\n{terr}: {name}, m (hours ' + ', '.join(f'{h:g}' for h in HRS) + ')')
        for f in fs:
            d = np.load(f)
            if 'cfrac' not in d:
                continue
            t = d['t'] / 3600
            row = [f'{get(d, int(np.argmin(abs(t - h)))):5.0f}' if abs(t - h).min() < 0.02 else '    -' for h in HRS]
            print(f'  {f.split("/")[-1][5 + len(terr) + 1:-4]:16s} ' + ' '.join(row) + f'   ({int(d["ntri"])} cells, {int(d["steps"])} steps, {float(d["wall"]):.0f} s)')
