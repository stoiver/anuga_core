"""Summarise the flat-plain bench: front position (furthest 10 m band more than half wet) by time."""
import glob
import numpy as np
HRS = [0.5, 1, 2, 3, 4]
for terr in ('flat', 'ridged', 'rough'):
    print(f'\n{terr}: front position, m (hours ' + ', '.join(f'{h:g}' for h in HRS) + ')')
    fs = sorted(glob.glob(f'results/flat_{terr}_*.npz'), key=lambda f: ('_sub' in f, float(f.split('_')[2]), f))
    for f in fs:
        d = np.load(f); t = d['t'] / 3600
        row = []
        for h in HRS:
            i = int(np.argmin(abs(t - h)))
            if abs(t[i] - h) > 0.02:
                row.append('   -'); continue
            w = d['frac'][i] > 0.5
            row.append(f'{10 * (np.nonzero(w)[0].max() + 1) if w.any() else 0:4.0f}')
        name = f.split('/')[-1][5:-4]
        print(f'  {name:18s} ' + ' '.join(row) + f'   ({int(d["ntri"])} cells, {float(d["wall"]):.0f} s)')
