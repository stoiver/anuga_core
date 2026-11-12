"""Score Phase 0 runs against the resolved reference.

    python evaluate_phase0.py [REFERENCE]      (default reference: fine if present, else resolved)

Metrics per run:
  peak discharge and its time at each section; arrival time (Q first exceeds
  half the reference peak); peak stage along the centreline (every 50 m);
  max flood extent downscaled to the 1 m DTM (pixel wet if its cell was wet
  and the cell's peak stage is above the pixel); CSI against the reference; mass balance;
  runtime. Writes phase0_summary.txt and phase0_*.png.
"""
import glob
import os
import sys

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.tri import Triangulation

import terrain as T

ORDER = ['band025TU', 'band1LF4', 'band025', 'fine', 'band1', 'resolved', 'coarse500_centroid', 'coarse500_trench',
         'coarse2000_centroid', 'coarse2000_trench',
         'coarse500_sub8', 'coarse500_sub16', 'coarse2000_sub8', 'coarse2000_sub16',
         'coarse500_sub8K', 'coarse500_sub16K', 'coarse2000_sub8K',
         'coarse500_sub8E', 'coarse500_sub8EK', 'coarse2000_sub8EK',
         'coarse500_sub8EKL', 'coarse2000_sub8EKL', 'coarse500_sub8EKLC', 'coarse2000_sub8EKLC',
         'coarse500_sub8EKLCtu', 'coarse2000_sub8EKLCtu',
         'coarse500_sub8EKLCT', 'coarse2000_sub8EKLCT', 'coarse500_sub8EKLCTflow']      # T: cell-problem tensor friction
T_SPIN = 5400.0          # arrival times ignore the spin-up transient
PROFILE_X = np.arange(25.0, 1975.0, 50.0)
DRY = 0.01


def load():
    runs = {}
    for name in ORDER:
        p = f'results/{name}.npz'
        if os.path.exists(p):
            runs[name] = dict(np.load(p))
    return runs


def owner(run, x, y):
    f = Triangulation(run['nodes'][:, 0], run['nodes'][:, 1], run['triangles']).get_trifinder()
    return f(x, y)


def downscaled_extent(run, px, py, pz):
    # a pixel is wet if its cell was wet and the cell's peak level is above it
    cell = owner(run, px, py)
    cell_wet = run['peak_stage'] - run['bed'] > DRY
    return cell_wet[cell] & (run['peak_stage'][cell] > pz + DRY)


def main():
    runs = load()
    if not runs:
        sys.exit('no results yet')
    ref_name = sys.argv[1] if len(sys.argv) > 1 else next(n for n in ('band025', 'fine', 'band1', 'resolved') if n in runs)
    ref = runs[ref_name]
    px, py, pz = T.dtm()
    # subsample the DTM 2x for extent (1 m pixels) to keep this quick
    keep = ((np.round(px / T.PIXEL - 0.5) % 2 == 0) & (np.round(py / T.PIXEL - 0.5) % 2 == 0))
    px, py, pz = px[keep], py[keep], pz[keep]
    pix_area = (2 * T.PIXEL) ** 2

    ref_ext = downscaled_extent(ref, px, py, pz)
    ref_peakQ = ref['q_sections'].max(axis=0)
    sec_x = ref['section_x']
    yc = T.centreline(PROFILE_X)

    out = []
    w = out.append
    w(f'Reference: {ref_name}\n')
    w(f'{"run":22s}{"ntri":>8s}{"steps":>9s}{"dt":>7s}{"wall s":>8s}{"gpu":>5s}'
      f'{"out/in":>10s}{"extent km2":>11s}{"CSI":>6s}')
    profiles = {}
    for name, r in runs.items():
        outvol = np.sum(r['q_sections'][:, -1] * np.diff(r['times']))
        mass = outvol / r['inflow_volume'][-1]
        ext = downscaled_extent(r, px, py, pz)
        csi = np.sum(ext & ref_ext) / np.sum(ext | ref_ext)
        dt = r['times'][-1] / int(r['nsteps'])
        w(f'{name:22s}{int(r["ntri"]):8d}{int(r["nsteps"]):9d}{dt:7.3f}{float(r["wall"]):8.0f}'
          f'{str(bool(r["gpu"])):>5s}{100 * mass:9.2f}%{np.sum(ext) * pix_area / 1e6:11.3f}{csi:6.3f}')
        cells = owner(r, PROFILE_X, yc)
        profiles[name] = r['peak_stage'][cells]

    w('\nPeak discharge (m3/s) at sections x =' + ''.join(f'{x:8.0f}' for x in sec_x))
    w(f'{"inflow peak":22s}{ref["q_in"].max():8.1f}')
    for name, r in runs.items():
        w(f'{name:22s}' + ''.join(f'{q:8.1f}' for q in r['q_sections'].max(axis=0)))

    w('\nTime of peak discharge (h)')
    for name, r in runs.items():
        w(f'{name:22s}' + ''.join(f'{t:8.2f}' for t in r['q_times'][np.argmax(r['q_sections'], axis=0)] / 3600))

    w('\nArrival time (h): Q first exceeds half the reference peak at that section')
    for name, r in runs.items():
        q = r['q_sections']
        arr = []
        for k in range(q.shape[1]):
            idx = np.nonzero((q[:, k] > 0.5 * ref_peakQ[k]) & (r['q_times'] > T_SPIN))[0]
            arr.append(r['q_times'][idx[0]] / 3600 if len(idx) else np.nan)
        w(f'{name:22s}' + ''.join(f'{a:8.2f}' for a in arr))

    w('\nPeak stage on the centreline, error against reference (m): mean, mean |.|, max |.|')
    for name in runs:
        e = profiles[name] - profiles[ref_name]
        w(f'{name:22s}{np.mean(e):8.3f}{np.mean(np.abs(e)):8.3f}{np.max(np.abs(e)):8.3f}')

    text = '\n'.join(out)
    print(text)
    with open('phase0_summary.txt', 'w') as f:
        f.write(text + '\n')

    # plots
    fig, axes = plt.subplots(1, 3, figsize=(17, 4.8))
    for k, ax in zip([2, 4], axes[:2]):
        ax.plot(ref['times'] / 3600, ref['q_in'], 'k:', label='inflow')
        for name, r in runs.items():
            ax.plot(r['q_times'] / 3600, r['q_sections'][:, k], label=name)
        ax.set_title(f'Discharge at x = {sec_x[k]:.0f} m')
        ax.set_xlabel('time (h)')
        ax.set_ylabel('Q (m3/s)')
    axes[0].legend(fontsize=8)
    ax = axes[2]
    for name in runs:
        ax.plot(PROFILE_X, profiles[name] - T.channel_bed(PROFILE_X), label=name)
    ax.axhline(T.DEPTH, color='grey', lw=0.8, ls='--')
    ax.set_title('Peak depth above thalweg along the channel')
    ax.set_xlabel('x (m)')
    ax.set_ylabel('m')
    fig.tight_layout()
    fig.savefig('phase0_hydrographs.png', dpi=110)

    fig, axes = plt.subplots(len(runs), 1, figsize=(12, 2.4 * len(runs)), sharex=True)
    axes = np.atleast_1d(axes)
    for ax, name in zip(axes, runs):
        ext = downscaled_extent(runs[name], px, py, pz)
        img = np.zeros(ext.shape, np.int8)
        img[ext & ref_ext] = 1
        img[ext & ~ref_ext] = 2
        img[~ext & ref_ext] = 3
        nx = int(T.LENGTH / (2 * T.PIXEL))
        ax.imshow(img.reshape(-1, nx), origin='lower', extent=[0, T.LENGTH, 0, T.WIDTH],
                  cmap=matplotlib.colors.ListedColormap(['white', '#4a90d9', '#d94a4a', '#e6b800']),
                  vmin=0, vmax=3, interpolation='nearest')
        ax.set_title(f'{name}: max extent (blue both, red extra, yellow missed vs {ref_name})', fontsize=9)
    fig.tight_layout()
    fig.savefig('phase0_extent.png', dpi=90)


if __name__ == '__main__':
    main()
