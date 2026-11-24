"""Mesh controls for the meander test, derived from the DEM by dtm_lines.derive: bank lines and the river region,
sills between the river and the floodplain compartments, crests and scarp rims.

    python meander_controls.py [--plot]   ->  results_meander/controls.json
"""
import argparse
import json
from pathlib import Path

import numpy as np
from matplotlib.path import Path as MplPath
from scipy.interpolate import RegularGridInterpolator
from scipy.ndimage import distance_transform_edt

from dtm_lines import derive
from meander_floodplain import EXAMPLE, HERE, read_asc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pixel", type=float, default=1.0)
    ap.add_argument("--inlet", type=float, nargs=2, default=[66.0, 1030.0])
    ap.add_argument("--out", type=Path, default=HERE / "results_meander" / "controls.json")
    ap.add_argument("--plot", action="store_true")
    args = ap.parse_args()
    zraw, xll, yll, pix0 = read_asc(EXAMPLE / "topo.asc")
    valid = np.isfinite(zraw)
    zf = zraw[tuple(distance_transform_edt(~valid, return_distances=False, return_indices=True))]
    ny0, nx0 = zf.shape
    xc0, yc0 = (np.arange(nx0) + 0.5) * pix0, (ny0 - np.arange(ny0) - 0.5) * pix0
    poly = np.loadtxt(EXAMPLE / "outline.csv", delimiter=",") - [xll, yll]
    pix = args.pixel
    n = int(round(nx0 * pix0 / pix)), int(round(ny0 * pix0 / pix))
    xs, ys = (np.arange(n[0]) + 0.5) * pix, (np.arange(n[1]) + 0.5) * pix          # row 0 at the south
    X, Y = np.meshgrid(xs, ys)
    z = RegularGridInterpolator((yc0[::-1], xc0), zf[::-1], bounds_error=False, fill_value=None)(np.stack([Y.ravel(), X.ravel()], -1)).reshape(X.shape)
    ok = RegularGridInterpolator((yc0[::-1], xc0), valid[::-1].astype(float), method="nearest", bounds_error=False, fill_value=0.0)(
        np.stack([Y.ravel(), X.ravel()], -1)).reshape(X.shape) > 0.5
    ok &= MplPath(poly).contains_points(np.c_[X.ravel(), Y.ravel()]).reshape(X.shape)
    z = np.where(ok, z, np.nan)
    c = derive(z, pix, (0.0, 0.0), inlet=tuple(args.inlet), verbose=True)
    out = dict(channel=np.asarray(c["channel"]).tolist(), left=np.asarray(c["banks"]["left"]).tolist(), right=np.asarray(c["banks"]["right"]).tolist(),
               river=[list(p) for p in c["river"].exterior.coords], sills=[np.asarray(s).tolist() for s in c["sills"]],
               crests=[np.asarray(s).tolist() for s in c["crests"]], rims=[np.asarray(s).tolist() for s in c["rims"]],
               river_area=float(c["river_area"]), fp_area=float(c["fp_area"]), width=float(c["width"]))
    json.dump(out, open(args.out, "w"))
    print(args.out)
    if args.plot:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(9, 9))
        ax.imshow(z, origin="lower", extent=[0, xs[-1] + pix / 2, 0, ys[-1] + pix / 2], cmap="terrain", vmin=1523.5, vmax=1530)
        ax.plot(*np.array(out["river"]).T, "k", lw=1.2, label="river region")
        for k, col in (("sills", "red"), ("crests", "magenta"), ("rims", "orange")):
            for i, ln in enumerate(out[k]):
                ax.plot(*np.array(ln).T, color=col, lw=1.4, label=k if i == 0 else None)
        ax.legend()
        fig.savefig(args.out.with_suffix(".png"), dpi=110, bbox_inches="tight")


if __name__ == "__main__":
    main()
