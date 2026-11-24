"""Score the runs of meander_floodplain.py against a reference run.

    python meander_floodplain_compare.py [--ref ref8] [--out results_meander] [--plot]

Outflow is the rate of change of the boundary flux integral over each yieldstep. Flood extent is the DEM pixels
whose maximum depth passed 0.05 m; fit = pixels wet in both / pixels wet in either. Levels are at the gauges
down the channel, compared where both runs are wet.
"""
import argparse
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent


def load(d):
    s = np.genfromtxt(d / "series.csv", delimiter=",", names=True)
    m = np.load(d / "maxima.npz")
    q = -np.gradient(s["boundary_m3"], s["t_s"])
    g = np.array([s[k] for k in s.dtype.names if k.startswith("gauge")])
    return dict(t=s["t_s"], q=q, vol=s["volume_m3"], g=g, depth=m["depth"], first=m["first_wet_s"], x=m["x"], y=m["y"],
                ncell=int(m["ncell"]), steps=int(m["steps"]), wall=float(m["wall"]), out=-s["boundary_m3"][-1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", default="ref8")
    ap.add_argument("--out", type=Path, default=HERE / "results_meander")
    ap.add_argument("--plot", action="store_true")
    args = ap.parse_args()
    names = sorted((p.parent.name for p in args.out.glob("*/maxima.npz")),
                   key=lambda s: (not s.startswith("ref"), "_t" in s, float(s.split("_")[0][3:])))
    R = {k: load(args.out / k) for k in names}
    ref = R[args.ref]
    pix = (ref["x"][1] - ref["x"][0]) ** 2
    A = ref["depth"] > 0.05
    kp = int(np.argmax(ref["q"]))
    print(f"reference {args.ref}: {ref['ncell']:,} cells; peak outflow {ref['q'].max():.1f} m3/s at {ref['t'][kp]/60:.0f} min; "
          f"flooded {A.sum()*pix/1e4:.1f} ha; held at the peak of storage {ref['vol'].max():,.0f} m3\n")
    print(f"{'run':12s} {'cells':>7s} {'steps':>8s} {'wall s':>7s} | {'Qpeak':>6s} {'t min':>5s} {'rmse/Qp':>7s} | {'Vmax %':>6s} | "
          f"{'ha':>5s} {'fit':>5s} {'|dd| m':>6s} | {'arrive s':>8s} | {'level rmse m':>12s} {'peak diff m':>11s}")
    for k, r in R.items():
        B = r["depth"] > 0.05
        U = A | B
        dd = np.abs(np.nan_to_num(r["depth"]) - np.nan_to_num(ref["depth"]))[U].mean()
        both = np.isfinite(r["first"]) & np.isfinite(ref["first"])
        arr = np.median((r["first"] - ref["first"])[both])
        ok = np.isfinite(r["g"]) & np.isfinite(ref["g"])
        lv = np.sqrt(np.mean((r["g"] - ref["g"])[ok] ** 2))
        with np.errstate(all="ignore"):
            pk = np.nanmax(r["g"], axis=1) - np.nanmax(ref["g"], axis=1)
        kq = int(np.argmax(r["q"]))
        print(f"{k:12s} {r['ncell']:7,d} {r['steps']:8,d} {r['wall']:7.0f} | {r['q'].max():6.1f} {r['t'][kq]/60:5.0f} "
              f"{np.sqrt(np.mean((r['q']-ref['q'])**2))/ref['q'].max():7.3f} | {100*(r['vol'].max()/ref['vol'].max()-1):+6.1f} | "
              f"{B.sum()*pix/1e4:5.1f} {(A & B).sum()/U.sum():5.3f} {dd:6.3f} | {arr:+8.0f} | {lv:12.3f} "
              f"{np.nanmean(pk):+6.3f}/{np.nanmax(np.abs(pk)):.3f}")
    if args.plot:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        show = [k for k in R if k != args.ref]
        fig, ax = plt.subplots(1, 2, figsize=(13, 4.5))
        for k in show:
            ax[0].plot(R[k]["t"] / 60, R[k]["q"], lw=1.3, ls="--" if k.startswith("ref") else "-", label=k)
            ax[1].plot(R[k]["t"] / 60, R[k]["vol"] / 1e3, lw=1.3, ls="--" if k.startswith("ref") else "-", label=k)
        ax[0].plot(ref["t"] / 60, ref["q"], "k", lw=2, label=args.ref)
        ax[1].plot(ref["t"] / 60, ref["vol"] / 1e3, "k", lw=2, label=args.ref)
        ax[0].set(xlabel="minutes", ylabel="outflow (m3/s)", title="Outflow")
        ax[1].set(xlabel="minutes", ylabel="volume held (1000 m3)", title="Storage")
        ax[0].legend(fontsize=7, ncol=2)
        fig.tight_layout()
        fig.savefig(args.out / "hydrographs.png", dpi=130)
        fig, axs = plt.subplots(2, (len(show) + 2) // 2, figsize=(4 * ((len(show) + 2) // 2), 8.5), squeeze=False)
        ext = [ref["x"][0], ref["x"][-1], ref["y"][-1], ref["y"][0]]
        for a, k in zip(axs.ravel(), [args.ref] + show):
            im = a.imshow(np.where(R[k]["depth"] > 0.05, R[k]["depth"], np.nan), extent=ext, vmin=0, vmax=2.5, cmap="Blues")
            a.contour(ref["x"], ref["y"], np.nan_to_num(ref["depth"]) > 0.05, levels=[0.5], colors="r", linewidths=0.5)
            a.set_title(f"{k}: {R[k]['ncell']:,} cells", fontsize=9)
            a.set_xticks([]); a.set_yticks([])
        for a in axs.ravel()[len(show) + 1:]:
            a.axis("off")
        fig.colorbar(im, ax=axs, shrink=0.6, label="maximum depth (m); red: reference extent")
        fig.savefig(args.out / "max_depth.png", dpi=130, bbox_inches="tight")
        print(f"\n{args.out}/hydrographs.png, max_depth.png")


if __name__ == "__main__":
    main()
