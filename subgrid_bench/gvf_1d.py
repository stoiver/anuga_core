"""gvf_1d.py Q: 1D steady water-surface profile of the straight-channel valley
(SUBGRID_AMPLITUDE=0) by the standard step method, from critical depth at the
slope break (x = 1900) upstream. Cross-sections are sampled from the analytic bed
every 0.1 m; conveyance by strips with n per strip; energy with the Coriolis
coefficient alpha = (sum K_i^3/A_i^2) / (K^3/A^2). Writes results/gvf_1d_Q<Q>.npz.
"""
import os
import sys

os.environ['SUBGRID_AMPLITUDE'] = '0'
import numpy as np  # noqa: E402

import terrain as T  # noqa: E402
assert T.AMPLITUDE == 0

G = 9.81
N_CH, N_FP = 0.035, 0.05
DY = 0.1
y = np.arange(DY / 2, T.WIDTH, DY)


def section(x):
    xx = np.full_like(y, x)
    z = T.bed(xx, y)
    n = np.where(T.distance_to_centreline(xx, y) < T.TOP_WIDTH / 2, N_CH, N_FP)
    return z, n


def props(eta, z, n, Q):
    h = np.maximum(eta - z, 0.0)
    a = h * DY
    k = h ** (5 / 3) / n * DY
    A, K = a.sum(), k.sum()
    wet = a > 0
    alpha = np.sum(k[wet] ** 3 / a[wet] ** 2) / (K ** 3 / A ** 2)
    V = Q / A
    return A, K, alpha, eta + alpha * V ** 2 / (2 * G)


def critical(x, Q):
    z, n = section(x)
    etas = np.linspace(z.min() + 0.05, z.min() + 4.0, 4000)
    E = [props(e, z, n, Q)[3] for e in etas]
    return etas[int(np.argmin(E))]


def profile(Q, x_ctrl=T.OUTLET_START, x_end=1000.0, dx=5.0):
    xs = np.arange(x_ctrl, x_end - 1e-9, -dx)
    eta = critical(x_ctrl, Q)
    z, n = section(x_ctrl)
    A, K, al, H = props(eta, z, n, Q)
    out = [eta]
    for x in xs[1:]:
        Sf1 = (Q / K) ** 2
        z, n = section(x)
        # solve H(eta) = H_down + dx * (Sf1 + Sf2)/2 on the subcritical branch
        lo, hi = eta, eta + 3.0
        for _ in range(80):
            mid = 0.5 * (lo + hi)
            A2, K2, al2, H2 = props(mid, z, n, Q)
            res = H2 - (H + dx * 0.5 * (Sf1 + (Q / K2) ** 2))
            lo, hi = (mid, hi) if res < 0 else (lo, mid)
        eta = 0.5 * (lo + hi)
        A, K, al, H = props(eta, z, n, Q)
        out.append(eta)
    return xs, np.array(out)


if __name__ == '__main__':
    Q = float(sys.argv[1]) if len(sys.argv) > 1 else 50.0
    xs, eta = profile(Q)
    np.savez(f'results/gvf_1d_Q{Q:g}.npz', x=xs, eta=eta)
    for x, e in zip(xs[::10], eta[::10]):
        print(f'x {x:6.0f}  level {e:7.3f}  depth above thalweg {e - T.channel_bed(x):6.3f}')
