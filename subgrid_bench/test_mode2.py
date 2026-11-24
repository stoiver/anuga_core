"""Subgrid step in compute mode unified (mode 2) against legacy (mode 1).

    OMP_NUM_THREADS=1 python test_mode2.py            # CPU build: both modes on the host
    python test_mode2.py                              # GPU build: mode 2 on the device

A valley with a channel narrower than a cell, rain, a free outfall and walls,
first order, with the conveyance split and the draining limiter. The two modes
run the same kernels, so levels, beds and momenta must agree closely; they
differ only in where the protection against tiny depths falls relative to the
rain, so the tolerance is not round-off.

The release is also run with ponds below a spill level (pond_storage) and a
conveyance tensor: the pond exchange and the tensor friction are device kernels
in mode 2. On a GPU build each mode-2 domain wants its own process (the NVHPC
runtime hangs after several in one), so there every case is run as a child.
"""
import sys

import numpy as np

import anuga
from subgrid_tables import SubgridTables, Subgrid_storage_operator, edge_subface_beds
from pond_storage import PondStore, Pond_exchange_operator

S, NSUB, MANNING = 8, 16, 0.05


def bed(x, y):
    valley = 20.0 - 0.004 * x + 0.02 * np.abs(y - 200.0)
    channel = -2.0 * np.exp(-((y - 200.0 - 30.0 * np.sin(x / 150.0)) / 8.0) ** 2)
    return valley + channel + 0.3 * np.sin(x / 37.0) * np.cos(y / 23.0)


def hollows(x, y):
    """Depth of closed hollows below their spill level (the ponds)."""
    return 0.4 * np.maximum(np.sin(x / 11.0) * np.sin(y / 9.0) - 0.6, 0.0)


def run(mode, rain=True, finaltime=900.0, yieldstep=300.0, nx=24, ny=8, ponds=False):
    domain = anuga.rectangular_cross_domain(nx, ny, len1=1200.0, len2=400.0)
    domain.set_compute_mode(mode)
    domain.set_flow_algorithm('DE0')
    domain.set_store(False)
    domain.set_low_froude(1)
    domain.set_betas(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    n = len(domain)
    v = domain.get_vertex_coordinates().reshape(n, 3, 2)
    b = np.array([((i + 1 / 3) / S, (j + 1 / 3) / S) for i in range(S) for j in range(S - i)]
                 + [((i + 2 / 3) / S, (j + 2 / 3) / S) for i in range(S) for j in range(S - i - 1)])
    w = np.c_[1 - b.sum(axis=1), b]                                       # (m, 3)
    px, py = (w @ v[:, :, 0].T).T, (w @ v[:, :, 1].T).T                   # (n, m)
    tables = SubgridTables(bed(px, py), domain.areas, nsub=np.full(px.shape, MANNING))   # the raised bed with ponds
    ez, ein = edge_subface_beds(domain, tables, NSUB, bed, manning=lambda x, y: np.full(np.shape(x), MANNING))
    domain.set_quantity('friction', MANNING)
    domain.set_boundary({'left': anuga.Reflective_boundary(domain), 'top': anuga.Reflective_boundary(domain),
                         'bottom': anuga.Reflective_boundary(domain),
                         'right': anuga.Dirichlet_boundary([float(tables.zmin.min()) - 5.0, 0.0, 0.0])})
    if rain:
        anuga.Rate_operator(domain, rate=lambda t: 50.0 / 1000.0 / 3600.0 if t < 600.0 else 0.0)
    pond = pop = None
    if ponds:
        pond = PondStore(bed(px, py) - hollows(px, py), bed(px, py), domain.areas, S)
        pop = Pond_exchange_operator(domain, tables, pond)                # before the storage operator: runs first
        L = 6                                                             # a tensor that turns with the level
        tables.tlev = tables.zmin[:, None] + np.linspace(0.05, 1.5, L)[None, :]
        ang = 0.7 * np.arange(n)[:, None] + 0.3 * np.arange(L)[None, :]
        l1, l2 = 0.9 - 0.05 * np.arange(L)[None, :] + 0 * ang, 0.2 + 0.1 * np.cos(ang)
        c, s = np.cos(ang), np.sin(ang)
        tables.tR = np.ascontiguousarray(np.stack([l1 * c * c + l2 * s * s, (l1 - l2) * c * s, l1 * s * s + l2 * c * c], -1))
    op = Subgrid_storage_operator(domain, tables, cap_min_fraction=0.0, cfl=0.5, conveyance_friction=True,
                                  edge_z=ez, cap_exponent=0.0, drain_limiter=True, edge_invn=ein, tensor=ponds)
    level = tables.zmin.copy()
    if not rain:                                                          # a pool in the upper valley, released
        level = np.where(domain.centroid_coordinates[:, 0] < 400.0, np.maximum(tables.zmin, 19.5), tables.zmin)
    op.set_initial_level(level)
    st, el = domain.quantities['stage'].centroid_values, domain.quantities['elevation'].centroid_values
    xm, ym = domain.quantities['xmomentum'].centroid_values, domain.quantities['ymomentum'].centroid_values
    out, steps = [], 0
    for t in domain.evolve(yieldstep=yieldstep, finaltime=finaltime):
        steps += domain.number_of_steps
        if pop is not None:
            pop.sync()
        out.append((st.copy(), el.copy(), xm.copy(), ym.copy()) + ((pond.V.copy(),) if ponds else ()))
    vol = float((np.maximum(st - el, 0.0) * domain.areas).sum()) + (float(pond.V.sum()) if ponds else 0.0)
    return out, steps, vol, domain.areas


def run_child(mode, rain, ponds):
    """run() in a fresh process: one mode-2 domain per process on a GPU build."""
    import pickle, subprocess
    r = subprocess.run([sys.executable, __file__, '--child', mode, str(int(rain)), str(int(ponds))], capture_output=True)
    if r.returncode != 0:
        raise RuntimeError(r.stderr.decode()[-2000:])
    return pickle.loads(r.stdout[r.stdout.index(b'PICKLE') + 6:])


def main():
    fails = 0
    for rain, ponds in ((False, False), (True, False), (False, True)):
        a, sa, va, area = run_child('legacy', rain, ponds)
        b, sb, vb, _ = run_child('unified', rain, ponds)
        print(f"{'rain' if rain else 'release'}{', ponds and tensor' if ponds else ''}: steps legacy {sa}, unified {sb}; "
              f"volume {va:.6f} against {vb:.6f} m3" + (f"; ponds hold {a[-1][4].sum():.3f} against {b[-1][4].sum():.3f} m3" if ponds else ''))
        for k, (p, q) in enumerate(zip(a, b)):
            h1, h2 = np.maximum(p[0] - p[1], 0.0), np.maximum(q[0] - q[1], 0.0)
            wet = (h1 > 1e-4) | (h2 > 1e-4)
            dl = np.abs(p[0] - q[0])[wet].max() if wet.any() else 0.0
            dh = np.abs(h1 - h2).max()
            dq = max(np.abs(p[2] - q[2]).max(), np.abs(p[3] - q[3]).max())
            print(f"   yield {k}: wet cells {int(wet.sum())}, max |level diff| {dl:.3e} m, |depth diff| {dh:.3e} m, |momentum diff| {dq:.3e} m2/s")
            tol = 1.0e-9 if not rain else 1.0e-3
            if not (dl < tol and dh < tol and dq < tol):
                fails += 1
            if ponds and not np.abs(p[4] - q[4]).max() < 1.0e-6:
                fails += 1
        if abs(va - vb) > 1e-6 * max(va, 1.0) + (0.0 if not rain else 1e-3 * va):
            fails += 1
    print('FAILED' if fails else 'passed')
    return 1 if fails else 0


if __name__ == '__main__':
    if len(sys.argv) > 1 and sys.argv[1] == '--child':
        import pickle
        res = run(sys.argv[2], rain=sys.argv[3] == '1', ponds=sys.argv[4] == '1')
        sys.stdout.flush()
        sys.stdout.buffer.write(b'PICKLE' + pickle.dumps(res))
        sys.exit(0)
    sys.exit(main())
