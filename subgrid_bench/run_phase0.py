"""Phase 0 dynamic runs: a flood hydrograph through the synthetic valley.

    python run_phase0.py VARIANT [--gpu]

VARIANT    mesh and bed
  coarse2000_centroid   max 2000 m2, bed sampled at the cell centroid
  coarse2000_trench     same, cells crossed by the centreline lowered to the thalweg
  coarse500_centroid    max 500 m2
  coarse500_trench
  resolved              floodplain 25 m2, channel band 4 m2   (reference)
  fine                  floodplain 6 m2,  channel band 1 m2   (convergence check)

Hydrograph: 1.5 h spin-up at 5 m3/s, rise to 60 m3/s over 1 h (cosine),
recede to 5 m3/s over 3 h; 5.5 h simulated. Downstream: the valley steepens
at x = 1900 (critical-depth control) and ends in a free outfall, a stage
1 m below the bed with the outgoing momentum only. (Earlier versions drove a
rating-curve stage at the outlet; the transmissive set-stage boundary then let
a reverse flow sustain itself on coarse500_trench, and a Flather boundary
collapsed the timestep in the recession on coarse2000_centroid.)

Discharge is measured conservatively from the volume budget (see main).

Writes results/VARIANT.npz: gauge stage and section discharge time series,
volume budget, per-cell peak stage, mesh, and run statistics.
"""
import argparse
import os
import time

import numpy as np
from matplotlib.tri import Triangulation

import anuga
from anuga.shallow_water.forcing import Cross_section
import terrain as T

N_CHANNEL = 0.035
N_FLOODPLAIN = 0.05
Q_BASE, Q_PEAK = 5.0, 60.0
T_SPIN, T_RISE, T_FALL = 5400.0, 3600.0, 3 * 3600.0
FINAL_TIME = T_SPIN + T_RISE + T_FALL
YIELDSTEP = 60.0
GAUGE_X = [250.0, 500.0, 1000.0, 1500.0, 1750.0]
OUTLET_SECTION_X = 1950.0
SECTION_OFFSET = 0.37   # keep sections off mesh vertices (Cross_section asserts on them)

VARIANTS = {
    'coarse2000_centroid': (2000.0, None, False),
    'coarse2000_trench': (2000.0, None, True),
    'coarse500_centroid': (500.0, None, False),
    'coarse500_trench': (500.0, None, True),
    'resolved': (25.0, 4.0, False),
    'fine': (6.0, 1.0, False),
    'band1': (25.0, 1.0, False),         # 1 m2 channel band, 25 m2 floodplain (matches fine)
    'band025': (25.0, 0.25, False),      # convergence check of the channel band
    # Phase 1 prototype: storage-only subgrid tables (mode 1, Python operator)
    'coarse2000_sub8': (2000.0, None, False),
    'coarse2000_sub16': (2000.0, None, False),
    'coarse500_sub8': (500.0, None, False),
    'coarse500_sub16': (500.0, None, False),
    # Phase 2 start: storage tables + cell conveyance friction (n_eff = h^5/3 / Kc)
    'coarse2000_sub8K': (2000.0, None, False),
    'coarse500_sub8K': (500.0, None, False),
    'coarse500_sub16K': (500.0, None, False),
    # Phase 2: + subgrid edge flux kernel (16 subfaces per edge); EK adds cell
    # conveyance friction
    'coarse500_sub8E': (500.0, None, False),
    'coarse500_sub8EK': (500.0, None, False),
    'coarse2000_sub8EK': (2000.0, None, False),
    # + draining-time limiter, wet fraction dropped from the step cap
    'coarse500_sub8EKL': (500.0, None, False),
    'coarse2000_sub8EKL': (2000.0, None, False),
    # + conveyance-weighted subface velocity in the edge flux
    'coarse500_sub8EKLC': (500.0, None, False),
    'coarse2000_sub8EKLC': (2000.0, None, False),
}
CAP_MIN_FRACTION = 0.05
SUBGRID_N = {'coarse2000_sub8': 8, 'coarse2000_sub16': 16,
             'coarse500_sub8': 8, 'coarse500_sub16': 16,
             'coarse2000_sub8K': 8, 'coarse500_sub8K': 8, 'coarse500_sub16K': 16}
for _v in ('coarse500_sub8E', 'coarse500_sub8EK', 'coarse2000_sub8EK',
           'coarse500_sub8EKL', 'coarse2000_sub8EKL', 'coarse500_sub8EKLC', 'coarse2000_sub8EKLC'):
    SUBGRID_N[_v] = 8
CONVEYANCE = {'coarse2000_sub8K', 'coarse500_sub8K', 'coarse500_sub16K',
              'coarse500_sub8EK', 'coarse2000_sub8EK', 'coarse500_sub8EKL', 'coarse2000_sub8EKL',
              'coarse500_sub8EKLC', 'coarse2000_sub8EKLC'}
EDGE_FLUX = {'coarse500_sub8E', 'coarse500_sub8EK', 'coarse2000_sub8EK',
             'coarse500_sub8EKL', 'coarse2000_sub8EKL', 'coarse500_sub8EKLC', 'coarse2000_sub8EKLC'}
DRAIN_LIMITER = {'coarse500_sub8EKL', 'coarse2000_sub8EKL', 'coarse500_sub8EKLC', 'coarse2000_sub8EKLC'}
CONVEYANCE_SPLIT = {'coarse500_sub8EKLC', 'coarse2000_sub8EKLC'}
EDGE_NSUB = 16
EDGE_CAP_CFL = 0.75     # the full cap, with margin: see test_phase2.py
# The conveyance split puts the channel's speed on the deep subfaces of partly
# wet channel cells: at 0.75 cells near the inlet oscillate (coarse2000, depth
# jumping 0.6-1.3 m every 10 s on the rising limb); 0.6 still rings, 0.5 is
# clean. A wet-fraction cap (p = 0.5, or p = 1 for f >= 0.2) also cures it but
# takes 2-3x the steps.
EDGE_CAP_CFL_SPLIT = 0.5
EDGE_CAP_P = 0.0        # wet-fraction exponent of the cap with the drain limiter
EDGE_CAP_MIN_FRACTION = 0.0


def inflow(t):
    if t < T_SPIN:
        return Q_BASE
    t = t - T_SPIN
    if t < T_RISE:
        return Q_BASE + (Q_PEAK - Q_BASE) * 0.5 * (1 - np.cos(np.pi * t / T_RISE))
    t = t - T_RISE
    if t < T_FALL:
        return Q_BASE + (Q_PEAK - Q_BASE) * 0.5 * (1 + np.cos(np.pi * t / T_FALL))
    return Q_BASE


def friction(x, y):
    return np.where(T.distance_to_centreline(x, y) < T.TOP_WIDTH / 2, N_CHANNEL, N_FLOODPLAIN)


def sinuosity():
    x = np.linspace(0, T.LENGTH, 200001)
    y = T.centreline(x)
    return np.sum(np.hypot(np.diff(x), np.diff(y))) / T.LENGTH


class Rating:
    """Divided-channel Manning rating Q(stage) for the transect at x."""

    def __init__(self, x):
        self.y = np.arange(0.25, T.WIDTH, 0.5)
        self.x = np.full_like(self.y, x)
        self.z = T.bed(self.x, self.y)
        self.n = friction(self.x, self.y)
        in_channel = T.distance_to_centreline(self.x, self.y) < T.TOP_WIDTH / 2
        self.s = np.where(in_channel, T.SLOPE / sinuosity(), T.SLOPE)
        stages = np.linspace(self.z.min() + 1e-3, self.z.max() + 2.0, 4000)
        self.stages = stages
        self.q = np.array([self.discharge(s) for s in stages])

    def discharge(self, stage):
        h = np.maximum(stage - self.z, 0.0)
        return np.sum(h ** (5 / 3) * np.sqrt(self.s) / self.n) * 0.5

    def stage(self, q):
        return np.interp(q, self.q, self.stages)


class Diagnostics:
    """Time series for the attenuation diagnosis (--diag), every yieldstep.

    sections  discharge through x = GAUGE_X split into channel (|d| < TOP_WIDTH/2
              from the centreline) and floodplain: the x-flux q_x(y) integrated
              along the section every 0.5 m. Standard cells give q_x = uh of the
              cell; subgrid cells give the model's own closure, the cell
              velocity uh/h times the local depth eta - z(y), or with the
              conveyance split uh * h_y^(5/3) / (n_y * Kc).
    storage   water over the DTM (1 m grid), eta of the owning cell minus the
              ground, summed per 250 m reach and region: channel, near
              floodplain (9-60 m from the centreline) and far floodplain.
    probes    water level at x = PROBE_X, offsets PROBE_OFF either side.
    """
    REACH = np.arange(0.0, T.LENGTH + 1.0, 250.0)
    NEAR = 60.0
    PROBE_X = [1000.0, 1500.0]
    PROBE_OFF = [0.0, 15.0, 30.0, 60.0, 90.0, 120.0]
    DY = 0.5

    def __init__(self, domain, subgrid):
        self.subgrid = subgrid
        self.domain = domain
        finder = Triangulation(domain.nodes[:, 0], domain.nodes[:, 1],
                               domain.triangles).get_trifinder()
        y = np.arange(self.DY / 2, T.WIDTH, self.DY)
        self.sections = []
        for gx in GAUGE_X:
            x = np.full_like(y, gx + SECTION_OFFSET)
            self.sections.append((finder(x, y), T.bed(x, y),
                                  T.distance_to_centreline(x, y) < T.TOP_WIDTH / 2,
                                  1.0 / friction(x, y)))
        gx, gy = np.meshgrid(np.arange(0.5, T.LENGTH, 1.0), np.arange(0.5, T.WIDTH, 1.0))
        gx, gy = gx.ravel(), gy.ravel()
        self.pix_own = finder(gx, gy)
        self.pix_z = T.bed(gx, gy)
        d = T.distance_to_centreline(gx, gy)
        region = np.where(d < T.TOP_WIDTH / 2, 0, np.where(d < self.NEAR, 1, 2))
        reach = np.clip(np.digitize(gx, self.REACH) - 1, 0, len(self.REACH) - 2)
        self.nreach = len(self.REACH) - 1
        self.pix_key = reach * 3 + region
        px, py = [], []
        for x0 in self.PROBE_X:
            yc = T.centreline(np.array([x0]))[0]
            for o in self.PROBE_OFF:
                for sgn in ((1,) if o == 0 else (1, -1)):
                    px.append(x0)
                    py.append(yc + sgn * o)
        self.probe_xy = np.column_stack([px, py])
        self.probe_own = finder(self.probe_xy[:, 0], self.probe_xy[:, 1])
        self.t, self.qch, self.qfp, self.store, self.probe = [], [], [], [], []

    def record(self, t, stage, elev, xmom):
        h_cell = stage - elev
        qch, qfp = [], []
        # conveyance split: the operator sets n_eff = h^(5/3)/Kc, so Kc = h^(5/3)/n_eff
        # (cells with no conveyance have no wet sub-triangle, so h_y = 0 there anyway)
        neff = getattr(self.domain, 'subgrid_cell_neff', None)
        kc = None
        if neff is not None:
            kc = np.where(neff > 0, np.maximum(h_cell, 0.0) ** (5 / 3) / np.maximum(neff, 1e-300), 0.0)
        for own, z, chan, invn in self.sections:
            if self.subgrid and kc is not None:
                k = kc[own]
                hy = np.maximum(stage[own] - z, 0.0)
                q = np.where(k > 0, xmom[own] * hy ** (5 / 3) * invn / np.maximum(k, 1e-300), 0.0)
            elif self.subgrid:
                u = np.where(h_cell[own] > 1e-6, xmom[own] / np.maximum(h_cell[own], 1e-6), 0.0)
                q = u * np.maximum(stage[own] - z, 0.0)
            else:
                q = xmom[own]
            qch.append(np.sum(q[chan]) * self.DY)
            qfp.append(np.sum(q[~chan]) * self.DY)
        hp = np.maximum(stage[self.pix_own] - self.pix_z, 0.0)
        st = np.bincount(self.pix_key, weights=hp, minlength=3 * self.nreach)
        self.t.append(t)
        self.qch.append(qch)
        self.qfp.append(qfp)
        self.store.append(st.reshape(self.nreach, 3))
        self.probe.append(stage[self.probe_own].copy())

    def arrays(self):
        return dict(diag_t=np.array(self.t), diag_qch=np.array(self.qch),
                    diag_qfp=np.array(self.qfp), diag_store=np.array(self.store),
                    diag_reach=self.REACH, diag_probe=np.array(self.probe),
                    diag_probe_xy=self.probe_xy,
                    diag_probe_ground=T.bed(self.probe_xy[:, 0], self.probe_xy[:, 1]))


def make_domain(variant):
    max_area, band, trench = VARIANTS[variant]
    regions = [[T.channel_band_polygon(), band]] if band else None
    # SUBGRID_OUTLET_AREA: refine x > SUBGRID_OUTLET_X (default 1800) to that area
    # (outlet-reach investigation: does the error follow the cells at the control?)
    outlet_area = float(os.environ.get('SUBGRID_OUTLET_AREA', 0))
    if outlet_area:
        x0 = float(os.environ.get('SUBGRID_OUTLET_X', 1800.0))
        poly = [[x0, 1.0], [T.LENGTH - 1.0, 1.0], [T.LENGTH - 1.0, T.WIDTH - 1.0], [x0, T.WIDTH - 1.0]]
        regions = (regions or []) + [[poly, outlet_area]]
    domain = anuga.create_domain_from_regions(
        T.BOUNDING_POLYGON, T.BOUNDARY_TAGS, maximum_triangle_area=max_area,
        interior_regions=regions)
    domain.set_name(variant)
    domain.set_store(False)
    if os.environ.get('SUBGRID_ALG'):          # e.g. DE_ader2 (default: config, DE0)
        domain.set_flow_algorithm(os.environ['SUBGRID_ALG'])
    x, y = domain.centroid_coordinates.T
    z = T.bed(x, y)
    n = friction(x, y)
    ncross = 0
    if trench:
        xs = np.clip(np.linspace(0, T.LENGTH, 200001), 1e-6, T.LENGTH - 1e-6)
        finder = Triangulation(domain.nodes[:, 0], domain.nodes[:, 1],
                               domain.triangles).get_trifinder()
        crossed = np.unique(finder(xs, T.centreline(xs)))
        crossed = crossed[crossed >= 0]
        z[crossed] = T.channel_bed(x[crossed])
        n[crossed] = N_CHANNEL
        ncross = len(crossed)
    domain.set_quantity('elevation', z, location='centroids')
    domain.set_quantity('friction', n, location='centroids')
    return domain, ncross


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('variant', choices=list(VARIANTS))
    ap.add_argument('--gpu', action='store_true')
    ap.add_argument('--final-time', type=float, default=FINAL_TIME)
    ap.add_argument('--outlet', choices=['tms', 'flather'], default='tms')
    ap.add_argument('--diag', action='store_true',
                    help='record attenuation diagnostics; writes results/VARIANT_diag.npz')
    args = ap.parse_args()
    out_name = args.variant + ('_diag' if args.diag else '') + os.environ.get('SUBGRID_TAG', '')

    anuga.set_gpu_offload(args.gpu)
    domain, ncross = make_domain(args.variant)
    sub_n = SUBGRID_N.get(args.variant)
    domain.set_compute_mode('legacy' if sub_n else 'unified')
    # low_froude mode, optionally | LOW_FROUDE_TANGENTIAL_UPWIND (4), e.g. 5 = LF1 +
    # tangential upwind (#510). Default 0, the standard central-upwind flux.
    lf = int(os.environ.get('SUBGRID_LOW_FROUDE', 0))
    if lf:
        domain.set_low_froude(lf)
    x, y = domain.centroid_coordinates.T
    z = domain.quantities['elevation'].centroid_values.copy()

    inlet_rating = Rating(0.0)
    h0 = inlet_rating.stage(Q_BASE) - T.channel_bed(0.0)
    domain.set_quantity('stage', np.maximum(z, T.water_surface(x, h0)), location='centroids')
    tables = None
    if sub_n:
        from static_storage import Cells
        from subgrid_tables import SubgridTables
        px, py, pz = T.dtm()
        cells = Cells(domain.nodes, domain.triangles, px, py)
        # SUBGRID_HOLLOWS=1: closed depressions in a cell fill only from their sills
        tables = SubgridTables.from_cells(cells, sub_n, px, py, pz,
                                          manning=friction if args.variant in CONVEYANCE else None,
                                          hollows=os.environ.get('SUBGRID_HOLLOWS', '0') == '1')
        del px, py, pz
        z = tables.zmin.copy()        # lowest point per cell: wet/dry and extent tests

    outfall_stage = T.channel_bed(T.LENGTH) - 1.0

    def outlet_stage(t):
        return outfall_stage

    br = anuga.Reflective_boundary(domain)
    if args.outlet == 'flather':
        # weakly reflecting; does not copy interior momentum, so a reverse flow
        # cannot sustain itself (the transmissive set-stage boundary let water
        # flow in at the outlet for over an hour on coarse500_trench)
        bo = anuga.Flather_external_stage_zero_velocity_boundary(
            domain, function=outlet_stage)
    else:
        bo = anuga.Transmissive_n_momentum_zero_t_momentum_set_stage_boundary(
            domain, function=outlet_stage)
    domain.set_boundary({'left': br, 'right': bo, 'top': br, 'bottom': br})

    yc0 = T.centreline(0.0)
    anuga.Inlet_operator(domain, [[1.0, yc0 - T.TOP_WIDTH / 2], [1.0, yc0 + T.TOP_WIDTH / 2]],
                         Q=inflow)
    sub_op = None
    if tables is not None:
        from subgrid_tables import Subgrid_storage_operator
        # cells under 5% wet (a corner clipping the channel) skip the wet-fraction
        # timestep cap: lake at rest stays at round-off for n=8 and 16, and the
        # step grows 16-92% (f >= 0.1 is unstable at n=16)
        edge = args.variant in EDGE_FLUX
        ez = ein = None
        if edge:
            from subgrid_tables import edge_subface_beds
            if args.variant in CONVEYANCE_SPLIT:
                ez, ein = edge_subface_beds(domain, tables, EDGE_NSUB, T.bed, manning=friction)
            else:
                ez = edge_subface_beds(domain, tables, EDGE_NSUB, T.bed)
        sub_op = Subgrid_storage_operator(domain, tables,       # after the inlet: runs last
                                          cap_min_fraction=EDGE_CAP_MIN_FRACTION if edge else CAP_MIN_FRACTION,
                                          cfl=(EDGE_CAP_CFL_SPLIT if args.variant in CONVEYANCE_SPLIT
                                               else EDGE_CAP_CFL) if edge else 1.0,
                                          conveyance_friction=args.variant in CONVEYANCE,
                                          edge_z=ez,
                                          cap_exponent=EDGE_CAP_P if args.variant in DRAIN_LIMITER else 1.0,
                                          drain_limiter=args.variant in DRAIN_LIMITER,
                                          edge_invn=ein)
        if os.environ.get('SUBGRID_P1', '0') == '1':
            # experimental: sloping level in each cell from its connected neighbours
            beta = os.environ.get('SUBGRID_P1_BETA')
            sub_op.enable_p1_level(beta=float(beta) if beta else None,
                                   min_sub=float(os.environ.get('SUBGRID_P1_MINSUB', 0.02)))
        sub_op.set_initial_level(T.water_surface(x, h0))
    elev_q = domain.quantities['elevation']

    sections = [Cross_section(domain, [[gx + SECTION_OFFSET, 0.0], [gx + SECTION_OFFSET, T.WIDTH]])
                for gx in GAUGE_X + [OUTLET_SECTION_X]]
    # Conservative discharge: Q(X) = Q_in - d/dt (volume in cells with centroid x < X),
    # i.e. the exact flux through that zig-zag of cell edges. X = LENGTH gives the outflow.
    budget_x = np.array(GAUGE_X + [OUTLET_SECTION_X, T.LENGTH])
    upstream = [x < bx for bx in budget_x]
    finder = Triangulation(domain.nodes[:, 0], domain.nodes[:, 1],
                           domain.triangles).get_trifinder()
    gauge_cells = finder(np.array(GAUGE_X), T.centreline(np.array(GAUGE_X)))

    stage_q = domain.quantities['stage']
    diag = Diagnostics(domain, tables is not None) if args.diag else None
    xmom_q = domain.quantities['xmomentum']
    times, q_in, q_sec, g_stage, volume, wet = [], [], [], [], [], []
    q_budget, q_times, v_up_prev, t_prev = [], [], None, None
    peak = np.full(len(x), -np.inf)
    t_peak = np.zeros(len(x))
    nsteps = 0
    print(f'{args.variant}: {len(x)} triangles, trench cells {ncross}, '
          f'h0 {h0:.3f} m, gpu {args.gpu}', flush=True)

    wall0 = time.time()
    for t in domain.evolve(yieldstep=YIELDSTEP, finaltime=args.final_time):
        nsteps += domain.number_of_steps
        s = stage_q.centroid_values
        qs = [sec.get_flow_through_cross_section() for sec in sections]
        vol_cells = domain.areas * (s - elev_q.centroid_values)   # bed is live under subgrid
        v_up = np.array([np.sum(vol_cells[m]) for m in upstream])
        if v_up_prev is not None:
            tt = np.linspace(t_prev, t, 61)
            qq = np.array([inflow(u) for u in tt])
            vin = np.sum(0.5 * (qq[1:] + qq[:-1]) * np.diff(tt))
            qb = (vin - (v_up - v_up_prev)) / (t - t_prev)
            q_budget.append(qb)
            q_times.append(0.5 * (t + t_prev))
        v_up_prev, t_prev = v_up, t
        times.append(t)
        q_in.append(inflow(t))
        q_sec.append(qs)
        g_stage.append(s[gauge_cells].copy())
        volume.append(domain.get_water_volume())
        h = s - z
        if tables is not None:
            wet.append(np.sum(tables.wet_area(np.maximum(s, tables.zmin))))
        else:
            wet.append(np.sum(domain.areas[h > 0.01]))
        if diag is not None:
            diag.record(t, s, elev_q.centroid_values, xmom_q.centroid_values)
        higher = s > peak
        peak[higher] = s[higher]
        t_peak[higher] = t
        if int(t) % 1800 == 0:
            qb = q_budget[-1] if q_budget else np.zeros(len(budget_x))
            print(f'  t={t / 3600:5.2f} h  Qin {q_in[-1]:5.1f}  '
                  f'Q@1750 {qb[-3]:6.1f}  Qout {qb[-1]:6.1f}  '
                  f'wall {time.time() - wall0:7.0f} s', flush=True)
    wall = time.time() - wall0

    times = np.array(times)
    q_in = np.array(q_in)
    inflow_volume = np.concatenate([[0.0], np.cumsum(0.5 * (q_in[1:] + q_in[:-1]) * np.diff(times))])
    os.makedirs('results', exist_ok=True)
    np.savez_compressed(
        f'results/{out_name}.npz',
        times=times, q_in=q_in, q_xsec=np.array(q_sec), xsec_x=np.array(GAUGE_X + [OUTLET_SECTION_X]) + SECTION_OFFSET,
        q_times=np.array(q_times), q_sections=np.array(q_budget), section_x=budget_x,
        gauge_x=np.array(GAUGE_X), gauge_stage=np.array(g_stage), volume=np.array(volume),
        inflow_volume=inflow_volume, wet_area=np.array(wet),
        peak_stage=peak, t_peak=t_peak, bed=z, centroids=np.column_stack([x, y]),
        areas=domain.areas, nodes=domain.nodes, triangles=domain.triangles,
        ntri=len(x), nsteps=nsteps, outlet=args.outlet, subgrid_n=sub_n or 0,
        min_f_dt=sub_op.min_f_dt if sub_op else np.nan, wall=wall, gpu=args.gpu, trench_cells=ncross,
        **(diag.arrays() if diag is not None else {}))
    print(f'{args.variant}: done, {nsteps} steps, wall {wall:.0f} s, '
          f'mean dt {args.final_time / nsteps:.3f} s', flush=True)


if __name__ == '__main__':
    main()
