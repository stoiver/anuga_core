"""Phase 1 prototype: per-cell subgrid storage tables and the volume-conserving update.

Each coarse triangle is split uniformly into m = n^2 equal-area sub-triangles,
each holding the mean of its DTM pixels (see static_storage.Cells.subgrid).
With a level water surface eta in the cell:

    V(eta) = a * sum_k max(eta - z_k, 0)        a = A / m
    A(eta) = a * #{k : z_k < eta}

Storing the sub-beds sorted, with their prefix sums, makes V exact and its
inverse eta(h) (for mean depth h = V / A) a search plus one division.

Conveyance (Phase 2 start): with a per-sub-triangle Manning n_k, the cell's
conveyance per unit width, assuming every sub-triangle flows under the same
energy slope, is

    Kc(eta) = (1/m) * sum_k max(eta - z_k, 0)^(5/3) / n_k

Then q = Kc * sqrt(S_f) and the friction force per area is g h S_f, so the
friction term is -g h |q| q / Kc^2. ANUGA's Manning term is -g n^2 |q| q / h^(7/3);
an effective n_eff = h^(5/3) / Kc gives exactly the conveyance form, so the
operator can apply it through the existing friction kernel. For a level bed and
uniform n, Kc = h^(5/3)/n and n_eff = n. Kc is evaluated directly here (O(m) per
cell); a production version would tabulate it.

The coarse scheme keeps conserving the mean depth h. After each step,

    h = w - d,   eta = eta(h),   d = eta - h,   w = eta

so the cell's "effective bed" d and stage w change but w - d (the volume) does
not. A dry cell (h = 0) gets d = min z_k: the cell's lowest point, so a channel
crossing it sets its bed.

Timestep: a partly wet cell has a free surface of only f*A (f = A(eta)/A), so
a volume change moves its level by 1/f times what the coarse scheme assumes.
The explicit update then needs dt <= CFL * f * r / (|u| + sqrt(g h)); without
it, lake at rest blows up from round-off in cells with small f (measured:
x75 growth per 120 s, saturating at ~1 mm stage oscillation). The operator
caps domain.evolve_max_timestep with this limit each step.

cap_min_depth / cap_min_fraction leave cells with mean depth h or wet fraction
f below the threshold out of that cap (they still get the ordinary flux CFL).
Such cells -- typically a coarse cell whose corner clips the channel, with one
wet sub-triangle -- hold little water but have the smallest f, so they set the
cap on almost every step. Both default to 0: every wet cell is capped.

Hollows (optional, sill=): a sub-triangle inside a closed depression of its cell
(no path to the cell's edges through lower ground) holds water and conveys only
once the cell level reaches its sill s_k, the lowest level at which it connects
to an edge (minimax over paths in the sub-triangle mesh; s_k = z_k when it is
connected). Then

    V(eta) = a * sum_k [eta >= s_k] (eta - z_k)

jumps by the hollow's volume at its sill; the inverse holds eta at the sill
while the hollow fills. No hysteresis: a hollow left above its sill's level
drains with the cell. The dry bed (zmin) becomes the lowest connected level.
"""
import os

import numpy as np

import anuga

# conveyance averaging over the wet sub-triangles: parallel (default), geometric, series
KMEAN = os.environ.get('SUBGRID_KMEAN', 'parallel')
from anuga.operators.base_operator import Operator


def subtriangle_graph(n, edge=None):
    """Neighbour pairs and edge-touching flags for the n^2 uniform split, in the
    used-slot order of static_storage.Cells.subgrid (slot = 2*(i*n + j) + down).

    edge: None flags sub-triangles on any cell edge; 0, 1 or 2 only those on
    edge e of the cell (opposite vertex e, as domain.neighbours and
    edge_subface_beds number them). Index i runs along vertex 0 -> 1 and j along
    0 -> 2, so i == 0 is edge 1, j == 0 is edge 2 and i + j == n - 1 is edge 0."""
    slots = {}
    for i in range(n):
        for j in range(n):
            for down in (0, 1):
                if i + j <= n - 1 - down:
                    slots[(i, j, down)] = 2 * (i * n + j) + down
    order = sorted(slots.values())
    col = {sl: c for c, sl in enumerate(order)}
    idx = {key: col[sl] for key, sl in slots.items()}
    pairs = []
    for (i, j, down), c in idx.items():
        if down:      # a down triangle borders three up triangles
            for key in ((i, j, 0), (i + 1, j, 0), (i, j + 1, 0)):
                pairs.append((c, idx[key]))
    boundary = np.zeros(len(idx), bool)
    for (i, j, down), c in idx.items():
        on = (i + j == n - 1, i == 0, j == 0)
        if not down and (any(on) if edge is None else on[edge]):
            boundary[c] = True
    return np.array(pairs), boundary


def hollow_sills(zsub, n, edge=None):
    """Sill level per sub-triangle (same layout as zsub, (ncell, n^2)): the lowest
    level at which it connects to the cell's edges (minimax path height).
    edge: only to that edge of the cell (see subtriangle_graph)."""
    pairs, boundary = subtriangle_graph(n, edge)
    m = len(boundary)
    nbr = [[k] for k in range(m)]                  # pad with self
    for a, b in pairs:
        nbr[a].append(b)
        nbr[b].append(a)
    nbr = np.array([v + [v[0]] * (4 - len(v)) for v in nbr])
    z = np.asarray(zsub, dtype=float)
    s = np.where(boundary[None, :], z, np.inf)
    for _ in range(m + 1):
        new = np.maximum(z, s[:, nbr].min(axis=2))
        if np.array_equal(new, s):
            break
        s = new
    return s


class SubgridTables:

    def __init__(self, zsub, area, nsub=None, sill=None):
        """zsub: (ncell, m) sub-triangle beds (equal areas); area: (ncell,);
        nsub: optional (ncell, m) Manning n per sub-triangle (enables conveyance);
        sill: optional (ncell, m) hollow sill levels (see hollow_sills)."""
        zsub = np.asarray(zsub, dtype=float)
        order = np.argsort(zsub, axis=1)
        z = np.take_along_axis(zsub, order, axis=1)
        self.order = order
        self.z = z
        self.zslot = None
        self.inv_n = None
        if nsub is not None:
            self.inv_n = 1.0 / np.take_along_axis(np.asarray(nsub, dtype=float), order, axis=1)
        self.area = np.asarray(area, dtype=float)
        self.m = z.shape[1]
        self.S = np.cumsum(z, axis=1)                 # prefix sums
        j1 = np.arange(1, self.m + 1)
        # m * (mean depth) at eta = z_j: sum_{k<=j} (z_j - z_k)
        self.Vb = j1[None, :] * z - self.S
        self.zmin = z[:, 0]
        self.zmean = self.S[:, -1] / self.m
        self.sill = None
        if sill is not None:
            sill = np.maximum(np.asarray(sill, dtype=float), zsub)
            self.set_sills(np.take_along_axis(sill, order, axis=1))
            self.zmin = self.s_sorted[:, 0]                       # the dry bed: lowest connected level
            self.n_hollow = (self.sill > self.z).sum(axis=1)

    def set_sills(self, sill, rows=None):
        """Sill per sub-triangle, aligned with self.z ((ncell, m), or (len(rows), m)
        for just those cells). Builds the inverse tables over the sills: sort by
        sill, prefix sums of z in that order."""
        if rows is None:
            self.sill = np.array(sill, dtype=float)
            self.s_sorted = np.empty_like(self.sill)
            self.Zp = np.empty_like(self.sill)
            self.Vm = np.empty_like(self.sill)
            rows = slice(None)
        else:
            self.sill[rows] = sill
        sl = self.sill[rows]
        o2 = np.argsort(sl, axis=1)
        self.s_sorted[rows] = np.take_along_axis(sl, o2, axis=1)
        Zp = np.cumsum(np.take_along_axis(self.z[rows], o2, axis=1), axis=1)
        self.Zp[rows] = Zp
        Zprev = np.concatenate([np.zeros((len(Zp), 1)), Zp[:, :-1]], axis=1)
        j1 = np.arange(1, self.m + 1)
        self.Vm[rows] = (j1[None, :] - 1) * self.s_sorted[rows] - Zprev   # m * mean depth just below each sill

    def add_connectivity(self, zsub, n, domain):
        """Data for the operator's connectivity rule: the sub-triangle graph, each
        cell edge's boundary sub-triangles in order along the edge, and the
        matching sub-triangle across each edge in the neighbour. Sets the
        all-edge hollow sills as the starting (fallback) state."""
        zsub = np.asarray(zsub, dtype=float)
        pairs, boundary = subtriangle_graph(n)
        m = len(boundary)
        nbr = [[k] for k in range(m)]
        for a, b in pairs:
            nbr[a].append(b)
            nbr[b].append(a)
        self.sub_nbr = np.array([v + [v[0]] * (4 - len(v)) for v in nbr])
        self.zslot = zsub                                  # used-slot order
        self.inv_order = np.argsort(self.order, axis=1)
        # boundary up-triangles of each local edge, ordered from vertex A_e to B_e
        # (edge 0: v1 -> v2, edge 1: v0 -> v2, edge 2: v0 -> v1)
        col = {}
        c = 0
        for i in range(n):
            for j in range(n):
                for down in (0, 1):
                    if i + j <= n - 1 - down:
                        col[(i, j, down)] = c
                        c += 1
        bsub = np.array([[col[(n - 1 - r, r, 0)] for r in range(n)],     # edge 0, j = r from v1
                         [col[(0, r, 0)] for r in range(n)],             # edge 1, j = r from v0
                         [col[(r, 0, 0)] for r in range(n)]])            # edge 2, i = r from v0
        ends = np.array([[1, 2], [0, 2], [0, 1]])
        tri, nb = domain.triangles, domain.neighbours
        ncell = len(tri)
        self.bsub = bsub
        self.across = np.full((ncell, 3, n), -1)          # neighbour's sub-triangle (slot col)
        for k in range(ncell):
            for e in range(3):
                q = nb[k, e]
                if q < 0:
                    continue
                e2 = int(np.nonzero(nb[q] == k)[0][0])
                same = tri[k, ends[e, 0]] == tri[q, ends[e2, 0]]
                self.across[k, e] = bsub[e2] if same else bsub[e2][::-1]
        self.edge_boundary = boundary
        s0 = np.maximum(hollow_sills(zsub, n), zsub)
        self.set_sills(np.take_along_axis(s0, self.order, axis=1))
        self.zmin = self.s_sorted[:, 0]
        self.n_hollow = (self.sill > self.z).sum(axis=1)

    def minimax_from(self, seed, rows):
        """Sills (slot order) for cells rows: the lowest level at which each
        sub-triangle connects to the seed sub-triangles (bool, (len(rows), m))."""
        z = self.zslot[rows]
        s = np.where(seed, z, np.inf)
        for _ in range(z.shape[1] + 1):
            new = np.maximum(z, s[:, self.sub_nbr].min(axis=2))
            if np.array_equal(new, s):
                break
            s = new
        return s

    @classmethod
    def from_cells(cls, cells, n, px, py, pz, manning=None, hollows=False):
        """manning: optional function (x, y) -> n, sampled at the sub-centroids;
        hollows: fill closed depressions only from their sills."""
        zsub, _, _, sx, sy = cells.subgrid(n, px, py, pz, return_centroids=True)
        used = ~np.isnan(zsub[0])
        nsub = manning(sx[:, used], sy[:, used]) if manning is not None else None
        sill = hollow_sills(zsub[:, used], n) if hollows else None
        return cls(zsub[:, used], cells.area, nsub, sill)

    def _active_depth(self, eta):
        """Per sub-triangle depth eta - z_k where it holds water (hollows: only above the sill)."""
        hk = np.maximum(eta[:, None] - self.z, 0.0)
        if self.sill is not None:
            hk = np.where(eta[:, None] >= self.sill, hk, 0.0)
        return hk

    # --- level -> storage ---------------------------------------------------
    def depth(self, eta):
        """Mean depth h = V(eta) / A for a level eta per cell."""
        eta = np.asarray(eta, dtype=float)
        return self._active_depth(eta).sum(axis=1) / self.m

    def volume(self, eta):
        return self.depth(eta) * self.area

    def wet_area(self, eta):
        eta = np.asarray(eta, dtype=float)
        return (self._active_depth(eta) > 0.0).sum(axis=1) * self.area / self.m

    # --- storage -> level ---------------------------------------------------
    def eta(self, h):
        """Level eta for mean depth h (inverse of depth); h <= 0 gives zmin."""
        h = np.maximum(np.asarray(h, dtype=float), 0.0)
        target = self.m * h
        if self.sill is not None:
            # j = last sill with V just below it <= target; eta is that sill while
            # its hollow fills (flat), then rises linearly over the active set
            j = (self.Vm <= target[:, None]).sum(axis=1) - 1
            rows = np.arange(len(h))
            return np.maximum(self.s_sorted[rows, j], (target + self.Zp[rows, j]) / (j + 1))
        # j = last breakpoint with Vb_j <= target (Vb_0 = 0, so j >= 0)
        j = (self.Vb <= target[:, None]).sum(axis=1) - 1
        rows = np.arange(len(h))
        return (target + self.S[rows, j]) / (j + 1)

    def effective_bed(self, h):
        return self.eta(h) - h

    # --- conveyance ---------------------------------------------------------
    def conveyance(self, eta):
        """Kc(eta): conveyance per unit width, (1/m) sum_k max(eta - z_k, 0)^(5/3) / n_k."""
        eta = np.asarray(eta, dtype=float)
        hk = self._active_depth(eta)
        k = hk ** (5.0 / 3.0) * self.inv_n
        mode = KMEAN
        if mode == 'parallel':
            return k.sum(axis=1) / self.m
        # experimental (#508 flat-plain bench): average the wet sub-triangles'
        # conveyance as if partly in series, times the wet fraction
        wet = hk > 1e-6
        nw = wet.sum(axis=1)
        f = nw / self.m
        with np.errstate(divide='ignore', invalid='ignore'):
            if mode == 'geometric':
                km = np.exp(np.where(wet, np.log(np.where(wet, k, 1.0)), 0.0).sum(axis=1) / np.maximum(nw, 1))
            elif mode == 'series':
                km = (np.where(wet, np.where(wet, k, 1.0) ** -2.0, 0.0).sum(axis=1) / np.maximum(nw, 1)) ** -0.5
            else:
                raise ValueError(mode)
        return np.where(nw > 0, f * km, 0.0)

    def add_directional(self, sx, sy, ndir=6, nstrip=None, cells=None):
        """Data for direction-dependent conveyance: sx, sy (ncell, m) sub-centroids in
        used-slot order. For ndir flow directions over [0, pi), each cell's
        sub-triangles are binned into nstrip (default sqrt(m)) equal strips across
        the flow. cells: bool mask or indices of the cells that use it (default
        all); the others keep the side-by-side conveyance."""
        allc = np.arange(len(self.z))
        self.dcells = allc if cells is None else allc[cells]
        sx = np.take_along_axis(np.asarray(sx, float)[self.dcells], self.order[self.dcells], axis=1)
        sy = np.take_along_axis(np.asarray(sy, float)[self.dcells], self.order[self.dcells], axis=1)
        ns = nstrip or int(round(np.sqrt(self.m)))
        ncell = len(sx)
        self.ndir, self.nstrip = ndir, ns
        self.dkey = np.empty((ndir, ncell, self.m), np.int64)
        self.dcount = np.empty((ndir, ncell, ns))
        for d in range(ndir):
            th = np.pi * d / ndir
            p = sx * np.cos(th) + sy * np.sin(th)
            lo, hi = p.min(axis=1, keepdims=True), p.max(axis=1, keepdims=True)
            b = np.clip(((p - lo) / np.maximum(hi - lo, 1e-12) * ns).astype(np.int64), 0, ns - 1)
            self.dkey[d] = np.arange(ncell)[:, None] * ns + b
            self.dcount[d] = np.bincount(self.dkey[d].ravel(), minlength=ncell * ns).reshape(ncell, ns)

    def directional_conveyance(self, eta, theta, floor=0.1):
        """Conveyance for flow in direction theta (per cell): within a strip across
        the flow the sub-triangles convey side by side (mean of k), strips act in
        series (resistance 1/K^2 averaged over the strips), interpolated linearly
        in resistance between the ndir directions. Not below floor x the
        side-by-side conveyance (a dry strip would otherwise stop all flow)."""
        c = self.dcells
        hk = np.maximum(np.asarray(eta, dtype=float)[c, None] - self.z[c], 0.0)
        if self.sill is not None:
            hk = np.where(eta[c, None] >= self.sill[c], hk, 0.0)
        k = hk ** (5.0 / 3.0) * self.inv_n[c]
        theta = np.asarray(theta)[c]
        ncell, ns = len(k), self.nstrip
        kpar = k.sum(axis=1) / self.m
        wet = kpar > 0
        kp = np.where(wet, kpar, 1.0)
        # resistance relative to side by side, (kpar / K_strip)^2, in [~1, 1/floor^2]
        R = np.empty((self.ndir, ncell))
        for d in range(self.ndir):
            cnt = self.dcount[d]
            ks = np.bincount(self.dkey[d].ravel(), weights=k.ravel(), minlength=ncell * ns).reshape(ncell, ns)
            rel = ks / np.maximum(cnt, 1) / kp[:, None]
            r = np.where(cnt > 0, 1.0 / np.maximum(rel, floor) ** 2, 0.0)
            R[d] = r.sum(axis=1) / (cnt > 0).sum(axis=1)
        x = np.mod(np.nan_to_num(theta), np.pi) / np.pi * self.ndir
        i0 = np.floor(x).astype(int) % self.ndir
        w = x - np.floor(x)
        rows = np.arange(ncell)
        Rt = (1 - w) * R[i0, rows] + w * R[(i0 + 1) % self.ndir, rows]
        return np.where(wet, kpar * np.minimum(Rt ** -0.5, 1.0), 0.0)   # for the cells dcells

    def effective_manning(self, eta, h, n_dry, kc=None):
        """n_eff = h^(5/3) / Kc(eta), so ANUGA's Manning term becomes -g h |q| q / Kc^2.
        Cells with no conveyance keep n_dry. kc: Kc(eta) if already computed."""
        if kc is None:
            kc = self.conveyance(eta)
        ok = (kc > 0.0) & (h > 0.0)
        out = np.array(n_dry, dtype=float, copy=True)
        out[ok] = h[ok] ** (5.0 / 3.0) / kc[ok]
        return out


def edge_subface_beds(domain, tables, nsub, bed, ds=0.1, manning=None):
    """Subface beds for the subgrid edge flux: (3n, nsub), row 3k+i is edge i of
    triangle k (opposite vertex i), split into nsub equal subfaces, each the mean
    of bed(x, y) sampled every <= ds along it.

    Each subface is clipped from below at the higher of the two adjacent cells'
    lowest sub-triangle (zmin): a cell at zmin holds no water, and the clip makes
    every subface depth max(eta - z_j, 0) zero there too, so the flux cannot
    drain an empty cell. Rows are sorted (the flux does not depend on order).

    manning: optional function (x, y) -> n. Then returns (z, inv_n), inv_n the
    mean of 1/n along each subface, in the same order as z, for the
    conveyance-weighted subface velocity.
    """
    nodes, tris = domain.nodes, domain.triangles
    n = len(tris)
    L = domain.edgelengths.reshape(-1)
    per = max(int(np.ceil(L.max() / nsub / ds)), 1)
    m = nsub * per
    s = (np.arange(m) + 0.5) / m
    a = nodes[tris[:, [1, 2, 0]]].reshape(-1, 2)          # edge i: vertex i+1 -> i+2
    b = nodes[tris[:, [2, 0, 1]]].reshape(-1, 2)
    z = np.empty((3 * n, nsub))
    inv_n = np.empty((3 * n, nsub)) if manning is not None else None
    chunk = 20000
    for c0 in range(0, 3 * n, chunk):
        aa, bb = a[c0:c0 + chunk], b[c0:c0 + chunk]
        x = aa[:, None, 0] + s[None, :] * (bb - aa)[:, None, 0]
        y = aa[:, None, 1] + s[None, :] * (bb - aa)[:, None, 1]
        z[c0:c0 + chunk] = bed(x, y).reshape(len(aa), nsub, per).mean(axis=2)
        if manning is not None:
            inv_n[c0:c0 + chunk] = (1.0 / manning(x, y)).reshape(len(aa), nsub, per).mean(axis=2)
    own = np.repeat(np.arange(n), 3)
    nb = domain.neighbours.reshape(-1)
    floor = tables.zmin[own].copy()
    inner = nb >= 0
    floor[inner] = np.maximum(floor[inner], tables.zmin[nb[inner]])
    z = np.maximum(z, floor[:, None])
    order = np.argsort(z, axis=1)
    z = np.ascontiguousarray(np.take_along_axis(z, order, axis=1))
    if manning is None:
        return z
    return z, np.ascontiguousarray(np.take_along_axis(inv_n, order, axis=1))


class Subgrid_storage_operator(Operator):
    """Applies the volume-conserving subgrid update after every step (mode 1).

    Call set_initial_level(eta) once instead of setting stage/elevation by hand.
    """

    def __init__(self, domain, tables, limit_timestep=True, cfl=1.0,
                 cap_min_depth=0.0, cap_min_fraction=0.0, conveyance_friction=False,
                 edge_z=None, cap_exponent=1.0, drain_limiter=False, edge_invn=None,
                 connectivity=False, directional=False, description='subgrid storage', label=None):
        Operator.__init__(self, domain, description=description, label=label)
        self.tables = tables
        self.stage = domain.quantities['stage']
        self.elev = domain.quantities['elevation']
        self.xmom = domain.quantities['xmomentum']
        self.ymom = domain.quantities['ymomentum']
        self.limit_timestep = limit_timestep
        self.cfl = cfl
        self.cap_min_depth = cap_min_depth
        self.cap_min_fraction = cap_min_fraction
        self.max_timestep = domain.evolve_max_timestep
        self.radii = domain.radii
        self.min_f_dt = np.inf        # smallest limit applied (for reporting)
        self.conveyance_friction = conveyance_friction
        self.cap_exponent = cap_exponent     # cap = cfl * f**p * r / c (p = 1 by design)
        self.edge_flux = edge_z is not None
        if self.edge_flux:
            # switches Domain.compute_fluxes to the subgrid edge flux kernel
            domain.subgrid_edge_z = np.ascontiguousarray(edge_z, dtype=float)
            # draining-time limiter keeps depths >= 0 for any step, so the cap
            # only has to keep the scheme stable: pass cap_exponent=0 with it
            domain.subgrid_drain_limiter = bool(drain_limiter)
        self.conveyance_split = edge_invn is not None
        if self.conveyance_split:
            # conveyance-weighted subface velocity in the edge flux: the kernel
            # reads subgrid_cell_neff, refreshed in place after every step
            if not self.edge_flux or tables.inv_n is None:
                raise ValueError('edge_invn needs edge_z and tables built with manning=')
            domain.subgrid_edge_invn = np.ascontiguousarray(edge_invn, dtype=float)
            domain.subgrid_cell_neff = np.zeros(len(tables.area))
            self.n_split_dry = domain.quantities['friction'].centroid_values.copy()
        self.directional = directional
        if directional and getattr(tables, 'dkey', None) is None:
            raise ValueError('directional needs tables.add_directional')
        self.connectivity = connectivity
        if connectivity:
            # a sub-triangle holds water once reached: the cell level has passed
            # its minimax level from the seeds -- sub-triangles already reached,
            # and boundary sub-triangles across from a reached sub-triangle of a
            # neighbour whose level is above them. A wet cell with no seeds (fed
            # by an inlet) starts from all its edges. Reached stays until the
            # cell dries. Call tables.add_connectivity(zsub, n, domain) first.
            if tables.zslot is None:
                raise ValueError('connectivity needs tables.add_connectivity')
            self.reached = np.zeros(tables.zslot.shape, bool)     # slot order
            self.seed = np.zeros(tables.zslot.shape, bool)
            self.use = np.broadcast_to(tables.edge_boundary, tables.zslot.shape).copy()
            self.nbr = domain.neighbours
            self.eta_prev = tables.zmin.copy()
            self.h_prev = np.zeros(len(tables.area))
            self.reach_events = 0
        if conveyance_friction:
            if tables.inv_n is None:
                raise ValueError('conveyance_friction needs tables built with manning=')
            self.friction = domain.quantities['friction']
            self.n_base = self.friction.centroid_values.copy()   # used where Kc = 0

    def enable_p1_level(self, beta=None, min_sub=0.02, h_dry=1.0e-3):
        """Experimental (#508): a sloping level in each cell for the edge fluxes.

        ANUGA's edge extrapolation drops the gradient where the mean depths of a
        cell and its neighbours differ by more than about 10:1, which it reads as
        a wet/dry front. Between subgrid cells that ratio says nothing: a channel
        cell and a bank cell share one level. So after every extrapolation this
        rebuilds the edge levels from the neighbours the cell is connected to:

          connected   both cells wet and min(eta_k, eta_j) more than min_sub above
                      the lowest subface bed of their shared edge
          gradient    least squares over the connected neighbours (a plane for
                      three, along the line of centroids for one, none for zero)
          limiter     ANUGA's: the edge levels stay within beta times the range
                      of the connected neighbours' levels

        The edge mean depth and momentum are set to the cell's own (a P1 level
        with a P0 discharge): mean depth and mean velocity jump between subgrid
        cells for geometric reasons, so their gradients mean nothing. Needs the
        edge flux kernel (edge_z)."""
        if not self.edge_flux:
            raise ValueError('p1 level needs edge_z')
        d = self.domain
        n = len(self.tables.area)
        nb = d.neighbours
        self.p1_beta = d.beta_w if beta is None else beta
        self.p1_min_sub, self.p1_h_dry = min_sub, h_dry
        self.p1_nb = np.where(nb >= 0, nb, np.arange(n)[:, None])
        self.p1_interior = nb >= 0
        c = d.centroid_coordinates
        self.p1_dxn = c[self.p1_nb] - c[:, None, :]
        self.p1_dxe = d.get_edge_midpoint_coordinates().reshape(n, 3, 2) - c[:, None, :]
        self.p1_ezmin = d.subgrid_edge_z.reshape(n, 3, -1).min(axis=2)
        self.p1_height = d.quantities['height']
        self.p1_cells = 0          # cell-steps with a nonzero gradient
        self.p1_calls = 0
        orig = d.distribute_to_vertices_and_edges

        def distribute(*args, **kwargs):
            orig(*args, **kwargs)
            self._p1_edges()
        d.distribute_to_vertices_and_edges = distribute

    def _p1_edges(self):
        w = self.stage.centroid_values
        h = np.maximum(w - self.elev.centroid_values, 0.0)
        wn = w[self.p1_nb]
        wet = h > self.p1_h_dry
        conn = (self.p1_interior & wet[:, None] & wet[self.p1_nb]
                & (np.minimum(w[:, None], wn) - self.p1_ezmin > self.p1_min_sub))
        dw = np.where(conn, wn - w[:, None], 0.0)
        dx = self.p1_dxn * conn[:, :, None]
        m11 = (dx[:, :, 0] ** 2).sum(axis=1)
        m22 = (dx[:, :, 1] ** 2).sum(axis=1)
        m12 = (dx[:, :, 0] * dx[:, :, 1]).sum(axis=1)
        b1 = (dx[:, :, 0] * dw).sum(axis=1)
        b2 = (dx[:, :, 1] * dw).sum(axis=1)
        det = m11 * m22 - m12 ** 2
        tr = m11 + m22
        full = det > 1.0e-6 * tr ** 2
        sd = np.where(full, det, 1.0)
        st = np.where(tr > 0.0, tr, 1.0)
        gx = np.where(full, (m22 * b1 - m12 * b2) / sd, b1 / st)   # rank 1: along the neighbours
        gy = np.where(full, (m11 * b2 - m12 * b1) / sd, b2 / st)
        dq = self.p1_dxe[:, :, 0] * gx[:, None] + self.p1_dxe[:, :, 1] * gy[:, None]
        qmax = self.p1_beta * np.maximum(dw.max(axis=1), 0.0)[:, None]
        qmin = self.p1_beta * np.minimum(dw.min(axis=1), 0.0)[:, None]
        with np.errstate(divide='ignore', invalid='ignore'):
            r = np.where(dq > qmax, qmax / dq, np.where(dq < qmin, qmin / dq, 1.0))
        phi = r.min(axis=1)
        self.stage.edge_values[:] = w[:, None] + phi[:, None] * dq
        self.p1_height.edge_values[:] = h[:, None]
        self.xmom.edge_values[:] = self.xmom.centroid_values[:, None]
        self.ymom.edge_values[:] = self.ymom.centroid_values[:, None]
        self.p1_cells += int((np.abs(phi[:, None] * dq).max(axis=1) > 1.0e-12).sum())
        self.p1_calls += 1

    def set_initial_level(self, eta):
        eta = np.maximum(np.asarray(eta, dtype=float), self.tables.zmin)
        # at the dry bed the cell is empty (a hollow whose sill is the dry bed counts
        # as full at eta = sill, so do not take depth() there)
        h = np.where(eta > self.tables.zmin, self.tables.depth(eta), 0.0)
        if self.connectivity:
            sl = np.take_along_axis(self.tables.sill, self.tables.inv_order, axis=1)
            self.reached[:] = (h > 0)[:, None] & (eta[:, None] >= sl)
            self.seed[:] = self.reached
            self.eta_prev, self.h_prev = eta.copy(), h.copy()
        d = eta - h
        self.domain.set_quantity('elevation', d, location='centroids')
        self.domain.set_quantity('stage', eta, location='centroids')
        self._update(eta, h)

    def __call__(self):
        w = self.stage.centroid_values
        d = self.elev.centroid_values
        h = np.maximum(w - d, 0.0)
        if self.connectivity:
            self._connect(h)
        eta = self.tables.eta(h)
        if self.connectivity:
            sl = np.take_along_axis(self.tables.sill, self.tables.inv_order, axis=1)
            new = (h > 0)[:, None] & (eta[:, None] >= sl) & ~self.reached
            self.reach_events += int(new.any(axis=1).sum())
            self.reached |= new
            dry = h <= 0
            self.reached[dry] = False
            self.seed[dry] = False
            self.eta_prev, self.h_prev = eta.copy(), h.copy()
        d[:] = eta - h
        w[:] = eta
        self._update(eta, h)

    def _connect(self, h):
        """Grow each wet cell's seeds and re-derive its sills where they changed."""
        T = self.tables
        has = T.across >= 0                                                 # (ncell, 3, n)
        q = np.where(has, self.nbr[:, :, None], 0)
        qs = np.where(has, T.across, 0)
        zown = T.zslot[:, T.bsub]                                           # (ncell, 3, n)
        entry = has & self.reached[q, qs] & (self.h_prev[q] > 0) & (self.eta_prev[q] > zown)
        seed = self.seed | self.reached
        for e in range(3):
            seed[:, T.bsub[e]] |= entry[:, e]
        seed[h <= 0] = False
        self.seed = seed
        use = seed.copy()
        use[~use.any(axis=1)] = T.edge_boundary          # no seed (inlet-fed or dry): any edge
        rows = np.nonzero((use != self.use).any(axis=1))[0]
        self.use = use
        if len(rows):
            ss = T.minimax_from(use[rows], rows)
            T.set_sills(np.take_along_axis(ss, T.order[rows], axis=1), rows)

    def _update(self, eta, h):
        kc = None
        if self.conveyance_split:
            kc = self.tables.conveyance(eta)
            self.domain.subgrid_cell_neff[:] = self.tables.effective_manning(
                eta, h, self.n_split_dry, kc)
        if self.conveyance_friction:
            self._set_friction(eta, h, kc)
        if self.limit_timestep:
            self._limit_timestep(eta, h)

    def _set_friction(self, eta, h, kc=None):
        # in place: the C friction kernel holds a pointer to this array
        if self.directional:
            # friction from the conveyance along the cell's flow direction
            th = np.arctan2(self.ymom.centroid_values, self.xmom.centroid_values)
            kc = self.tables.conveyance(eta) if kc is None else np.array(kc, dtype=float)
            kc[self.tables.dcells] = self.tables.directional_conveyance(eta, th)
        self.friction.centroid_values[:] = self.tables.effective_manning(eta, h, self.n_base, kc)

    def _limit_timestep(self, eta, h):
        f_all = self.tables.wet_area(eta) / self.tables.area
        wet = (h > max(1.0e-6, self.cap_min_depth)) & (f_all >= self.cap_min_fraction)
        if not np.any(wet):
            self.domain.evolve_max_timestep = self.max_timestep
            return
        hw = h[wet]
        f = f_all[wet]
        speed = np.hypot(self.xmom.centroid_values[wet], self.ymom.centroid_values[wet]) / hw
        c = speed + np.sqrt(anuga.g * hw)
        if self.edge_flux and getattr(self.domain, 'max_speed', None) is not None:
            # the edge kernel's wave speed comes from the deepest wet subface, and
            # a cell drains through the true flow area: f * r / s keeps it positive
            # (max_speed is from the step just taken; None before the first flux)
            c = np.maximum(c, self.domain.max_speed[wet])
        limit = self.cfl * np.min(f ** self.cap_exponent * self.radii[wet] / c)
        self.domain.evolve_max_timestep = min(self.max_timestep, limit)
        self.min_f_dt = min(self.min_f_dt, limit)

    def statistics(self):
        return 'Subgrid_storage_operator'

    def timestepping_statistics(self):
        return 'Subgrid_storage_operator'
