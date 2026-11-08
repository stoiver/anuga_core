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
"""
import numpy as np

import anuga
from anuga.operators.base_operator import Operator


class SubgridTables:

    def __init__(self, zsub, area, nsub=None):
        """zsub: (ncell, m) sub-triangle beds (equal areas); area: (ncell,);
        nsub: optional (ncell, m) Manning n per sub-triangle (enables conveyance)."""
        zsub = np.asarray(zsub, dtype=float)
        order = np.argsort(zsub, axis=1)
        z = np.take_along_axis(zsub, order, axis=1)
        self.z = z
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

    @classmethod
    def from_cells(cls, cells, n, px, py, pz, manning=None):
        """manning: optional function (x, y) -> n, sampled at the sub-centroids."""
        zsub, _, _, sx, sy = cells.subgrid(n, px, py, pz, return_centroids=True)
        used = ~np.isnan(zsub[0])
        nsub = manning(sx[:, used], sy[:, used]) if manning is not None else None
        return cls(zsub[:, used], cells.area, nsub)

    # --- level -> storage ---------------------------------------------------
    def depth(self, eta):
        """Mean depth h = V(eta) / A for a level eta per cell."""
        eta = np.asarray(eta, dtype=float)
        return np.maximum(eta[:, None] - self.z, 0.0).sum(axis=1) / self.m

    def volume(self, eta):
        return self.depth(eta) * self.area

    def wet_area(self, eta):
        eta = np.asarray(eta, dtype=float)
        return (self.z < eta[:, None]).sum(axis=1) * self.area / self.m

    # --- storage -> level ---------------------------------------------------
    def eta(self, h):
        """Level eta for mean depth h (inverse of depth); h <= 0 gives zmin."""
        h = np.maximum(np.asarray(h, dtype=float), 0.0)
        target = self.m * h
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
        hk = np.maximum(eta[:, None] - self.z, 0.0)
        return (hk ** (5.0 / 3.0) * self.inv_n).sum(axis=1) / self.m

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
                 description='subgrid storage', label=None):
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
        if conveyance_friction:
            if tables.inv_n is None:
                raise ValueError('conveyance_friction needs tables built with manning=')
            self.friction = domain.quantities['friction']
            self.n_base = self.friction.centroid_values.copy()   # used where Kc = 0

    def set_initial_level(self, eta):
        eta = np.maximum(np.asarray(eta, dtype=float), self.tables.zmin)
        h = self.tables.depth(eta)
        d = eta - h
        self.domain.set_quantity('elevation', d, location='centroids')
        self.domain.set_quantity('stage', eta, location='centroids')
        self._update(eta, h)

    def __call__(self):
        w = self.stage.centroid_values
        d = self.elev.centroid_values
        h = np.maximum(w - d, 0.0)
        eta = self.tables.eta(h)
        d[:] = eta - h
        w[:] = eta
        self._update(eta, h)

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
