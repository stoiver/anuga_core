"""Two-compartment subgrid storage (#508 bench): ground behind a sill fills through a weir.

The connected store is an ordinary SubgridTables built on beds raised to their
spill level s (the lowest level at which a pixel connects to the water source),
so the coarse scheme only sees water above the sills. The pond store holds the
water below the spill level:

    V_p(eta_p) = a * sum_k max(min(eta_p, s_k) - z_k, 0)

Each cell's pond sub-triangles are grouped by sill level (within `tol`), and
each group is its own pond, filling only once the connected level passes its
crest z_s (the group's lowest sill), through a broad-crested weir:

    Q = C_w w sqrt(g) sgn(eta_c - eta_p) H^(3/2) psi,   H = max(eta_c, eta_p) - z_s
    psi = (1 - (H_d / H)^(3/2))^0.385                   (Villemonte, drowned weir)

Conveyance over a drowned pond (deep=True, with PondEdges): the connected tables
and the edge subfaces are built on the raised beds, so on their own they let
water flow only in the layer above the sills. Once a pond holds water and the
connected level is above a sill, the water under it flows too: the cell
conveyance and the edge subfaces then use the depth down to the pond's own
level over the real bed (the real bed once the pond is full), ramped in over
the first tenth of that depth above the sill. Storage and mass are unchanged.

At equal levels the two stores together hold exactly the single-store volume.
Mass is exact: the operator moves Q dt between stage (connected) and V_p.
Create the Pond_exchange_operator before the Subgrid_storage_operator so it
runs first in each step.
"""
import heapq

import numpy as np

import anuga
from anuga.operators.base_operator import Operator

from subgrid_tables import EXTRA_RAMP, edge_subface_beds, subtriangle_graph


def spill_levels(z, seed):
    """Minimax level from the seed pixels over a 2-D grid (4-connected): the
    lowest water level at which each pixel connects to a seed. NaN = no data."""
    z = np.asarray(z, float)
    ny, nx = z.shape
    s = np.full(z.shape, np.inf)
    heap = []
    for r, c in zip(*np.nonzero(seed & ~np.isnan(z))):
        s[r, c] = z[r, c]
        heap.append((z[r, c], r, c))
    heapq.heapify(heap)
    while heap:
        v, r, c = heapq.heappop(heap)
        if v > s[r, c]:
            continue
        for rr, cc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
            if 0 <= rr < ny and 0 <= cc < nx and not np.isnan(z[rr, cc]):
                nv = max(v, z[rr, cc])
                if nv < s[rr, cc]:
                    s[rr, cc] = nv
                    heapq.heappush(heap, (nv, rr, cc))
    return s


class PondStore:
    """Ponds from sub-triangle beds z and spill levels s ((ncell, m), used-slot
    order, equal sub-areas). Each cell's pond sub-triangles (s > z + delta) are
    grouped by sill level: sorted sills split where they jump by more than tol.
    Each group is one pond (index g), in cell pc[g], with member mask mem[g]."""

    def __init__(self, zsub, ssub, area, n, delta=0.01, tol=0.05, cw=0.5445):   # 0.385 sqrt(2): broad-crested weir
        z = np.asarray(zsub, float)
        s = np.maximum(np.asarray(ssub, float), z)
        self.area = np.asarray(area, float)
        ncell, self.m = z.shape
        pond = s > z + delta
        pc, mem = [], []
        for c in np.nonzero(pond.any(axis=1))[0]:
            k = np.nonzero(pond[c])[0]
            k = k[np.argsort(s[c, k])]
            cut = np.nonzero(np.diff(s[c, k]) > tol)[0] + 1
            for grp in np.split(k, cut):
                row = np.zeros(self.m, bool)
                row[grp] = True
                pc.append(c)
                mem.append(row)
        self.pc = np.array(pc, int)
        self.mem = np.array(mem, bool).reshape(-1, self.m)
        self.cells = np.unique(self.pc)
        npd = len(self.pc)
        self.a = self.area[self.pc] / self.m
        self.z = np.where(self.mem, z[self.pc], np.inf)        # non-members: never wet
        self.s = np.where(self.mem, s[self.pc], -np.inf)
        self.vmax = self.a * np.where(self.mem, self.s - self.z, 0.0).sum(axis=1)
        self.zs = np.where(self.mem, self.s, np.inf).min(axis=1)   # weir crest: the group's lowest sill
        self.zlow = np.where(self.mem, self.z, np.inf).min(axis=1)
        self.shigh = np.where(self.mem, self.s, -np.inf).max(axis=1)
        # weir width: sub-edges between the group and the rest of the cell, plus the
        # group's sub-triangles on the cell's edges (water can arrive from a neighbour)
        pairs, boundary = subtriangle_graph(n)
        pairs = np.asarray(pairs)
        cnt = (self.mem[:, pairs[:, 0]] != self.mem[:, pairs[:, 1]]).sum(axis=1) + (self.mem & boundary[None, :]).sum(axis=1)
        self.w = cnt * np.sqrt(4.0 * self.area[self.pc] / np.sqrt(3.0)) / n
        self.cw = cw
        self.V = np.zeros(npd)
        self.Vzs = self.volume(self.zs, np.arange(npd))
        # padded per-cell view, for pixel levels: crest of each of a cell's ponds
        self.G = int(np.bincount(self.pc, minlength=ncell).max()) if npd else 0
        self.slot = np.zeros(npd, int)
        for g in range(npd):
            self.slot[g] = (self.pc[:g] == self.pc[g]).sum()
        self.crest = np.full((ncell, max(self.G, 1)), np.inf)
        self.crest[self.pc, self.slot] = self.zs

    def volume(self, eta, rows):
        return self.a[rows] * np.maximum(np.minimum(eta[:, None], self.s[rows]) - self.z[rows], 0.0).sum(axis=1)

    def level(self, rows):
        """Pond level for its volume (bisection; the pond's lowest bed when empty)."""
        rows = np.asarray(rows)
        V = self.V[rows]
        out = np.where(V > 0, self.shigh[rows], self.zlow[rows])     # full or empty: no search
        part = np.nonzero((V > 0) & (V < self.vmax[rows] * (1.0 - 1e-12)))[0]
        if len(part):
            r = rows[part]
            lo = self.zlow[r].copy()
            hi = self.shigh[r].copy()
            for _ in range(30):
                mid = 0.5 * (lo + hi)
                big = self.volume(mid, r) > V[part]
                hi = np.where(big, mid, hi)
                lo = np.where(big, lo, mid)
            out[part] = 0.5 * (lo + hi)
        return out

    def fill_depth(self, lev, ncell):
        """(ncell, m), used-slot order: depth of pond water over each pond
        sub-triangle for pond levels lev, min(lev, s_k) - z_k, 0 elsewhere."""
        if not hasattr(self, '_mr'):
            self._mr, self._mc = np.nonzero(self.mem)
        f = np.maximum(np.minimum(lev[self._mr], self.s[self._mr, self._mc]) - self.z[self._mr, self._mc], 0.0)
        out = np.zeros((ncell, self.m))
        out[self.pc[self._mr], self._mc] = f
        return out

    def surface(self, eta, rows):
        """Free-surface area of the pond at level eta."""
        return self.a[rows] * ((self.z[rows] < eta[:, None]) & (eta[:, None] < self.s[rows])).sum(axis=1)

    def cell_levels(self):
        """(ncell, G) level of each cell's ponds, NaN where empty or absent."""
        lev = np.full(self.crest.shape, np.nan)
        full = np.nonzero(self.V > 0)[0]
        if len(full):
            lev[self.pc[full], self.slot[full]] = self.level(full)
        return lev

    def pixel_level(self, own, spill, lev=None):
        """Water level from the ponds on pixels in cells own with spill levels spill:
        the pond of that cell whose crest is nearest the pixel's spill level, capped
        at it; -inf where that pond is empty (or the cell has none)."""
        lev = self.cell_levels() if lev is None else lev
        cr = self.crest[own]
        g = np.argmin(np.abs(np.where(np.isfinite(cr), cr, 1e9) - spill[:, None]), axis=1)
        out = np.minimum(lev[own, g], spill)
        return np.where(np.isfinite(cr[np.arange(len(own)), g]) & ~np.isnan(out), out, -np.inf)


class PondEdges:
    """Edge subface beds that drop from the raised bed to the real one where the
    ground under a subface is drowned pond.

    raised, real: bed functions (x, y) with and without the spill levels. Build
    it before the Subgrid_storage_operator, give that operator .z and .inv_n as
    edge_z and edge_invn, and pass this object to the Pond_exchange_operator,
    which calls update() every step. A subface is lowered by
        min(pond fill under it, EXTRA_RAMP x (lower of the two cell levels - raised bed))
    with the pond fill the smaller of the two sides', so both sides of an edge
    keep the same beds and the flux stays conservative."""

    def __init__(self, domain, tables, pond, nsub, raised, real, ds=0.5, manning=None, delta=0.01):
        self.domain, self.pond = domain, pond
        n = len(tables.area)
        zr, inv_n = edge_subface_beds(domain, tables, nsub, raised, ds=ds, manning=manning, sort=False) if manning is not None \
            else (edge_subface_beds(domain, tables, nsub, raised, ds=ds, sort=False), None)
        zlow = tables.zmin.copy()
        np.minimum.at(zlow, pond.pc, pond.zlow)                   # lowest real bed of each cell
        zb = np.minimum(edge_subface_beds(domain, tables, nsub, real, ds=ds, sort=False, zmin=zlow), zr)
        order = np.argsort(zr, axis=1)
        self.z = np.ascontiguousarray(np.take_along_axis(zr, order, axis=1))
        self.inv_n = None if inv_n is None else np.ascontiguousarray(np.take_along_axis(inv_n, order, axis=1))
        own = np.repeat(np.arange(n), 3)
        nb = domain.neighbours.reshape(-1)
        mate = np.where(nb >= 0, 3 * nb + domain.neighbour_edges.reshape(-1), -1)
        low = (zr - zb > delta).any(axis=1)
        low[mate[low & (mate >= 0)]] = True
        R = np.nonzero(low)[0]
        pos = np.full(3 * n, -1)
        pos[R] = np.arange(len(R))
        self.R, self.zr, self.zb = R, zr[R], zb[R]
        self.inv_u = None if inv_n is None else inv_n[R]
        self.own, self.nb = own[R], nb[R]
        self.mate = np.where(mate[R] >= 0, pos[np.maximum(mate[R], 0)], -1)
        # the pond under each subface, on its own cell's side: the cell's pond whose crest is nearest the raised bed
        gid = np.full(pond.crest.shape, -1)
        gid[pond.pc, pond.slot] = np.arange(len(pond.pc))
        cr = pond.crest[self.own]                                  # (nR, G)
        g = np.argmin(np.abs(np.where(np.isfinite(cr), cr, 1e9)[:, None, :] - self.zr[:, :, None]), axis=2)
        self.g = np.where(np.isfinite(np.take_along_axis(cr, g, axis=1)), gid[self.own[:, None], g], -1)

    def update(self, lev, eta):
        """lev: level of every pond; eta: connected level per cell, -inf where dry."""
        if len(self.R) == 0:
            return
        fill = np.where(self.g >= 0, np.maximum(np.minimum(lev[np.maximum(self.g, 0)], self.zr) - self.zb, 0.0), np.inf)
        inner = self.mate >= 0
        fill[inner] = np.minimum(fill[inner], fill[self.mate[inner]][:, ::-1])
        e = eta[self.own]
        e[inner] = np.minimum(e[inner], eta[self.nb[inner]])
        drop = np.minimum(fill, EXTRA_RAMP * np.maximum(e[:, None] - self.zr, 0.0))
        z = np.maximum(self.zr - drop, self.zb)
        order = np.argsort(z, axis=1)
        self.domain.subgrid_edge_z[self.R] = np.take_along_axis(z, order, axis=1)
        if self.inv_u is not None:
            self.domain.subgrid_edge_invn[self.R] = np.take_along_axis(self.inv_u, order, axis=1)
        self.lowered = float(drop.max()) if drop.size else 0.0


class Pond_exchange_operator(Operator):

    def __init__(self, domain, tables, pond, edges=None, deep=False, description='pond exchange', label=None):
        """deep: water under a drowned sill conveys (see the module docstring);
        edges: the PondEdges that does the same for the edge subfaces."""
        Operator.__init__(self, domain, description=description, label=label)
        self.tables, self.pond = tables, pond
        self.deep, self.edges = deep, edges
        self.stage = domain.quantities['stage']
        self.elev = domain.quantities['elevation']
        self.moved = 0.0

    def __call__(self):
        self._exchange()
        if self.deep and len(self.pond.pc):
            P, T = self.pond, self.tables
            h = np.maximum(self.stage.centroid_values - self.elev.centroid_values, 0.0)
            lev = P.level(np.arange(len(P.pc)))
            T.extra = np.take_along_axis(P.fill_depth(lev, len(h)), T.order, axis=1)
            if self.edges is not None and getattr(self.domain, 'subgrid_edge_z', None) is not None:
                self.edges.update(lev, np.where(h > 0, T.eta(h), -np.inf))

    def _exchange(self):
        P, T = self.pond, self.tables
        if len(P.pc) == 0:
            return
        dt = self.get_timestep()
        w = self.stage.centroid_values
        hall = np.maximum(w - self.elev.centroid_values, 0.0)
        etall = T.eta(hall)
        # the weir only runs where a level is above the crest
        hc, ec = hall[P.pc], etall[P.pc]
        r = np.nonzero(((hc > 0) & (ec > P.zs)) | (P.V > P.Vzs))[0]
        if len(r) == 0:
            return
        c = P.pc[r]
        h = hall[c]
        eta_c = np.where(h > 0, etall[c], -np.inf)
        eta_p = P.level(r)
        zs = P.zs[r]
        hu = np.maximum(np.maximum(eta_c, eta_p) - zs, 0.0)
        hd = np.clip(np.minimum(eta_c, eta_p) - zs, 0.0, None)
        psi = np.where(hu > 0, np.maximum(1.0 - (hd / np.where(hu > 0, hu, 1.0)) ** 1.5, 0.0) ** 0.385, 0.0)
        Q = P.cw * P.w[r] * np.sqrt(anuga.g) * np.sign(eta_c - eta_p) * hu ** 1.5 * psi
        dV = Q * dt
        # no overshoot: not past equal levels, not more than either store holds
        A = T.area[c]
        ac = np.maximum(T.wet_area(np.where(hall > 0, etall, T.zmin))[c], 1e-9)
        # filling: up to the pond volume at the connected level, and no more than the
        # connected water above the crest (the pond surface at the connected level is
        # zero once that level passes every sill of the group, so it cannot be used here)
        ec_ = np.where(np.isfinite(eta_c), eta_c, zs)
        fill = np.minimum(P.volume(np.minimum(ec_, P.shigh[r]), r) - P.V[r], ac * (ec_ - np.maximum(eta_p, zs)))
        # draining: the linearised equal-level transfer, pond surface at its own level
        ap = np.maximum(P.surface(eta_p, r), 1e-9)
        drain = np.where(np.isfinite(eta_c), (eta_c - eta_p) * ac * ap / (ac + ap), -P.V[r])
        dV = np.where(dV > 0, np.minimum(dV, np.maximum(fill, 0.0)), np.maximum(dV, np.minimum(drain, 0.0)))
        dV = np.clip(dV, -P.V[r], P.vmax[r] - P.V[r])
        # a cell's ponds together take no more than the cell holds
        take = np.bincount(c, weights=np.maximum(dV, 0.0), minlength=len(w))
        scale = np.where(take > 0, np.minimum(1.0, hall * T.area / np.where(take > 0, take, 1.0)), 1.0)
        dV = np.where(dV > 0, dV * scale[c], dV)
        P.V[r] += dV
        np.add.at(w, c, -dV / A)
        self.moved += np.abs(dV).sum()

    def parallel_safe(self):
        return False

    def statistics(self):
        return f'pond exchange: {self.moved:.0f} m3 moved'

    def timestepping_statistics(self):
        return f'ponds hold {self.pond.V.sum():.0f} m3'
