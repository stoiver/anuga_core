"""Travel time inside a subgrid cell: take the stream's fall inside the cell out of the cell's beds.

A subgrid cell holds its water as a level pool. Where a stream falls several metres on its way across the cell,
the pool wets only the stream's low end, so the cell stores a small part of the water the flowing stream holds
and passes a wave on without the time the stream takes to cross it. Here the DEM is drained (priority flood from
the outfall), streams are the pixels draining more than a threshold area, and every pixel is lowered by the fall
of its stream between the point where the pixel's water joins it and the point where the stream leaves the cell.
A stream then lies level along its length in the cell, at its exit's elevation, with its banks and slopes at their
own height above it; the level pool over these beds has the depth at the exit along the whole stream. Edges keep
their real beds, so a cell still delivers to the next over its real lip.
"""
import heapq
import numpy as np


def drainage(z, seeds):
    """Priority flood of z (nan = outside) from the seed pixels. Returns the filled surface, each pixel's
    downstream pixel (flat index, -1 at a seed or outside) and the pixels in the order they were drained."""
    ny, nx = z.shape
    zf = np.full(z.size, np.nan)
    par = np.full(z.size, -1, np.int64)
    zr = z.ravel()
    ok = np.isfinite(zr)
    done = ~ok
    heap = [(float(zr[i]), int(i)) for i in np.flatnonzero(seeds.ravel() & ok)]
    for _, i in heap:
        done[i] = True
        zf[i] = zr[i]
    heapq.heapify(heap)
    order = []
    pop, push = heapq.heappop, heapq.heappush
    while heap:
        lev, i = pop(heap)
        order.append(i)
        r, c = divmod(i, nx)
        for dr in (-1, 0, 1):
            rr = r + dr
            if rr < 0 or rr >= ny:
                continue
            for dc in (-1, 0, 1):
                cc = c + dc
                if cc < 0 or cc >= nx:
                    continue
                j = rr * nx + cc
                if done[j]:
                    continue
                done[j] = True
                par[j] = i
                v = zr[j]
                if v < lev:
                    v = lev
                zf[j] = v
                push(heap, (float(v), j))
    return zf.reshape(z.shape), par.reshape(z.shape), np.asarray(order, np.int64)


def stream_fall(z, own, seeds, pix, min_area, share=0.25):
    """The fall to take out of each pixel's bed, and the stream mask.

    z: DEM (nan outside); own: the cell holding each pixel (-1 outside); seeds: the outfall pixels;
    min_area: a stream drains at least this (m2); share: inside a cell a tributary counts as part of the stream
    only if it drains at least this share of what the stream drains where it leaves the cell (a smaller one keeps
    its rise above the junction, so it does not take the main stream's depth).
    """
    zf, par, order = drainage(np.where(own >= 0, z, np.nan), seeds)
    zf_, par_, own_ = zf.ravel(), par.ravel(), own.ravel()
    acc = np.zeros(z.size)
    acc[order] = pix * pix
    for i in order[::-1]:
        p = par_[i]
        if p >= 0:
            acc[p] += acc[i]
    stream = np.zeros(z.size, bool)
    exit_z = np.zeros(z.size)
    exit_a = np.zeros(z.size)
    fall = np.zeros(z.size)
    for i in order:                                          # downstream first
        p = par_[i]
        same = p >= 0 and own_[p] == own_[i]
        if acc[i] >= min_area:
            if same and stream[p] and acc[i] >= share * exit_a[p]:
                stream[i] = True
                exit_z[i], exit_a[i] = exit_z[p], exit_a[p]
                fall[i] = zf_[i] - exit_z[i]
                continue
            if not (same and stream[p]):                     # the stream leaves the cell here
                stream[i] = True
                exit_z[i], exit_a[i] = zf_[i], acc[i]
                continue
        if same:
            fall[i] = fall[p]
    return fall.reshape(z.shape), stream.reshape(z.shape), acc.reshape(z.shape), zf
