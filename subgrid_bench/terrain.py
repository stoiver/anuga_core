"""Synthetic narrow-channel valley for the subgrid Phase 0 benchmark.

All geometry is analytic and synthetic (safe to publish). The "DTM" is the
analytic bed sampled on a 0.5 m raster, as a real DTM would be.

    valley      2000 m x 500 m, bed falling 1:1000 in +x
    channel     meandering trapezoid along x: base 10 m, depth 2 m, banks 1:2
                (top width 18 m), centreline y = 250 + 40 sin(2 pi x / 500)
    floodplain  rises 1:250 away from the channel, with small hollows
    outlet      the last 100 m steepen to 1:50 (2 m extra drop), so the flow
                passes through critical depth at x = 1900: a physical,
                mesh-independent downstream control, then a free outfall
"""
import os

import numpy as np

LENGTH = 2000.0
WIDTH = 500.0
Z0 = 10.0              # floodplain level at x = 0 next to the channel
SLOPE = 1.0e-3         # valley slope (down in +x)
CROSS_SLOPE = 4.0e-3   # floodplain rises away from the channel (1 m at 250 m)

BASE = 10.0            # channel base width
DEPTH = 2.0            # channel depth below the floodplain
BANK = 2.0             # bank slope, horizontal : vertical
TOP_WIDTH = BASE + 2 * BANK * DEPTH

OUTLET_START = 1900.0  # start of the steep outlet reach
OUTLET_SLOPE = 0.02    # extra slope there

AMPLITUDE = float(os.environ.get('SUBGRID_AMPLITUDE', 40.0))   # meander amplitude (env: straight-channel test)
WAVELENGTH = 500.0     # meander wavelength

PIXEL = 0.5            # DTM pixel size

BOUNDING_POLYGON = [[0.0, 0.0], [LENGTH, 0.0], [LENGTH, WIDTH], [0.0, WIDTH]]
BOUNDARY_TAGS = {'bottom': [0], 'right': [1], 'top': [2], 'left': [3]}


def centreline(x):
    return WIDTH / 2 + AMPLITUDE * np.sin(2 * np.pi * x / WAVELENGTH)


def distance_to_centreline(x, y):
    """Approximate perpendicular distance to the centreline."""
    dydx = AMPLITUDE * 2 * np.pi / WAVELENGTH * np.cos(2 * np.pi * x / WAVELENGTH)
    return np.abs(y - centreline(x)) / np.sqrt(1.0 + dydx ** 2)


def valley(x):
    """Floodplain level next to the channel."""
    return Z0 - SLOPE * x - OUTLET_SLOPE * np.maximum(x - OUTLET_START, 0.0)


def channel_bed(x):
    """Thalweg elevation."""
    return valley(x) - DEPTH


def _hollows():
    rng = np.random.default_rng(508)
    out = []
    while len(out) < 16:
        x = rng.uniform(100, LENGTH - 100)
        y = rng.uniform(30, WIDTH - 30)
        if distance_to_centreline(x, y) < 50:
            continue
        out.append((x, y, rng.uniform(0.2, 0.5), rng.uniform(6.0, 15.0)))
    return np.array(out)


HOLLOWS = _hollows()   # x, y, depth, sigma


def bed(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    d = distance_to_centreline(x, y)
    z = valley(x) + CROSS_SLOPE * d
    cut = np.clip(DEPTH - (d - BASE / 2) / BANK, 0.0, DEPTH)
    z = z - cut
    for hx, hy, hd, hs in HOLLOWS:
        z = z - hd * np.exp(-((x - hx) ** 2 + (y - hy) ** 2) / (2 * hs ** 2))
    return z


def dtm():
    """Pixel-centre coordinates and elevations of the 0.5 m raster (flattened)."""
    xs = np.arange(PIXEL / 2, LENGTH, PIXEL)
    ys = np.arange(PIXEL / 2, WIDTH, PIXEL)
    X, Y = np.meshgrid(xs, ys)
    X = X.ravel()
    Y = Y.ravel()
    return X, Y, bed(X, Y)


def channel_band_polygon(half_width=20.0, dx=10.0):
    """Polygon around the channel for mesh refinement."""
    xs = np.arange(5.0, LENGTH - 5.0 + dx / 2, dx)   # inset: no slivers at the ends
    yc = centreline(xs)
    upper = np.column_stack([xs, yc + half_width])
    lower = np.column_stack([xs, yc - half_width])[::-1]
    poly = np.vstack([upper, lower])
    return poly.tolist()


def water_surface(x, depth_above_thalweg):
    """A water surface parallel to the valley, given depth above the thalweg."""
    return channel_bed(x) + depth_above_thalweg
