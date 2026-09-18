"""Entrainment of bed sediment into still water: the reference solution.

A tank of still water stands over a fixed bed that is flat on one half and
slopes gently on the other. There is no flow, so the quadratic-drag closure
would give zero shear; the run instead selects the depth-slope closure
[T-7], under which the bed shear stress of a cell is a function of its
depth and its bed slope alone:

    tau_b / rho = g h S .                                   [T-7]

On the flat half S = 0, the Shields stress is below the critical value and
nothing happens: that half is the control. On the sloped half the Shields
stress

    tau* = (tau_b / rho) / (R g d) = h S / (R d)             [T-3]

exceeds tau_c*, and the non-cohesive entrainment law fires,

    E = v_s E*,   E* = 0.65 gamma0 X / (1 + gamma0 X),   X = tau*/tau_c* - 1
                                                           [E-1], [E-2]

while deposition returns sediment to the bed at the well-mixed rate

    D = d* v_s c .                                          [D-1]

With no flow there is no advection, so every cell is a closed system, and
with the bed held fixed h is constant. The sediment mass per unit area
m = h c then obeys the linear equation

    dm/dt = E - D = E - d* v_s m / h                        [G-3]

whose solution from c = 0 is a relaxation to the equilibrium concentration

    c_eq = E / (d* v_s) = E* / d* ,
    c(t) = c_eq (1 - exp(-t / T)),      T = h / (d* v_s) .

The bed is held fixed deliberately. With bed evolution on, the cells along
the far wall (whose reconstructed bed slope is zero) would not erode while
their neighbours did, and the steps that opened between them would enter the
depth-slope closure through the reconstructed bed edge values. That is the
model doing what it should, but the slope of each cell would then no longer
be a known constant and the case would have no closed-form reference. The
bed update [G-4] is checked by the settling case instead.
"""
import numpy as np


def entrainment_rate(h, S, v_s, R, d, tau_c_star, gamma0):
    """E in m/s for depth h and bed slope S, [T-7], [T-3], [E-1], [E-2]."""
    h = np.asarray(h, dtype=float)
    S = np.asarray(S, dtype=float)
    tau_star = h * S / (R * d)
    X = tau_star / tau_c_star - 1.0
    gX = gamma0 * np.maximum(X, 0.0)
    return v_s * 0.65 * gX / (1.0 + gX)


def equilibrium_concentration(h, S, v_s, R, d, tau_c_star, gamma0, d_star=1.0):
    """c_eq = E / (d* v_s), the concentration at which deposition balances."""
    return entrainment_rate(h, S, v_s, R, d, tau_c_star, gamma0) / (d_star * v_s)


def timescale(h, v_s, d_star=1.0):
    """The e-folding time h / (d* v_s) of the relaxation."""
    return np.asarray(h, dtype=float) / (d_star * v_s)


def erosion(t, h, S, v_s, R, d, tau_c_star, gamma0, d_star=1.0):
    """Reference c(t) for cells of depth h and bed slope S, from c = 0.

    h and S are arrays with one entry per cell (scalars are broadcast); the
    result has shape (len(t), ncells).
    """
    t = np.asarray(t, dtype=float)
    h = np.atleast_1d(np.asarray(h, dtype=float))
    S = np.atleast_1d(np.asarray(S, dtype=float))
    c_eq = equilibrium_concentration(h, S, v_s, R, d, tau_c_star, gamma0, d_star)
    T = timescale(h, v_s, d_star)
    return c_eq[None, :] * (1.0 - np.exp(-t[:, None] / T[None, :]))


def read_centroid_series(filename, name='sand'):
    """time, concentration, bed, stage and momenta at the centroids.

    All arrays are (ntime, ncells). A fixed bed is stored once in the sww;
    it is broadcast over time here so the caller need not care.
    """
    from netCDF4 import Dataset
    with Dataset(filename) as f:
        t = f.variables['time'][:].astype(float)
        c = f.variables[name + '_c'][:].astype(float)
        z = f.variables['elevation_c'][:].astype(float)
        w = f.variables['stage_c'][:].astype(float)
        uh = f.variables['xmomentum_c'][:].astype(float)
        vh = f.variables['ymomentum_c'][:].astype(float)
    if z.ndim == 1:
        z = np.broadcast_to(z, c.shape).copy()
    return t, c, z, w, uh, vh
