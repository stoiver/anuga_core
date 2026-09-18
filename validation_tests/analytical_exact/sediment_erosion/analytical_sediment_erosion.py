"""Entrainment of bed sediment into still water: the reference solution.

A tank of still water stands over a fixed plane bed of gentle slope S.
There is no flow, so the quadratic-drag closure would give zero shear; the
run instead selects the depth-slope closure [T-7], under which the bed
shear stress of a cell is a function of its depth and its bed slope alone:

    tau_b / rho = g h S .                                   [T-7]

The Shields stress

    tau* = (tau_b / rho) / (R g d) = h S / (R d)             [T-3]

exceeds the critical value tau_c* of the sand fraction, and the
non-cohesive entrainment law fires,

    E = v_s E*,   E* = 0.65 gamma0 X / (1 + gamma0 X),   X = tau*/tau_c* - 1
                                                           [E-1], [E-2]

while deposition returns sediment to the bed at the well-mixed rate

    D = d* v_s c .                                          [D-1]

A second fraction of the same grain but a critical stress far above tau*
has X < 0 and is never entrained: that is the threshold control.

With no flow there is no advection, so every cell is a closed system, and
with the bed held fixed h is constant. The sediment mass per unit area
m = h c then obeys the linear equation

    dm/dt = E - D = E - d* v_s m / h                        [G-3]

whose solution from c = 0 is a relaxation to the equilibrium concentration

    c_eq = E / (d* v_s) = E* / d* ,
    c(t) = c_eq (1 - exp(-t / T)),      T = h / (d* v_s) .

Every cell follows its own curve, since h varies along the slope.

The bed is held fixed deliberately. The depth-slope closure takes S from
the bed itself, so once the bed evolves any non-uniform erosion steepens
local slopes, which raises tau_b, which erodes faster: the case has no
closed-form reference and, left to run, scours metres. That is a property
of the closure (anugaSed contains it with a domain-global clamp), not of
the discretisation. The bed update [G-4] is checked by the settling case.
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
