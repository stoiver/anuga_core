# Sediment transport examples

Ported from the examples shipped with **anugaSed** (Mariela Perignon, 2016),
which is MIT licensed. `topo.asc`, `topo.prj` and `outline.csv` are their data
files, copied unchanged.

The originals are Python 2 and drive `Sed_transport_operator` through an
`evolved_quantities` list containing `'concentration'`. These ports keep their
geometry, boundary conditions, durations and parameters, and drive our API
instead: `domain.add_sediment_class()`, which registers the fractional-step
`Sediment_operator` for you.

## One deliberate difference in the physics

anugaSed erodes with the **cohesive** Hanson & Simon route `[E-3]`
(`tau_crit = 0.088` Pa, dimensional). These ports use the **non-cohesive**
Shields / Wong-Parker route `[E-1]`, because that is the bed material this work
targets -- a sand bed. Spec 4.1.1 is explicit that these are models for
*different sediment*, not competing formulations of the same physics, so these
numbers are **not** expected to reproduce anugaSed's.

Their other defaults are carried over: porosity 0.3, rho_s 2650, rho_w 1000,
D50 = 65 um (Griffin et al. 2014).

| file | ported from |
|------|-------------|
| `example_1_channel.py` | `run_simple_sed_transport.py` |
| `example_2_raster.py`  | `run_raster_sed_transport.py` |

`run_simple_veg.py` is not ported: vegetation drag (spec 8) is Phase 5 and is
not implemented.

## A note on the data files

`topo.asc` (594 KB) and `outline.csv` are force-added: the repository's root
`.gitignore` excludes `*.asc` and `*.csv` as build artefacts, and without `-f`
example 2 would have been committed without the data it needs.

## Status

These demonstrate the module running end to end. They are **not validation**:
nothing here has been checked against measured data or against a published
result. The validation rungs of spec 10 -- Rio Puerco (rung 8) and the crater
breach (rung 7) -- have not been attempted.
