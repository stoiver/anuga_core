# Sediment transport examples

Ported from the examples shipped with **anugaSed** (Mariela Perignon, 2016),
which is MIT licensed. `topo.asc`, `topo.prj` and `outline.csv` are their data
files, copied unchanged.

The originals are Python 2 and drive `Sed_transport_operator` through an
`evolved_quantities` list containing `'concentration'`. These ports keep their
geometry, boundary conditions, durations and parameters, and drive our API
instead: `domain.add_sediment_class()`, which registers the fractional-step
`Sediment_operator` for you.

## Bed material: both routes are available

anugaSed erodes with the **cohesive** Hanson & Simon route `[E-3]`
(`tau_crit = 0.088` Pa, dimensional). The default here is the **non-cohesive**
Shields route `[E-1]`, the bed material this work targets. Spec 4.1.1 is
explicit that these describe *different sediment*, not competing formulations
of the same physics.

`example_1_channel.py --bed cohesive` runs anugaSed's own configuration.
The difference is not a detail:

| `--bed` | erosion law | bed change over 30 s |
|---------|-------------|----------------------|
| `noncohesive` (default) | `[E-1]` Shields, `tau_c*` = 0.04 | **erodes** 3.1 to 5.8 cm, all 16 cells |
| `cohesive` | `[E-3]` Hanson & Simon, `tau_c` = 0.088 Pa | **accretes** 0.6 to 1.3 mm, all 16 cells |

The sign reverses. A cohesive bed resists erosion strongly enough
(`K_e = 6.74e-7 m3/N/s`) that the sediment-laden inflow deposits instead of
scouring. This is what spec 4.1.1 means by "a physics error, not a tuning
error".

## How far the anugaSed comparison goes

`test_cohesive.py` checks our `[E-3]` against their `erosion()` transcribed
from `sed_transport_operator.py`, and reproduces their `edot` **exactly**
(difference 0.000e+00) across `tau_b` in [0, 20] Pa, with the same
`K_e = 6.742e-7`.

It stops there, deliberately. anugaSed cannot be *run* here -- it is Python 2
and drives a `Sed_transport_operator` this ANUGA does not have -- and a
whole-model comparison would differ anyway for a reason unrelated to erosion:
they compute bed shear from the depth-slope closure `[T-7]`, we use quadratic
drag `[T-1]` (divergence **D1**). Equal erosion laws fed unequal `tau_b` give
unequal answers.

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
