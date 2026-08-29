# Sediment transport in ANUGA -- user guide

This documents the user-facing interface: every parameter, its units, its
default, the spec clause it comes from, and how to choose between the
alternative methods. It assumes you know ANUGA; it does not derive the
physics, which is `PHYSICS_SPEC.md`. Spec labels like `[E-1]` are that
document's, and each one below is a pointer into it.

Verification evidence for these terms is in `sandpit/tracer_spike/`; runnable
examples are in `sandpit/sediment_examples/`.

---

## 1. The shortest useful program

```python
import anuga

domain = anuga.rectangular_cross_domain(40, 10, len1=100.0, len2=25.0)
domain.set_flow_algorithm('DE0')
domain.set_quantity('elevation', lambda x, y: -0.01 * x)
domain.set_quantity('stage', 0.5)
domain.set_quantity('friction', 0.03)
domain.set_boundary({t: anuga.Reflective_boundary(domain)
                     for t in domain.get_boundary_tags()})

domain.add_sediment_class('sand', diameter=2.0e-4)   # <- the only new line

for t in domain.evolve(yieldstep=1.0, finaltime=30.0):
    pass
```

`add_sediment_class` is the entry point. One call gives you a transported
concentration, erosion, deposition, the settling velocity, the bed exchange,
and the limiters, with defaults chosen for a sand bed. It registers the
fractional-step `Sediment_operator` for you, so there is nothing else to wire
up.

Everything below is about changing those defaults.

## 2. Print what you configured

```python
print(domain.sediment_summary())
```

This is the single most useful call in the interface. It reports the active
configuration -- every law selected, every scalar in force, and each class's
derived settling velocity -- as text:

```
sediment configuration
  classes            : 2  ['fine', 'coarse']
  erosion            : [E-1] Shields / Smith-McLean, non-cohesive (sand, gravel)
  deposition         : [D-1] D = d* c v_s
  near-bed d*        : [S-4] Rouse profile
  shear closure      : [T-1] quadratic drag, tau_b = rho f_c |v|^2
  friction closure   : wilson [T-8..10], bed=gravel, D=0.02 m
  bedload            : [K-1] power law, K=3.97 m=1.5 tau_c*=0.0495
  bed evolution      : True  (spec 2.4 Phase 4, evolving)
  porosity lambda    : 0.28
  c_max      [L-2]   : 0.35
  c_pack     [L-4]   : 0.65
  rho_w              : 1000 kg/m3
  a/h floor          : 0.01
  per class:
    fine       d=0.0001 m  v_s=8.0040e-03 m/s  R=1.65  tau_c*=0.04
    coarse     d=0.0005 m  v_s=9.4839e-02 m/s  R=1.65  tau_c*=0.04
```

Settling velocity in particular is *derived*, not set: if `v_s` is not what
you expected, the diameter or the fluid properties are not what you thought.
Print this at the top of every run.

## 3. The interface at a glance

Choices are made by naming the **physics**, never by setting a flag:

| call | chooses | spec |
|---|---|---|
| `add_sediment_class(name, diameter, ...)` | a grain size to transport | 2.2 |
| `set_bed_material(material, ...)` | the erosion law | 4.1.1 |
| `set_deposition(law, near_bed, ...)` | the deposition law and near-bed ratio | 4.4 |
| `set_shear_closure(closure)` | how `tau_b` is formed | 3.2 |
| `set_sediment_friction(mode, ...)` | the friction factor feeding `tau_b` | 3.3 |
| `set_bedload(formula, ...)` | bedload transport, or off | 5 |
| `set_sediment_parameters(...)` | the scalar physical properties | 2.4, 6 |
| `set_erodible_base(...)` | the depth below which nothing erodes | 4.5 |
| `set_erodible_region(...)` | where erosion may act at all | 4.5 |
| `set_tracer_source(name, values)` | an external source | 2.6 |
| `set_tracer_boundary(name, value)` | inflow concentration | 2.5 |

Order does not matter, with one exception noted in §4.1: call them before
`evolve()`, in whatever order reads best.

**Anything not in that table is internal.** The domain carries roughly fifteen
`sediment_*` arrays (`sediment_qbx`, `sediment_settling_velocity`,
`sediment_erosion_mode`, ...) that exist to be handed to the C kernel. Setting
them directly can leave the GPU mapping stale, and no validation runs. Use the
setters; they invalidate the device mapping for you.

---

## 4. Sediment classes

### 4.1 `add_sediment_class`

```python
domain.add_sediment_class(name, diameter, d_star=1.0, beta=None,
                          initial_concentration=0.0, rho_s=2650.0,
                          rho_w=1000.0, tau_c_star=0.04,
                          reference_height=None, auto_operator=True,
                          **settling_kwargs)
```

| parameter | units | default | meaning |
|---|---|---|---|
| `name` | -- | required | label; also the tracer name |
| `diameter` | m | required | grain diameter `d`; sets `v_s` via `[S-1]` |
| `rho_s` | kg/m3 | 2650 | sediment density (quartz) |
| `rho_w` | kg/m3 | 1000 | fluid density; `R = rho_s/rho_w - 1` |
| `tau_c_star` | -- | 0.04 | critical Shields stress, `[E-1]` |
| `d_star` | -- | 1.0 | near-bed ratio `c_b/c`; 1.0 is well-mixed |
| `initial_concentration` | -- | 0.0 | volumetric, uniform |
| `beta` | -- | domain's | limiter coefficient `[L-3]` |
| `reference_height` | m | `None` | Rouse reference height `a`; see §6.2 |
| `auto_operator` | -- | `True` | register `Sediment_operator` |

Multiple classes are independent: each has its own concentration, settling
velocity and critical stress, and each exchanges with the same bed. Call it
once per grain size.

Classes occupy tracer slots in call order, so class `s` is tracer `s`. **The
one ordering rule**: do not interleave `add_tracer` and `add_sediment_class`
on the same domain if you rely on that correspondence.

### 4.2 Choosing `tau_c_star`

0.04 is a reasonable default for sand. It is the threshold at which grains
begin to move, and erosion is zero below it, so it sets *when* the bed becomes
active, not how fast. If the bed does not erode when you expect it to, check
this against the Shields curve for your grain size before adjusting anything
else.

### 4.3 Initial and boundary concentrations

```python
domain.set_tracer('sand', 0.001)          # uniform, or an array of centroids
domain.set_tracer_boundary('sand', 0.02)  # concentration entering the domain
```

Both are volumetric concentration `c` (dimensionless), not `h*c`. The
conserved quantity is `m = h*c`; the interface works in `c` throughout.

---

## 5. Erosion: naming the bed material

```python
domain.set_bed_material('noncohesive')   # default
domain.set_bed_material('cohesive', tau_crit=0.088, K_e=6.742e-7)
domain.set_bed_material('partheniades', tau_crit=0.088, K_e=...)
```

The argument is the **material**, not the formula, because spec 4.1.1 is
explicit that these describe different sediment rather than competing
descriptions of the same sediment. Picking the wrong one is a physics error.

| material | law | when |
|---|---|---|
| `noncohesive` | `[E-1]`/`[E-2]` Shields, Smith & McLean | sand, gravel, boulders |
| `cohesive` | `[E-3]` Hanson & Simon | clay, silt, consolidated mud |
| `partheniades` | `[E-4]` Partheniades | cohesive, where you have a site-calibrated `K_e` |

`tau_crit` (Pa, default 0.088) and `K_e` (m3/N/s) apply to the cohesive
routes only; the non-cohesive route takes its threshold per class from
`tau_c_star` instead. The default `K_e = 6.742e-7` is anugaSed's.

The choice is not a small correction. On the same channel over 30 s, the
non-cohesive route scours 3-6 cm while the cohesive route accretes about a
millimetre -- the sign of the bed change reverses. See
`sandpit/sediment_examples/README.md`.

---

## 6. Deposition

```python
domain.set_deposition(law='d_star', tau_d=0.0, near_bed='constant',
                      reference_height_floor=0.01)
```

### 6.1 `law`

| value | expression | when |
|---|---|---|
| `'d_star'` | `D = d* c v_s`, `[D-1]` | default; always deposits |
| `'threshold'` | `[D-2]`, deposition only where `tau_b < tau_d` | when you need deposition suppressed under strong flow |

`tau_d` (Pa) is the threshold for `'threshold'` and is ignored otherwise.

### 6.2 `near_bed` -- the `d*` ratio

Deposition is driven by the concentration *at the bed*, but the transported
quantity is depth-averaged. `d* = c_b/c` bridges them.

| value | meaning |
|---|---|
| `'constant'` | `d*` is whatever each class was given (1.0 = well-mixed). Default. |
| `'rouse'` | `d*` from the Rouse profile `[S-4]`, recomputed per cell per step |

`'constant'` with `d* = 1` is the well-mixed assumption: fine sediment,
vigorous mixing, shallow flow. It is also what the analytic decay solutions
assume, so use it when comparing against them.

`'rouse'` is the physical choice when the profile is stratified -- coarser
grains, or deeper and slower flow, where near-bed concentration genuinely
exceeds the mean. It costs an evaluation of the fitted `d*(Z, a/h)`
polynomial per cell per class per step (§9.5 of the spec; 28 terms, maximum
error 0.82% over `Z` in [0.01, 2.5], `a/h` in [1e-3, 0.15], clamped at the
edges rather than extrapolated).

`reference_height_floor` (default 0.01) is the floor on `a/h`. The Rouse
profile is singular as the reference height approaches the bed, so `a/h` is
never allowed below this. Lowering it admits more stratification and more
near-bed concentration; it is a numerical guard, not a physical parameter, and
0.01 sits comfortably inside the fit range.

Near-bed concentration is bounded by `c_pack` `[L-4]` regardless. That bound
exists because equilibrium Rouse `d*` at vanishing shear will otherwise
deposit the entire water column in under a second.

---

## 7. Bed shear stress

Two independent choices feed `tau_b`: how the stress is formed, and what
friction factor goes into it.

### 7.1 `set_shear_closure` -- how

```python
domain.set_shear_closure('quadratic_drag')   # default
domain.set_shear_closure('depth_slope')
```

| value | expression | spec |
|---|---|---|
| `'quadratic_drag'` | `tau_b = rho f_c |v|^2` | `[T-1]` |
| `'depth_slope'` | `tau_b = rho g h S` | `[T-7]` |

`'quadratic_drag'` is the default and the right choice for unsteady or
rapidly varying flow -- dam breaks, floods, anything with significant
inertia.

`'depth_slope'` assumes locally uniform flow, where friction balances gravity.
It is what anugaSed uses, so choose it when reproducing their results
(divergence **D1** in the spec). It degrades where that balance does not hold.

The two are interchangeable by construction: the kernel returns `tau_b/rho`,
so everything downstream is unchanged by the choice.

### 7.2 `set_sediment_friction` -- what

```python
domain.set_sediment_friction('constant')    # default
domain.set_sediment_friction('wilson', bed='gravel', grain_size=0.02)
domain.set_sediment_friction('larsen_lamb', k_s=0.05, r_d=2.0, r_br=2.0)
```

`'wilson'` and `'larsen_lamb'` are not callable with the mode alone -- they
require a length scale and refuse without one, rather than inventing a
default:

* `'wilson'` needs `grain_size > 0` (D50 for sand, D84 for gravel or boulder);
* `'larsen_lamb'` needs either `k_s` or `sigma_br`. There is no universal
  `sigma_br`: it is site-measured, and LL16 report about 5 m at Moses Coulee.

| mode | spec | when |
|---|---|---|
| `'constant'` | `[T-6]` | default: `f_c` from the domain's Manning `n`. Ordinary flood and channel work. |
| `'wilson'` | `[T-8..T-12]` | depth-dependent, from grain size. Shallow flow over coarse beds, where relative submergence matters. |
| `'larsen_lamb'` | `[T-13..T-15]` | partitions total stress into grain and form drag. Bedforms or roughness elements, where only the grain part drives sediment. |

`bed` is `'sand'` or `'gravel'`; `grain_size` (m) is the roughness length
scale; `k_s` (m) is the roughness height; `r_d` and `r_br` (default 2.0) are
Larsen-Lamb's drag partitioning ratios.

This affects **only** the sediment source term. The hydrodynamic friction
operator is untouched, so momentum still sees the domain's Manning `n`
whatever you choose here.

---

## 8. Bedload

```python
domain.set_bedload('wong_parker_eq24')   # K=3.97, m=1.5, tau_c*=0.0495
domain.set_bedload('wong_parker_eq23')   # K=4.93, m=1.6, tau_c*=0.0470
domain.set_bedload('engelund_hansen')
domain.set_bedload('off')                # default
```

Bedload `[K-1]`-`[K-4]` transports sediment along the bed rather than in
suspension, and drives its own bed evolution term `[G-5]`. It is **off by
default**: it is a separate transport mode, not a refinement of suspension,
and enabling it changes what the model represents.

Enable it when the grain size is coarse enough that a significant fraction of
the load moves without going into suspension -- sand and gravel beds under
moderate flow. Leave it off for fine, fully suspended sediment.

The two Wong & Parker variants are their corrected Meyer-Peter-Muller fits;
Eq 24 is the default. `K`, `m` and `tau_c_star` override the formula's
constants if you have a calibration.

Bedload only redistributes: it moves sediment between cells and conserves the
total exactly. The flux across each edge is centred, which is what makes it
antisymmetric and therefore conservative; see `test_bedload.py`.

---

## 9. Scalar parameters

```python
domain.set_sediment_parameters(porosity=0.30, c_max=0.30, c_pack=0.65,
                               bed_evolution=True, rho_w=1000.0)
```

All optional; only what you pass is changed. All are validated.

| parameter | units | default | meaning |
|---|---|---|---|
| `porosity` | -- | 0.30 | bed porosity `lambda`, `[G-4]`. Sediment volume leaving suspension is `(1-lambda) dz`; the rest is pore space filled from the water column. LM15 use 0.28. |
| `c_max` | -- | 0.30 | `[L-2]`, ceiling on depth-averaged concentration (FG21; anugaSed use 0.20). |
| `c_pack` | -- | 0.65 | `[L-4]`, maximum packing bounding *near-bed* `c_b = d* c`. Only bites when `d* != 1`. |
| `bed_evolution` | -- | `True` | whether the bed moves |
| `rho_w` | kg/m3 | 1000 | fluid density used to form dimensional `tau_b` |

### 9.1 `bed_evolution` is the coupling stage

`False` gives a **fixed bed**: sediment is entrained and deposited, and
concentration evolves, but elevation never changes. Choose it when

* comparing against analytic solutions, which assume constant depth;
* comparing against RDycore v1.0, which is configured this way;
* isolating a transport question from a morphology question.

`True` (the default) evolves the bed through `[G-4]` and `[G-5]`. Choose it
for any real morphological problem. Both bed terms are applied in a single
fractional step.

---

## 10. The non-erodible base

```python
domain.set_erodible_base(depth=0.5)          # 0.5 m of erodible material
domain.set_erodible_base(elevation=z_rock)   # or an absolute surface, (n,)
domain.set_erodible_base()                   # remove it again
```

By default the bed is **bottomless**: erosion lowers it for as long as the flow
has the strength to. That is right for a deep alluvial channel and wrong
wherever the erodible layer is finite -- a reach floored by an outcrop, a lined
culvert, a dam apron, a soil layer of known depth over rock. `[L-5]` gives it a
floor.

The base is a **per-centroid field**, because bedrock is a surface. `depth=`
measures down from the elevation set so far and is recorded as an elevation at
the moment of the call, so later changes to the elevation quantity do not drag
it around. `elevation=` gives the surface directly, in the domain's datum.
Scalars broadcast; arrays must be `(n,)`. Give exactly one.

A base above the bed is rejected rather than silently accepted -- it would mean
negative erodible thickness, which is a mistake, not a configuration.

```python
domain.erodible_thickness()   # (n,) metres remaining; 0 means bedrock
```

`sediment_summary()` reports the range and how many cells have reached bedrock.

### 10.1 What it guarantees

The limit is applied to the **source**, not by clamping elevation. Erosion is
scaled back to what the remaining thickness can supply, so the sediment that is
not eroded never enters the water column and the budget still closes to machine
precision. Clamping `z` afterwards would leave suspended sediment that came
from nowhere.

Where several classes compete for the last of the material they are scaled by
one shared proportional factor, not served in registration order: the bed
carries no per-class stratigraphy, so no class has a better claim, and the
answer must not depend on the order you called `add_sediment_class`.
Deposition is never scaled -- it is what replenishes the bed.

The two transport routes give **different strengths of guarantee**, and it is
worth knowing which you are relying on:

| route | floor is | why |
|---|---|---|
| suspended exchange `[G-4]` | **exact** | the limit is on the exchange term itself |
| bedload `[G-5]` | within one step's flux | bedload is a divergence; see below |

Bedload only redistributes, and stays exactly conservative with a base present,
because the limit is applied to the transport vector and to whole edges -- both
of which the two cells sharing an edge evaluate identically. The price is that
the floor is not exact: closing an edge for a cell that cannot pay also cancels
its neighbour's inflow, so the deficit migrates. Measured overshoot is
5.1e-6 m on a 1.0e-2 m layer. If you need bedload's floor exact, that is a
known limitation with a known fix (iterating the exhaustion flag to a fixed
point), not a mystery.

### 10.2 Restricting erosion to a region

The base says how *deep* erosion may go; a region says *where* it may happen
at all.

```python
domain.set_erodible_region(polygon=breach)                 # ONLY here erodes
domain.set_erodible_region(polygon=apron, erodible=False)  # everywhere BUT here
domain.set_erodible_region(center=[x, y], radius=25.0)     # a circle
domain.set_erodible_region(indices=ids)                    # triangles directly
domain.set_erodible_region(my_region)                      # a Region object
domain.set_erodible_region()                               # remove it
```

The keyword arguments are the ones the region-based operators
(`Erosion_operator` and friends) already take, resolved by the same `Region`
class, so a polygon that selects a set of cells there selects the same set
here. A region that selects no cells is rejected rather than silently doing
nothing -- that is almost always a polygon in the wrong coordinates.

**Passing a `Region` is the general form.** `Region` understands more than the
keywords above -- `line=`, `poly=`, `expand_polygon=` -- so build one and hand
it over when you need those:

```python
from anuga.abstract_2d_finite_volumes.region import Region

domain.set_erodible_region(Region(domain, line=thalweg))
domain.set_erodible_region(Region(domain, polygon=reach, expand_polygon=True))
```

Both do reach through: a line selects the cells it crosses (83 of 960 on a test
mesh), and `expand_polygon` genuinely changes the selection (504 cells against
480), because it intersects on vertices rather than centroids.

It must be a `Region` built on **this** domain -- one built elsewhere carries
triangle indices that mean nothing here, and is refused rather than silently
mis-selecting. A bare list of points passed positionally is refused too, with a
message pointing at `polygon=`: taking it as a region would select every cell
and look like it worked.

**Locked means unscourable, not inert.** A locked cell is held at the
elevation it has when you call this, by giving it zero erodible thickness --
the restriction is `[L-5]` with the layer set to nothing, not a separate
mechanism. Sediment may still settle onto it, which is what a concrete apron
or a rock bar does in the field, and that new material is erodible again
because it now sits above the base. Under genuinely erosive flow such a cell
sits at exactly net zero: the limiter scales erosion back until it just
cancels deposition, so nothing piles up on a scoured apron.

The two compose, in either order, and neither discards the other:

```python
domain.set_erodible_base(depth=0.4)        # 0.4 m of erodible material
domain.set_erodible_region(polygon=reach)  # but only inside this reach
```

Where they disagree the stricter wins. `sediment_summary()` reports both, and
the thickness range it prints covers only the erodible cells -- locked ones
carry zero thickness and would otherwise drag the minimum to zero whatever the
layer is.

### 10.3 Cost

None when unset. With no base configured the kernels take the path they took
before the feature existed, and produce bitwise identical results -- which is
asserted, not assumed, in `test_erodible_base.py` check E1.

---

## 11. External sources

```python
domain.set_tracer_source('sand', values)   # array over centroids, or a scalar
```

Adds a source to the tracer equation directly (spec 2.6), in units of `m`
(that is, `h*c`) per second. This is what the manufactured-solution tests use
to impose an analytic forcing, and it is the hook for anything the bed
exchange does not describe -- a lateral inflow, a point discharge, a
prescribed release.

---

## 12. Running on the GPU

```python
domain.set_multiprocessor_mode(2)
```

Mode 1 is the legacy CPU/OpenMP path; mode 2 is the unified path that runs on
the device. **Both paths share `core_kernels.c`, so the physics is the same
code**, and `test_mode1_vs_mode2.py` holds them to agreement.

Nothing about the sediment configuration changes between modes: set it up the
same way and switch the mode.

Mode 2 selects the *unified* code path; whether that path actually offloads to
a device is a property of the **build**, not of this call. A build without
offload compiles the same kernels under `CPU_ONLY_MODE` and runs them on the
host. `set_multiprocessor_mode(2)` therefore does not fail on a machine with
no GPU, but it also does not report one: `domain.multiprocessor_mode` will
read 2 either way. To find out what you are actually running on, ask the
build:

```python
import anuga
anuga.gpu_offload_enabled()    # True if this build offloads
```

To confirm kernels are reaching the device on a run, set
`NVCOMPILER_ACC_NOTIFY=1` in the environment. Polling `nvidia-smi` is
unreliable for this -- the sampling interval misses short kernel bursts.

Call `set_multiprocessor_mode` **after** the sediment setup. Each setter
invalidates the device mapping, so configuring sediment after selecting mode 2
simply forces the mapping to be rebuilt.

---

## 13. Choosing a configuration

If you do not know where to start:

* **Sand bed, flood or dam break, morphology wanted.** Defaults, plus a class:
  `add_sediment_class('sand', diameter=2e-4)`. Add
  `set_bedload('wong_parker_eq24')` if the grains are coarse enough to move
  along the bed.
* **Fine cohesive sediment, muddy estuary.** `set_bed_material('cohesive')`
  with a `tau_crit` you trust, `d*` left at 1.0.
* **Deep, slow, stratified flow.** `set_deposition(near_bed='rouse')`, and give
  each class a `reference_height`.
* **Shallow flow over gravel.** `set_sediment_friction('wilson', bed='gravel',
  grain_size=...)`.
* **Reproducing anugaSed.** `set_bed_material('cohesive')` and
  `set_shear_closure('depth_slope')`; see `sandpit/sediment_examples/`.
* **Comparing against an analytic solution.**
  `set_sediment_parameters(bed_evolution=False)` and leave `d*` at 1.0.
* **A finite erodible layer over rock.** `set_erodible_base(depth=...)`, and
  check `erodible_thickness()` afterwards to see where it bit.
* **Scour confined to one structure or reach.** `set_erodible_region(polygon=...)`,
  or `erodible=False` to lock an apron while the rest of the domain erodes.

Then print `sediment_summary()` and check it says what you meant.

---

## 14. What is not implemented

Vegetation drag (spec 8) is Phase 5 and absent. Neither validation rung of
spec 10 -- Rio Puerco, the crater breach -- has been attempted; the evidence
in `sandpit/tracer_spike/` is verification (the equations are solved
correctly), which is a different claim from validation (they are the right
equations for the field case).
