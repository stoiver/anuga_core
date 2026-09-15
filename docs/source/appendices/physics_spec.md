# Sediment transport physics specification

This is the specification the sediment implementation was written from. Every
equation is transcribed from a published, citable source, so each line of the
implementation can point at an equation label here and at the paper behind it.
The bracketed labels used throughout the ANUGA documentation -- `[E-1]`,
`[T-7]`, `[G-4]` and the rest -- are defined in this document.

```{note}
This is the reference half of a longer internal document. Two sections are
omitted: an audit of a third-party implementation against this specification,
and the working list of open items. Neither defines any term the
documentation refers to. What remains is the notation, the governing
equations, every closure, the numerical scheme and the parameter defaults.
```

## Sources

| Tag | Source |
|-----|--------|
| **P14** | Perignon, M.C. (2014), *A Rolling Stone Gathers No Moss*, PhD thesis, University of Colorado Boulder, §3.3.2. Equations 3.1–3.11. |
| **FG21** | Fassett, C.I. & Goudge, T.A. (2021), *Modeling the Hydrodynamics, Sediment Transport, and Valley Incision of Outlet-Forming Floods From Martian Crater Lakes*, JGR Planets 126, e2021JE006979. Equations 1–9 + Supporting Information S1–S5. |
| **RDy26** | Feng, D., Tan, Z., Xu, D., Johnson, J. & Bisht, G. (2026), *RDycore-sediment v1.0*, EGUsphere preprint 2026-4859 (CC BY 4.0). Equations 1–13 + Appendix A1–A7. |
| **aSM16** | Perignon, M.C. (2016), *Using the Sediment Transport and Vegetation Operators in ANUGA*, `anugaSed/docs/anugaSed_manual.pdf`, 4 May 2016. Equations 1–14. **This is the authoritative specification for the shipped code** and supersedes P14 where they differ. |
| **DL09** | Davy, P. & Lague, D. (2009), *Fluvial erosion/transport equation of landscape evolution models revisited*, JGR Earth Surface 114, F03007, doi:10.1029/2008JF001146. Equations 2–8, 19. Source of the E–D framework and of `[S-4]`. |
| **P13** | Perignon, M.C., Tucker, G.E., Griffin, E.R. & Friedman, J.M. (2013), JGR Earth Surface 118(3), 1193–1209, doi:10.1002/jgrf.20073. Rio Puerco lidar differencing — the field data for validation rung 8. |
| **W04** | Wilson, L., Ghatan, G.J., Head, J.W. & Mitchell, K.L. (2004), JGR Planets 109, E09003, doi:10.1029/2004JE002281. Eqs 4, 13–17: Darcy–Weisbach `f_c` by bed type. |
| **LL16** | Larsen, I.J. & Lamb, M.P. (2016), Nature 538, 229–232, doi:10.1038/nature19817. Methods: Manning–Strickler roughness closure. **Uses ANUGA.** |
| **EH67** | Engelund, F. & Hansen, E. (1967), *A Monograph on Sediment Transport in Alluvial Streams*, Teknisk Forlag. Eqs 3.1.3, 4.3.5. |
| **aS16** | `anugaSed` source (Perignon 2016, MIT). |

## Clean-room statement

No code from `ANUGA-erosion` was read in preparing this document. FG21 states that
their implementation of the bedload equations is "nearly literal" from the paper,
so the paper is a sufficient specification. Commit messages implementing sections of
this document should cite the equation label and source tag.

---

## 1. Notation

| Symbol | Meaning | Units |
|--------|---------|-------|
| `h` | water depth | m |
| `u, v` | depth-averaged velocity components | m s⁻¹ |
| `\|v\|` | velocity magnitude √(u²+v²) | m s⁻¹ |
| `z` | bed elevation | m |
| `g` | gravitational acceleration (**domain parameter, not a constant**) | m s⁻² |
| `ρ` | water density (1000) | kg m⁻³ |
| `ρ_s` | sediment particle density | kg m⁻³ |
| `R` | submerged specific gravity, (ρ_s − ρ)/ρ | – |
| `D`, `d_g` | grain diameter | m |
| `ν` | kinematic viscosity of water (1×10⁻⁶) | m² s⁻¹ |
| `κ` | von Kármán constant (0.41) | – |
| `λ`, `φ` | bed porosity | – |
| `n` | Manning coefficient | s m⁻¹ᐟ³ |
| `f` | Darcy–Weisbach friction factor | – |
| `f_c`, `C_D` | friction coefficient, ≡ f/8 | – |
| `τ_b`, `τ` | bed shear stress | Pa |
| `τ*` | Shields (dimensionless) shear stress | – |
| `τ_c*` | critical Shields stress for motion | – |
| `u*` | shear velocity | m s⁻¹ |
| `c_s` | depth-averaged volumetric concentration, class s | – |
| `m_s` | conserved sediment variable, ≡ h·c_s | m |
| `c_b` | near-bed concentration | – |
| `w_s`, `v_s` | settling velocity | m s⁻¹ |
| `Z` | Rouse number, w_s/(κu*) | – |
| `d*`, `D*` | near-bed concentration profile factor, c_b/c | – |
| `E_s`, `Ė` | entrainment (erosion) flux | m s⁻¹ |
| `D_s`, `Ḋ` | deposition flux | m s⁻¹ |
| `q_b` | volumetric bedload flux per unit width | m² s⁻¹ |
| `q_b*` | dimensionless bedload flux (Einstein number) | – |
| `M_ℓ,s` | bed mass per unit area, layer ℓ class s | kg m⁻² |
| `β` | ratio of sediment transport speed to water speed | – |
| `N_s` | number of sediment classes | – |

**Sign convention (fixed for the whole document):** `E` is positive from bed into
the water column; `D` is positive from water column onto the bed. Bed elevation
rises under net deposition.

---

## 2. Governing equations

### 2.1 Hydrodynamics — unchanged

ANUGA already solves the conservative shallow water equations; this is stated only
to fix notation. `[P14 3.1–3.3]`, `[RDy26 3–5]`:

```
∂U/∂t + ∂E/∂x + ∂G/∂y = S ,     U = [h, hu, hv]ᵀ
```

with bed friction slope and bed slope in `S`. Sediment does **not** modify these
equations in any phase of the planned work (see §2.4).

### 2.2 Suspended sediment transport

The prognostic sediment variable is **mass per unit area**, not concentration
`[RDy26 1]`:

```
[G-1]   m_s = h · c_s                                    s = 1 … N_s
```

State vector `[RDy26 2]`:

```
[G-2]   U = [ h, hu, hv, m_1, m_2, …, m_N_s ]ᵀ
```

Conservative transport, one equation per class `[RDy26 6]`, which is `[DL09 4]`
written per class — DL09 give it in Lagrangian form,
`D(c_S h)/Dt = ∂(c_S h)/∂t + div(q_S) = ė − ḋ`:

```
[G-3]   ∂m_s/∂t + ∂(u m_s)/∂x + ∂(v m_s)/∂y = E_s − D_s + S_ms
```

`S_ms` is an optional external source (hillslope yield, tributary load, rainfall
washoff).

> **Reconciliation note.** P14 Eq 3.4 writes the same balance as
> `∂(Ch)/∂t = Ė − Ḋ − (∂q_sx/∂x + ∂q_sy/∂y)` with `q_s(x,y) = β C q(x,y)`, where
> `q` is specific discharge and **β is the ratio of sediment transport speed to
> water speed**. `[G-3]` is the β = 1 case. β is retained as an exposed parameter —
> see the note on `beta` in the ANUGA sediment documentation.

### 2.3 Bed evolution (Exner)

`[P14 3.5]`, `[FG21 6]`:

```
[G-4]   ∂z/∂t = (D − E)/(1 − λ)                         suspended contribution
[G-5]   ∂z/∂t = −(1/(1 − λ)) ∇·q_b                      bedload contribution
```

Both act on the same `z`; when both operators are active the contributions sum.

### 2.4 Coupling stages

| Stage | Bed elevation | Bed → flow | Sediment → momentum | Reference model |
|-------|---------------|-----------|---------------------|-----------------|
| Phase 3 | fixed | no | no | RDy26 v1.0 |
| Phase 4 | evolving via `[G-4]`,`[G-5]` | yes | no | FG21 |
| not scoped | evolving | yes | yes | — |

Neither reference model implements the momentum feedback. FG21 flags this as the
assumption most likely to fail for very energetic flows that "could easily have
debris flow-like behaviors". Fluid density is held at ρ regardless of `c_s` in all
scoped phases.

---

## 3. Bed shear stress — the reconciled closure

**This is the most important section in the document.** Every erosion and deposition
rate depends on `τ_b`. The shipped `anugaSed` uses a *different but documented*
closure — see §3.4 — which this specification recommends superseding.

### 3.1 The quadratic drag law

FG21, RDy26 and P14 all specify the same quadratic drag law, in different notation:

```
[T-1]   τ_b = ρ · f_c · |v|²           where f_c ≡ f/8
```

| Source | As written | Equivalent to |
|--------|-----------|---------------|
| FG21 Eq 1 | `τ = ρ f_c v²`, "f_c is the Darcy–Weisbach friction factor divided by 8" | `[T-1]` |
| P14 Eq 3.7 | `τ* = u*²/(R g d_g)` with `u* = \|v\|√(f/8)` | `[T-1]` via `[T-3]` |
| RDy26 A5–A7 | `C_D = g n² h⁻¹ᐟ³` ; `τ_b = ρ_w C_D (u² + v²)` | `[T-1]` with `f_c = C_D` |
| aSM16 Eq 6 | `τ_b = ρ_w u*²` | `[T-1]` via `[T-2]`, but `u*` from `[T-7]` not `[T-2]` |

Shear velocity:

```
[T-2]   u* = √(τ_b/ρ) = |v| √(f_c) = |v| √(f/8)
```

Shields stress `[FG21 2]`, `[P14 3.7]`:

```
[T-3]   τ* = τ_b / ((ρ_s − ρ) g D) = u*² / (R g D)
```

Excess Shields stress `[FG21 3]`:

```
[T-4]   τ_x = τ* − τ_c*
```

> **Typographic note on P14 Eq 3.7.** The thesis prints `u* = √(f/8)`, omitting the
> velocity. `u*` must be a velocity and `√(f/8)` is dimensionless, so the intended
> form is `[T-2]`. FG21 Eq 1 and RDy26 A7 both confirm the intent.

### 3.2 Velocity regularisation

Velocity must be recovered from momentum with the standard ANUGA depth-limiting
form to avoid division by near-zero depth. RDy26 A22–A23 explicitly adopts
"the ANUGA-style form (Mungkasi & Roberts, 2012)":

```
[T-5]   u = (uh)·h / (h² + h_ε²)        v = (vh)·h / (h² + h_ε²)
```

with `h_ε` a regularisation depth (`aS16` uses 1×10⁻⁶). Set `u = v = c_s = 0`
below the wet/dry threshold.

### 3.3 Friction closures

`f_c` varies **per cell, per timestep** as a function of the evolving flow depth.
It must be recomputed inside the sediment kernel, not read once from Manning's `n`.
The Manning ↔ Darcy–Weisbach bridge `[FG21 9]` is:

```
[T-6]   f = 8 g n² / h^(1/3)        ⟺   f_c = C_D = g n² h^(−1/3)
```

which is exactly RDy26 A5. Three selectable closures `[FG21 §2.2.3]`:

| Mode | Definition | `n` varies? |
|------|-----------|-------------|
| `constant` | user supplies `n` | no |
| `larsen_lamb` | constant `n` from bed roughness parameters `k_s`, `r_d`, `r_br`, `σ_br` — Larsen & Lamb (2016) Eqs 3–4 | no |
| `wilson` | `f` as an empirical function of grain size and depth — Wilson et al. (2004) Eqs 13–15; `n` then follows from `[T-6]` | yes, spatially and temporally |

**In all three modes `f_c` still varies per cell per timestep**, because `[T-6]`
depends on `h`. This is the coupling most easily missed.

### 3.3.1 ⚠ Friction-factor conventions — a factor-of-2 and factor-of-8 trap

Three conventions appear across these sources, all describing the same physics.
**Getting this wrong silently scales every transport rate.**

| Convention | Definition | Relation |
|------------|-----------|----------|
| Darcy–Weisbach `f` | `τ₀ = ρ f V²/8` | — |
| **This spec, `f_c`** (= FG21, RDy26 `C_D`) | `τ₀ = ρ f_c V²` | `f_c = f/8` |
| Engelund–Hansen `f_EH` `[EH67 3.1.3]` | `I = f_EH V²/(2gD)` | `f_EH = f/4 = 2 f_c` |

W04 explicitly warns of this: "some theoretical treatments adopt factors other than 8
in the above expression and thus quote systematically different, but still
dimensionless, values for the friction factor." Convert before substituting.

### 3.3.2 `wilson` mode — `f_c` by bed type  `[W04 13–17]`

W04 give `f_c` for five bed types. `R` is hydraulic radius (`R ≈ h` for wide
channels); `D50`/`D84`/`D90` are bed clast percentiles.

```
[T-8]    Sand bed         (8/f_c)^½ = 8.46 (R/D50)^0.1005
[T-9]    Gravel bed       (8/f_c)^½ = 5.75 log₁₀(R/D84) + 3.514
[T-10]   Boulder bed      (8/f_c)^½ = 5.62 log₁₀(R/D84) + 4.0
[T-11]   Steep pool-fall  (8/f_c)^½ = 4.60 log₁₀(d_s/D90) + 4.203
[T-12]   Fixed roughness  (8/f_c)^½ = 5.657 log₁₀(R/r) + 6.6303
```

FG21 cite Eqs 13–15, i.e. `[T-8]`–`[T-10]`. `[T-11]` uses `d_s` = depth of water
*plus* sediment; `[T-12]` uses roughness element size `r`. Velocity follows from
`[W04 4]`: `U = ((8gRS)/f_c)^½`, which is `[T-1]` rearranged.

> ### ⚠⚠ `f_c` ABOVE IS **NOT** THIS SPEC'S `f_c`.
>
> `[T-8]`–`[T-12]` reproduce W04's symbol verbatim, and **W04's `f_c` is the
> Darcy–Weisbach `f`, not `f/8`.** Their Eq 4, quoted immediately above, is
> `U = ((8gRS)/f_c)^½` — the standard Darcy–Weisbach velocity, which carries `f`.
> This spec's `f_c` is defined in §3.3.1 as `f/8`. The two collide on one symbol,
> which is precisely the trap §3.3.1 exists to warn about — and §3.3.2 walked
> into it by adopting W04's notation without converting.
>
> **The conversion, and a convenient cancellation.** Writing `X` for the
> right-hand side of `[T-8]`–`[T-12]`:
>
> ```
>     X = (8/f_W04)^½   with  f_W04 = f      ⟹   f = 8/X²
>     f_c(this spec) = f/8                   ⟹   f_c = 1/X²
> ```
>
> The eights cancel: **`f_c = 1/X²`**. Taking W04's `f_c` as ours would make
> `f_c` — and therefore `τ_b`, and therefore every transport rate — **8× too
> large**.
>
> **Confirmed against W04's own headline result.** They derive
> `n = 0.0545 s m⁻¹ᐟ³` for Martian channels. Under `[T-6]`,
> `n = √(f_c h^⅓ / g)` with *this spec's* `f_c`. Taking a sand bed at
> `R/D50 = 1000` and Martian `g = 3.71 m s⁻²`, `X = 8.46 × 1000^0.1005 = 16.94`:
>
> | reading | `f_c` | depth giving `n = 0.0545` |
> |---------|------:|--------------------------:|
> | **correct**, `f_c = 1/X²` | 0.00349 | **31.6 m** — an outflow channel |
> | literal, `f_c = 8/X²` | 0.02788 | 0.062 m — not an outflow channel |
>
> Martian outflow channels are tens of metres deep, so the correct reading is
> the one that reproduces W04's own number.

These are simplifications of W04's fuller forms, obtained by fixing weakly-varying
terms at Martian averages (slope `S`, clast spread `σ_g`, and the `a` term in the
gravel formula). W04 quantify the resulting error as ~2%, <5% and ~13% respectively.
**On Earth those fixed values may not apply** — for terrestrial work prefer W04's
unsimplified Eqs 7a/7b, 8–9, 10, 11 (also in §14's source).

### 3.3.3 `larsen_lamb` mode — Manning–Strickler  `[LL16 Methods]`

```
[T-13]   u/u* = 8.1 (h/k_s)^(1/6)              Manning–Strickler
[T-14]   n = k_s^(1/6) / (8.1 √g)              rearranged with the Manning equation
[T-15]   k_s = r_d · r_br · σ_br               bedrock roughness length scale
```

`r_d` and `r_br` are hydraulic roughness scaling parameters and `σ_br` is one
standard deviation of bedrock bed elevation. LL16 set `r_d = 2`, `r_br = 2`, and
measured `σ_br ≈ 5 m` across five reaches, giving `k_s = 20 m` and:

```
n = 20^(1/6)/(8.1 × 9.81^½) = 0.0649 ≈ 0.065     ✓ matches LL16's stated value
```

`n` is then **spatially and temporally uniform**; only `f_c` varies, through `[T-6]`.

> **LL16 used ANUGA.** Their Methods state "ANUGA implements bed friction with
> Manning's roughness coefficient (n)" and that ANUGA generated their triangular
> mesh from a USGS 10 m DEM. This is a *Nature* paper built on this codebase — worth
> knowing both as provenance for `[T-13]`–`[T-15]` and as a citable precedent for
> ANUGA in megaflood work. Note `σ_br` is a **site-measured** quantity: 5 m is Moses
> Coulee, not a universal default.

> **Both non-constant closures are megaflood parameterisations, not general-purpose
> terrestrial flood closures.** Wilson et al. (2004) is a *Mars
> outflow channel* study — its `n = 0.0545 s m⁻¹ᐟ³` is derived for Martian channels
> under Martian gravity, and is FG21's constant-`n` default for exactly that reason.
> Larsen & Lamb (2016) is the Channeled Scablands, a terrestrial *megaflood*. Neither
> is calibrated for ordinary river or urban flood modelling, which is ANUGA's main
> use. **For standard flood work, `constant` with an `n` from conventional tables
> remains the right default**; treat `wilson` and `larsen_lamb` as specialist modes
> for planetary and megaflood applications and label them as such in the API.

> **Open item F1.** Wilson et al. (2004) Eqs 13–15 and Larsen & Lamb (2016) Eqs 3–4
> are not reproduced in FG21's main text (they are in Supporting Information S3).
> Transcribe from the primary papers — both are now fully identified in §14 — before
> implementing those modes. `constant` is sufficient for Phases 1–3.

### 3.4 The depth–slope closure used by anugaSed

`aSM16` Eq 7 specifies a different shear velocity — the **depth–slope product**:

```
[T-7]   τ_b = ρ g h S       ⟹   u* = √(g S h)              [aSM16 6–7]
```

This is a documented closure, not undocumented drift. It is the
**steady uniform (normal) flow** approximation:
it assumes the energy slope equals the bed slope and that the flow is locally in
equilibrium.

**It should nonetheless be superseded by `[T-1]`, for three reasons:**

1. **Validity.** Normal-flow equilibrium is exactly what does not hold in a
   dam-breach, outburst or flash flood — the cases this add-on exists to model.
   `[T-1]` makes no equilibrium assumption.
2. **`S` is taken from the bed, not the energy grade line.** `aSM16` says "`S` is the
   local slope of the bed". The energy slope `S_f = f_c|v|²/(gh)` is the correct
   quantity, and substituting it into `[T-7]` recovers `[T-1]` identically. The
   commented-out energy-slope block in `aS16` shows this was recognised.
3. **The clamp is not in the manual.** `aS16` applies `S ← min(S, mean(S)/2)`, a
   domain-global rescaling with no counterpart in `aSM16` and no physical basis.
   Under `[T-1]` it is unnecessary.

Retain `[T-7]` behind a `legacy_depth_slope` flag for reproducing published
anugaSed results. Default to `[T-1]`.

---

## 4. Suspended sediment

### 4.1 Entrainment — two published formulations

The sources do **not** agree here, and the disagreement is physical rather than
typographic. Both formulations should be implemented behind a selector.

**(a) Saturating reference-concentration form** — Smith & McLean (1977), Parker
(1998), as used by `[FG21 7]`:

```
[E-1]   E* = 0.65 · γ₀ S / (1 + γ₀ S)        S = τ*/τ_c* − 1
[E-2]   E  = v_s · E*
```

with `γ₀ = 0.0024` (empirical) and `τ_c* = 0.04` (FG21's choice for suspension).
`E*` is a dimensionless near-bed reference concentration; 0.65 is the maximum
packing fraction, so `E*` saturates rather than growing without bound. `E` is
recovered as a flux by multiplying by the settling velocity `[E-2]`.

**(b) Cohesive / fine-grained form** — Hanson (1990), Hanson & Simon (2001), as
specified by `[aSM16 3–5]` and implemented in `aS16`:

```
[E-3]   Ė = K_e (τ_b − τ_c)                             DIMENSIONAL stress, Pa
[E-5]   K_e = 0.2×10⁻⁶ / τ_c^0.5                        [m³ N⁻¹ s⁻¹]
[E-6]   τ_c = τ_c* (ρ_s − ρ) g D50
```

`[E-5]` is the jet-test erodibility relation of Hanson & Simon (2001) for cohesive
streambeds, `k_d = 0.2 τ_c^(−0.5)` in cm³ N⁻¹ s⁻¹; the `10⁻⁶` converts cm³ to m³.

**(c) Partheniades form on dimensional stress** — as used by `[RDy26 A8]`, per bed
layer ℓ:

```
[E-4]   E_ℓ^pot = K_e,ℓ · (τ_b − τ_c,ℓ)/τ_c,ℓ    for τ_b > τ_c,ℓ ; else 0
```

Note `[E-4]` and `[E-3]` both use *dimensional* stress; `[E-1]` uses Shields stress.

> **On the dimensions of `K_e`.**
> Reading P14 alone suggests a dimensional inconsistency in `K_e`. It is an
> artefact of that reading. P14 Eq 3.6 applies
> `K_e` to *Shields* stress (dimensionless), which cannot balance; `aSM16` Eq 3
> applies it to *dimensional* stress, which balances exactly:
>
> ```
>   [m³ N⁻¹ s⁻¹] × [N m⁻²]  =  m s⁻¹   ✓
> ```
>
> The shipped code implements `[E-3]`/`[E-5]` correctly and with proper provenance.
> **No action needed.** P14 Eq 3.8 (`K_e = K_e* d_g √(Rgd_g)`, `K_e* = 12` after
> Wilson 1966) is a different, non-cohesive parameterisation that the manual
> abandoned; do not implement it.

### 4.1.1 Which erosion law? — a regime choice, not a preference

`[E-1]` and `[E-3]` are **not competing formulations of the same physics**. They
describe different sediment:

| | `[E-3]` cohesive | `[E-1]`/`[K-1]` non-cohesive |
|---|---|---|
| Sediment | silt, clay, cohesive bank material | sand, gravel, boulders |
| Threshold physics | inter-particle cohesion, jet-test calibrated | grain weight, Shields entrainment |
| Calibration source | Hanson & Simon (2001) | Smith & McLean (1977) / Wong & Parker (2006) |
| Validation case | Rio Puerco (rung 8) | Mars crater breach (rung 7) |
| Typical `D` | < 0.1 mm | mm to cm |

Selecting between them is a **statement about the bed material**, and the API should
present it that way (`bed_material='cohesive'|'noncohesive'`) rather than as an
opaque numerical switch. Getting this wrong is a physics error, not a tuning error.

### 4.2 Settling velocity

Two published options; they differ in provenance, not in intent.

**Ferguson & Church (2004)** — used by `[P14 3.10]` and implemented in `aS16`:

```
[S-1]   v_s = R g d_g² / ( C₁ ν + √(0.75 C₂ R g d_g³) )
```

with `C₁ = 18`, `C₂ = 0.4` for smooth spheres (1.0 and 1.1 respectively for natural
grains — expose both). This form is smooth across the Stokes → turbulent transition
and is cheap, branch-free and GPU-friendly.

**Dietrich (1982) Eqs 5–9** — used by `[FG21 §2.2.2]` via Supporting Information S1.
A polynomial fit in the dimensionless particle size; more accurate for natural
irregular grains, but requires transcription from Dietrich or FG21 S1.

**Recommendation:** implement `[S-1]` as the default. It is fully specified here,
dimensionally clean, and already the basis of the existing code. Add Dietrich as an
option only if rung 7 (crater breach) needs it to reproduce FG21. Note DL09 themselves
take settling velocities from Dietrich (1982).

> **`[S-1]` numerically verified.** P13 §[54] reports `v_s = 0.00175 m s⁻¹` for
> 0.045 mm quartz using Ferguson & Church (2004). Evaluating `[S-1]` at
> `D = 4.5×10⁻⁵ m`, `R = 1.65`, `g = 9.81`, `ν = 10⁻⁶`:
>
> | `C₁` | `C₂` | `v_s` | error |
> |------|------|-------|-------|
> | **18** | **0.4** | **0.00176** | **+0.3%** |
> | 18 | 1.0 | 0.00172 | −1.7% |
> | 20 | 1.1 | 0.00155 | −11.2% |
>
> This confirms both the transcription of `[S-1]` and that the smooth-sphere
> constants `C₁ = 18`, `C₂ = 0.4` are the ones in use. Use this as a unit test.

### 4.3 Rouse number and the near-bed concentration profile

```
[S-2]   Z = v_s / (κ u*)                    κ = 0.41
[S-3]   c_b = c · d*(Z)
```

`d*` is the ratio of near-bed to depth-averaged concentration. Physically: at low
`Z` sediment is well mixed through the column (`d* → 1`, washload); at high `Z` it
concentrates near the bed (`d* ≫ 1`).

The sources diverge:

| Source | `d*` treatment |
|--------|---------------|
| P14 Eq 3.9 | `D* = 1` — uniform suspension assumed, citing Nordin (1963), Griffin et al. (2014) |
| **aSM16 Eq 10** | **the defining Rouse–Vanoni integral, from Davy & Lague (2009)** |
| FG21 Fig 2 / S2 | `d*` a function of `Z`, plotted; numeric form only in Supporting Information S2 |
| aS16 | 8-term polynomial in `Z`, plus a linear branch for `Z > 4` — a *fit* to `[S-4]` |

**The defining expression** `[aSM16 10]`, after Davy & Lague (2009), obtained by
requiring that sediment discharge be the depth integral of concentration × velocity
over a Rouse–Vanoni concentration profile and a logarithmic velocity profile:

```
                 ∫ₐʰ ln(z/z₀) dz
[S-4]   d*  =  ─────────────────────────────────
               ∫ₐʰ ((h−z)/(h−a) · a/z)^Z ln(z/z₀) dz
```

with `Z` the Rouse number `[S-2]`, `z₀` the roughness length and `a` the reference
height near the bed.

> **The profile factor is misprinted in DL09 as published.**
> `[DL09 19]` prints it as `((z−a)/(h−a) · a/z)^Z`. That is wrong, and
> DL09's own paper contains the proof.
>
> *Internal contradiction in DL09.* Immediately above their Eq 19 they write the
> sediment flux as
>
> ```
>     q_S = ∫ₐʰ c_s(z) u(z) dz = c_S(a) ∫ₐʰ [ … ]^Z u(z) dz
> ```
>
> Factoring `c_S(a)` out requires the bracket to equal **1** at `z = a`. The
> printed factor `(z−a)/(h−a) · a/z` is **0** there, which would make
> `c_s(a) = 0 ≠ c_S(a)`. The reference height is by definition where the near-bed
> concentration is evaluated, so the profile cannot vanish there.
>
> *The intended expression.* The Rouse–Vanoni profile
> `c(z)/c_a = [(h−z)/z · a/(h−a)]^Z` rearranges **exactly** to
> `((h−z)/(h−a) · a/z)^Z`, which is 1 at `z = a` and 0 at `z = h`. So `(z−a)` is a
> slip for `(h−z)`. The model was never in doubt — only one glyph.
>
> *It is not a small numerical difference.* At `z₀/h = 10⁻⁴`:
>
> | `Z` | as printed `(z−a)`, `a/h`=0.005 | corrected `(h−z)`, `a/h`=0.005 |
> |-----|-------------------------------:|-------------------------------:|
> | 0.10 | 1.28 | 1.29 |
> | 0.50 | 14.3 | 11.1 |
> | 2.00 | **41 227** | **356** |
>
> Both forms satisfy DL09's Figure 4 caption for `Z < 0.1` (they agree closely
> there), so the caption alone does not discriminate. The `Z = 2` column does:
> `d* ≈ 4×10⁴` is not a value that appears on their plotted figure, whereas the
> corrected form stays in a plottable range. **DL09's Figure 4 was evidently
> computed with the correct profile; their Eq 19 as typeset is a typo.**
>
> Everything downstream in this document, and the implementation, uses the
> corrected form.

**`[S-4]` is confirmed as `[DL09 19]`** — the manual's Eq 10 reproduces it exactly.
DL09 derive it by assuming a Rouse–Vanoni concentration profile and a logarithmic
velocity profile, and integrating `c_s(z)u(z)` from a reference height `a` to `h`.

**DL09's own assessment of the expression** (their §4.1, Figure 4):

> "The above expression depends mostly on `Z`, and only slightly on the ratios
> `a/h` and `z₀/h`."

and from the Figure 4 caption:

> "`d*` is always larger than 1 and smaller than 3 for Rouse numbers `Z` smaller
> than 0.1. For `Z` between 0.1 and 2, `d*` increases rapidly mainly as a function
> of `a/h`."

**Documented implementation choices in `aSM16`** — these are approximations, and
each is a candidate for improvement:

- `z₀ = D50/30`  (standard hydraulically-rough wall)
- **flow depth assumed to be 1 m when evaluating the integral**, irrespective of the
  actual local depth
- `d*` recomputed only **once every 10 timesteps**, for speed

Physical expectation stated in `aSM16`: `d*` lies between 1 and 3 for `Z < 0.1`, is
close to 1 for large rivers or fine particles, and becomes ≫ 1 for small rivers with
coarse particles where entrainment is largely bedload.

> **On the shipped polynomial.**
> It is a fit to `[S-4]`. Its fixed 1 m depth is a limitation rather than a
> defect:
>
> - `d*` depends *mostly on `Z`*, and `Z` is computed per cell from the actual `u*`.
>   The fixed depth enters only through the weakly-dependent ratios `a/h` and `z₀/h`.
>   Fitting `d*` as a function of `Z` alone is therefore what DL09's own analysis
>   supports, and the shipped polynomial is a **defensible** engineering choice.
> - The caveat is the mid-range: for `0.1 < Z < 2`, DL09 state `d*` "increases
>   rapidly mainly as a function of `a/h`". A single-depth fit is weakest precisely
>   there. **Quantify `d*` sensitivity to `a/h` across the `Z` range of interest
>   before deciding whether a depth-dependent fit is warranted.**
> - `a`, the near-bed reference height, is a required input to `[S-4]` that `aSM16`
>   does not state. Recover it from the code, document it, and expose it.
>
> Remaining engineering (tracked as **S1a**): `[S-4]` is a 1-D quadrature per cell,
> far too expensive inside a GPU kernel every step, so a fitted form is the right
> answer. Regenerate the fit from `[S-4]`, document its `Z` range, and make
> out-of-range behaviour explicit rather than extrapolating a polynomial. Keep
> `d* = 1` available as the P14/P13 limiting case and the 10-timestep refresh as a
> tunable rather than a hard-coded stride.

> **Implemented and tested**; see
> `test_rouse.py` (12/12) and `core_rouse_d_star()`.
>
> **S1b first, because it changes S1a.** The `a/h` sensitivity, from the
> corrected `[S-4]` at `z₀/h = 10⁻⁴`:
>
> | `Z` | `a/h`=0.01 | 0.02 | 0.05 | 0.10 | spread |
> |-----|-----------:|-----:|-----:|-----:|-------:|
> | 0.05 | 1.268 | 1.226 | 1.174 | 1.139 | 1.11× |
> | 0.25 | 3.064 | 2.598 | 2.109 | 1.821 | 1.68× |
> | 1.00 | 34.237 | 19.654 | 9.967 | 6.270 | **5.46×** |
> | 4.00 | 507.489 | 225.822 | 80.775 | 38.604 | **13.15×** |
>
> `z₀/h` over two decades moves `d*` by 1.01×–1.14× — genuinely weak, exactly as
> DL09 say. **`a/h` is not.** So the Draft-3 conclusion above, that a `Z`-only fit
> is what DL09's analysis supports, holds only for **small `Z`**. Above `Z ≈ 0.25`
> it fails, and DL09's other remark — that for `0.1 < Z < 2`, `d*` "increases
> rapidly mainly as a function of `a/h`" — is the one that governs the range of
> interest. **A `Z`-only fit is not adequate; the fit must take `a/h` too.**
>
> **S1a.** Fitted in both variables. Using the structure of the integral rather
> than fitting blind is what makes it cheap enough for a kernel: near the bed the
> Rouse profile behaves like `z^(−Z)`, so `d*` diverges roughly as `(a/h)^(−Z)`.
> Factoring that out,
>
> ```
>       ln d*  =  −Z ln(a/h)  +  P(Z, ln(a/h))
> ```
>
> leaves a mild remainder that a low-order polynomial captures well:
>
> | form | terms | max err |
> |------|------:|--------:|
> | direct polynomial in `(ln Z, ln(a/h))`, `Z ≤ 4`, `a/h ≥ 0.002` | 36 | 10.0% |
> | direct polynomial, suspension range | 18 | 3.1% |
> | **structural form above** | **15** | **0.84%** |
>
> Fitted range `Z ∈ [0.01, 2.5]`, `a/h ∈ [0.01, 0.15]`. Beyond `Z ≈ 2.5` transport
> is essentially bedload and this ratio is the wrong model. Out-of-range inputs are
> **clamped, not extrapolated**. `d* = 1` remains available as the P14/P13 limiting
> case (`sediment_d_star_mode = 0`, still the default).
>
> **`a` is exposed and remains a judgement call.** `aSM16` require it and never
> state it, and it was not recovered from their code — the default used here
> (`a = 2 d_g`, with the standard van Rijn floor `a ≥ 0.01 h`) is **ours**. Given
> the 13× spread above this is a first-order modelling choice, not a detail, and
> recovering `aSM16`'s actual value remains worth doing before any validation run
> that claims to reproduce them.
>
> The fixed-1 m-depth and 10-timestep-refresh approximations of `aSM16` are both
> now moot: `d*` is evaluated per cell per step from the local `h` and `u*`, and
> costs nothing measurable (Ns=0 gate unchanged, and the evaluation is a Horner
> in `Z` over coefficients quadratic in `ln(a/h)`).

### 4.3.1 Transport length — a scale that constrains model setup

DL09's central quantity is the **disequilibrium (transport) length** `ξ`: the mean
distance a particle travels in the flow before being trapped on the bed.

```
[S-5]   ξ = q / (d* v_s)                                 [DL09 8]
```

It is the length over which the sediment load relaxes toward capacity. `ξ → 0`
recovers transport-limited (local-capacity) behaviour; `ξ → ∞` recovers
detachment-limited. **The explicit `E − D` balance of `[G-3]` exists precisely
because `ξ` is finite** — the flow is *not* instantaneously at capacity.

Evaluated for the Rio Puerco 2006 flood from P13's own parameters
(`q = 2.66 m² s⁻¹`, `d* = 1`, `v_s = 0.00175 m s⁻¹`):

```
ξ ≈ 1520 m  —  about 6.5× the 235 m mean arroyo width
```

DL09 note the same generally: "except for coarse sand, the disequilibrium length is
about larger than the river width".

**Two practical consequences:**

1. **Mesh and domain sizing.** A domain shorter than a few `ξ` cannot develop the
   deposition pattern; a cell much larger than `ξ` makes the local-equilibrium error
   irrelevant, while `Δx ≪ ξ` is where the explicit balance earns its keep. Report
   `ξ` at setup as a diagnostic — it is cheap and immediately tells a user whether
   their domain resolves the process they are modelling.
2. **A sanity check on results.** Deposition concentrated within one or two cells of
   a source, when `ξ` spans hundreds of cells, indicates a bug rather than physics.

### 4.4 Deposition

```
[D-1]   D = c_b · v_s = d*(Z) · c · v_s          [P14 3.9], [FG21 §2.2.2]
```

RDy26 A12 uses a different, threshold-based form with a critical *deposition* stress:

```
[D-2]   D_s = w_s c_s (1 − τ_b/τ_d,s)     if τ_d,s > 0 and τ_b < τ_d,s ; else 0
```

Setting `τ_d,s = 0` disables deposition entirely — RDy26 uses this for their passive
transport benchmarks, which is a useful test hook worth preserving.

**Recommendation:** `[D-1]` as default (consistent with the `d*` machinery of §4.3),
`[D-2]` available for the layered bed model and required to reproduce RDy26's
passive-transport validation cases.

⚠ When `d* ≠ 1`, `[D-1]` **must** be evaluated with the near-bed ceiling `[L-4]`
of §4.5(d). Without it the equilibrium `d*` makes the deposition rate diverge as
shear vanishes.

### 4.5 Limiters

Four distinct limiters, easily confused. All four are needed.

**(a) Positivity — physical, mandatory** `[RDy26 9]`:

```
[L-1]   F_s^net = E_s − D_s  ≥  − m_s / Δt
```

Deposition can never remove more suspended sediment than is present. This is a
hard physical constraint, not a stability patch, and it replaces `aS16`'s ad-hoc
concentration clamp.

**(b) Maximum concentration — physical ceiling** `[FG21 §2.2.2]`:

```
[L-2]   c_s ≤ c_max                     FG21: c_max = 0.30 by volume
```

FG21 notes this matters mainly for the finest grain sizes (1–2 mm in their runs).
`aS16` uses 0.2. Expose it; do not hard-code.

**(c) Maximum bed change rate — numerical, diagnostic** `[FG21 §2.2.1, Table 2]`:

```
[L-3]   |∂z/∂t| ≤ max_dz               FG21: 5 m/s (deliberately non-physical)
```

FG21 is explicit that this is a stability device set "nonphysically large" so it
"seldom comes into effect", and that when it does fire it acts as an artificial
reduction in transport efficiency. Requirements for our implementation:

- user-settable, default deliberately large;
- **count and report firings** — a run where `[L-3]` fires often is not trustworthy;
- rescale consistently (FG21 rescale erosion in *all* changing cells, preserving
  relative pattern) rather than clipping cell-by-cell.

**(d) Near-bed concentration ceiling — physical, mandatory when `d* ≠ 1`**
**[not in the source literature]**:

```
[L-4]   c_b = d*(Z) · c_s  ≤  c_pack               c_pack = 0.65
```

`[D-1]` is `D = c_b v_s`, and **no source bounds `c_b`**. It needs bounding, for a
reason that only appears once `[S-4]` is used inside a transient solver.

`d*` is derived from the **equilibrium** Rouse–Vanoni profile, which presupposes a
suspension maintained by shear. As shear vanishes that premise fails: `u* → 0`, so
`Z = v_s/(κu*) → ∞` `[S-2]`, and `d*` grows without bound — in the fitted
implementation it saturates at its range clamp, `d* ≈ 250` at `a/h = 0.01`. The
deposition rate `d* c v_s` then becomes enormous *precisely where the physical
settling rate should be at its most ordinary*.

Observed before the bound was added: a tilted lake starting from rest, `d_g = 0.1 mm`
(`v_s = 8.0×10⁻³ m s⁻¹`, `h = 1 m`), deposited its **entire** suspended load in under
one second. The physical timescale is `h/v_s ≈ 125 s`. At `t = 1 s` the median Rouse
number was 2.34 while `max|u|` was still 0.11 m s⁻¹ — the flow had barely started.

`c_b` is a concentration, so it cannot exceed maximum packing. `0.65` is the same
constant that bounds `E*` in `[E-1]`, where it appears for exactly this reason, so
`[L-4]` introduces no new parameter. With the bound the same case removes 26.0% in
the first second, matching the packing-limited rate `c_pack v_s / m` to two figures.

Scope and interactions:

- **Inactive in the well-mixed limit.** With `d* = 1`, `c_b = c_s ≤ c_max = 0.30`
  by `[L-2]`, comfortably below `c_pack`. `[L-4]` therefore changes nothing in the
  P14/P13 `d* = 1` configuration, and only engages at moderate-to-high `Z`.
- **Does not conflict with `[L-1]`.** `[L-4]` bounds the deposition *rate*;
  `[L-1]` bounds the *net source* by what is present. `[L-4]` applies first and
  can only reduce a removal, so `[L-1]` remains the binding positivity guarantee.
- **`c_pack` should be exposed**, defaulting to 0.65, alongside `c_max`.

> **Provenance.** `[L-4]` is **not** in P14, FG21, RDy26, DL09 or `aSM16`. It is
> required by the combination of an equilibrium profile (`[S-4]`) with a transient
> solver, which none of those sources does in this form: DL09 derive `d*` for a
> steady sediment-discharge relation, and `aS16` evaluate it at a fixed 1 m depth
> every 10 steps. Flagged here rather than buried in code, because it is a
> modelling addition and future validation against FG21 or `anugaSed` must know it
> is present.

**(e) Non-erodible base — physical, optional**
**[not in the source literature]**:

```
[L-5]   z(x, y, t) ≥ z_base(x, y)            z_base a per-cell field, default −∞
```

Erosion in `[G-4]`/`[G-5]` is otherwise unbounded below: the bed lowers for as long
as the flow can lift material. That is right for a deep alluvial bed and wrong
wherever the erodible layer is finite — a reach floored by an outcrop, a lined
culvert, a dam apron, a soil layer of known depth over rock. `z_base` is a **field,
not a constant**, because bedrock is a surface.

Applied to the **source**, never by clamping `z`. The erodible thickness
`T = z − z_base` bounds the net removal over a step, so for the suspended exchange

```
        Σ_s F_s^net  ≤  T (1 − λ) / Δt
```

and the erosive part of the source is scaled to satisfy it. Sediment that is not
eroded never enters the water column, so `[G-3]` and `[G-4]` still use the same
limited source and the budget closes exactly, as it does for `[L-1]` and `[L-2]`.
Clamping `z` afterwards would instead leave suspended sediment that came from
nowhere — the same failure mode as the `[L-1]` sign bug (spec 12, D-obs).

Scope and interactions:

- **Shared across classes, proportionally.** `T` is a property of the cell, not of a
  class, so the classes are limited *together*: their erosive sources are scaled by
  one common factor. Serving them in registration order would make the answer depend
  on the order `add_sediment_class` was called, which is not physics. The bed carries
  no per-class stratigraphy in this model, so no class has a stronger claim on the
  last millimetre; proportional is the only choice that invents nothing. **If bed
  stratigraphy is ever added, this rule is what must change.**
- **Deposition is never scaled.** A shortage of bed material does not restrain
  deposition — deposition is what relieves it.
- **Ordered after `[L-1]` and `[L-2]`.** Those bound the source by what the water
  column holds and can hold; `[L-5]` bounds it by what the bed holds. It can only
  reduce an erosive source, so it cannot defeat `[L-1]`'s positivity.
- **Bedload `[K-3]`/`[G-5]` is limited differently, and less strictly.** Bedload is a
  divergence: clipping it cell-by-cell would create bed material, because one side of
  an edge would refuse to give up what the other has already received. The limit is
  therefore applied to the transport vector `q_b` and to whole edges — both of which
  the two cells sharing an edge evaluate identically, so the flux stays antisymmetric
  and `[K-3]` stays exactly conservative. The cost is that the floor is **not exact
  under bedload**: closing an edge for a cell that cannot pay also cancels its
  neighbour's inflow, so the deficit migrates one cell per sweep. Measured overshoot
  5.1×10⁻⁶ m on a 1.0×10⁻² m layer (14 cells of 960). Driving it to zero requires the
  exhaustion flag iterated to a fixed point with double buffering, so the result stays
  independent of thread order; **not implemented**.

**Region restriction.** *Where* the bed may erode is the same constraint as *how
deep*, so it is expressed through the same field rather than as a second
mechanism: a locked cell is one whose erodible thickness is zero, `z_base = z` at
the moment it is locked. It follows that locked means **unscourable, not inert** —
deposition still lands on it, and that new material is erodible again, being above
the base. Under erosive flow such a cell settles at exactly net zero, the limiter
scaling erosion back until it cancels deposition. This matches ANUGA's existing
region-based erosion operators in interface (the same `Region` polygon/circle/
indices arguments) while differing from them in kind: those clamp `z` directly,
which is sound only because they carry no sediment budget to violate.

> **Provenance.** `[L-5]` is **not** in P14, FG21, RDy26, DL09 or `aSM16`, none of
> which limits the erodible depth. It is an engineering requirement rather than a
> closure, and it is off by default, so every configuration in this spec is
> unaffected unless a base is set.

---

## 5. Bed layer model

From `[RDy26 §2.2, A1–A3]`. One **active layer** exchanging with the water column,
over up to 32 **substrate layers**. Per class per layer the state is mass per unit
area `M_ℓ,s` [kg m⁻²].

### 5.1 Initialisation

```
[B-1]   Σ_s f_s = 1                                    class fractions
[B-2]   M_ℓ,s(0) = f_s · C_ℓ · H_ℓ                     C_ℓ layer concentration, H_ℓ thickness
[B-3]   ρ_grain(0) = Σ_s f_s ρ_s                       bulk particle density
[B-4]   φ_ℓ = 1 − C_ℓ/ρ_grain                          layer porosity
```

### 5.2 Erosion through the stack

Layers erode **sequentially**: the active layer first; if it is exhausted within the
timestep, the remaining time is applied to the next substrate layer, and so on.

```
[B-5]   ΔM_ℓ^ero  = min( M_ℓ , E_ℓ^pot · Δt_ℓ )        Δt_ℓ = time layer ℓ is exposed
[B-6]   ΔM_ℓ,s^ero = ΔM_ℓ^ero · (M_ℓ,s / M_ℓ)          partition by composition
[B-7]   E_s = (1/Δt) Σ_ℓ ΔM_ℓ,s^ero
```

with `E_ℓ^pot` from `[E-4]` and `M_ℓ = Σ_s M_ℓ,s` `[RDy26 7]`.

Erosion is disabled in shallow water `[RDy26 A8]`:

```
[B-8]   h < h_min,ero = max(0.1 m, 10·h_wet)   ⟹   no erosion
```

where `h_wet` is the solver's wet/dry threshold. **This threshold matters** — it is
far more conservative than `aS16`'s `h > 0.1` gate and prevents spurious erosion in
thin films where velocity is poorly resolved.

### 5.3 Deposition and active-layer update

```
[B-9]    ΔM_A,s^dep = max(0, (E_s − F_s^net)·Δt)
[B-10]   H_ℓ = M_ℓ / (ρ̄_ℓ (1 − φ_ℓ))
[B-11]   ρ̄_ℓ = Σ_s (M_ℓ,s/M_ℓ) ρ_s
```

After exchange, if the active layer exceeds its prescribed thickness `H_A`, the
excess is transferred down to the first substrate layer; if thinner, mass is drawn
up until refilled or the substrate is exhausted. This conserves class-specific bed
mass locally and **does not itself change bed elevation**. Active-layer concentration
is updated by mixing with the donor layer `[RDy26 A16]`.

### 5.4 Memory

```
bytes = N_classes × N_layers × N_cells × 8
```

5 classes × 32 layers × 5×10⁶ cells = **6.4 GB** for the bed alone, before
hydrodynamics. The API must surface this estimate at setup and fail with a clear
message rather than a mid-run allocation failure. Cap `N_layers`.

---

## 6. Bedload

From `[FG21 §2.2.1]`, Eqs 1–6. Uses the shear stress chain of §3 directly.

```
[K-1]   q_b* = K · τ_x^m                                       [FG21 4]
[K-2]   q_b  = q_b* · √((ρ_s/ρ − 1) g) · D^1.5                 [FG21 5]
[K-3]   ∂z/∂t = −(1/(1−λ)) ∇·q_b                               [FG21 6]
```

Parameter sets for `[K-1]`:

| Set | `K` | `m` | `τ_c*` | Notes |
|-----|-----|-----|--------|-------|
| Wong & Parker (2006) Eq 24 | 3.97 | 1.5 | 0.0495 | MPM-tradition, reanalysed. Bedload only. Their Eq 23 is the alternative free-exponent fit (`K` = 4.93, `m` = 1.60, `τ_c*` = 0.0470) — confirm which FG21 used. |
| Engelund & Hansen (1967) `[EH67 4.3.5]` | `0.05/f_c` | 2.5 | 0 (no threshold) | **Total load** — includes suspension implicitly. See `[K-5]`. |

> **Critical usage rule** `[FG21 §2.2.2]`: the suspended operator is **not** applied
> when using Engelund & Hansen, because E&H already incorporates suspended
> transport. Running both double-counts. The API must enforce this.

**Engelund & Hansen total load** `[EH67 4.3.5]` —

EH67 write their transport relation as

```
        f_EH · Φ = 0.1 θ^(5/2)                                   [EH67 4.3.5]
```

where `Φ = q_T/√((s−1)g d³)` is dimensionless **total** sediment discharge, `θ` is
the Shields stress, and `f_EH` is EH67's own friction factor from their Eq 3.1.3.
Converting to this spec's convention using §3.3.1 (`f_EH = 2 f_c`):

```
[K-5]   q_b* = 0.05 · τ*^(5/2) / f_c                     total load, no threshold
```

so in the `[K-1]` power-law form, `K = 0.05/f_c` and `m = 2.5`, with `τ_c* = 0`.

> **Open item K1 — RESOLVED.** `K` is indeed friction-dependent, not constant, and
> the familiar literature coefficient `0.05` is EH67's `0.1` after the factor-of-2
> convention conversion. Two cautions carried forward:
>
> 1. **Do not apply a threshold.** `[K-5]` has none; subtracting `τ_c*` would be a
>    different model. This is also why EH67 must replace, not supplement, the
>    suspended operator.
> 2. **`θ` vs `θ'`.** EH67 derive `[K-5]` via the *effective* (skin-friction) bed
>    shear `θ'` — their Eq 4.2.4, `θ' − 0.06 = 0.4 θ²` — but the final empirical fit
>    4.3.5 is stated in terms of **total** `θ`, which is what `[K-5]` uses. Do not
>    mix the two.

**Vectorisation** `[FG21 §2.2.1]`: `q_b` is a scalar magnitude and must be
partitioned into components. FG21 follow Parker (1998) Eqs 2.11–2.12: **the sediment
transport vector is parallel to the boundary shear stress vector**. With `[T-1]`
that means parallel to `(u, v)`:

```
[K-4]   q_bx = q_b · u/|v|          q_by = q_b · v/|v|
```

> **Known defect to avoid.** FG21 report ~1% sediment mass conservation error in
> their bedload operator, "equivalent to a sediment volume lost by advection out of
> the domain without intermediate deposition", which they were unable to correct.
> Computing `[K-3]` as a **flux difference across shared edges** — the same
> `q_b·n·L` debited from one cell and credited to its neighbour — makes conservation
> structural. Do not compute `∇·q_b` from a reconstructed cell-centred gradient.

---

## 7. Angle-of-repose relaxation

From `[FG21 §2.2.4]`. Where the cell-to-cell bed slope exceeds a threshold angle
(FG21 use 35°), topography is "near-instantaneously diffused until it falls below
the threshold".

FG21 are explicit that this is **a numerical heuristic, not physics** — real bed
slope failures are advective. It exists to stop other parts of the model breaking on
over-steep slopes, and it has a side effect worth recording: it limits the steepness
of canyon walls and knickpoints, suppressing knickpoint retreat that may be real.

Implementation requirements:

- iterative neighbour diffusion — **the only non-cell-local sediment kernel**;
- MPI halo exchange **per sweep**;
- hard iteration cap, with a report when the cap is hit;
- conserve mass: material removed from over-steep cells is deposited on neighbours,
  never discarded.

### 7.1 Implementation status

Implemented as `core_apply_repose`, enabled by `domain.set_angle_of_repose(angle)`
and **off by default** — it is a heuristic, and it suppresses knickpoint retreat.

Three of the four requirements are met as written. Mass conservation is exact
(measured drift 2.3×10⁻¹³ m³ on 4.0×10² m³ of bed) and structural rather than
incidental: the transfer is computed per **edge** from data both cells share, so
both compute the identical volume,

```
        V = relax (|Δz| − tan θ_c d) / (1/A_k + 1/A_nb) / 3
```

and the pair balances by construction. This is the same device that keeps `[K-3]`
conservative, and it is where this differs from ANUGA's `sanddune_erosion_operator`,
which lowers an over-steep cell and lets the material vanish.

Sweeps are **Jacobi**, not Gauss–Seidel: each sweep reads elevation and writes
increments to a separate array, so no cell sees a neighbour that has already moved.
Reading live elevation would make the result depend on thread order and the two
compute modes would disagree.

Two properties found by measurement rather than assumed, both recorded because they
are not obvious from FG21's one-sentence description:

- **The `/3` above is a stability limit.** A cell has three edges and can be the
  donor on all of them, so without it a relaxation factor of 1 moves up to three
  times what any single edge intended — an explicit diffusion step past its
  stability limit. It overshoots, creates fresh over-steep edges, and oscillates.
- **Convergence is asymptotic and slow.** An over-steep cone (36.8°) needed **793
  sweeps** to reach a 30° limit from cold. Convergence is therefore declared within
  a 10⁻³ relative tolerance on the threshold slope; a strict test never terminates.
  In a running model the bed is already near-relaxed and a step needs a handful of
  sweeps, so the cap is for the pathological case. Hitting it is not fatal —
  progress carries over between timesteps — but it is reported, as required.

**Deviation from the requirements: the halo exchange is per TIMESTEP, not per
sweep.** The sweep loop lives inside the kernel so that it stays on the device;
lifting it into Python to exchange between sweeps would put a host round trip in
the middle of every sweep and drop the run off the GPU path entirely. The
consequence in parallel is that relaxation crosses a subdomain boundary one sweep
per timestep rather than one per sweep, so a slump spanning a boundary relaxes more
slowly there than it would in serial. Serial and single-subdomain results are
unaffected. Resolving this properly needs a device-side halo exchange callable from
within the sweep loop.

---

## 8. Vegetation drag

From `[aSM16 §4.2–4.3, Eqs 12–14]`, after Kean & Smith (2004) and Nepf (1999).
`aSM16` is fuller than P14 Eq 3.11 and should be preferred.

### 8.1 Drag force

```
[V-1]   F_D = ½ ρ C_D α U_ref²                          [aSM16 12]
[V-2]   α  = d_s / λ²                                   projected plant area per unit volume [m⁻¹]
```

`U_ref` is the flow velocity in the absence of vegetation, `d_s` the stem diameter,
`λ` the mean stem spacing.

> P14 Eq 3.11 omits `ρ` from `[V-1]`. `aSM16` Eq 12
> includes it. The manual's form is dimensionally correct and is the one to implement.

### 8.2 Velocity reduction

```
[V-3]   U = U_ref − F_D Δt                              [aSM16 13]
```

**Constraint stated in `aSM16`:** the drag force may reduce the flow velocity to
zero but must **not reverse its direction**. Clamp at zero; do not allow sign change.
This is a hard requirement, and an obvious source of instability if missed.

### 8.3 Drag coefficient as a function of stem density

For a sparse regular array of emergent cylinders `C_D = 1.2`. `aSM16` follows
Nepf (1999) to capture the effect of stem population density, via the force balance
`[aSM16 14]`:

```
[V-4]   (1 − αd) C_B U² + ½ C_D α d (h/d) U² = g h ∂h/∂x
```

where `C_B` is the bed drag coefficient. `aSM16` fits `∂h/∂x` as a cubic in `αd`
matched to Figure 6 of Nepf (1999), and solves `[V-4]` to obtain:

```
[V-5]   C_D = 1.2                                                   if αd ≤ 0.006
        C_D = 56.11(αd)² − 15.28(αd) + 1.3 − 5.465×10⁻⁴ (αd)⁻¹      if αd > 0.006
```

`C_D` is evaluated per cell whenever the stem-diameter or stem-spacing fields change.

> **Caveat.** `[V-5]` is a fit to a fit — a cubic fit to a digitised figure, fed
> through a force balance. It is reasonable within the range of Nepf's experiments
> and should not be extrapolated. Note the `(αd)⁻¹` term diverges as `αd → 0`, so the
> `αd ≤ 0.006` branch is doing real work, not just simplification. Guard the boundary.

Field values from P14 (Rio Puerco, sandbar willow and young tamarisk, after Griffin
et al. 2005): `d_s = 0.02 m`, `λ = 0.13 m`.

This operator is cell-local and drops directly into the L2 kernel pattern.

---

## 9. Numerical scheme

### 9.1 Tracer flux — upwind on the water flux

This is the L1 design, and `[RDy26 §2.3, 12–13]` specifies it exactly:

```
[N-1]   c_s^up = c_s,L   if  F̂_water ≥ 0
                 c_s,R   if  F̂_water < 0
[N-2]   F̂_s = F̂_water · c_s^up
```

RDy26 states plainly: *"This upwind sediment-transport scheme is conservative for
the transported sediment mass m_s = h c_s."*

Key ordering constraint `[RDy26 A4]`: the hydrodynamic flux is computed **first**,
from `h, u, v` only; sediment concentrations are used only afterwards. This
guarantees sediment cannot alter the hydrodynamics, which is what makes the one-way
coupling of Phase 3 exact rather than approximate.

For ANUGA the same principle applies to whichever Riemann solver is in use — take
the already-computed edge water flux and multiply by the upwind concentration. The
flux is shared between the two cells sharing the edge, so mass balance is exact
regardless of cell size ratio.

### 9.2 Semi-discrete update

`[RDy26 11]`, matching ANUGA's existing finite-volume form:

```
[N-3]   dU_i/dt = −(1/A_i) Σ_{e∈∂i} σ_i,e L_e F̂_e + S_i
```

### 9.3 Boundary conditions

`[RDy26 §2.3]`: apply hydrodynamic BCs to `h, hu, hv`; apply sediment BCs through
the **conserved** tracer `h·c_s`. A prescribed inflow concentration is converted to
`h·c_s` using the boundary depth. Dirichlet (prescribed concentration) is the
minimum required set.

### 9.4 Wet/dry

Below the wet/dry threshold set `u = v = c_s = 0` `[RDy26 A21–A23]`. Note the
separate, more conservative erosion threshold `[B-8]`.

### 9.5 Well-balancedness

RDy26 added hydrostatic reconstruction specifically because sediment amplifies
hydrodynamic error: *"small hydrodynamic errors can be amplified through
shear-stress-dependent erosion and deposition"* — erosion goes as `τ^m` with
`m` up to 2.5. Since `τ ∝ v²`, transport goes as `v^2m` — at `m = 2.5` that is `v⁵`,
so a 10% velocity error becomes a **61%** transport error (1.1⁵ ≈ 1.61). Verify
ANUGA's existing reconstruction is adequate under an erodible bed as part of Phase 4.

`LM15` is the constructive counterpart to that warning, and on the same mesh
topology (triangular, unstructured). It shows that once `Z` becomes
time-varying the bed-slope term needs its **own** discretisation, matched to the
reconstruction, for the lake-at-rest state to stay exact — a fixed-bed
well-balanced scheme does not remain well-balanced for free when the bed starts
to move. Their Example 2 (sand deposition into quiescent water) is the test that
isolates this: the bed aggrades while the free surface must remain flat and the
velocity exactly zero. **Use it as the Phase 4 acceptance test** — it is cheap,
needs no field data, and fails loudly if the coupling breaks well-balancedness.

Note the scope limit: `LM15` derives this for a *fully coupled* system, whereas
§9.1 here advects `c` with the already-computed water flux. The bed-slope
requirement is a property of a moving `Z`, not of the coupling, so it applies
either way — but their specific discretisation is not transferable as-is.

---

## 10. Parameter defaults

Defaults are FG21 Table 2 / P14 §3.3.2 where stated. **All must be settable.**

| Parameter | Symbol | Default | Source |
|-----------|--------|---------|--------|
| Water density | ρ | 1000 kg m⁻³ | FG21 T2 |
| Sediment density | ρ_s | 2650 (P14) / 3000 (FG21, Mars) kg m⁻³ | P14, FG21 T2 |
| Porosity | λ | 0.2 (FG21) / 0.3 (aS16) | FG21 T2 |
| Kinematic viscosity | ν | 1×10⁻⁶ m² s⁻¹ | P14 |
| von Kármán | κ | 0.41 | FG21 |
| Grain size | D | 0.1 mm (P14 suspension) … 8 mm–1 cm (FG21 bedload) | P14, FG21 T2 |
| Settling C₁, C₂ | | 18, 0.4 (smooth spheres) | P14 3.10 |
| Suspension critical stress | τ_c* | 0.04 | FG21 |
| Smith–McLean constant | γ₀ | 0.0024 | FG21 |
| Bedload critical stress | τ_c* | 0.0495 | Wong & Parker 2006 |
| Bedload coefficient | K, m | 3.97, 1.5 | Wong & Parker 2006 |
| Max concentration | c_max | 0.30 | FG21 |
| Max bed change rate | max_dz | large; report firings | FG21 T2 |
| Angle of repose | | 35° | FG21 T2 |
| Vegetation drag | C_D | 1.2 | P14 3.11 |
| Sediment speed ratio | β | 1.0 | P14 3.4 |
| Active layer thickness | H_A | user | RDy26 |
| Substrate layers | | ≤ 32 | RDy26 |
| Gravity | g | **from domain** (9.81 Earth, 3.711 Mars) | FG21 T2 |

---

## 10.1 Validation rung 8 — Lower Rio Puerco, 2006 flood

`[P13]` specifies this case completely enough to build it. Values are theirs.

| Quantity | Value | Source |
|----------|-------|--------|
| Peak discharge `Q` (Highway 6 bridge) | 625 m³ s⁻¹ | P13 §[55] |
| Mean active flow width `b` | 235 m | P13 §[55] |
| Unit discharge `q` | 2.66 m² s⁻¹ | P13 §[55] |
| Peak duration `t` | 50 400 s (14 h, Bernardo NM gauge) | P13 §[54] |
| Representative grain size | **0.045 mm** (coarse silt) | P13 §[54], after Nordin (1963) |
| Settling velocity `v_s` | 0.00175 m s⁻¹ | P13 §[54], Ferguson & Church (2004) |
| Bed porosity `φ` | 0.30 | P13 §[54], Beard & Weyl (1973) |
| Profile factor `d*` (`p`) | 1 (uniform) | P13 §[54], Nordin (1963) |
| Study reach | 12 km | P13 abstract |
| Deposit thickness (field) | 1–5 cm unconsolidated sand | P13 §[73] |
| Field control | 26 pits, measured 2006 deposit thickness | P13 Figure 4 |

**Volumetric budget** (P13 §[68]–[69], with Vincent et al. 2009):

| Component | Volume |
|-----------|--------|
| Eroded, arroyo bottom (sprayed reach) | 680 000 m³ |
| Eroded, arroyo walls (sprayed reach) | 56 000 m³ |
| **Total eroded, sprayed + study area** | **≈ 1 150 000 m³** |
| Aggraded, floodplains of sprayed reach | 500 000 m³ |
| **Total aggradation, both reaches** | **≈ 1 500 000 m³** |

**What rung 8 must reproduce**, in order of stringency:

1. **Qualitative** — deposition depth correlates with vegetation density, and its
   *variability* correlates with vegetation type (P13 Figure 7). This is the result
   the vegetation operator exists to capture.
2. **Profile** — median deposit thickness decaying exponentially downstream from the
   sediment source, per P13's 1-D model:
   ```
   [P-1]   C(x) = (C₀ − S_P/(p v_s)) · exp(−p v_s x / q) + S_P/(p v_s)     [P13 6]
   ```
   with `S_P` a local source term. Deviations from `[P-1]` are what P13 attribute to
   arroyo morphology and vegetation — so a 2-D model should reproduce the deviations,
   not just the profile.
3. **Budget** — total aggradation within the study area, against the table above.

> **Caution on the budget.** P13 are explicit that the sprayed-reach volumes of
> Vincent et al. (2009) are "not as well constrained" as their own lidar
> differencing. Treat the 1.5×10⁶ m³ total as an order-of-magnitude check; the lidar
> differencing over the study area is the defensible target.

> **Third grain size.** P13 use 0.045 mm, P14 uses 0.1 mm, `aS16` uses 0.065 mm — all
> for the same river, all citing the same body of work. See divergence D8; none is
> wrong, they are choices for different purposes, but the spec must not silently
> inherit one.

---

## 11. Equation citation map

Implementation should cite these labels. Suggested mapping to the layer plan:

| Layer | Sections |
|-------|----------|
| **L1** core tracer kernel | §2.2 `[G-1..G-3]`, §9 `[N-1..N-3]` |
| **L2** device bed operator | §3 `[T-1..T-6]`, §4 `[E-*, S-*, D-*, L-*]`, §5 `[B-*]`, §6 `[K-*]`, §8 `[V-*]` |
| **L2** separate kernel | §7 angle of repose (non-cell-local) |
| **L3** Python API | §10 parameters, closure selection, §4.3 pluggable `d*` |

---

## 14. References

- Davy, P. & Lague, D. (2009), *Fluvial erosion/transport equation of landscape evolution models revisited*, JGR Earth Surface 114, F03007. **doi:10.1029/2008JF001146**.
- Dietrich, W.E. (1982), *Settling velocity of natural particles*, Water Resources Research 18(6), 1615–1626.
- Engelund, F. & Hansen, E. (1967), *A Monograph on Sediment Transport in Alluvial Streams*, Teknisk Forlag, Copenhagen, 62 pp. Scanned copy held by TU Delft Repository, `uuid:81101b08-04b5-4082-9121-861949c336c9` — **redistribution restricted**, access it directly rather than circulating a copy.
- Fassett, C.I. & Goudge, T.A. (2021), JGR Planets 126, e2021JE006979. **doi:10.1029/2021JE006979**.
- Ferguson, R.I. & Church, M. (2004), *A simple universal equation for grain settling velocity*, Journal of Sedimentary Research 74(6), 933–937.
- Feng, D. et al. (2026), *RDycore-sediment v1.0*, EGUsphere preprint 2026-4859. **doi:10.5194/egusphere-2026-4859**.
- Ganti, V. et al. (2014) — cited by FG21 for the `d*` concept.
- Hanson, G.J. (1990) — excess-shear entrainment for cohesive soils.
- Hanson, G.J. & Simon, A. (2001), *Erodibility of cohesive streambeds in the loess area of the midwestern USA*, Hydrological Processes 15, 23–38. Source of `[E-5]`.
- Nepf, H.M. (1999), *Drag, turbulence, and diffusion in flow through emergent vegetation*, Water Resources Research 35(2), 479–489. Figure 6; source of `[V-4]`,`[V-5]`.
- Griffin, E.R. et al. (2005, 2014) — Rio Puerco vegetation and grain size.
- Kean, J.W. & Smith, J.D. (2004) — vegetation drag.
- Larsen, I.J. & Lamb, M.P. (2016), *Progressive incision of the Channeled Scablands by outburst floods*, Nature **538**(7624), 229–232. **doi:10.1038/nature19817**. Moses Coulee, Washington. Finding relevant here: including erosion thresholds gives flood discharges **5–10× smaller** than full-to-the-brim estimates.
- Liu, X., Mohammadian, A., Kurganov, A. & Infante Sedano, J.A. (2015), *Well-balanced central-upwind scheme for a fully coupled shallow water system modeling flows over erodible bed*, Journal of Computational Physics **300**, 202–218. **doi:10.1016/j.jcp.2015.07.043**. `LM15`. Fully **coupled** 5×5 hyperbolic system — SWE + nonequilibrium suspended-sediment transport + Exner bed evolution — solved by a Godunov-type central-upwind scheme **on triangular grids**, i.e. the same mesh topology as ANUGA. Relevant to §9: (a) a special bed-slope discretisation is what buys well-balancedness once `Z` is time-varying, which is the §9.5 problem this specification inherits as soon as the bed moves; (b) wave speeds come from Lagrange-theorem **bounds** on the Jacobian eigenvalues rather than a full eigen-decomposition of the coupled system — the cheap way to keep a coupled CFL. Deposition `D = ω₀(1−C_a)^m C_a` with near-bed `C_a = αc`, `α = min[2,(1−p)/c]`; settling `ω₀` from Eq 7 (a Zhang-type formula, **not** Ferguson & Church); entrainment `E ∝ φ(θ−θ_c)|u|` above threshold, coefficients fitted for **non-cohesive** sediment (after Cao et al.). Note this is a *fully coupled* formulation: the present specification advects `c` as a passive tracer with the already-computed water flux (§9.1), which is the decoupled path LM15 argues breaks down when erosion is of the same order as the flow.
- Meyer-Peter, E. & Mueller, R. (1948), *Formulas for bed-load transport*.
- Mungkasi, S. & Roberts, S.G. (2012) — velocity regularisation used in ANUGA.
- Nordin, C.F. (1963) — uniform suspension observations.
- Parker, G. (1998), *1D Sediment Transport Morphodynamics* e-book, Eqs 2.11–2.12.
- Parker, G. (2005); Kleinhans, M.G. (2005) — transport law reviews.
- Perignon, M.C. (2014), PhD thesis, University of Colorado Boulder.
- Perignon, M.C. (2016), *Using the Sediment Transport and Vegetation Operators in ANUGA* — `anugaSed/docs/anugaSed_manual.pdf`.
- Perignon, M.C., Tucker, G.E., Griffin, E.R. & Friedman, J.M. (2013), *Effects of riparian vegetation on topographic change during a large flood event, Rio Puerco, New Mexico, USA*, JGR Earth Surface 118(3), 1193–1209. **doi:10.1002/jgrf.20073**. USGS record: pubs.usgs.gov/publication/70176296.
- Simpson, G. & Castelltort, S. (2006) — coupled flow/sediment/topography framework cited by aSM16.
- Smith, J.D. & McLean, S.R. (1977), *Spatially averaged flow over a wavy surface*, JGR 82(12), 1735–1746.
- Wilson, K.C. (1966) — erosion coefficient `K_e*` for high-shear-stress flows, cited by P14 Eq 3.8. **Not the same Wilson as the 2004 friction paper above.**
- Wilson, L., Ghatan, G.J., Head, J.W. & Mitchell, K.L. (2004), *Mars outflow channels: A reappraisal of the estimation of water flow velocities from water depths, regional slopes, and channel floor properties*, JGR Planets **109**, E09003. **doi:10.1029/2004JE002281**. Eqs 3–4 (Darcy–Weisbach ↔ Manning) and 13–15 (`f` from grain size and depth). Derives **n = 0.0545 s m⁻¹ᐟ³** for Martian channels.
- Wong, M. & Parker, G. (2006), *Reanalysis and Correction of Bed-Load Relation of Meyer-Peter and Müller Using Their Own Database*, Journal of Hydraulic Engineering **132**(11), 1159–1168. **doi:10.1061/(ASCE)0733-9429(2006)132:11(1159)**. Eq 24. Corrected rates are **≤ half** the original MPM values — hence FG21's "lower K".
