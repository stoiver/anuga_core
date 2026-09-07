.. _sediment_physics:

.. currentmodule:: anuga

Sediment physics: choosing the laws
===================================

.. note::

   **You can skip this page to begin with.** ``Sediment_transport_operator`` picks a
   working set of laws for a sand bed, and :ref:`sediment` shows how to run a
   model with them. This appendix is for when you need to say *which* physics,
   rather than accept the defaults.

Each section below covers one choice: what the alternatives are, what they
assume, and how to tell which one your problem wants. Labels like ``[E-1]``
and the section numbers refer to the internal sediment specification, which is
not distributed with ANUGA; they are stable identifiers for each term rather
than links you can follow.

Choosing ``tau_c_star``
~~~~~~~~~~~~~~~~~~~~~~~

0.04 is a reasonable default for sand. It is the threshold at which grains
begin to move, and erosion is zero below it, so it sets *when* the bed becomes
active, not how fast. If the bed does not erode when you expect it to, check
this against the Shields curve for your grain size before adjusting anything
else.



--------------

Erosion: naming the bed material
--------------------------------

.. code-block:: python

   domain.set_bed_material('noncohesive')   # default
   domain.set_bed_material('cohesive', tau_crit=0.088, K_e=6.742e-7)
   domain.set_bed_material('partheniades', tau_crit=0.088, K_e=...)

The argument is the **material**, not the formula, because spec 4.1.1 is
explicit that these describe different sediment rather than competing
descriptions of the same sediment. Picking the wrong one is a physics error.

+------------------+-------------------------+-------------------------+
| material         | law                     | when                    |
+==================+=========================+=========================+
| ``noncohesive``  | ``[E-1]``/``[E-2]``     | sand, gravel, boulders  |
|                  | Shields, Smith & McLean |                         |
+------------------+-------------------------+-------------------------+
| ``cohesive``     | ``[E-3]`` Hanson &      | clay, silt,             |
|                  | Simon                   | consolidated mud        |
+------------------+-------------------------+-------------------------+
| ``partheniades`` | ``[E-4]`` Partheniades  | cohesive, where you     |
|                  |                         | have a site-calibrated  |
|                  |                         | ``K_e``                 |
+------------------+-------------------------+-------------------------+

``tau_crit`` (Pa, default 0.088) and ``K_e`` (m3/N/s) apply to the cohesive
routes only; the non-cohesive route takes its threshold per class from
``tau_c_star`` instead. The default ``K_e = 6.742e-7`` is anugaSed's.

The choice is not a small correction. On the same channel over 30 s, the
non-cohesive route scours 3-6 cm while the cohesive route accretes about a
millimetre -- the sign of the bed change reverses. See
``sandpit/sediment_examples/README.md``.

--------------

Deposition
----------

.. code-block:: python

   domain.set_deposition(law='d_star', tau_d=0.0, near_bed='constant',
                         reference_height_floor=0.01)

``law``
~~~~~~~

+-----------------+-------------------------+-------------------------+
| value           | expression              | when                    |
+=================+=========================+=========================+
| ``'d_star'``    | ``D = d* c v_s``,       | default; always         |
|                 | ``[D-1]``               | deposits                |
+-----------------+-------------------------+-------------------------+
| ``'threshold'`` | ``[D-2]``, deposition   | when you need           |
|                 | only where              | deposition suppressed   |
|                 | ``tau_b < tau_d``       | under strong flow       |
+-----------------+-------------------------+-------------------------+

``tau_d`` (Pa) is the threshold for ``'threshold'`` and is ignored otherwise.

.. _62-near_bed----the-d-ratio:

``near_bed`` -- the ``d*`` ratio
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Deposition is driven by the concentration *at the bed*, but the transported
quantity is depth-averaged. ``d* = c_b/c`` bridges them.

+----------------+-----------------------------------------------------+
| value          | meaning                                             |
+================+=====================================================+
| ``'constant'`` | ``d*`` is whatever each class was given (1.0 =      |
|                | well-mixed). Default.                               |
+----------------+-----------------------------------------------------+
| ``'rouse'``    | ``d*`` from the Rouse profile ``[S-4]``, recomputed |
|                | per cell per step                                   |
+----------------+-----------------------------------------------------+

``'constant'`` with ``d* = 1`` is the well-mixed assumption: fine sediment,
vigorous mixing, shallow flow. It is also what the analytic decay solutions
assume, so use it when comparing against them.

``'rouse'`` is the physical choice when the profile is stratified -- coarser
grains, or deeper and slower flow, where near-bed concentration genuinely
exceeds the mean. It costs an evaluation of the fitted ``d*(Z, a/h)``
polynomial per cell per class per step (§9.5 of the spec; 28 terms, maximum
error 0.82% over ``Z`` in [0.01, 2.5], ``a/h`` in [1e-3, 0.15], clamped at the
edges rather than extrapolated).

``reference_height_floor`` (default 0.01) is the floor on ``a/h``. The Rouse
profile is singular as the reference height approaches the bed, so ``a/h`` is
never allowed below this. Lowering it admits more stratification and more
near-bed concentration; it is a numerical guard, not a physical parameter, and
0.01 sits comfortably inside the fit range.

Near-bed concentration is bounded by ``c_pack`` ``[L-4]`` regardless. That bound
exists because equilibrium Rouse ``d*`` at vanishing shear will otherwise
deposit the entire water column in under a second.

--------------

Bed shear stress
----------------

Two independent choices feed ``tau_b``: how the stress is formed, and what
friction factor goes into it.

.. _71-set_shear_closure----how:

``set_shear_closure`` -- how
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   domain.set_shear_closure('quadratic_drag')   # default
   domain.set_shear_closure('depth_slope')

==================== ===================== =========
value                expression            spec
==================== ===================== =========
``'quadratic_drag'`` \`tau_b = rho f_c     v
``'depth_slope'``    ``tau_b = rho g h S`` ``[T-7]``
==================== ===================== =========

``'quadratic_drag'`` is the default and the right choice for unsteady or
rapidly varying flow -- dam breaks, floods, anything with significant
inertia.

``'depth_slope'`` assumes locally uniform flow, where friction balances gravity.
It is what anugaSed uses, so choose it when reproducing their results
(divergence **D1** in the spec). It degrades where that balance does not hold.

The two are interchangeable by construction: the kernel returns ``tau_b/rho``,
so everything downstream is unchanged by the choice.

.. _72-set_sediment_friction----what:

``set_sediment_friction`` -- what
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   domain.set_sediment_friction('constant')    # default
   domain.set_sediment_friction('wilson', bed='gravel', grain_size=0.02)
   domain.set_sediment_friction('larsen_lamb', k_s=0.05, r_d=2.0, r_br=2.0)

``'wilson'`` and ``'larsen_lamb'`` are not callable with the mode alone -- they
require a length scale and refuse without one, rather than inventing a
default:

- ``'wilson'`` needs ``grain_size > 0`` (D50 for sand, D84 for gravel or boulder);
- ``'larsen_lamb'`` needs either ``k_s`` or ``sigma_br``. There is no universal
  ``sigma_br``: it is site-measured, and LL16 report about 5 m at Moses Coulee.

+-------------------+------------------+-------------------------------+
| mode              | spec             | when                          |
+===================+==================+===============================+
| ``'constant'``    | ``[T-6]``        | default: ``f_c`` from the     |
|                   |                  | domain's Manning ``n``.       |
|                   |                  | Ordinary flood and channel    |
|                   |                  | work.                         |
+-------------------+------------------+-------------------------------+
| ``'wilson'``      | ``[T-8..T-12]``  | depth-dependent, from grain   |
|                   |                  | size. Shallow flow over       |
|                   |                  | coarse beds, where relative   |
|                   |                  | submergence matters.          |
+-------------------+------------------+-------------------------------+
| ``'larsen_lamb'`` | ``[T-13..T-15]`` | partitions total stress into  |
|                   |                  | grain and form drag. Bedforms |
|                   |                  | or roughness elements, where  |
|                   |                  | only the grain part drives    |
|                   |                  | sediment.                     |
+-------------------+------------------+-------------------------------+

``bed`` is ``'sand'`` or ``'gravel'``; ``grain_size`` (m) is the roughness length
scale; ``k_s`` (m) is the roughness height; ``r_d`` and ``r_br`` (default 2.0) are
Larsen-Lamb's drag partitioning ratios.

This affects **only** the sediment source term. The hydrodynamic friction
operator is untouched, so momentum still sees the domain's Manning ``n``
whatever you choose here.

--------------

Bedload
-------

.. code-block:: python

   domain.set_bedload('wong_parker_eq24')   # K=3.97, m=1.5, tau_c*=0.0495
   domain.set_bedload('wong_parker_eq23')   # K=4.93, m=1.6, tau_c*=0.0470
   domain.set_bedload('engelund_hansen')
   domain.set_bedload('off')                # default

Bedload ``[K-1]``-``[K-4]`` transports sediment along the bed rather than in
suspension, and drives its own bed evolution term ``[G-5]``. It is **off by
default**: it is a separate transport mode, not a refinement of suspension,
and enabling it changes what the model represents.

Enable it when the grain size is coarse enough that a significant fraction of
the load moves without going into suspension -- sand and gravel beds under
moderate flow. Leave it off for fine, fully suspended sediment.

The two Wong & Parker variants are their corrected Meyer-Peter-Muller fits;
Eq 24 is the default. ``K``, ``m`` and ``tau_c_star`` override the formula's
constants if you have a calibration.

Bedload only redistributes: it moves sediment between cells and conserves the
total exactly. The flux across each edge is centred, which is what makes it
antisymmetric and therefore conservative; see ``test_sediment_bedload.py``.


.. _sediment_labels:

What the bracketed labels mean
------------------------------

Labels like ``[E-1]`` name a **term in the physics**, not a reference. They
appear throughout this page, in the source comments, and in the output of
``domain.sediment_summary()``, so that a given term can be pointed at
unambiguously wherever it comes up. The list below is what each one names.

They originate in an internal specification that is not distributed with
ANUGA; the numbering is kept because it is already in the code and the
summaries, and renumbering would only break the correspondence.

**Settling and suspension**

.. list-table::
   :header-rows: 1
   :widths: 14 86

   * - Label
     - Term
   * - ``[S-1]``
     - settling velocity ``v_s``, Ferguson & Church (2004)
   * - ``[S-2]``
     - Rouse number ``Z = v_s / (kappa u*)``
   * - ``[S-4]``
     - the Rouse near-bed concentration ratio ``d*(Z, a/h)``

**Erosion**

.. list-table::
   :header-rows: 1
   :widths: 14 86

   * - Label
     - Term
   * - ``[E-1]``, ``[E-2]``
     - non-cohesive entrainment: Shields threshold with a Smith & McLean
       reference concentration
   * - ``[E-3]``, ``[E-5]``
     - cohesive erosion, Hanson & Simon excess-stress form
   * - ``[E-4]``
     - Partheniades cohesive erosion

**Deposition**

.. list-table::
   :header-rows: 1
   :widths: 14 86

   * - Label
     - Term
   * - ``[D-1]``
     - ``D = d* c v_s``
   * - ``[D-2]``
     - threshold deposition, ``D = v_s c (1 - tau_b/tau_d)``

**Bed shear and friction**

.. list-table::
   :header-rows: 1
   :widths: 14 86

   * - Label
     - Term
   * - ``[T-1]``
     - quadratic drag, ``tau_b = rho f_c |v|^2`` -- the default closure
   * - ``[T-2]``
     - shear velocity ``u* = |v| sqrt(f_c)``
   * - ``[T-3]``
     - dimensionless stress ``tau* = f_c |v|^2 / (R g d)``
   * - ``[T-4]``
     - excess stress ``tau_x = tau* - tau_c*``
   * - ``[T-5]``
     - the depth-limiting velocity form ANUGA uses
   * - ``[T-6]``
     - constant Manning ``n``, taken from the domain's friction quantity
   * - ``[T-7]``
     - depth-slope closure, ``tau_b = rho g h S``
   * - ``[T-8]``..``[T-10]``
     - the ``'wilson'`` friction closure
   * - ``[T-13]``..``[T-15]``
     - the ``'larsen_lamb'`` friction closure

**Bedload**

.. list-table::
   :header-rows: 1
   :widths: 14 86

   * - Label
     - Term
   * - ``[K-1]``, ``[K-2]``
     - power law, ``q_b* = K tau_x^m``
   * - ``[K-3]``
     - bed change from bedload, ``dz/dt = -(1/(1-lambda)) div q_b``
   * - ``[K-4]``
     - the per-cell bedload transport vector ``q_b``
   * - ``[K-5]``
     - Engelund & Hansen total load, no threshold

**Coupling to the flow and the bed**

.. list-table::
   :header-rows: 1
   :widths: 14 86

   * - Label
     - Term
   * - ``[G-3]``
     - the suspended source term, ``m_s <- m_s + dt (E_s - D_s)``, including
       any external source
   * - ``[G-4]``
     - Exner bed evolution from the suspended exchange
   * - ``[G-5]``
     - the bedload divergence contribution to the bed

**Limiters**

.. list-table::
   :header-rows: 1
   :widths: 14 86

   * - Label
     - Term
   * - ``[L-1]``
     - positivity
   * - ``[L-2]``
     - the concentration ceiling ``c_max``
   * - ``[L-3]``
     - the edge reconstruction limiter ``beta``
   * - ``[L-4]``
     - the packing fraction ``c_pack`` bounding near-bed concentration
   * - ``[L-5]``
     - the non-erodible base

.. _sediment_references:

References
----------

The laws above are standard sediment-transport formulations; these are the
sources for each.

**Settling velocity**

* Ferguson, R. I. and Church, M. (2004). A simple universal equation for grain
  settling velocity. *Journal of Sedimentary Research*, 74(6), 933-937.
  The ``[S-1]`` settling velocity, chosen over the Dietrich fit because it is
  smooth across the Stokes/turbulent transition and branch-free.
* Dietrich, W. E. (1982). Settling velocity of natural particles.
  *Water Resources Research*, 18(6), 1615-1626.
  The polynomial fit the above is preferred to.

**Erosion**

* Shields, A. (1936). *Anwendung der Ähnlichkeitsmechanik und der
  Turbulenzforschung auf die Geschiebebewegung.* Mitteilungen der
  Preussischen Versuchsanstalt für Wasserbau und Schiffbau, Berlin.
  The critical shear stress ``tau_c_star`` is a Shields parameter.
* Smith, J. D. and McLean, S. R. (1977). Spatially averaged flow over a wavy
  surface. *Journal of Geophysical Research*, 82(12), 1735-1746.
  The near-bed reference concentration in the ``[E-1]`` non-cohesive route.
* Partheniades, E. (1965). Erosion and deposition of cohesive soils.
  *Journal of the Hydraulics Division, ASCE*, 91(1), 105-139.
  The ``[E-4]`` cohesive erosion route.
* Hanson, G. J. and Simon, A. (2001). Erodibility of cohesive streambeds in
  the loess area of the midwestern USA. *Hydrological Processes*, 15(1),
  23-38. The ``[E-3]``/``[E-5]`` cohesive route and its excess-stress form.

**Suspension and deposition**

* Rouse, H. (1937). Modern conceptions of the mechanics of fluid turbulence.
  *Transactions of the ASCE*, 102, 463-505.
  The ``[S-4]`` Rouse profile behind the ``d*`` near-bed ratio.
* van Rijn, L. C. (1984). Sediment transport, part II: suspended load
  transport. *Journal of Hydraulic Engineering*, 110(11), 1613-1641.
  The reference height ``a`` and the ``a >= 0.01 h`` floor.

**Bedload and bed evolution**

* Exner, F. M. (1925). Über die Wechselwirkung zwischen Wasser und Geschiebe
  in Flüssen. *Sitzungsberichte der Akademie der Wissenschaften*, Vienna.
  The ``[G-4]`` bed evolution equation.
* Wong, M. and Parker, G. (2006). Reanalysis and correction of bed-load
  relation of Meyer-Peter and Müller using their own database. *Journal of
  Hydraulic Engineering*, 132(11), 1159-1168.
  The default ``'wong_parker_eq24'`` bedload formula.
* Engelund, F. and Hansen, E. (1967). *A Monograph on Sediment Transport in
  Alluvial Streams.* Teknisk Forlag, Copenhagen.
  The ``[K-5]`` total-load option.

**Bed roughness closures**

* Larsen, I. J. and Lamb, M. P. (2016). Progressive incision of the Channeled
  Scablands by outburst floods. *Nature*, 538, 229-232.
  The ``'larsen_lamb'`` friction closure and the Moses Coulee ``sigma_br``
  value quoted above.
* Wilson, L., Ghatan, G. J., Head, J. W. and Mitchell, K. L. (2004). Mars
  outflow channels: a reappraisal of the estimation of water flow velocities
  from water depths, regional slopes, and channel floor properties. *Journal
  of Geophysical Research: Planets*, 109, E09003.
  The ``'wilson'`` friction closure. Note the caveat above: these are
  Mars outflow channel closures, not general-purpose flood closures.

.. note::

   **Still to be filled in.** The following short codes appear in the text and
   in the source comments and come from the internal sediment specification,
   which is not distributed with ANUGA. They have deliberately not been
   guessed at:

   ``FG21``, ``RDy26``, ``LM15``, ``DL09``, ``P13``, ``P14``, ``FP64``,
   ``A22``, ``A23``.

   ``RDy26`` refers to the RDycore configuration the benchmarks are compared
   against; the rest are literature the specification cites. Replace this note
   with the full citations when they are to hand.
