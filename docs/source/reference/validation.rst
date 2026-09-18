.. _anuga-validation:

.. currentmodule:: anuga

Validation Test Suite
=====================

ANUGA ships a validation test suite in the ``validation_tests/`` directory of
the repository.  The tests verify the correctness of the numerical solver
against analytical solutions, published experimental data, and benchmark
problems.  They are separate from the unit tests (run with ``pytest``) and are
intended to be run periodically, or whenever a significant change is made to
the solver.

.. contents:: Contents
   :local:
   :depth: 2


Running the validation tests
-----------------------------

Change into the ``validation_tests/`` directory and run the suite runner:

.. code-block:: bash

   cd validation_tests
   python run_auto_validation_tests.py

The runner searches all subdirectories recursively for scripts whose names
start with ``validate_`` and end with ``.py``, then executes them in
alphabetical order.  Each script uses Python's ``unittest`` framework to check
that the numerical solution meets a specified tolerance.

**Selecting a flow algorithm**

Many tests accept a ``--alg`` argument to select the flow algorithm:

.. code-block:: bash

   python run_auto_validation_tests.py --alg DE1

Valid options are ``DE0`` (first-order, default) and ``DE1`` (second-order).

**Verbose output**

.. code-block:: bash

   python run_auto_validation_tests.py --verbose

**Running a single test**

Each ``validate_*.py`` script can also be run in isolation from its own
directory:

.. code-block:: bash

   cd validation_tests/analytical_exact/dam_break_dry
   python validate_dam_break_dry.py

**Producing a PDF report**

Each test directory contains a ``produce_results.py`` script that runs the
numerical simulation and writes a LaTeX report.  To produce reports for all
tests (excluding the large case studies):

.. code-block:: bash

   cd validation_tests/reports
   python all_tests_produce_report.py


Nightly run in CI
-----------------

The suite also runs every night in GitHub Actions (``validation.yml``), on
Linux with one Python, against both ``main`` and ``develop``, including the
long HEC-RAS behaviour cases. The unit tests say the code does what the code
says; this run says the solver still reproduces the benchmarks it is
validated against, which a limiter or wet/dry change can break while every
unit test stays green.

Each run's job summary shows the ``VALIDATION SUMMARY`` table, and the full
log plus the plots each case wrote are kept as an artifact for two weeks. A
failing scheduled run opens, or comments on, an issue titled *Validation
tests failing*. The workflow can also be started by hand from the Actions
tab for any branch, algorithm (``-alg``) and with or without the long cases,
which is the way to validate a feature branch before merging it.

Test suite structure
---------------------

The tests are organised into five categories:

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Directory
     - Contents
   * - ``analytical_exact/``
     - Comparisons against closed-form analytical solutions to the shallow
       water equations.  These tests are fast and provide the most rigorous
       correctness check.
   * - ``experimental_data/``
     - Comparisons against physical laboratory experiments.  Results are
       validated against measured water-surface time series and runup
       extents.
   * - ``case_studies/``
     - Larger, real-world simulations (e.g. patong beach tsunami).  These
       can take many hours and require 8–16 GB of RAM.  They are excluded
       from the standard automated run.
   * - ``behaviour_only/``
     - Tests that verify qualitative behaviour against results from a
       previous ANUGA run (no independent analytical reference).
   * - ``other_references/``
     - Comparisons against results published in the literature other than
       the experimental datasets above.


Benchmark descriptions
-----------------------

Dry avalanche — analytical solution
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Directory:** ``analytical_exact/avalanche_dry/``

**Physical scenario:** A mass of water on a dry, frictionless inclined plane
released from rest.  The exact solution for the runup and rundown of the
water front is known analytically.

**What is checked:** The simulated water-surface profile and front position at
several output times are compared against the analytical solution.  The
normalised RMS error must be below a set tolerance.

**Key techniques:** Rectangular cross mesh, ``Dirichlet_boundary`` inflow,
wet/dry front tracking.

**Run time:** < 1 minute.

Wet avalanche — analytical solution
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Directory:** ``analytical_exact/avalanche_wet/``

**Physical scenario:** Same inclined-plane geometry as the dry avalanche but
with a pre-existing thin layer of water on the slope.  The analytical solution
accounts for the interaction between the incoming and resident water.

**What is checked:** Water-surface profile versus analytical solution at
multiple times.

**Run time:** < 1 minute.

Dry dam break — analytical solution
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Directory:** ``analytical_exact/dam_break_dry/``

**Physical scenario:** Instantaneous removal of a dam separating still water
from a dry bed.  The Riemann solution provides exact depth and velocity
profiles for the resulting shock and rarefaction wave.

**What is checked:** Depth and velocity profiles at a fixed output time are
compared against the Ritter (1892) analytical solution.  Peak error must be
below tolerance.

**Key techniques:** Rectangular cross mesh, zero-friction flat bed, initial
depth step as ``Dirichlet_boundary``.

**Run time:** < 1 minute.

Wet dam break — analytical solution
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Directory:** ``analytical_exact/dam_break_wet/``

**Physical scenario:** Same as the dry dam break but with a shallow layer of
water on the downstream side.  The exact solution includes a reflected shock
wave in addition to the rarefaction fan.

**What is checked:** Depth and velocity profiles versus the exact Stoker (1957)
wet dam-break solution.

**Run time:** < 1 minute.

Okushiri Island tsunami runup — experimental data
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Directory:** ``experimental_data/okushiri/``

**Physical scenario:** The 1993 Okushiri Island tsunami (Benchmark 2 from the
Third International Workshop on Long-Wave Runup Models, Catalina Island 2004).
A physical scale model was used to measure water-surface time series at gauges
5, 7, and 9 and maximum runup around the island.

**What is checked:** Simulated water-surface elevation time series at the three
gauge locations are compared against the experimental measurements.  The
normalised error at each gauge must be below a set tolerance.

**Key techniques:** Unstructured mesh from ``Benchmark_2.msh``, bathymetry
from ``Benchmark_2_Bathymetry.asc``, incoming wave from
``Benchmark_2_input.tms`` via
``Transmissive_n_momentum_zero_t_momentum_set_stage_boundary``, gauge
extraction via ``file_function``.

**Data source:** `<http://www.cee.cornell.edu/longwave/>`_ (Third IWLWRM
benchmark problem 2).

**Run time:** 10–30 minutes depending on mesh resolution and hardware.


Sediment settling in still water — analytical solution
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Directory:** ``analytical_exact/sediment_settling/``

**Physical scenario:** A tank of still water 1 m deep carries a uniform
volumetric concentration of 0.01 of 100 micron sediment. There is no flow, so
nothing is entrained; the sediment settles onto the bed at the deposition
rate :math:`D = d^{*} v_s c` with a constant near-bed profile factor, and the
bed rises by the deposited volume over the packing fraction. The depth falls
as the bed rises, which feeds back on the depth-averaged concentration; the
reference solution integrates that coupled system to machine precision, and
reduces to a plain exponential when the feedback is dropped.

**What is checked:** The cell-mean concentration and bed rise against the
reference at every output time (relative :math:`L^1` error below 1%); that
every cell follows the same curve; that the free surface does not move; and
that the sediment mass in the water column plus the bed stays at its initial
value. It exercises the deposition term, the sediment mass balance and the
bed update of the sediment transport operator.

**Key techniques:** ``add_sediment_fraction`` with ``initial_concentration``,
``set_deposition(near_bed='constant')``, reading the tracer and the
time-varying bed back from the SWW file.

**Run time:** a few seconds.


Sediment entrainment into still water — analytical solution
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Directory:** ``analytical_exact/sediment_erosion/``

**Physical scenario:** A tank of still water 1 m deep over a fixed plane
bed of slope :math:`10^{-3}`, with 0.5 mm sand and no initial load. There
is no flow, so the case selects the depth-slope shear closure, under which
the bed shear stress of a cell depends only on its depth and bed slope. The
Smith-McLean entrainment law :math:`E = v_s E^{*}` fires and the
concentration relaxes to the equilibrium :math:`c_{eq} = E/(d^{*} v_s)`
along a closed-form exponential with time constant :math:`h/(d^{*} v_s)`,
each cell on its own curve. A second fraction of the same grain with a
critical Shields stress of 10, far above the stress the slope gives, is the
threshold control and must stay at zero. The bed is held fixed because the
depth-slope closure takes its slope from the bed itself and feeds back on
any non-uniform erosion, so an evolving bed has no closed-form reference.

**What is checked:** Every cell's concentration against its own curve at
every output time (relative :math:`L^1` error below 1%, measured at a few
:math:`10^{-4}`, worst cell included, so the slope estimate must reproduce
the plane along the walls and in the corners); that the control fraction is
never entrained; that the final concentration is the equilibrium value; and
that the free surface, the bed and the momenta do not move. It exercises the
entrainment law and its threshold, the depth-slope closure, and the
entrainment-deposition balance in the sediment mass equation.

**Key techniques:** ``set_shear_closure('depth_slope')``,
``set_bed_material('noncohesive')``, a second ``add_sediment_fraction``
with a high ``tau_c_star`` as the control,
``initialize_sediment_operator(bed_evolution=False)``.

**Run time:** a few seconds.

Adding a new validation test
------------------------------

1. Create a subdirectory under the appropriate category
   (``analytical_exact/``, ``experimental_data/``, etc.).
2. Write a numerical simulation script (e.g. ``numerical_mytest.py``) that
   saves an SWW file.
3. Write a ``validate_mytest.py`` script using Python's ``unittest`` that
   loads the SWW output, extracts the relevant quantities, and asserts that
   errors are within tolerance.
4. Optionally write ``produce_results.py`` (calling
   ``anuga.validation_utilities.produce_report``) and a ``report.tex``
   template for the PDF report generator.

The runner will pick up ``validate_mytest.py`` automatically on the next run.
