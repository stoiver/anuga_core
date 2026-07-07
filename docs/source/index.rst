

.. currentmodule:: anuga

===================
ANUGA documentation
===================

ANUGA (pronounced "AHnooGAH") is open-source software for the simulation of the 
shallow water equation, 
in particular it can be used to model tsunamis and floods.

ANUGA is a Python 3 package (Python 3.10 – 3.14) with some C and Cython
extensions.

ANUGA was created in a collaboration by Geoscience Australia 
and Mathematical Sciences Institute at the Australian National University. 
It is now developed and maintained by a community of volunteers.

Quick start
-----------

.. code-block:: bash

   # 1. install (conda-forge, recommended)
   conda create -n anuga -c conda-forge anuga
   conda activate anuga

.. code-block:: python

   # 2. run a minimal model
   import anuga
   domain = anuga.rectangular_cross_domain(10, 5)
   domain.set_quantity('elevation', lambda x, y: -x / 10)
   domain.set_quantity('stage', 0.0)
   domain.set_boundary({b: anuga.Reflective_boundary(domain)
                        for b in domain.get_boundary_tags()})
   for t in domain.evolve(yieldstep=1.0, finaltime=10.0):
       print(domain.timestepping_statistics())

Then view the ``.sww`` output (see :doc:`Visualisation <visualisation/index>`).
The sections below build this up step by step; new to ANUGA? Start with
:doc:`Examples <examples/index>`.

.. note::

   The main documentation (**Contents**) covers the standard workflow most users
   need: install ANUGA, try an example, write a script (or use the TOML scenario
   interface), run it, view the results, and parallelise if needed. The
   **Appendices** hold advanced topics — the developer and GPU installs, advanced
   parallel methods, compute modes and GPU offloading, and advanced scripting.


.. toctree::
   :maxdepth: 2
   :caption: Contents:

   background
   installation/index
   examples/index
   setup_anuga_script/index
   toml_scenario/index
   visualisation/index
   parallel/index
   reference/index
   troubleshooting
   genindex

.. toctree::
   :maxdepth: 2
   :caption: Appendices:

   installation/install_anuga_developers
   appendices/install_gpu
   parallel/advanced
   appendices/compute_modes
   appendices/advanced_script
   mathematical_background


   



