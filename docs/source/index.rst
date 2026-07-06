

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

.. note::

   The main documentation (**Contents**) covers the standard, single-process
   workflow that most users need: install ANUGA, write a script, run it, and
   view the results. Advanced topics live in the **Appendices** — parallel
   simulation (OpenMP / MPI ``distribute``), compute modes and GPU offloading,
   checkpointing, the developer install, and the TOML scenario interface.


.. toctree::
   :maxdepth: 2
   :caption: Contents:

   background
   installation/index
   setup_anuga_script/index
   examples/index
   visualisation/index
   reference/index
   troubleshooting
   genindex

.. toctree::
   :maxdepth: 2
   :caption: Appendices:

   parallel/index
   parallel/advanced
   appendices/compute_modes
   installation/install_anuga_developers
   toml_scenario/index
   appendices/advanced_script
   appendices/advanced_visualisation
   mathematical_background


   



