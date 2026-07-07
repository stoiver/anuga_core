

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
   workflow that most users need: install ANUGA, try an example, write a script
   (or use the TOML scenario interface), run it, and view the results. Advanced
   topics live in the **Appendices** — the developer and GPU installs, parallel
   simulation (OpenMP / MPI ``distribute``), compute modes and GPU offloading,
   and checkpointing.


.. toctree::
   :maxdepth: 2
   :caption: Contents:

   background
   installation/index
   examples/index
   setup_anuga_script/index
   toml_scenario/index
   visualisation/index
   reference/index
   troubleshooting
   genindex

.. toctree::
   :maxdepth: 2
   :caption: Appendices:

   installation/install_anuga_developers
   appendices/install_gpu
   parallel/index
   parallel/advanced
   appendices/compute_modes
   appendices/advanced_script
   mathematical_background


   



