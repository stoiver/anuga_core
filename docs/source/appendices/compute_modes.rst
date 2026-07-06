.. _compute_modes:

.. currentmodule:: anuga

Compute modes: legacy vs unified (and GPU offload)
==================================================

.. note::

   **Standard users can skip this page.** By default ANUGA runs in the
   **legacy** compute path on the CPU, which is stable and needs no
   configuration. This appendix is for advanced users who want the newer
   unified C kernels or GPU offloading.

ANUGA v4.0 exposes two *orthogonal* knobs:

1. **Per-domain compute path** — *how* each step is computed: the **legacy**
   path (mode 1: the historical ``sw_domain_openmp`` kernels plus Python-level
   operators) or the **unified** path (mode 2: a single set of C kernels shared
   by CPU and GPU).
2. **Process-global GPU offload** — *where* the unified kernels run: on the CPU
   (multicore OpenMP) or offloaded to a GPU. This is a process-wide setting, not
   a per-domain one, because OpenMP target offload is a process-level runtime
   state.

The two combine: ``cpu`` = unified + offload off; ``gpu`` = unified + offload on.
"cpu" and "gpu" are *compositions* of these knobs, not extra modes.


Per-domain compute path
-----------------------

.. code-block:: python

   domain.set_compute_mode('legacy')    # = set_multiprocessor_mode(1)
   domain.set_compute_mode('unified')   # = set_multiprocessor_mode(2)
   domain.get_compute_mode()            # 'legacy' or 'unified'

- **legacy** (default) — the historical CPU path. Widely exercised and the
  reference for correctness.
- **unified** — the mode-2 C kernels. On a CPU-only build these run as multicore
  OpenMP; on a GPU build they can offload (see below). Results match legacy to
  round-off; validate before relying on unified for production.

The process-wide default can be set with the environment variable
``ANUGA_DEFAULT_COMPUTE_MODE=unified`` (default ``legacy``). This is honoured for
**serial** runs only — under MPI (``numprocs > 1``) new domains stay ``legacy``.


Process-global GPU offload
--------------------------

Call these *before* the first ``evolve()`` (and before building the first
unified domain, since its arrays are mapped to the chosen device at init):

.. code-block:: python

   anuga.set_gpu_offload(True)      # offload unified kernels to a GPU (GPU build only)
   anuga.set_gpu_offload(False)     # run unified on the CPU
   anuga.gpu_offload_enabled()      # resolved offload state
   anuga.set_omp_num_threads(16)    # OpenMP thread count for the whole process
   domain.compute_capabilities()    # {gpu_offload, num_gpu_devices, mpi, modes}

On a standard conda/pip install the GPU extension is absent and offload requests
degrade gracefully to the CPU path — ``set_gpu_offload(True)`` warns and returns
``False`` rather than failing. GPU offload requires a GPU-enabled build (see
:ref:`use_gpu_offloading` for hardware/compiler requirements and build steps).


Command-line flags
------------------

The standard argument parser (used by the benchmark/example scripts) exposes:

- ``-mpm 1|2`` / ``--multiprocessor_mode`` — legacy vs unified
- ``-nt N`` / ``--num_threads`` — OpenMP thread count
- ``-go`` / ``-ngo`` — GPU offload on / off
- ``-ro <method>`` — mesh reordering for cache locality (e.g. ``metis_rcm``,
  ``hilbert``)


.. seealso::

   :ref:`use_gpu_offloading`
      Hardware/compiler requirements, the GPU build, and CPU-only fallback.

   :ref:`use_parallel_openmp`, :ref:`use_parallel_mpi`
      Multi-threaded and multi-process parallelism (independent of the
      legacy/unified choice).
