"""Fractional-step operator for suspended sediment transport and bed evolution.

Applies, once per timestep and after the hydrodynamic step:

    m_s  <-  m_s + dt (E_s - D_s)          [G-3] source term
    z    <-  z   + dt (D - E)/(1 - lambda) [G-4] Exner, suspended contribution

Both use the same limited source, so the sediment volume leaving suspension is
exactly `(1 - lambda) dz` and the mass budget closes by construction, whatever
timestepping method the flow algorithm uses.

The advective transport of `m` is NOT here: that lives in the flux kernel and
runs every substep with the rest of the shallow-water system. This operator
carries only the bed exchange.

GPU safety
----------
The work is done by `core_apply_sediment_source`, the same kernel in both
compute modes. In mode 2 it is called through the device path, so the tracer and
bed arrays are updated in place on the GPU with no host round trip. That matters
more than it looks: a fractional-step operator that writes on the host forces
`_gpu_host_writes_suppressed` on, which reactivates the sync bracket and drops
the whole run onto the host path -- a large, silent performance loss. This
operator must therefore stay on the GPU-safe list in
`Domain._has_cpu_only_fractional_operators`.
"""

from anuga.operators.base_operator import Operator


class Sediment_operator(Operator):
    """Apply sediment bed exchange and Exner bed evolution as a fractional step.

    Parameters
    ----------
    domain : Domain
        Must already have at least one class registered via
        `domain.add_sediment_class()`.
    """

    def __init__(self, domain, description=None, label=None, logging=False,
                 verbose=False):
        Operator.__init__(self, domain, description, label, logging, verbose)

        if getattr(domain, 'n_sediment_classes', 0) == 0:
            raise ValueError(
                'Sediment_operator requires at least one sediment class; '
                'call domain.add_sediment_class(...) first')

    def __call__(self):
        timestep = self.domain.get_timestep()
        if timestep <= 0.0:
            return

        domain = self.domain
        if domain.multiprocessor_mode == 2 and domain.gpu_interface is not None:
            from anuga.shallow_water.sw_domain_gpu_ext import (
                apply_sediment_source_gpu)
            apply_sediment_source_gpu(domain.gpu_interface.gpu_dom, timestep)
        else:
            from anuga.shallow_water.sw_domain_openmp_ext import (
                apply_sediment_source)
            apply_sediment_source(domain, timestep)

    def parallel_safe(self):
        """Safe in parallel: the kernel is cell-local.

        It reads only quantities the halo exchange already keeps current, and
        writes only to the cell it is working on -- no neighbour access, so
        ghost cells need no special treatment.
        """
        return True

    def statistics(self):
        return '%s: %d sediment class(es)' % (self.label,
                                              self.domain.n_sediment_classes)

    def timestepping_statistics(self):
        return ''
