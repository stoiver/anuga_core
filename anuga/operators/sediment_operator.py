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

import math

import anuga.utilities.log as log
from anuga.operators.base_operator import Operator


class Sediment_transport_operator(Operator):
    """Suspended sediment transport and bed evolution, as a fractional step.

    This is the entry point: creating one both switches sediment transport on
    and registers a grain size for it to carry.

    .. code-block:: python

        from anuga import Sediment_transport_operator

        Sediment_transport_operator(domain, name='sand', diameter=2.0e-4)

    A grain size is a tracer with settling parameters attached, so it inherits
    the transport, boundary and conservation machinery described under
    :ref:`tracers`.

    ONE OPERATOR PER DOMAIN. A second call adds another grain size to the same
    operator and returns it, rather than creating a second one:

    .. code-block:: python

        Sediment_transport_operator(domain, name='sand', diameter=2.0e-4)
        Sediment_transport_operator(domain, name='silt', diameter=2.0e-5)
        # one operator, two grain sizes

    That is not a convenience -- the kernel makes a single pass over every
    registered grain size, so two operators in the fractional-step list would
    apply the bed exchange twice per timestep.

    Parameters
    ----------
    domain : Domain
        The domain to transport sediment on.
    name : str
        Identifier for this grain size, e.g. 'sand'. Also its tracer name.
    diameter : float
        Grain diameter in metres.

    The remaining parameters are the per-grain-size settling and threshold
    properties; see :ref:`sediment` for what they mean and how to choose them.
    """

    def __new__(cls, domain, *args, **kwargs):
        # One per domain: return the existing operator so a second call adds a
        # grain size to it instead of registering a duplicate fractional step.
        for op in getattr(domain, 'fractional_step_operators', ()):
            if isinstance(op, cls):
                return op
        return super().__new__(cls)

    def __init__(self, domain, name=None, diameter=None, d_star=1.0, beta=None,
                 initial_concentration=0.0, rho_s=2650.0, rho_w=1000.0,
                 tau_c_star=0.04, reference_height=None,
                 description=None, label=None, logging=False, verbose=False,
                 **settling_kwargs):

        registering = dict(d_star=d_star, beta=beta,
                           initial_concentration=initial_concentration,
                           rho_s=rho_s, rho_w=rho_w, tau_c_star=tau_c_star,
                           reference_height=reference_height,
                           **settling_kwargs)

        # __new__ may have handed back the operator this domain already has, in
        # which case Python still calls __init__ on it. Add the grain size and
        # leave everything else alone -- re-running Operator.__init__ would
        # register a second fractional step.
        if getattr(self, '_sediment_initialised', False):
            if name is not None:
                domain._register_sediment_fraction(name, diameter, **registering)
            return

        Operator.__init__(self, domain, description, label, logging, verbose)
        self._sediment_initialised = True

        # Reporting for the repose relaxation (spec 7).
        self.repose_sweeps = 0
        self.repose_sweeps_total = 0
        self.repose_cap_hits = 0

        if name is not None:
            domain._register_sediment_fraction(name, diameter, **registering)
        elif getattr(domain, 'n_sediment_classes', 0) == 0:
            raise ValueError(
                'Sediment_transport_operator needs a grain size: pass '
                "name= and diameter=, e.g. "
                "Sediment_transport_operator(domain, name='sand', "
                'diameter=2.0e-4)')

    def add_grain_size(self, name, diameter, **kwargs):
        """Register another grain size on this operator. Returns its index."""
        return self.domain._register_sediment_fraction(name, diameter, **kwargs)

    def __call__(self):
        timestep = self.domain.get_timestep()
        if timestep <= 0.0:
            return

        domain = self.domain
        suspended = getattr(domain, '_sediment_suspended_enabled', True)
        # _gpu_host_writes_suppressed marks a HOST-COHERENT window: because a
        # CPU-only operator is registered, apply_fractional_steps has already
        # pulled the device state to the host and will push the host state back
        # when the loop finishes. Running on the device inside that window is
        # not merely redundant -- the trailing push overwrites whatever the
        # device computed with the host copy, so the work is silently lost.
        # That is what made an external source contribute nothing in mode 2
        # whenever any CPU-only operator was present (#288).
        on_gpu = (domain.multiprocessor_mode == 2
                  and domain.gpu_interface is not None
                  and not getattr(domain, '_gpu_host_writes_suppressed', False))

        if on_gpu:
            from anuga.shallow_water.sw_domain_gpu_ext import (
                apply_sediment_source_gpu, apply_bedload_gpu)
            if suspended:
                apply_sediment_source_gpu(domain.gpu_interface.gpu_dom, timestep)
            if domain.sediment_bedload_mode:
                apply_bedload_gpu(domain.gpu_interface.gpu_dom, timestep)
        else:
            from anuga.shallow_water.sw_domain_openmp_ext import (
                apply_sediment_source, apply_bedload)
            if suspended:
                apply_sediment_source(domain, timestep)
            if domain.sediment_bedload_mode:
                apply_bedload(domain, timestep)

        # Angle-of-repose relaxation LAST, so it relaxes the bed this step
        # actually produced rather than the one it started from. It is the only
        # non-cell-local sediment kernel, and the only one that can iterate.
        if domain.sediment_repose_tan > 0.0:
            if on_gpu:
                from anuga.shallow_water.sw_domain_gpu_ext import (
                    apply_repose_gpu)
                sweeps = apply_repose_gpu(domain.gpu_interface.gpu_dom)
            else:
                from anuga.shallow_water.sw_domain_openmp_ext import (
                    apply_repose)
                sweeps = apply_repose(domain)

            self.repose_sweeps = sweeps
            self.repose_sweeps_total += sweeps
            if sweeps >= domain.sediment_repose_max_sweeps:
                # Spec 7 requires this be reported, not swallowed: hitting the
                # cap means the bed may still be over-steep where the whole
                # point of the kernel was that it should not be.
                self.repose_cap_hits += 1
                if self.repose_cap_hits == 1 or self.verbose:
                    log.critical(
                        '%s: angle-of-repose relaxation hit its %d-sweep cap '
                        'at t = %g s; the bed may still exceed %.1f degrees. '
                        'Raise max_sweeps, or relax less aggressively.'
                        % (self.label, domain.sediment_repose_max_sweeps,
                           domain.get_time(),
                           math.degrees(math.atan(domain.sediment_repose_tan))))

            # Spec 7 requires a halo exchange per sweep. This exchanges once
            # per timestep instead: the sweep loop lives inside the kernel so
            # that it stays on the device, and pulling it into Python to
            # exchange between sweeps would put a host round trip in the middle
            # of every sweep and drop the run off the GPU path. In parallel the
            # consequence is that relaxation propagates across a subdomain
            # boundary one sweep per TIMESTEP rather than one per sweep, so a
            # slump spanning a boundary relaxes more slowly there. Recorded in
            # PHYSICS_SPEC 7.1; serial results are unaffected.
            if domain.parallel:
                domain.update_ghosts(['elevation'])

    def parallel_safe(self):
        """Safe in parallel.

        The suspended exchange is cell-local. The bedload divergence DOES read
        neighbours, but only their transport vector, which is computed from
        stage, momentum and friction -- all quantities the halo exchange already
        keeps current -- so ghost cells carry a valid q_b and the divergence is
        correct on owned cells.

        Angle-of-repose relaxation also reads neighbours, and exchanges
        elevation after its sweeps. See the note at the call site for what it
        does NOT do, which is exchange between them.
        """
        return True

    def statistics(self):
        return '%s: %d sediment class(es)' % (self.label,
                                              self.domain.n_sediment_classes)

    def timestepping_statistics(self):
        if self.domain.sediment_repose_tan > 0.0:
            return ', repose sweeps %d' % self.repose_sweeps
        return ''
