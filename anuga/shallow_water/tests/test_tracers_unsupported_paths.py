"""The paths that would silently mishandle tracers must refuse instead.

Tracers are not Quantity objects, so machinery that walks `domain.quantities`
does not see them:

  distribute() copies state by walking quantities, so the sub-domains would be
               built with no tracers at all (#278)

It raises until that is implemented. This test exists so the guard cannot be
removed without replacing it.

reorder() had the same problem (#277) and now permutes the tracer blocks
properly -- see test_tracers_reorder.py.
"""

import numpy as num
import pytest

import anuga


def _domain(n=4):
    d = anuga.rectangular_cross_domain(n, n)
    d.set_quantity('elevation', 0.0)
    d.set_quantity('stage', 1.0)
    b = anuga.Reflective_boundary(d)
    d.set_boundary({'left': b, 'right': b, 'top': b, 'bottom': b})
    return d


def test_reorder_without_tracers_still_works():
    # The guard must not disturb the ordinary path.
    d = _domain()
    n = len(d)
    order = num.arange(n - 1, -1, -1)      # reverse
    d.reorder(order)                        # must not raise
    assert len(d) == n


def test_distribute_with_tracers_refuses():
    # distribute() returns the domain unchanged on 1 process, so the guard is
    # checked ahead of that bypass; on 1 process it must NOT fire, since a
    # serial run is unaffected.
    from anuga.parallel.parallel_api import distribute
    from anuga import numprocs

    d = _domain()
    d.add_tracer('salinity', initial_value=0.02)

    if numprocs == 1:
        assert distribute(d) is d
    else:
        with pytest.raises(NotImplementedError, match='278'):
            distribute(d)
