"""Regression tests for fractional-step operator evaluation timing in the
shallow-water evolve loop.

These run in mode-1 (legacy) only, so they execute on any build (no GPU
extension required) — unlike the mode-1-vs-mode-2 checks in
``test_DE_gpu_omp.py::Test_GPU_OperatorTimeAlignment``.

Background: fractional-step operators are applied by the evolve loop *before* it
advances ``relative_time`` from t to t+dt, so they should evaluate forcing at the
pre-step time ``t``. A latent bug left the mode-1 rk2 (DE1) body with
``relative_time`` advanced to t+dt, so DE1's operators evaluated forcing "one step
too far" (t+dt) — unlike DE0/DE_ader2 (and the mode-2 GPU loops), and diverging
from mode-2 by ~4e-4 for a time-varying rate. No prior test used a time-varying
operator, so the bug was invisible. Reverting the mode-1 rk2 fix makes
``test_rk2_operator_evaluated_at_pre_step_time`` fail.
"""

import tempfile

import anuga
from anuga import rectangular_cross_domain, Reflective_boundary, Rate_operator


def _max_operator_eval_time(algorithm, finaltime=3.0, yieldstep=1.0):
    """Return (max time a time-varying Rate_operator was evaluated at, finaltime).

    With pre-step evaluation the last inner step's operator sees t < finaltime;
    with the buggy post-step evaluation it lands exactly on finaltime.
    """
    seen = []
    d = rectangular_cross_domain(12, 8, len1=100.0, len2=100.0)
    d.set_flow_algorithm(algorithm)
    d.set_name('op_timing_%s' % algorithm)
    d.set_datadir(tempfile.mkdtemp())
    d.store = False
    d.set_quantity('elevation', 0.0)
    d.set_quantity('friction', 0.03)
    d.set_quantity('stage', 0.3)
    # rate is a function of time so the pre/post-step distinction matters; the
    # value is irrelevant here, we only record the times it is evaluated at.
    Rate_operator(d, rate=lambda t: (seen.append(float(t)) or 0.001))
    d.set_boundary({b: Reflective_boundary(d) for b in d.get_boundary_tags()})
    d.set_multiprocessor_mode(1)  # legacy / CPU path
    for _ in d.evolve(yieldstep=yieldstep, finaltime=finaltime):
        pass
    return max(seen), finaltime


def test_rk2_operator_evaluated_at_pre_step_time():
    """DE1 (rk2) must evaluate a time-varying operator at the pre-step time t.

    The last inner step advances to finaltime, so a pre-step operator sees a time
    strictly less than finaltime. Reverting the mode-1 rk2 relative_time fix makes
    DE1 evaluate at t+dt and land exactly on finaltime -> this assertion fails.
    """
    mx, ft = _max_operator_eval_time('DE1')
    assert mx < ft, (
        'DE1 evaluated a time-varying operator at t=%r (>= finaltime=%r); '
        'expected the pre-step time t (< finaltime).' % (mx, ft))


def test_de0_operator_evaluated_at_pre_step_time():
    """DE0 (Euler) reference — already pre-step; guards a shared regression."""
    mx, ft = _max_operator_eval_time('DE0')
    assert mx < ft, 'DE0 operator evaluated at t=%r (>= finaltime=%r).' % (mx, ft)


def test_de_ader2_operator_evaluated_at_pre_step_time():
    """DE_ader2 reference — pre-step."""
    mx, ft = _max_operator_eval_time('DE_ader2')
    assert mx < ft, 'DE_ader2 operator evaluated at t=%r (>= finaltime=%r).' % (mx, ft)
