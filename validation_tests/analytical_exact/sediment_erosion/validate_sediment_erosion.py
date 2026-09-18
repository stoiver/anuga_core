"""Automatic verification of ANUGA flows.
See functions exercised by this wrapper for more details
"""
import json
import os
import sys
import unittest

import numpy
import anuga

args = anuga.get_args()
indent = anuga.indent
verbose = args.verbose


class Test_results(unittest.TestCase):

    def setUp(self):
        for file in os.listdir('.'):
            if file.endswith('.stdout') or \
                    file.endswith('.sww') or \
                    file.endswith('.msh') or \
                    file.endswith('.png'):
                os.remove(file)

    def tearDown(self):
        pass

    def test_sediment_erosion(self):
        if verbose:
            print()
            print(indent + 'Running simulation script')

        s = 'numerical_sediment_erosion.py'
        res = anuga.run_anuga_script(s, args=args)

        # Test that script runs ok
        assert res == 0

        if verbose:
            print(indent + 'Testing accuracy')

        import analytical_sediment_erosion as analytic

        with open('erosion_parameters.json') as f:
            prm = json.load(f)

        t, c_num, z_num, w_num, uh, vh = analytic.read_centroid_series('sediment_erosion.sww')

        S_cell = numpy.array(prm['S_cell'])
        h_cell = numpy.array(prm['h_cell'])
        x_cell = numpy.array(prm['x_cell'])
        law = (prm['v_s'], prm['R'], prm['diameter'], prm['tau_c_star'],
               prm['gamma0'], prm['d_star'])

        # The slope each cell was built with: 0 on the flat half, S_bed beyond
        S_built = numpy.where(x_cell > prm['L_flat'], prm['S_bed'], 0.0)
        # The solver reconstructs bed edge values from stage - height, and the
        # limiter reduces the slope of the cells at the kink and along the
        # far wall. Those are compared against the slope they were given;
        # every other cell against the constructed one, which must agree.
        recon = numpy.abs(S_cell - S_built) > 1.0e-12
        flat = S_built == 0.0
        sloped = (S_built > 0.0) & ~recon

        c_ref = analytic.erosion(t, h_cell, S_cell, *law)
        c_eq = analytic.equilibrium_concentration(h_cell, S_cell, *law)

        # Relative L1 errors over the time series, cell by cell
        ec = numpy.sum(numpy.abs(c_num[:, sloped] - c_ref[:, sloped])) \
            / numpy.sum(numpy.abs(c_ref[:, sloped]))
        ec_recon = numpy.sum(numpy.abs(c_num[:, recon] - c_ref[:, recon])) \
            / numpy.sum(numpy.abs(c_ref[:, recon]))
        # Below threshold: nothing may be entrained
        c_flat = numpy.abs(c_num[:, flat]).max()
        # Equilibrium at the end (t_final >> h/v_s)
        e_eq = numpy.abs(c_num[-1, sloped] / c_eq[sloped] - 1.0).max()
        # Still water over a fixed bed: nothing else may move
        dw = numpy.abs(w_num - w_num[0]).max()
        dz = numpy.abs(z_num - z_num[0]).max()
        mom = max(numpy.abs(uh).max(), numpy.abs(vh).max())

        print()
        print(indent + 'Reconstruction-limited cells (kink and far wall): %d of %d'
              % (recon.sum(), len(S_cell)))
        print(indent + 'Relative L1 error in concentration, sloped interior: %.3e' % ec)
        print(indent + 'Relative L1 error in concentration, limited cells:  %.3e' % ec_recon)
        print(indent + 'Max concentration on the flat (control) half:       %.3e' % c_flat)
        print(indent + 'Max relative departure from c_eq at the end:        %.3e' % e_eq)
        print(indent + 'Max free-surface movement (m):                      %.3e' % dw)
        print(indent + 'Max bed movement (m):                               %.3e' % dz)
        print(indent + 'Max |momentum|:                                     %.3e' % mom)
        print(indent + 'Concentration at the end: numerical %.4e, reference %.4e'
              % (c_num[-1, sloped].mean(), c_ref[-1, sloped].mean()))

        # The source is a first-order fractional step, so the error scales
        # with dt / (h / v_s): a few 1e-4 here.
        assert ec < 0.01, 'concentration relaxation off by %.2e' % ec
        assert ec_recon < 0.01, 'limited cells off by %.2e' % ec_recon
        assert c_flat < 1.0e-12, 'entrainment below threshold: %.2e' % c_flat
        assert e_eq < 0.01, 'equilibrium concentration off by %.2e' % e_eq
        assert dw < 1.0e-6, 'free surface moved by %.2e m' % dw
        assert dz < 1.0e-12, 'fixed bed moved by %.2e m' % dz
        assert mom < 1.0e-8, 'still water acquired momentum %.2e' % mom


if __name__ == '__main__':
    suite = unittest.TestLoader().loadTestsFromTestCase(Test_results)
    runner = unittest.TextTestRunner(verbosity=2)
    runner.run(suite)
