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
        _, c_ctl, _, _, _, _ = analytic.read_centroid_series('sediment_erosion.sww',
                                                             name='control')

        h_cell = numpy.array(prm['h_cell'])
        law = (prm['v_s'], prm['R'], prm['diameter'], prm['tau_c_star'],
               prm['gamma0'], prm['d_star'])

        # Every cell against its own curve, at the one slope the bed was
        # built with: the kernel's slope estimate must reproduce it in
        # every cell, the ones along the walls and in the corners included.
        c_ref = analytic.erosion(t, h_cell, prm['S_bed'], *law)
        c_eq = analytic.equilibrium_concentration(h_cell, prm['S_bed'], *law)

        # Relative L1 error over the time series, all cells
        ec = numpy.sum(numpy.abs(c_num - c_ref)) / numpy.sum(numpy.abs(c_ref))
        # ... and the worst single cell, so a wall cell cannot hide in the mean
        ec_cell = (numpy.abs(c_num - c_ref).sum(axis=0)
                   / numpy.abs(c_ref).sum(axis=0)).max()
        # Below threshold: nothing may be entrained into the control fraction
        c_control = numpy.abs(c_ctl).max()
        # Equilibrium at the end (t_final >> h/v_s)
        e_eq = numpy.abs(c_num[-1] / c_eq - 1.0).max()
        # Still water over a fixed bed: nothing else may move
        dw = numpy.abs(w_num - w_num[0]).max()
        dz = numpy.abs(z_num - z_num[0]).max()
        mom = max(numpy.abs(uh).max(), numpy.abs(vh).max())

        print()
        print(indent + 'Relative L1 error in concentration, all cells:  %.3e' % ec)
        print(indent + 'Relative L1 error in concentration, worst cell: %.3e' % ec_cell)
        print(indent + 'Max concentration of the control fraction:      %.3e' % c_control)
        print(indent + 'Max relative departure from c_eq at the end:    %.3e' % e_eq)
        print(indent + 'Max free-surface movement (m):                  %.3e' % dw)
        print(indent + 'Max bed movement (m):                           %.3e' % dz)
        print(indent + 'Max |momentum|:                                 %.3e' % mom)
        print(indent + 'Concentration at the end: numerical %.4e, reference %.4e'
              % (c_num[-1].mean(), c_ref[-1].mean()))

        # The source is a first-order fractional step, so the error scales
        # with dt / (h / v_s): a few 1e-4 here.
        assert ec < 0.01, 'concentration relaxation off by %.2e' % ec
        assert ec_cell < 0.01, 'a cell is off by %.2e' % ec_cell
        assert c_control < 1.0e-12, 'entrainment below threshold: %.2e' % c_control
        assert e_eq < 0.01, 'equilibrium concentration off by %.2e' % e_eq
        assert dw < 1.0e-6, 'free surface moved by %.2e m' % dw
        assert dz < 1.0e-12, 'fixed bed moved by %.2e m' % dz
        assert mom < 1.0e-8, 'still water acquired momentum %.2e' % mom


if __name__ == '__main__':
    suite = unittest.TestLoader().loadTestsFromTestCase(Test_results)
    runner = unittest.TextTestRunner(verbosity=2)
    runner.run(suite)
