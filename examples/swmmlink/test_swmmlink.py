#!/usr/bin/env python

""" Test suite to test polygon functionality. """

import unittest
import numpy as np
import os

from pyswmm import Simulation, Nodes, Links

class Test_SwmmLink(unittest.TestCase):
    
    def setUp(self):
        sim = Simulation('./swmm_pipe_test.inp')

        node_names = ['Inlet', 'Outlet']
        link_names = ['Culvert']
    
        self.nodes = [Nodes(sim)[names] for names in node_names]
        self.links = [Links(sim)[names] for names in link_names]


    def tearDown(self):
        self.nodes = None
        self.links = None

    def test_establishment_of_a_swmmlink(self):
        for node in self.nodes:
            print node


        
################################################################################

if __name__ == "__main__":
    suite = unittest.makeSuite(Test_SwmmLink, 'test')
    runner = unittest.TextTestRunner()
    runner.run(suite)
