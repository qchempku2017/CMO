#!/usr/bin/python
import unittest
import numpy as np

import pymatgen as mg
structure = mg.Structure.from_file("LiCoO2.cif")
from pyabinitio.compressive.sense_matrix.structures import CoordData, ExpColumns, \
                                                           RInvColumns, SenseMatrix, \
                                                           AxilrodTellerColumns, \
                                                           GaussianTripletColumns

np.set_printoptions(threshold = np.NAN, linewidth = np.NAN, suppress = np.NAN)

class TestStructures(unittest.TestCase):

    def setUp(self):
        self.s = read_structure('LiCoO2.cif')
        coords = np.array(self.s.cart_coords)
        coords = np.concatenate([coords[:, None, :]] * 2, axis = 1)
        self.cd = CoordData(coords, self.s, cutoff = 5, ntol = 1e-8)

    def testCoordData(self):
        self.assertEqual(len(self.cd.vectors[self.cd.distances < 5]), 616)
        self.assertEqual(len(self.cd.vectors[self.cd.distances < 2]), 24)
        self.assertEqual(len(self.cd.vectors[self.cd.distances < 2.15]), 48)

    def testExpColumns(self):
        ec = ExpColumns(2)
        sense = ec.make(self.cd)
        expected = np.zeros((24, 9))
        expected[13][-3:] = [0.02714011, -0.0330112, -0.0074453]
        expected[16][-3:] = [0.02714011, -0.0330112, -0.0074453]
        expected[19][-3:] = [-0.02714011, 0.0330112, 0.0074453]
        expected[22][-3:] = [-0.02714011, 0.0330112, 0.0074453]
        self.assertTrue(np.allclose(sense, expected))

    def testRInvColumns(self):
        ec = RInvColumns(6)
        sense = ec.make(self.cd)
        expected = np.zeros((24, 9))
        expected[13][-3:] = [0.02102591, -0.02988213, -0.00454469]
        expected[16][-3:] = [0.02102591, -0.02988213, -0.00454469]
        expected[19][-3:] = [-0.02102591, 0.02988213, 0.00454469]
        expected[22][-3:] = [-0.02102591, 0.02988213, 0.00454469]
        self.assertTrue(np.allclose(sense, expected))

    def testSenseMatrix(self):
        sm = SenseMatrix([RInvColumns(6), ExpColumns(2)])
        sense = sm.make(self.cd)
        expected = np.zeros((24, 18))
        expected[13][6:9] = [0.02102591, -0.02988213, -0.00454469]
        expected[16][6:9] = [0.02102591, -0.02988213, -0.00454469]
        expected[19][6:9] = [-0.02102591, 0.02988213, 0.00454469]
        expected[22][6:9] = [-0.02102591, 0.02988213, 0.00454469]
        expected[13][-3:] = [0.02714011, -0.0330112, -0.0074453]
        expected[16][-3:] = [0.02714011, -0.0330112, -0.0074453]
        expected[19][-3:] = [-0.02714011, 0.0330112, 0.0074453]
        expected[22][-3:] = [-0.02714011, 0.0330112, 0.0074453]
        self.assertTrue(np.allclose(sense, expected))

    def testBondOrder(self):
        return
        n_images = self.cd.positions.shape[1]
        b_o = self.cd.voronoi_bond_orders
        print (b_o[:, :, n_images/2, 0])

    def testAxilrodTellerColumns(self):
        return
        at = AxilrodTellerColumns()
        print (at.make(self.cd))

    def testGaussianTripletColumns(self):
        gtc = GaussianTripletColumns()
        gtc.make(self.cd).shape
