#!/usr/bin/python
import unittest
import numpy as np

from pymatgen import Structure, Lattice, PeriodicSite
from pyabinitio.compressive.sense_matrix.structures import CoordData
from pyabinitio.compressive.sense_matrix.gaussian import GaussianTripletColumns

np.set_printoptions(threshold = np.NAN, linewidth = np.NAN, suppress = np.NAN)

class TestGaussian(unittest.TestCase):
    
    def test1(self):
        l = Lattice.cubic(100)
        s1 = PeriodicSite('Li', [99.5,0,1], l, coords_are_cartesian=True)
        s2 = PeriodicSite('Sn', [99.5,1,0], l, coords_are_cartesian=True)
        s3 = PeriodicSite('Li', [0.5,0,0], l, coords_are_cartesian=True)
        
        s = Structure.from_sites([s1, s2, s3])
        coords = np.array(s.cart_coords)[:, None, :]
        cd = CoordData(coords, s)
        gtc = GaussianTripletColumns(rrange = np.arange(1, 1.1, 1), c = 0.5)
        f = gtc.make(cd)
        
        self.assertAlmostEqual(f[0,0], 0)
        self.assertAlmostEqual(f[1,1], -0.41849511)
        self.assertAlmostEqual(f[2,1], 0.83699023)
        self.assertAlmostEqual(f[3,4], -0.83699023)
        
        gtc = GaussianTripletColumns(rrange = np.arange(1, 1.1, 1), c = 1)
        f = gtc.make(cd)
        self.assertAlmostEqual(f[0,0], 0)
        self.assertAlmostEqual(f[1,1], -0.2264329)
        self.assertAlmostEqual(f[2,1], 0.45286586)
        self.assertAlmostEqual(f[3,4], -0.45286586)
        
    def test2(self):
        l = Lattice.cubic(100)
        s1 = PeriodicSite('Li', [99.5,0,1.1], l, coords_are_cartesian=True)
        s2 = PeriodicSite('Sn', [99.5,1,0], l, coords_are_cartesian=True)
        s3 = PeriodicSite('Li', [0.5,0,0], l, coords_are_cartesian=True)
        s4 = PeriodicSite('Li', [59.5,0,1.1], l, coords_are_cartesian=True)
        s5 = PeriodicSite('Sn', [59.5,1,0], l, coords_are_cartesian=True)
        s6 = PeriodicSite('Li', [60.5,0,0], l, coords_are_cartesian=True)
        s = Structure.from_sites([s1, s2, s3, s4, s5, s6])
        coords = np.array(s.cart_coords)[:, None, :]
        cd = CoordData(coords, s)
        gtc = GaussianTripletColumns(rrange = np.arange(1, 1.1, 1), c = 0.5)
        f = gtc.make(cd)
        self.assertAlmostEqual(f[9,0], 0)
        self.assertAlmostEqual(f[10,1], -0.36030969)
        self.assertAlmostEqual(f[11,1], 0.79268132)
        self.assertAlmostEqual(f[12,4], -0.64481206)
        self.assertAlmostEqual(f[13,4], 1.36543144)
        
        self.assertAlmostEqual(np.sum(f), 0)
        
    def test3(self):    
        l = Lattice.cubic(100)
        s1 = PeriodicSite('Li', [99.5,0,1], l, coords_are_cartesian=True)
        s2 = PeriodicSite('Sn', [99.5,1,0], l, coords_are_cartesian=True)
        s3 = PeriodicSite('Li', [0.5,0,0], l, coords_are_cartesian=True)
        
        s = Structure.from_sites([s1, s2, s3])
        coords = np.array(s.cart_coords)[:, None, :]
        cd = CoordData(coords, s)
        gtc = GaussianTripletColumns(rrange = [1,10], c = 0.5)
        f = gtc.make(cd)
        self.assertAlmostEqual(f[0,0], 0)
        self.assertAlmostEqual(f[1,8], -0.41849511)
        self.assertAlmostEqual(f[2,8], 0.83699023)
        self.assertAlmostEqual(f[3,32], -0.83699023)

    def test4(self):    
        l = Lattice.cubic(100)
        s1 = PeriodicSite('Li', [99.5,0,1], l, coords_are_cartesian=True)
        s2 = PeriodicSite('Sn', [99.5,1,0], l, coords_are_cartesian=True)
        s3 = PeriodicSite('Li', [0.5,0,0], l, coords_are_cartesian=True)
        
        s = Structure.from_sites([s1, s2, s3])
        coords = np.array(s.cart_coords)[:, None, :]
        cd = CoordData(coords, s)
        gtc = GaussianTripletColumns(rrange = [1,1.01], c = 0.5)
        f = gtc.make(cd)
        self.assertAlmostEqual(f[0,0], 0)
        self.assertAlmostEqual(f[1,8], -0.41849511)
        self.assertAlmostEqual(f[2,8], 0.83699023)
        self.assertAlmostEqual(f[3,32], -0.83699023)
        
        self.assertAlmostEqual(np.sum(f), 0)
        
    def test5(self):  
        l = Lattice.cubic(100)
        s1 = PeriodicSite('Li', [99.5,0,1], l, coords_are_cartesian=True)
        s2 = PeriodicSite('Sn', [99.5,1,0], l, coords_are_cartesian=True)
        s3 = PeriodicSite('Mg', [0.5,0,0], l, coords_are_cartesian=True)
        s = Structure.from_sites([s1, s2, s3])
        coords = np.array(s.cart_coords)[:, None, :]
        cd = CoordData(coords, s)
        gtc = GaussianTripletColumns(rrange = [1], c = 0.5)
        f = gtc.make(cd)
        self.assertAlmostEqual(f[0,0], 0)
        self.assertAlmostEqual(f[1,5], -0.41849511)
        self.assertAlmostEqual(f[2,7], 0.83699023)
        self.assertAlmostEqual(f[8,11], -0.41849511)
        
        self.assertAlmostEqual(np.sum(f), 0)
        
        
        
    