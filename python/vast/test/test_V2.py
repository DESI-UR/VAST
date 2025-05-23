# -*- coding: utf-8 -*-
"""Unit tests of utilities in the vsquared submodule.
"""
import unittest

from vast.vsquared import util, classes, zobov

import os
import copy
import numpy as np
import configparser

class TestV2(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # Set up global varaibles for tests.
        TestV2.inifile = 'python/vast/vsquared/tests/test_config.ini'
        TestV2.catfile = 'python/vast/vsquared/tests/test_data.fits'

        TestV2.cat = None
        TestV2.nside = 16
        TestV2.tess = None
        TestV2.zones = None
        TestV2.voids = None
        TestV2.zobov = None

    def test_cat_coord(self):
        """Check catalog coordinate access
        """
        config = configparser.ConfigParser()
        config.read(TestV2.inifile)
        TestV2.zobov = zobov.Zobov(TestV2.inifile, save_intermediate=False)
        TestV2.cat = classes.Catalog(TestV2.catfile, TestV2.nside, 0.03, 0.1, config['Galaxy Column Names'],zobov=TestV2.zobov)

        mcoord = np.array([-158.0472400951847,-19.01100010666949,94.40978960900837])
        self.assertTrue(np.isclose(np.mean(TestV2.cat.coord.T[0]), mcoord[0]))
        self.assertTrue(np.isclose(np.mean(TestV2.cat.coord.T[1]), mcoord[1]))
        self.assertTrue(np.isclose(np.mean(TestV2.cat.coord.T[2]), mcoord[2]))

    def test_cat_nnls(self):
        """Check catalog nnls
        """
        self.assertEqual(len(self.cat.nnls[self.cat.nnls==-1]), 1800)
        self.assertEqual(len(self.cat.nnls[self.cat.nnls==np.arange(10000)]), 8200)

    def test_cat_masks(self):
        """Check catalog mask
        """
        self.assertEqual(len(self.cat.mask[self.cat.mask]), 576)

    def test_cat_imsk(self):
        """Check catalog imsk
        """
        self.assertEqual(len(self.cat.imsk[self.cat.imsk]), 8200)

    def test_zobov_1_tess(self):
        """Test ZOBOV tesselation
        """
        TestV2.tess = classes.Tesselation(TestV2.cat, TestV2.nside)

        self.assertTrue(np.isclose(np.mean(TestV2.tess.volumes), 1600.190056988941))
        
        # Switched over to using the scipy data structures for this, so this
        # assertion will just not pass anymore, replaced it with code below
        #self.assertTrue(np.isclose(np.mean([len(nn) for nn in TestV2.tess.neighbors]), 16.06))
        
        indptr, neigh_gal_indices = TestV2.tess.neighbors
        num_gals = TestV2.tess.num_gals
        #print("Derp", np.mean([len(nn) for nn in TestV2.tess.neighbors]))
        
        out = []
        for idx in range(num_gals):
            
            neighs = neigh_gal_indices[indptr[idx]:indptr[idx+1]]
            
            #Should really be checking that these are the correct neighbors
            #in some way rather than just the mean of the lengths
            out.append(len(neighs))
            
        self.assertTrue(np.isclose(np.mean(out), 15.06))
        
        
        



    def test_zobov_2_zones(self):
        """Test ZOBOV zone formation
        """
        TestV2.zones = classes.Zones(TestV2.tess, catalog=TestV2.cat)

        # Test zone cells
        
        diff = np.abs(np.mean([len(zc) for zc in self.zones.zcell]) - 87.23404255319149)
        
        print(diff)
        self.assertTrue(diff/87.23404255319149 <= .01)

        # Test zone volumes
        
        #print("TestZobov 2 Zones", np.mean(self.zones.zvols))
        #Py 3.10 - 6897.755361237265
        #Py 3.11 - 6897.755361237265
        
        diff = np.abs(np.mean(self.zones.zvols) - 6897.767791048626)
        
        self.assertTrue(diff/6897.767791048626 <= .01)

        # Test zone links
        diff = np.abs(np.mean([len(zl0) for zl0 in self.zones.zlinks[0]]) - 9.617021276595745)
        self.assertTrue(diff/9.617021276595745 <= 0.01)
        
        diff = np.abs(np.mean([np.mean(zl1) for zl1 in self.zones.zlinks[1][:-1]]) - 3285.6313303024826)
        self.assertTrue(diff/3285.6313303024826 <= 0.01)

    def test_zobov_3_voids(self):
        """Test ZOBOV void creation
        """
        TestV2.voids = classes.Voids(TestV2.zones)

        # Test voids
        self.assertTrue(np.isclose(np.mean([len(v) for v in self.voids.voids]), 1.978494623655914))
        self.assertTrue(np.isclose(np.mean([np.mean([len(vv) for vv in v]) for v in self.voids.voids]), 1.060548559599793))

        # Test mvols
        self.assertTrue(np.isclose(np.mean(self.voids.mvols), 6971.937337188934))

        # Test ovols
        self.assertTrue(np.isclose(np.mean([np.mean(ov) for ov in self.voids.ovols]), 5875.14756763797))

    def test_zobov_4_zobov(self):
        """Test full ZOBOV algorithm
        """
        TestV2.zobov = zobov.Zobov(TestV2.inifile, save_intermediate=False)

        # Sort voids.
        self.assertFalse(hasattr(TestV2.zobov, 'vrads')) # before sortVoids

        zobov1 = copy.deepcopy(TestV2.zobov)
        zobov2 = copy.deepcopy(TestV2.zobov)
        zobov3 = copy.deepcopy(TestV2.zobov)
        zobov4 = copy.deepcopy(TestV2.zobov)
        TestV2.zobov.sortVoids()
        zobov1.sortVoids(1)
        zobov2.sortVoids(2)
        #zobov3.sortVoids(3)
        zobov4.sortVoids(4)
        self.assertEqual(len(TestV2.zobov.vrads), 62)
        self.assertEqual(len(zobov1.vrads), 87)
        self.assertEqual(len(zobov2.vrads), 14)
        #self.assertEqual(len(zobov3.vrads), 87)
        self.assertEqual(len(zobov4.vrads), 87)

        # Save voids.
        self.assertTrue(hasattr(TestV2.zobov, 'vrads')) # after sortVoids

        TestV2.zobov.saveVoids()

        self.assertTrue(os.path.exists('TEST_V2_VIDE_Output.fits'))

        # Save zones.
        TestV2.zobov.saveZones()
        
        self.assertTrue(os.path.exists('TEST_V2_VIDE_Output.fits'))

    def tearDown(self):
        """Delete files produced for the unit tests.
        """
        files = [ 'TEST_V2_VIDE_Output.fits' ]
        for f in files:
            if os.path.exists(f):
                os.remove(f)
