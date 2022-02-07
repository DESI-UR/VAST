# -*- coding: utf-8 -*-
"""Unit tests of utilities in the vsquared submodule.
"""
import unittest

from vast.vsquared import util, classes, zobov

import numpy as np

class TestV2(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # Set up global varaibles for tests.
        TestV2.catfile = 'python/vast/vsquared/data/test_data.fits'
        TestV2.cat = None
        TestV2.tess = None
        TestV2.zones = None
        TestV2.voids = None

    def test_cat_coord(self):
        """Check catalog coordinate access
        """
        TestV2.cat = classes.Catalog(TestV2.catfile, 16, 0.03, 0.1)

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
        TestV2.tess = classes.Tesselation(TestV2.cat)

        self.assertTrue(np.isclose(np.mean(TestV2.tess.volumes), 1600.190056988941))
        self.assertTrue(np.isclose(np.mean([len(nn) for nn in TestV2.tess.neighbors]), 16.06))

    def test_zobov_2_zones(self):
        """Test ZOBOV zone formation
        """
        TestV2.zones = classes.Zones(TestV2.tess)

        # Test zone cells
        self.assertTrue(np.isclose(np.mean([len(zc) for zc in self.zones.zcell]), 82.82828282828282))

        # Test zone volumes
        self.assertTrue(np.isclose(np.mean(self.zones.zvols), 6549.395680389604))

        # Test zone links
        self.assertTrue(np.isclose(np.mean([len(zl0) for zl0 in self.zones.zlinks[0]]), 12.626262626262626))
        self.assertTrue(np.isclose(np.mean([np.mean(zl1) for zl1 in self.zones.zlinks[1]]), 2345.915247204872))

    def test_zobov_3_voids(self):
        """Test ZOBOV void creation
        """
        TestV2.voids = classes.Voids(TestV2.zones)

        # Test voids
        self.assertTrue(np.isclose(np.mean([len(v) for v in self.voids.voids]), 1.9191919191919191))
        self.assertTrue(np.isclose(np.mean([np.mean([len(vv) for vv in v]) for v in self.voids.voids]), 1.0578692450350204))

        # Test mvols
        self.assertTrue(np.isclose(np.mean(self.voids.mvols), 6549.3956803896035))

        # Test ovols
        self.assertTrue(np.isclose(np.mean([np.mean(ov) for ov in self.voids.ovols]), 5519.078018084154))

