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
        TestV2.cat = classes.Catalog(TestV2.catfile, 16, 0.03, 0.1)
        TestV2.tess = classes.Tesselation(TestV2.cat)

    def test_cat_coord(self):
        """Check catalog coordinate access
        """
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

    def test_zobov_tess(self):
        """Test ZOBOV tesselation
        """
        self.assertTrue(np.isclose(np.mean(TestV2.tess.volumes), 1600.190056988941))
        self.assertTrue(np.isclose(np.mean([len(nn) for nn in TestV2.tess.neighbors]), 16.06))

