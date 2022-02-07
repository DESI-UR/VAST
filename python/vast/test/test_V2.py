# -*- coding: utf-8 -*-
"""Unit tests of utilities in the vsquared submodule.
"""
import unittest

from vast.vsquared import util, classes, zobov

import numpy as np

class TestV2Catalog(unittest.TestCase):

    def setUp(self):
        TestV2Catalog.catfile = 'python/vast/vsquared/data/test_data.fits'
        TestV2Catalog.cat = classes.Catalog(TestV2Catalog.catfile, 16, 0.03, 0.1)

    def testCoord(self):
        return self.assertTrue(True)
