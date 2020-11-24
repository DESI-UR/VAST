# Licensed under a 3-clause BSD-style license - see LICENSE.
# -*- coding: utf-8 -*-
"""
==========
VoidFinder
==========
Implementation of the galaxy VoidFinder algorithm by Hoyle and Vogeley (2002),
based on the algorithm described by El-Ad and Piran (1997).
"""
from __future__ import absolute_import

name = 'voidfinder'
__version__ = '0.3.7'

# This line allows you to do "from vast.voidfinder import filter_galaxies" 
# instead of "from vast.voidfinder.voidfinder import filter galaxies"
from .voidfinder import filter_galaxies, \
                        ra_dec_to_xyz, \
                        calculate_grid, \
                        wall_field_separation, \
                        find_voids
