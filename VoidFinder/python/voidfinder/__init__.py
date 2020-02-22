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
from ._version import __version__


#This line allows you to do "from voidfinder import filter_galaxies" instead of
#"from voidfinder.voidfinder import filter galaxies"
from .voidfinder import filter_galaxies, \
                        find_voids, \
                        filter_galaxies_2, \
                        ra_dec_to_xyz, \
                        calculate_grid, \
                        wall_field_separation