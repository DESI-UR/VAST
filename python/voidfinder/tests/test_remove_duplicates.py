from astropy.table import Table

import sys
sys.path.insert(1, '/Users/kellydouglass/Documents/Research/VoidFinder/python/')

from voidfinder.hole_combine import remove_duplicates



spheres = Table.read('../data/vollim_dr7_cbp_102709_holes.txt', format='ascii.commented_header')


unique_spheres = remove_duplicates(spheres)

unique_spheres.write('../data/vollim_dr7_cbp_102709_holes_unique.txt', format='ascii.commented_header')