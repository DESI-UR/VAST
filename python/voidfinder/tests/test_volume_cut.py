from astropy.table import Table

import sys
sys.path.insert(1, '/Users/kellydouglass/Documents/Research/VoidFinder/python/')

from voidfinder.voidfinder_functions import save_maximals
from voidfinder.hole_combine import combine_holes

maskra = 360
maskdec = 180
min_dist = 0.
max_dist = 300.
dec_offset = -90


################################################################################
# MASK HAS CHANGED --- WILL NOT RUN WITH LATEST UPDATE
#-------------------------------------------------------------------------------
maskfile = Table.read('cbpdr7mask.dat', format='ascii.commented_header')
mask = np.zeros((maskra, maskdec))
mask[maskfile['ra'].astype(int), maskfile['dec'].astype(int) - dec_offset] = 1
################################################################################

holes_table = Table.read('potential_voids_list.txt', format='ascii.commented_header')
potential_voids_table = volume_cut(holes_table, mask, [min_dist, max_dist])

maximal_spheres_table, myvoids_table = combine_holes(potential_voids_table, 0.1)

print('Number of unique voids is', len(maximal_spheres_table))
print('Number of void holes is', len(myvoids_table))

save_maximals(maximal_spheres_table, 'maximal_spheres_test.txt')



'''import numpy as np
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt

num_bins = 20
n, bins, patches = plt.hist(volumes_list, num_bins, facecolor='mediumvioletred', histtype='stepfilled')
plt.xlabel('Percent of Volume Outside of Mask')
plt.ylabel('Number of Holes')
plt.title('Histogram of Volume Percentages of Holes Around Border of Survey')
plt.show()
'''