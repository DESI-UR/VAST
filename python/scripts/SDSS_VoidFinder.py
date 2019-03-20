'''VoidFinder - Hoyle & Vogeley (2002)'''

################################################################################
#
#   IMPORT MODULES
#
################################################################################


import sys
sys.path.insert(1, '/home/moose/VoidFinder/python/')

from voidfinder import filter_galaxies, find_voids
from astropy.table import Table
import pickle
import numpy as np


################################################################################
#
#   USER INPUTS
#
################################################################################


survey_name = 'SDSS_dr7_'

# File header
directory = '/home/moose/VoidFinder/python/voidfinder/data/'

# Input file names
in_filename = directory + 'vollim_dr7_cbp_102709.dat'  # File format: RA, dec, redshift, comoving distance, absolute magnitude
#mask_filename = directory + 'cbpdr7mask.dat'           # File format: RA, dec
mask_filename = directory + 'vollim_dr7_cbp_102709_mask.npy'


# Output file names
out1_filename = in_filename[:-4] + '_maximal.txt'                 # List of maximal spheres of each void region: x, y, z, radius, distance, ra, dec
out2_filename = in_filename[:-4] + '_holes.txt'                   # List of holes for all void regions: x, y, z, radius, flag (to which void it belongs)
#out3_filename = in_filename[:-4] + 'out3_vollim_dr7.txt'          # List of void region sizes: radius, effective radius, evolume, x, y, z, deltap, nfield, vol_maxhole
#voidgals_filename = in_filename[:-4] + 'vollim_voidgals_dr7.txt'  # List of the void galaxies: x, y, z, void region


# Survey parameters
min_dist = 0
max_dist = 300. # z = 0.107 -> 313 h-1 Mpc   z = 0.087 -> 257 h-1 Mpc


################################################################################
#
#   OPEN FILES
#
################################################################################


infile = Table.read(in_filename, format='ascii.commented_header')
#maskfile = Table.read(mask_filename, format='ascii.commented_header')
maskfile = np.load(mask_filename)


################################################################################
#
#   FILTER GALAXIES
#
################################################################################


#print(maskfile)
#exit()

'''
coord_min_table, mask, ngrid = filter_galaxies(infile, maskfile, min_dist, max_dist, survey_name, True, True)

temp_outfile = open("filter_galaxies_output.pickle", 'wb')
pickle.dump((coord_min_table, mask, ngrid), temp_outfile)
temp_outfile.close()
'''



################################################################################
#
#   FIND VOIDS
#
################################################################################


temp_infile = open("filter_galaxies_output.pickle", 'rb')
coord_min_table, mask, ngrid = pickle.load(temp_infile)
temp_infile.close()


find_voids(ngrid, min_dist, max_dist, coord_min_table, mask, out1_filename, out2_filename, survey_name)
