'''VoidFinder - Hoyle & Vogeley (2002)'''

################################################################################
#
#   IMPORT MODULES
#
################################################################################


import sys
#sys.path.insert(1, '/home/moose/VoidFinder/python/')
sys.path.insert(1, '/Users/kellydouglass/Documents/Research/VoidFinder/python/')

from voidfinder import filter_galaxies, find_voids
from voidfinder.absmag_comovingdist_functions import Distance

from astropy.table import Table
import pickle
import numpy as np


################################################################################
#
#   USER INPUTS
#
################################################################################


# Number of CPUs available for analysis
num_cpus = 1

#-------------------------------------------------------------------------------
survey_name = 'SDSS_dr7_'

# File header
in_directory = '/Users/kellydouglass/Documents/Research/VoidFinder/python/voidfinder/data/'
out_directory = '/Users/kellydouglass/Documents/Research/VoidFinder/python/voidfinder/data/'

# Input file names
galaxies_filename = 'vollim_dr7_cbp_102709.dat'  # File format: RA, dec, redshift, comoving distance, absolute magnitude
#mask_filename = 'cbpdr7mask.dat'           # File format: RA, dec
mask_filename = 'vollim_dr7_cbp_102709_mask.npy'

in_filename = in_directory + galaxies_filename
mask_filename = in_directory + mask_filename
#-------------------------------------------------------------------------------

#-------------------------------------------------------------------------------
# Survey parameters
determine_parameters = False
min_dist = 0
max_dist = 300. # z = 0.107 -> 313 h-1 Mpc   z = 0.087 -> 257 h-1 Mpc

# Cosmology
Omega_M = 0.3
h = 1
#-------------------------------------------------------------------------------

#-------------------------------------------------------------------------------
# Remove faint galaxies?
mag_cut = True

# Remove isolated galaxies?
rm_isolated = True
#-------------------------------------------------------------------------------

#-------------------------------------------------------------------------------
# Output file names
if mag_cut and rm_isolated:
    out1_suffix = '_maximal.txt'
    out2_suffix = '_holes.txt'
elif rm_isolated:
    out1_suffix = '_maximal_noMagCut.txt'
    out2_suffix = '_holes_noMagCut.txt'
elif mag_cut:
    out1_suffix = '_maximal_keepIsolated.txt'
    out2_suffix = '_holes_keepIsolated.txt'
else:
    out1_suffix = '_maximal_noFiltering.txt'
    out2_suffix = 'holes_noFiltering.txt'

out1_filename = out_directory + galaxies_filename[:-5] + out1_suffix  # List of maximal spheres of each void region: x, y, z, radius, distance, ra, dec
out2_filename = out_directory + galaxies_filename[:-5] + out2_suffix  # List of holes for all void regions: x, y, z, radius, flag (to which void it belongs)
#out3_filename = out_directory + 'out3_vollim_dr7.txt'                # List of void region sizes: radius, effective radius, evolume, x, y, z, deltap, nfield, vol_maxhole
#voidgals_filename = out_directory + 'vollim_voidgals_dr7.txt'        # List of the void galaxies: x, y, z, void region
#-------------------------------------------------------------------------------

################################################################################
#
#   OPEN FILES
#
################################################################################


infile = Table.read(in_filename, format='ascii.commented_header')
#maskfile = Table.read(mask_filename, format='ascii.commented_header')
maskfile = np.load(mask_filename)


#-------------------------------------------------------------------------------
# Print min and max distances
if determine_parameters:

    # Minimum distance
    min_z = min(infile['z'])

    # Maximum distance
    max_z = max(infile['z'])

    # Convert redshift to comoving distance
    dist_limits = Distance([min_z, max_z], Omega_M, h)

    print('Minimum distance =', dist_limits[0], 'Mpc/h')
    print('Maximum distance =', dist_limits[1], 'Mpc/h')

    exit()
#-------------------------------------------------------------------------------

#-------------------------------------------------------------------------------
# Rename columns
if 'rabsmag' not in infile.columns:
    infile['magnitude'].name = 'rabsmag'
#-------------------------------------------------------------------------------

#-------------------------------------------------------------------------------
# Calculate comoving distance
if 'Rgal' not in infile.columns:
    infile['Rgal'] = Distance(infile['z'], Omega_M, h)
#-------------------------------------------------------------------------------

################################################################################
#
#   FILTER GALAXIES
#
################################################################################


coord_min_table, mask, ngrid = filter_galaxies(infile, maskfile, min_dist, max_dist, survey_name, mag_cut, rm_isolated)

temp_outfile = open("filter_galaxies_output.pickle", 'wb')
pickle.dump((coord_min_table, mask, ngrid), temp_outfile)
temp_outfile.close()




################################################################################
#
#   FIND VOIDS
#
################################################################################


temp_infile = open("filter_galaxies_output.pickle", 'rb')
coord_min_table, mask, ngrid = pickle.load(temp_infile)
temp_infile.close()


find_voids(ngrid, min_dist, max_dist, coord_min_table, mask, out1_filename, out2_filename, survey_name, num_cpus)
