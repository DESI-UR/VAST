'''VoidFinder - Hoyle & Vogeley (2002)'''

################################################################################
#
#   IMPORT MODULES
#
################################################################################






import sys
sys.path.insert(1, '/home/moose/VoidFinder/python/')
#sys.path.insert(1, '/Users/kellydouglass/Documents/Research/VoidFinder/python/')

from voidfinder import filter_galaxies, find_voids
from voidfinder.multizmask import generate_mask
from voidfinder.absmag_comovingdist_functions import Distance

from astropy.table import Table
import pickle
import numpy as np


################################################################################
#
#   USER INPUTS
#
################################################################################


# Number of CPUs available for analysis.
# A value of None will use one less than all available CPUs.
num_cpus = None

#-------------------------------------------------------------------------------
survey_name = 'SDSS_dr12_'

# File header
#in_directory = '/Users/kellydouglass/Documents/Research/VoidFinder/python/voidfinder/data/'
#out_directory = '/Users/kellydouglass/Documents/Research/VoidFinder/python/voidfinder/data/'

in_directory = '/home/moose/VoidFinder/python/voidfinder/data/'
out_directory = '/home/moose/VoidFinder/python/voidfinder/data/'


# Input file name
galaxies_filename = 'dr12r.dat'  # File format: RA, dec, redshift, comoving distance, absolute magnitude

in_filename = in_directory + galaxies_filename
#-------------------------------------------------------------------------------

#-------------------------------------------------------------------------------
# Survey parameters
determine_parameters = False
min_dist = 1000.0
max_dist = 1800.0 # z = 0.107 -> 313 h-1 Mpc   z = 0.087 -> 257 h-1 Mpc

# Cosmology
Omega_M = 0.3
h = 1

# Distance metric
dist_metric = 'comoving'
#dist_metric = 'redshift'
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
    out1_suffix = '_' + dist_metric + '_maximal.txt'
    out2_suffix = '_' + dist_metric + '_holes.txt'
elif rm_isolated:
    out1_suffix = '_' + dist_metric + '_maximal_noMagCut.txt'
    out2_suffix = '_' + dist_metric + '_holes_noMagCut.txt'
elif mag_cut:
    out1_suffix = '_' + dist_metric + '_maximal_keepIsolated.txt'
    out2_suffix = '_' + dist_metric + '_holes_keepIsolated.txt'
else:
    out1_suffix = '_' + dist_metric + '_maximal_noFiltering.txt'
    out2_suffix = '_' + dist_metric + 'holes_noFiltering.txt'

out1_filename = out_directory + galaxies_filename[:-4] + out1_suffix  # List of maximal spheres of each void region: x, y, z, radius, distance, ra, dec
out2_filename = out_directory + galaxies_filename[:-4] + out2_suffix  # List of holes for all void regions: x, y, z, radius, flag (to which void it belongs)
#out3_filename = out_directory + 'out3_vollim_dr7.txt'                # List of void region sizes: radius, effective radius, evolume, x, y, z, deltap, nfield, vol_maxhole
#voidgals_filename = out_directory + 'vollim_voidgals_dr7.txt'        # List of the void galaxies: x, y, z, void region
#-------------------------------------------------------------------------------

################################################################################
#
#   OPEN FILES
#
################################################################################


infile = Table.read(in_filename, format='ascii.commented_header')


#-------------------------------------------------------------------------------
# Print min and max distances
if determine_parameters:

    # Minimum distance
    min_z = min(infile['redshift'])

    # Maximum distance
    max_z = max(infile['redshift'])

    if dist_metric == 'comoving':
        # Convert redshift to comoving distance
        dist_limits = Distance([min_z, max_z], Omega_M, h)
        units = 'Mpc/h'
    else:
        dist_limits = [min_z, max_z]
        units = ''

    print('Minimum distance =', dist_limits[0], units)
    print('Maximum distance =', dist_limits[1], units)

    exit()
#-------------------------------------------------------------------------------

#-------------------------------------------------------------------------------
# Rename columns
if 'rabsmag' not in infile.columns:
    infile['magnitude'].name = 'rabsmag'
#-------------------------------------------------------------------------------

#-------------------------------------------------------------------------------
# Calculate comoving distance
if dist_metric == 'comoving' and 'Rgal' not in infile.columns:
    infile['Rgal'] = Distance(infile['z'], Omega_M, h)
#-------------------------------------------------------------------------------



################################################################################
#
#   GENERATE MASK
#
################################################################################

'''
maskfile, mask_resolution = generate_mask(infile, dist_metric, h, Omega_M)

temp_outfile = open(survey_name + 'mask.pickle', 'wb')
pickle.dump((mask_resolution, maskfile), temp_outfile)
temp_outfile.close()

'''

################################################################################
#
#   FILTER GALAXIES
#
################################################################################


temp_infile = open(survey_name + 'mask.pickle', 'rb')
mask_resolution, maskfile = pickle.load(temp_infile)
temp_infile.close()

'''
coord_min_table, mask, ngrid = filter_galaxies(infile, maskfile, 
                                               mask_resolution, min_dist, 
                                               max_dist, survey_name, mag_cut, 
                                               rm_isolated, dist_metric, h)


temp_outfile = open(survey_name + "filter_galaxies_output.pickle", 'wb')
pickle.dump((coord_min_table, mask, ngrid), temp_outfile)
temp_outfile.close()
'''



################################################################################
#
#   FIND VOIDS
#
################################################################################


temp_infile = open(survey_name + "filter_galaxies_output.pickle", 'rb')
coord_min_table, mask, ngrid = pickle.load(temp_infile)
temp_infile.close()


find_voids(ngrid, 
           min_dist, 
           max_dist, 
           coord_min_table, 
           mask, 
           mask_resolution, 
           out1_filename, 
           out2_filename, 
           survey_name, 
           num_cpus,
           verbose=1,
           print_after=1000000)










