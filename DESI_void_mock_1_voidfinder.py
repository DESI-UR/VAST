'''VoidFinder - Hoyle & Vogeley (2002)'''

################################################################################
#
#   IMPORT MODULES
#
################################################################################

from voidfinder import filter_galaxies, find_voids

from astropy.io import fits
from astropy.table import Table

from absmag_comovingdist_functions import Distance

################################################################################
#
#   USER INPUTS
#
################################################################################


survey_name = 'DESI_void_flatmock_1_'

# File header
in_directory = 'DESI/mocks/'
out_directory = 'DESI/mocks/'

# Input file names
in_filename = in_directory + 'void_flatmock_1.fits'  # File format: RA, dec, redshift, comoving distance, absolute magnitude
mask_filename = in_directory + 'void_1_mask.dat'     # File format: RA, dec

# Output file names
out1_filename = out_directory + in_filename[:-5] + '_maximal.txt'  # List of maximal spheres of each void region: x, y, z, radius, distance, ra, dec
out2_filename = out_directory + in_filename[:-5] + '_holes.txt'    # List of holes for all void regions: x, y, z, radius, flag (to which void it belongs)
#out3_filename = out_directory + 'out3_vollim_dr7.txt'              # List of void region sizes: radius, effective radius, evolume, x, y, z, deltap, nfield, vol_maxhole
#voidgals_filename = out_directory + 'vollim_voidgals_dr7.txt'      # List of the void galaxies: x, y, z, void region


# Survey parameters
determine_parameters = False
min_dist = 1015  # z = 0.37 --> 1013 Mpc/h
max_dist = 2600  # z = 1.2 --> 2634 Mpc/h

# Cosmology
Omega_M = 0.3
h = 1


################################################################################
#
#   OPEN FILES
#
################################################################################


gal_file = fits.open(in_filename)
infile = Table(gal_file[1].data)

maskfile = Table.read(mask_filename, format='ascii.commented_header')


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



# Rename columns
if 'rabsmag' not in infile.columns:
    '''
    print(infile.columns)
    print('Please rename columns')
    '''
    infile['magnitude'].name = 'rabsmag'

# Calculate comoving distance
if 'Rgal' not in infile.columns:
    infile['Rgal'] = Distance(infile['z'], Omega_M, h)


################################################################################
#
#   FILTER GALAXIES
#
################################################################################


coord_min_table, mask, ngrid = filter_galaxies(infile, maskfile, min_dist, max_dist, survey_name)


################################################################################
#
#   FIND VOIDS
#
################################################################################


find_voids(ngrid, min_dist, max_dist, coord_min_table, mask, out1_filename, out2_filename, survey_name)
