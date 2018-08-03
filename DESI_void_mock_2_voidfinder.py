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


survey_name = 'DESI_mock_2_'

# File header
in_directory = ''
out_directory = '/scratch/mguzzett/VoidFinder/'

# Input file names
galaxies_filename = 'DESI_void_mock_2.fits'  # File format: RA, dec, redshift, comoving distance, absolute magnitude
mask_filename = 'void_2_mask.dat'            # File format: RA, dec

in_filename = in_directory + galaxies_filename
mask_filename = in_directory + mask_filename

# Output file names
out1_filename = out_directory + galaxies_filename[:-5] + '_maximal.txt'  # List of maximal spheres of each void region: x, y, z, radius, distance, ra, dec
out2_filename = out_directory + galaxies_filename[:-5] + '_holes.txt'    # List of holes for all void regions: x, y, z, radius, flag (to which void it belongs)
#out3_filename = out_directory + 'out3_vollim_dr7.txt'              # List of void region sizes: radius, effective radius, evolume, x, y, z, deltap, nfield, vol_maxhole
#voidgals_filename = out_directory + 'vollim_voidgals_dr7.txt'      # List of the void galaxies: x, y, z, void region


# Survey parameters
min_dist = 0     # z = 
max_dist = 2300. # z = 0.7 --> 2388 Mpc/h

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


# Rename columns
if 'rabsmag' not in infile.columns:
    '''
    print(infile.columns)
    print('Please rename columns')
    exit()
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
