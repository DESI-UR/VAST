'''VoidFinder - Hoyle & Vogeley (2002)'''

################################################################################
#
#   IMPORT MODULES
#
################################################################################

from voidfinder import filter_galaxies, find_voids
from astropy.table import Table


################################################################################
#
#   USER INPUTS
#
################################################################################

# File header
directory = ''

# Input file names
in_filename = directory + 'vollim_dr7_cbp_102709.dat' # File format: RA, dec, redshift, comoving distance, absolute magnitude
mask_filename = directory + 'cbpdr7mask.dat' # File format: RA, dec

# Output file names
out1_filename = in_filename[:-4] + '_maximal.txt' # List of maximal spheres of each void region: x, y, z, radius, distance, ra, dec
out2_filename = in_filename[:-4] + '_holes.txt' # List of holes for all void regions: x, y, z, radius, flag (to which void it belongs)
'''out3_filename = 'out3_vollim_dr7.txt' # List of void region sizes: radius, effective radius, evolume, x, y, z, deltap, nfield, vol_maxhole
voidgals_filename = 'vollim_voidgals_dr7.txt' # List of the void galaxies: x, y, z, void region '''

#ngrid = 128     # Number of grid cells
max_dist = 300. # z = 0.107 -> 313 h-1 Mpc   z = 0.087 -> 257 h-1 Mpc
#box = 630.      # Size of survey/simulation box
dl = 5.         # Cell side length [Mpc/h]


################################################################################
#
#   OPEN FILES
#
################################################################################


infile = Table.read(in_filename, format='ascii.commented_header')
maskfile = Table.read(mask_filename, format='ascii.commented_header')


################################################################################
#
#   FILTER GALAXIES
#
################################################################################


#coord_min_table, mask = filter_galaxies(in_filename, mask_filename, ngrid, box, max_dist)
coord_min_table, mask, ngrid = filter_galaxies(infile, maskfile, dl, max_dist)


################################################################################
#
#   FIND VOIDS
#
################################################################################


find_voids(ngrid, dl, max_dist, coord_min_table, mask, out1_filename, out2_filename)
