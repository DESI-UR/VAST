'''VoidFinder - Hoyle & Vogeley (2002)'''



################################################################################
#
# If you have control over your python environment, voidfinder can be installed
# as a normal python package via 'python setup.py install', in which case the 
# below import of 'sys' and 'sys.path.insert(0, '/abspath/to/VoidFinder/python'
# is unnecessary.  If you aren't able to install the voidfinder package,
# you can use the sys.path.insert to add it to the list of available packages
# in your python environment.
#
# Alternately, "python setup.py develop" will 'install' some symlinks which
# point back to the current directory and you can run off the same voidfinder
# repository that you're working on as if it was installed
#
#
#
################################################################################


#import sys
#sys.path.insert(1, '/home/oneills2/VoidFinder/python/')
#sys.path.insert(1, '/Users/kellydouglass/Documents/Research/VoidFinder/python/')

################################################################################
#
#   IMPORT MODULES
#
################################################################################



from voidfinder import find_voids, \
                       ra_dec_to_xyz, \
                       calculate_grid, \
                       wall_field_separation
from voidfinder.multizmask import generate_mask
from voidfinder.preprocessing import file_preprocess
from voidfinder.table_functions import to_vector, to_array

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
num_cpus = 4

#-------------------------------------------------------------------------------
#survey_name = 'SDSS_dr7_'
survey_name = 'SDSS_dr12_'

# File header

if survey_name == 'SDSS_dr7_':
    in_directory = '/home/moose/VoidFinder/VoidFinder/data/SDSS/'
    out_directory = '/home/moose/VoidFinder/VoidFinder/data/SDSS/'
elif survey_name == 'SDSS_dr12_':
    in_directory = '/home/moose/VoidFinder/VoidFinder/data/'
    out_directory = '/home/moose/VoidFinder/VoidFinder/data/'

#in_directory = '/Users/kellydouglass/Documents/Research/VoidFinder/VoidFinder/data/SDSS/'
#out_directory = '/Users/kellydouglass/Documents/Research/VoidFinder/VoidFinder/data/SDSS/'


# Input file name
if survey_name == 'SDSS_dr7_':
    galaxies_filename = 'vollim_dr7_cbp_102709.dat'  # File format: RA, dec, redshift, comoving distance, absolute magnitude
elif survey_name == 'SDSS_dr12_':
    galaxies_filename = 'dr12r.dat'
#-------------------------------------------------------------------------------

#-------------------------------------------------------------------------------
# Survey parameters
# Note: These can be set to None, in which case VoidFinder will use the limits 
# of the galaxy catalog.
if survey_name == 'SDSS_dr7_':
    min_z = 0
    max_z = 0.1026
elif survey_name == 'SDSS_dr12_':
    min_z = None
    max_z = None

# Cosmology (uncomment and change values to change cosmology)
#Omega_M = 0.3
#h = 1

# Uncomment if you do NOT want to use comoving distances
#dist_metric = 'redshift'
#-------------------------------------------------------------------------------

#-------------------------------------------------------------------------------
# Uncomment if you do NOT want to remove galaxies with Mr > -20
mag_cut = True
magnitude_limit = -20.0

# Uncomment if you do NOT want to remove isolated galaxies
rm_isolated = True

hole_grid_edge_length = 5.0
#-------------------------------------------------------------------------------




################################################################################
#
#   PREPROCESS DATA
#
################################################################################

galaxy_data_table, dist_limits, out1_filename, out2_filename = file_preprocess(galaxies_filename, 
                                                                               in_directory, 
                                                                               out_directory, 
                                                                               min_z=min_z, 
                                                                               max_z=max_z,
                                                                               verbose=1)

################################################################################
#
#   GENERATE MASK
#
################################################################################

mask, mask_resolution = generate_mask(galaxy_data_table, verbose=0)

################################################################################
#
#   FILTER GALAXIES
#
################################################################################

if mag_cut:
    
    galaxy_data_table = galaxy_data_table[galaxy_data_table['rabsmag'] < magnitude_limit]


galaxy_coords_xyz = ra_dec_to_xyz(galaxy_data_table)


hole_grid_shape, coords_min = calculate_grid(galaxy_coords_xyz,
                                             hole_grid_edge_length)


if rm_isolated:
    
    wall_coords_xyz, field_coords_xyz = wall_field_separation(galaxy_coords_xyz,
                                                              sep_neighbor=3,
                                                              verbose=1)
    
else:
    
    wall_coords_xyz = galaxy_coords_xyz

del galaxy_data_table

################################################################################
#
#   FIND VOIDS
#
################################################################################

find_voids(wall_coords_xyz, 
           dist_limits,
           mask, 
           mask_resolution,
           coords_min,
           hole_grid_shape,
           hole_grid_edge_length=hole_grid_edge_length,
           galaxy_map_grid_edge_length=None,
           hole_center_iter_dist=1.0,
           maximal_spheres_filename=out1_filename,
           void_table_filename=out2_filename,
           potential_voids_filename=survey_name+'potential_voids_list.txt',
           num_cpus=num_cpus,
           batch_size=10000,
           verbose=0,
           print_after=5.0)












