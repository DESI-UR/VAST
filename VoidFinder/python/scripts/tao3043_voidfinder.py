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
#sys.path.insert(0, '/home/moose/VoidFinder/VoidFinder/python/')
#sys.path.insert(1, '/Users/kellydouglass/Documents/Research/VoidFinder/VoidFinder/python/')

################################################################################
#
#   IMPORT MODULES
#
################################################################################

from voidfinder import filter_galaxies, find_voids
from voidfinder.preprocessing import file_preprocess
from voidfinder.multizmask import generate_mask
from voidfinder.dist_funcs_cython import z_to_comoving_dist
from voidfinder.table_functions import to_vector, to_array

from astropy.table import Table
import pickle
import numpy as np


import time

################################################################################
#
#   USER INPUTS
#
################################################################################

start_time = time.time()

# Number of CPUs available for analysis.
# A value of None will use one less than all available CPUs.
num_cpus = 4

#-------------------------------------------------------------------------------
survey_name = 'tao3043_'

# File header
#in_directory = '/Users/kellydouglass/Documents/Research/VoidFinder/python/voidfinder/data/'
#out_directory = '/Users/kellydouglass/Documents/Research/VoidFinder/python/voidfinder/data/'

#in_directory = '/scratch/sbenzvi_lab/desi/dylanbranch/vfp8/VoidFinder/python/scripts/'
#out_directory = '/scratch/sbenzvi_lab/desi/dylanbranch/vfp8/VoidFinder/python/scripts/'

in_directory = '/home/moose/VoidFinder/VoidFinder/data/'
out_directory = '/home/moose/VoidFinder/VoidFinder/data/'


# Input file name
galaxies_filename = 'tao3043.dat'  # File format: RA, dec, redshift, comoving distance, absolute magnitude


#-------------------------------------------------------------------------------

#-------------------------------------------------------------------------------
# Survey parameters

min_z = 0.0
max_z = 0.7


################################################################################
#
#   PREPROCESS DATA
#
################################################################################


galaxy_data_table, dist_limits, out1_filename, out2_filename = file_preprocess(galaxies_filename, in_directory, out_directory, min_z=min_z, max_z=max_z)


print("Preprocess done at: ", time.time() - start_time)



################################################################################
#
#   GENERATE MASK
#
################################################################################


pre_mask, mask_resolution = generate_mask(galaxy_data_table)

temp_outfile = open(survey_name + 'mask.pickle', 'wb')
pickle.dump((pre_mask, mask_resolution), temp_outfile)
temp_outfile.close()

print("Generating mask done at:", time.time() - start_time)

################################################################################
#
#   FILTER GALAXIES
#
################################################################################


temp_infile = open(survey_name + 'mask.pickle', 'rb')
pre_mask, mask_resolution = pickle.load(temp_infile)
temp_infile.close()


coord_min_table, mask, grid_shape = filter_galaxies(galaxy_data_table, 
                                                    pre_mask, 
                                                    mask_resolution,
                                                    survey_name)


temp_outfile = open(survey_name + "filter_galaxies_output.pickle", 'wb')
pickle.dump((coord_min_table, mask, grid_shape), temp_outfile)
temp_outfile.close()


print("Filter galaxies done at:", time.time() - start_time)


################################################################################
#
#   FIND VOIDS
#
################################################################################


temp_infile = open(survey_name + "filter_galaxies_output.pickle", 'rb')
coord_min_table, mask, grid_shape = pickle.load(temp_infile)
temp_infile.close()







w_coord_table = Table.read(survey_name + 'wall_gal_file.txt', format='ascii.commented_header')

galaxy_coords = to_array(w_coord_table)

coord_min = to_vector(coord_min_table)


print("Starting VoidFinder at:", time.time() - start_time)


find_voids(grid_shape, 
           dist_limits,
           galaxy_coords,
           coord_min, 
           mask, 
           mask_resolution,
           void_grid_edge_length=5,
           search_grid_edge_length=None,
           hole_center_iter_dist=1.0,
           maximal_spheres_filename=out1_filename,
           void_table_filename=out2_filename,
           potential_voids_filename=survey_name+'potential_voids_list.txt',
           num_cpus=num_cpus,
           batch_size=10000,
           verbose=1,
           print_after=5.0)










