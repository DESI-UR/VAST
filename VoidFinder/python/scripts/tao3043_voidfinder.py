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

from voidfinder import find_voids, \
                       filter_galaxies_2
                       
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

in_directory = '/home/oneills2/Desktop/data/'
out_directory = '/home/oneills2/Desktop/data/'


# Input file name
galaxies_filename = 'tao3043.h5'  # File format: RA, dec, redshift, comoving distance, absolute magnitude


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


galaxy_data_table, dist_limits, out1_filename, out2_filename = file_preprocess(galaxies_filename, 
                                                                               in_directory, 
                                                                               out_directory, 
                                                                               min_z=min_z, 
                                                                               max_z=max_z,
                                                                               verbose=1)


print("#"*80)
print("Preprocess done at: ", time.time() - start_time)
print("#"*80)


################################################################################
#
#   GENERATE MASK
#
################################################################################


mask, mask_resolution = generate_mask(galaxy_data_table,
                                      verbose=1)

'''
temp_outfile = open(survey_name + 'mask.pickle', 'wb')
pickle.dump((mask, mask_resolution), temp_outfile)
temp_outfile.close()
'''
print("#"*80)
print("Generating mask done at:", time.time() - start_time)
print("#"*80)
################################################################################
#
#   FILTER GALAXIES
#
################################################################################

'''
temp_infile = open(survey_name + 'mask.pickle', 'rb')
mask, mask_resolution = pickle.load(temp_infile)
temp_infile.close()
'''

wall_coords_xyz, field_coords_xyz, hole_grid_shape, coords_min  = filter_galaxies_2(galaxy_data_table,
                                                                                    survey_name,
                                                                                    write_table=False,
                                                                                    verbose=1)

del galaxy_data_table








'''
temp_outfile = open(survey_name + "filter_galaxies_output.pickle", 'wb')
pickle.dump((wall_coords_xyz, field_coords_xyz, hole_grid_shape, coords_min), temp_outfile)
temp_outfile.close()
'''

print("#"*80)
print("Filter galaxies done at:", time.time() - start_time)
print("#"*80)

################################################################################
#
#   FIND VOIDS
#
################################################################################

'''
temp_infile = open(survey_name + "filter_galaxies_output.pickle", 'rb')
wall_coords_xyz, field_coords_xyz, hole_grid_shape, coords_min = pickle.load(temp_infile)
temp_infile.close()
'''



print("#"*80)
print("Starting VoidFinder at:", time.time() - start_time)
print("#"*80)

find_voids(wall_coords_xyz, 
           dist_limits,
           mask, 
           mask_resolution,
           coords_min,
           hole_grid_shape,
           survey_name,
           save_after=20000000,
           use_start_checkpoint=False,
           hole_grid_edge_length=5,
           galaxy_map_grid_edge_length=None,
           hole_center_iter_dist=1.0,
           maximal_spheres_filename=out1_filename,
           void_table_filename=out2_filename,
           potential_voids_filename=survey_name+'potential_voids_list.txt',
           num_cpus=num_cpus,
           batch_size=10000,
           verbose=1,
           print_after=5.0)










