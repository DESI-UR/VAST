################################################################################
# VoidFinder - Hoyle & Vogeley (2002)
#
# This is a working example script for running VoidFinder on an input data set 
# with Cartesian coordinates.
################################################################################




################################################################################
# Import modules
#-------------------------------------------------------------------------------
import numpy as np

from vast.voidfinder import find_voids, wall_field_separation

from vast.voidfinder.preprocessing import load_data_to_Table
################################################################################




################################################################################
# User inputs
#-------------------------------------------------------------------------------
# Input data file name
galaxies_filename = "gadget_sim_100_256_wall.dat"

# "Survey" name - this will be used as the prefix for all output files
survey_name = "Gadget_100_256_"

# Change this directory paths to where you want the output to be saved.
out_directory = ''

# Coordinate limits of the simulation
xyz_limits = np.array([[-50.,-50.,-50.],[50.,50.,50.]])

# Size of a single grid cell
hole_grid_edge_length = 5.0

# Number of CPUs available for analysis.
# A value of None will use one less than all available CPUs.
num_cpus = 1
################################################################################




################################################################################
# Read in data
#-------------------------------------------------------------------------------
# Read in the simulated data
coords_xyz = load_data_to_Table(galaxies_filename)

#-------------------------------------------------------------------------------
# Restructure the data for the find_voids function
#-------------------------------------------------------------------------------
x = coords_xyz['x']
y = coords_xyz['y']
z = coords_xyz['z']

num_gal = x.shape[0]

coords_xyz = np.concatenate((x.reshape(num_gal,1),
                             y.reshape(num_gal,1),
                             z.reshape(num_gal,1)), axis=1)
#-------------------------------------------------------------------------------

#-------------------------------------------------------------------------------
# Remove isolated galaxies
#-------------------------------------------------------------------------------
wall_coords_xyz, field_coords_xyz = wall_field_separation(coords_xyz,
                                                          survey_name=survey_name, 
                                                          out_directory=out_directory)
#-------------------------------------------------------------------------------
################################################################################




################################################################################
# Find voids
#-------------------------------------------------------------------------------
find_voids(wall_coords_xyz,
           survey_name,
           out_directory,
           mask_type='xyz',
           xyz_limits=xyz_limits,
           #save_after=50000,
           #use_start_checkpoint=True,
           num_cpus=num_cpus,
           batch_size=10000)
################################################################################







