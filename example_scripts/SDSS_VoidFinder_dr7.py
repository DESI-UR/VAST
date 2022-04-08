################################################################################
# VoidFinder - Hoyle & Vogeley (2002)
#
# This is a working example script for running VoidFinder on an observed 
# galaxy catalog.
################################################################################




################################################################################
# IMPORT MODULES
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
#-------------------------------------------------------------------------------
#import sys
#sys.path.insert(1, 'local/path/VAST/VoidFinder/vast/voidfinder/')

from vast.voidfinder import find_voids, filter_galaxies

from vast.voidfinder.multizmask import generate_mask
from vast.voidfinder.preprocessing import file_preprocess

import pickle
################################################################################




################################################################################
# USER INPUTS
#-------------------------------------------------------------------------------
# Number of CPUs available for analysis.
# A value of None will use one less than all available CPUs.
num_cpus = 1

#-------------------------------------------------------------------------------
# File name details
#-------------------------------------------------------------------------------
# File header
survey_name = 'vollim_dr7_cbp_102709_'


# Change these directory paths to where your data is stored, and where you want 
# the output to be saved.
in_directory = ''
out_directory = ''


# Input file name
# File format: RA, dec, redshift, comoving distance, absolute magnitude
galaxies_filename = 'vollim_dr7_cbp_102709.dat'
#-------------------------------------------------------------------------------


#-------------------------------------------------------------------------------
# Survey parameters
#-------------------------------------------------------------------------------
# Redshift limits
# Note: These can be set to None, in which case VoidFinder will use the limits 
# of the galaxy catalog.
min_z = 0
max_z = 0.1026


# Cosmology (uncomment and change values to change cosmology)
# Need to also uncomment relevent inputs in function calls below
Omega_M = 0.26
#h = 1


# Uncomment if you do NOT want to use comoving distances
# Need to also uncomment relevent inputs in function calls below
dist_metric = 'comoving'
#-------------------------------------------------------------------------------


#-------------------------------------------------------------------------------
# Galaxy pruning details
#-------------------------------------------------------------------------------
# Uncomment if you do NOT want to remove galaxies with Mr > -20
# Need to also uncomment relevent input in function calls below
#mag_cut = False


# Uncomment if you do NOT want to remove isolated galaxies
# Need to also uncomment relevent input in function calls below
#rm_isolated = False
#-------------------------------------------------------------------------------
################################################################################




################################################################################
# PREPROCESS DATA
#-------------------------------------------------------------------------------
galaxy_data_table, dist_limits, out1_filename, out2_filename = file_preprocess(galaxies_filename, 
                                                                               in_directory, 
                                                                               out_directory, 
                                                                               #mag_cut=mag_cut,
                                                                               #rm_isolated=rm_isolated,
                                                                               dist_metric=dist_metric,
                                                                               min_z=min_z, 
                                                                               max_z=max_z,
                                                                               Omega_M=Omega_M,
                                                                               #h=h,
                                                                               )

print("Dist limits: ", dist_limits)
################################################################################




################################################################################
# GENERATE MASK
#-------------------------------------------------------------------------------
mask, mask_resolution = generate_mask(galaxy_data_table, 
                                      max_z, 
                                      dist_metric=dist_metric, 
                                      smooth_mask=True,
                                      #h=h,
                                      )

# Save the mask and mask resolution so that it can be used elsewhere
temp_outfile = open(out_directory + survey_name + 'mask.pickle', 'wb')
pickle.dump((mask, mask_resolution, dist_limits), temp_outfile)
temp_outfile.close()
################################################################################




################################################################################
# FILTER GALAXIES
#-------------------------------------------------------------------------------
# If you are rerunning the code, you can comment out the mask generation step 
# above and just load it here instead.
#temp_infile = open(out_directory + survey_name + 'mask.pickle', 'rb')
#mask, mask_resolution, dist_limits = pickle.load(temp_infile)
#temp_infile.close()

wall_coords_xyz, field_coords_xyz = filter_galaxies(galaxy_data_table,
                                                    survey_name,
                                                    out_directory,
                                                    dist_limits=dist_limits,
                                                    #mag_cut_flag=mag_cut,
                                                    #rm_isolated_flag=rm_isolated,
                                                    dist_metric=dist_metric,
                                                    #h=h,
                                                    )

del galaxy_data_table

# Save the catalog details needed for finding voids, so that the code can be 
# restarted from the next step if needed.
temp_outfile = open(survey_name + "filter_galaxies_output.pickle", 'wb')
pickle.dump((wall_coords_xyz, field_coords_xyz), temp_outfile)
temp_outfile.close()
################################################################################





################################################################################
# FIND VOIDS
#-------------------------------------------------------------------------------
# Again, if you are running the code and have not changed any of the above steps 
# from a previous run, you can comment out most of the above function calls and 
# load all the details in here to start over.
#temp_infile = open(survey_name + "filter_galaxies_output.pickle", 'rb')
#wall_coords_xyz, field_coords_xyz = pickle.load(temp_infile)
#temp_infile.close()



find_voids(wall_coords_xyz, 
           survey_name,
           mask_type='ra_dec_z',
           mask=mask, 
           mask_resolution=mask_resolution,
           dist_limits=dist_limits,
           #save_after=50000,
           #use_start_checkpoint=True,
           maximal_spheres_filename=out1_filename,
           void_table_filename=out2_filename,
           potential_voids_filename=out_directory + survey_name + 'potential_voids_list.txt',
           num_cpus=num_cpus)
################################################################################











