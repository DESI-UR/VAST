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
import numpy as np
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
galaxy_data_table, dist_limits = file_preprocess(
    galaxies_filename, 
    survey_name,
    in_directory, 
    out_directory, 
    #mag_cut=mag_cut,
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
                                      survey_name,
                                      out_directory,
                                      dist_metric=dist_metric, 
                                      smooth_mask=True,
                                      #h=h,
                                      )

# The mask is automatically saved in the survey_name+'VoidFinder_Output.fits'
# file and can be read in for future use
################################################################################




################################################################################
# FILTER GALAXIES
#-------------------------------------------------------------------------------
# If you are rerunning the code, you can comment out the mask generation step 
# above and just load it here instead. Use something in the vein of the below 
# (untested) code:
# with fits.open(out_directory+survey_name+'_VoidFinder_Output.fits') as output:
#   mask = output['MASK'].data
#   mask_resolution  = output['MASK'].header['MSKRES']
#   dist_limits = (output['PRIMARY'].header['DLIML'], output['PRIMARY'].header['DLIMU'])


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

# The galaxies are automatically saved in the survey_name+'VoidFinder_Output.fits'
# file and can be read in for future use, o long as the write_table parameter of
# filter_galaxies is True (it is True by default)
################################################################################


coords_min = np.min(np.concatenate([wall_coords_xyz, field_coords_xyz]), axis=0)


################################################################################
# FIND VOIDS
#-------------------------------------------------------------------------------
# Again, if you are running the code and have not changed any of the above steps 
# from a previous run, you can comment out most of the above function calls and 
# load all the details in here to start over. Use something in the vein of the 
# below (untested) code:
# with fits.open(out_directory+survey_name+'_VoidFinder_Output.fits') as output:
#   wall_coords_xyz = np.array([output['WALL'].data['X'],
#                               output['WALL'].data['Y'],
#                               output['WALL'].data['Z']]).T



find_voids(wall_coords_xyz, 
           survey_name,
           out_directory,
           mask_type='ra_dec_z',
           mask=mask, 
           mask_resolution=mask_resolution,
           dist_limits=dist_limits,
           grid_origin=coords_min,
           #save_after=50000,
           #use_start_checkpoint=True,
           verbose=1,
           num_cpus=num_cpus)
################################################################################

"""
from vast.voidfinder.postprocessing import open_fits_file
from astropy.table import Table
from astropy.io import fits


import sys
sys.path.insert(1, '/global/homes/h/hrincon/python_tools')

import VoidVolume as vol

catalog, log_filename = open_fits_file(out_directory, survey_name)
maxs = Table(catalog['MAXIMALS'].data)
holes=Table(catalog['HOLES'].data)
maxs['EFFECTIVE_RADIUS']=-1.
maxs['EFFECTIVE_RADIUS_UNCERT']=-1.
maxs['EFFECTIVE_RADIUS'].unit = 'Mpc/h'
maxs['EFFECTIVE_RADIUS_UNCERT'].unit = 'Mpc/h'

for flag in maxs['FLAG']:
    hole_in_max = holes[holess['FLAG']==flag]
    x = np.array([holes['X'], holes['Y'],holes['Z']]).T
    R = holes['RADIUS'].data
    vol_info = vol.volume_of_spheres(x, R)
    maxs['EFFECTIVE_RADIUS'][flag] = ((3/4) * vol_info[2] / np.pi) ** (1/3) 
    maxs['EFFECTIVE_RADIUS_UNCERT'][flag] = vol_info[3] * ((3 * vol_info[2]) ** -2 / (4 * np.pi)) ** (1/3) 

catalog['MAXIMALS'].data = fits.BinTableHDU(maxs).data
hdul.writeto(log_filename, overwrite=True)

"""







