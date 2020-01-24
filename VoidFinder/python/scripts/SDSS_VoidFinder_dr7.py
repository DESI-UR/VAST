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



from voidfinder import filter_galaxies, find_voids
from voidfinder.multizmask import generate_mask
from voidfinder.absmag_comovingdist_functions import Distance
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
num_cpus = 1

#-------------------------------------------------------------------------------
survey_name = 'SDSS_dr7_'

# File header
#in_directory = '/home/moose/VoidFinder/VoidFinder/data/'
#out_directory = '/home/moose/VoidFinder/VoidFinder/data/'

#in_directory = '/home/oneills2/VoidFinder/python/voidfinder/data/'
#out_directory = '/home/oneills2/VoidFinder/python/voidfinder/data/'

in_directory = '/Users/kellydouglass/Documents/Research/VoidFinder/VoidFinder/data/SDSS/'
out_directory = '/Users/kellydouglass/Documents/Research/VoidFinder/VoidFinder/data/SDSS/'


# Input file name
galaxies_filename = 'vollim_dr7_cbp_102709.dat'  # File format: RA, dec, redshift, comoving distance, absolute magnitude

in_filename = in_directory + galaxies_filename
#-------------------------------------------------------------------------------

#-------------------------------------------------------------------------------
# Survey parameters
determine_parameters = False
min_dist = 0
max_dist = 300. # z = 0.107 -> 313 h-1 Mpc   z = 0.087 -> 257 h-1 Mpc

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


galaxy_data_table = Table.read(in_filename, format='ascii.commented_header')


#-------------------------------------------------------------------------------
# Print min and max distances
if determine_parameters:

    # Minimum distance
    min_z = min(galaxy_data_table['z'])

    # Maximum distance
    max_z = max(galaxy_data_table['z'])

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
if 'rabsmag' not in galaxy_data_table.columns:
    galaxy_data_table['magnitude'].name = 'rabsmag'
#-------------------------------------------------------------------------------

#-------------------------------------------------------------------------------
# Calculate comoving distance
if dist_metric == 'comoving' and 'Rgal' not in galaxy_data_table.columns:
    galaxy_data_table['Rgal'] = Distance(galaxy_data_table['z'], Omega_M, h)
#-------------------------------------------------------------------------------



################################################################################
#
#   GENERATE MASK
#
################################################################################


pre_mask, mask_resolution = generate_mask(galaxy_data_table, dist_metric, h, Omega_M)

temp_outfile = open(survey_name + 'mask.pickle', 'wb')
pickle.dump((pre_mask, mask_resolution), temp_outfile)
temp_outfile.close()



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
                                                    min_dist, 
                                                    max_dist, 
                                                    survey_name, 
                                                    mag_cut, 
                                                    rm_isolated, 
                                                    dist_metric, 
                                                    h)


temp_outfile = open(survey_name + "filter_galaxies_output.pickle", 'wb')
pickle.dump((coord_min_table, mask, grid_shape), temp_outfile)
temp_outfile.close()





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




find_voids(grid_shape, 
           min_dist, 
           max_dist,
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












