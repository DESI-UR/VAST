'''
Test various functions in VoidFinder by running through a pre-set fake galaxy 
list with artificial voids removed.
'''


################################################################################
# Import modules
#
# All modules from vast.voidfinder imported here are going to be tested below.
#-------------------------------------------------------------------------------
import numpy as np

from astropy.table import Table, setdiff, vstack

from sklearn import neighbors

from vast.voidfinder.constants import c

from vast.voidfinder import find_voids, filter_galaxies

from vast.voidfinder.multizmask import generate_mask
from vast.voidfinder.preprocessing import file_preprocess
################################################################################




################################################################################
# Generate our set of "fake" galaxies and save to disk
#-------------------------------------------------------------------------------
ra_range = np.arange(10, 30, 0.5)
dec_range = np.arange(-10, 10, 0.5)
redshift_range = np.arange(0, 0.011, 0.0005)

RA, DEC, REDSHIFT = np.meshgrid(ra_range, dec_range, redshift_range)

galaxies_table = Table()
galaxies_table['ra'] = np.ravel(RA)
galaxies_table['dec'] = np.ravel(DEC)
galaxies_table['redshift'] = np.ravel(REDSHIFT)

# Shuffle the table (so that the KDtree does not die)
rng = np.random.default_rng()
galaxies_shuffled = Table(rng.permutation(galaxies_table))

galaxies_shuffled['Rgal'] = c*galaxies_shuffled['redshift']/100.

N_galaxies = len(galaxies_shuffled)

# All galaxies will be brighter than the magnitude limit, so that none of them 
# are removed
galaxies_shuffled['rabsmag'] = 5*np.random.rand(N_galaxies) - 25.1

galaxies_filename = 'test_galaxies.txt'
galaxies_shuffled.write(galaxies_filename, 
                        format='ascii.commented_header', 
                        overwrite=True)

gal = np.zeros((N_galaxies,3))
gal[:,0] = galaxies_shuffled['Rgal']*np.cos(galaxies_shuffled['ra']*np.pi/180.)*np.cos(galaxies_shuffled['dec']*np.pi/180.)
gal[:,1] = galaxies_shuffled['Rgal']*np.sin(galaxies_shuffled['ra']*np.pi/180.)*np.cos(galaxies_shuffled['dec']*np.pi/180.)
gal[:,2] = galaxies_shuffled['Rgal']*np.sin(galaxies_shuffled['dec']*np.pi/180.)
################################################################################




################################################################################
# Test vast.voidfinder.preprocessing.file_preprocess
# 
# This function takes in a file name (pointing to the galaxy data file) and 
# returns the data table.  It also computes the redshift range in comoving 
# coordinates, and generates the output filenames.
#-------------------------------------------------------------------------------
f_galaxy_table, f_dist_limits, f_out1_filename, f_out2_filename = file_preprocess(galaxies_filename, 
                                                                                  '', 
                                                                                  '', 
                                                                                  dist_metric='redshift')

# Check the galaxy table
assert(len(setdiff(f_galaxy_table, galaxies_shuffled)) == 0)

# Check the distance limits
dist_limits = np.zeros(2)
dist_limits[1] = c*redshift_range[-1]/100.
assert(np.isclose(f_dist_limits, dist_limits).all())

# Check the first output file name
assert(f_out1_filename == 'test_galaxies_redshift_maximal.txt')

# Check the second output file name
assert(f_out2_filename == 'test_galaxies_redshift_holes.txt')
################################################################################




################################################################################
# Test vast.voidfinder.multizmask.generate_mask
#
# This function takes in an astropy table containing the galaxy coordinates and 
# the maximum redshift and returns a boolean mask and the resolution of the 
# mask.
#-------------------------------------------------------------------------------
f_mask, f_mask_resolution = generate_mask(galaxies_shuffled, 
                                          redshift_range[-1], 
                                          dist_metric='redshift')

# Check the mask
mask = np.zeros((360,180), dtype=bool)
for i in range(int(ra_range[0]), int(ra_range[-1]+1)):
    for j in range(int(dec_range[0] + 90), int(dec_range[-1] + 90)+1):
        mask[i, j] = True
assert((f_mask == mask).all())

# Check the mask resolution
assert(np.isclose(f_mask_resolution, 1))
################################################################################




################################################################################
# Test vast.voidfinder.filter_galaxies
#
# This function takes in an astropy table containing the galaxy coordinates, 
# the name of the survey, and the output directory and returns astropy tables of 
# the Cartesian coordinates of the wall and field galaxies as well as the shape 
# of the grid on which the galaxies will be placed and the coordinates of the 
# lower left corner of the grid.
#-------------------------------------------------------------------------------
f_wall, f_field, f_grid_shape, f_min = filter_galaxies(galaxies_shuffled, 
                                                       'test_', 
                                                       '', 
                                                       dist_metric='redshift', 
                                                       hole_grid_edge_length=1.0)

# Check the wall galaxy coordinates
gal_tree = neighbors.KDTree(gal)
distances, indices = gal_tree.query(gal, k=4)
dist3 = distances[:,3]
wall = gal[dist3 < (np.mean(dist3) + 1.5*np.std(dist3))]
assert(np.isclose(f_wall, wall).all())

# Check the field galaxy coordinates
field = gal[dist3 >= (np.mean(dist3) + 1.5*np.std(dist3))]
assert(np.isclose(f_field, field).all())

# Check the grid shape
n_cells = (np.max(gal, axis=0) - np.min(gal, axis=0))
grid_shape = tuple(np.ceil(n_cells).astype(int))
assert(f_grid_shape == grid_shape)

# Check the minimum coordinates
assert(np.isclose(f_min, np.min(gal, axis=0)).all())
################################################################################




################################################################################
# Test vast.voidfinder.find_voids
#
# This function takes in a set of Cartesian coordinates, the mask, and various 
# other parameters and returns a list of the maximal spheres and holes in the 
# galaxy distribution.
#-------------------------------------------------------------------------------
maximals = Table()
maximals['x'] = [25., 10.]
maximals['y'] = [8., 3.]
maximals['z'] = [0., -1.]
maximals['r'] = [2.5, 1.5]
maximals['flag'] = [0, 1]

holes = Table()
holes['x'] = [24., 10.5]
holes['y'] = [7.9, 3.2]
holes['z'] = [0.1, -0.5]
holes['r'] = [2., 0.5]
holes['flag'] = [0, 1]
holes = vstack([holes, maximals])

# Remove points which fall inside holes
remove_boolean = np.zeros(len(wall), dtype=bool)
for i in range(len(holes)):
    d = (holes['x'][i] - wall[:,0])**2 + (holes['y'][i] - wall[:,1])**2 + (holes['z'][i] - wall[:,2])**2
    remove_boolean = np.logical_or(remove_boolean, (d < holes['r'][i]))

find_voids(wall[~remove_boolean], 
           dist_limits, 
           mask, 
           1, 
           np.min(gal, axis=0), 
           grid_shape, 
           'test_', 
           hole_grid_edge_length=1.0,
           hole_center_iter_dist=0.2, 
           min_maximal_radius=1.0, 
           num_cpus=1, 
           void_table_filename='test_galaxies_redshift_holes.txt', 
           maximal_spheres_filename='test_galaxies_redshift_maximal.txt')

# Check maximal spheres
f_maximals = Table.read('test_galaxies_redshift_maximal.txt', 
                        format='ascii.commented_header')
maximals_truth = Table.read('test_galaxies_redshift_maximal_truth.txt', 
                            format='ascii.commented_header')
assert(len(setdiff(f_maximals, maximals_truth)) == 0)

# Check holes
f_holes = Table.read('test_galaxies_redshift_holes.txt', 
                     format='ascii.commented_header')
holes_truth = Table.read('test_galaxies_redshift_holes_truth.txt', 
                         format='ascii.commented_header')
assert(len(setdiff(holes_truth, f_holes)) == 0)
















