'''Functions used in voids_sdss.py'''

import numpy as np
from astropy.table import Table

from table_functions import add_row, subtract_row, table_divide, table_dtype_cast, row_cross, row_dot, to_vector


################################################################################
#
#   DEFINE FUNCTIONS
#
################################################################################

def mesh_galaxies(galaxy_coords, coord_min, grid_side_length, N_boxes):
    '''Sort galaxies onto a grid'''

    # Initialize the 3D bins that will contain the number of galaxies in each bin
    ngal = np.zeros((N_boxes, N_boxes, N_boxes), dtype=int)

    # Initialize the 3D bins that will contain the galaxy indices
    chainlist = -np.ones((N_boxes, N_boxes, N_boxes), dtype=int)

    # Initialize a list that will store the galaxy's index that previously occupied the cell
    linklist = np.zeros(len(galaxy_coords), dtype=int)

    # Conver the galaxy coordinates to grid indices
    mesh_indices = table_dtype_cast(table_divide(subtract_row(galaxy_coords, coord_min), grid_side_length), int)

    for igal in xrange(len(galaxy_coords)): # Change to range() for python 3
        # Increase the number of galaxies in corresponding cell in ngal
        ngal[mesh_indices['x'][igal], mesh_indices['y'][igal], mesh_indices['z'][igal]] += 1

        # Store the index of the last galaxy that was saved in corresponding cell 
        linklist[igal] = chainlist[mesh_indices['x'][igal], mesh_indices['y'][igal], mesh_indices['z'][igal]]

        # Store the index of current galaxy in corresponding cell
        chainlist[mesh_indices['x'][igal], mesh_indices['y'][igal], mesh_indices['z'][igal]] = igal

    return mesh_indices, ngal, chainlist, linklist

################################################################################
################################################################################

def in_mask(coordinates, survey_mask, r_limits):
    '''
    Determine whether the specified coordinates are within the masked area.
    '''

    # Convert coordinates to table if not already
    if not isinstance(coordinates, Table):
        coordinates = Table(coordinates, names=['x','y','z'])

    good = True

    r = np.linalg.norm(to_vector(coordinates))
    ra = np.arctan(coordinates['y']/coordinates['x'])*RtoD
    dec = np.arcsin(coordinates['z']/r)*RtoD

    if (coordinates['x'] < 0) and (coordinates['y'] != 0):
        ra += 180.
    if ra < 0:
        ra += 360.

    if (survey_mask[ra.astype(int), dec.astype(int)-dec_offset] == 0) or (r > r_limits[1]) or (r < r_limits[0]):
        good = False

    return good

################################################################################
################################################################################

def in_survey(coordinates, min_limit, max_limit):
    '''
    Determine whether the specified coordinates are within the minimum and 
    maximum limits.
    '''
    good = np.ones(len(coordinates), dtype=bool)
    
    for name in coordinates.colnames:
        check_min = coordinates[name] > min_limit[name]
        check_max = coordinates[name] < max_limit[name]

        good = np.all([good, check_min, check_max], axis=0)

    return good