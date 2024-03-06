

import numpy as np
from astropy.table import Table, Row

from .table_functions import add_row, \
                             subtract_row, \
                             table_divide, \
                             table_dtype_cast, \
                             row_cross, \
                             row_dot, \
                             to_vector, \
                             to_array
'''
import warnings
warnings.filterwarnings('error')
'''

RtoD = 180./np.pi
DtoR = np.pi/180.
dec_offset = -90



################################################################################
#
#   DEFINE FUNCTIONS
#
################################################################################

def mesh_galaxies(galaxy_coords, coord_min, grid_side_length, N_boxes):
    '''
    Sort galaxies onto a cubic grid

    Parameters:
    ===========

    galaxy_coords: astropy table
        galaxy Cartesian coordinates (columns x, y, and z)

    coord_min: one-row astropy table
        the minima in each of the three coordinates

    grid_side_length: float
        length of a grid cell

    N_boxes: integer
        number of cells in the grid


    Output:
    =======

    mesh_indices: astropy table
        the cell coordinates for each galaxy

    ngal: 3D numpy array
        the number of galaxies in each cell

    chainlist: 3D numpy array (same size as ngal)
        the index value of the last galaxy to be stored in that cell

    linklist: 1D numpy array
        length of the number of galaxies that stores the index value of the 
        previous galaxy stored in the cell of the current galaxy.  If the 
        galaxy is the first one to be put in the cell, then its value in 
        linklist is -1.  Using both chainlist and linklist, one can discern all 
        the galaxies that live in a given cell.
    '''

    # Initialize the 3D bins that will contain the galaxy indices

    #ngal = np.zeros((N_boxes, N_boxes, N_boxes), dtype=int)
    ngal = np.zeros(N_boxes, dtype=int)

    # Initialize the 3D bins that will contain the galaxy indices

    #chainlist = -np.ones((N_boxes, N_boxes, N_boxes), dtype=int)
    #chainlist = -np.ones(N_boxes, dtype=int)

    # Initialize a list that will store the galaxy's index that previously occupied the cell
    #linklist = np.zeros(len(galaxy_coords), dtype=int)

    # Convert the galaxy coordinates to grid indices
    mesh_indices = table_dtype_cast(table_divide(subtract_row(galaxy_coords, coord_min), grid_side_length), int)

    for igal in range(len(galaxy_coords)):
        
        # Increase the number of galaxies in corresponding cell in ngal
        ngal[mesh_indices['x'][igal], mesh_indices['y'][igal], mesh_indices['z'][igal]] += 1
        
        # Store the index of the last galaxy that was saved in corresponding cell 
        #linklist[igal] = chainlist[mesh_indices['x'][igal], mesh_indices['y'][igal], mesh_indices['z'][igal]]

        # Store the index of current galaxy in corresponding cell
        #chainlist[mesh_indices['x'][igal], mesh_indices['y'][igal], mesh_indices['z'][igal]] = igal
    
    #return mesh_indices, ngal, chainlist, linklist
    return ngal#, chainlist, linklist


################################################################################
################################################################################

def mesh_galaxies_dict(galaxy_coords, coord_min, grid_side_length):
    '''
    Build a dictionary of the galaxies' cell coordinates
    '''

    # Convert the galaxy coordinates to grid indices
    mesh_indices = ((galaxy_coords - coord_min)/grid_side_length).astype(int)
    #mesh_indices = table_dtype_cast(table_divide(subtract_row(galaxy_coords, coord_min), grid_side_length), int)

    # Initialize dictionary of cell IDs with at least one galaxy in them
    cell_ID_dict = {}

    for idx in range(len(mesh_indices)):

        #x = mesh_indices['x'][idx]
        #y = mesh_indices['y'][idx]
        #z = mesh_indices['z'][idx]

        #bin_ID = (x,y,z)
        bin_ID = tuple(mesh_indices[idx])

        cell_ID_dict[bin_ID] = 1

    return cell_ID_dict


################################################################################
################################################################################




################################################################################
################################################################################

def in_mask_table(coordinates, survey_mask, r_limits):
    '''
    Determine whether the specified coordinates are within the masked area.
    '''

    # Convert coordinates to table if not already
    if not isinstance(coordinates, Table):
        coordinates = Table(coordinates, names=['x','y','z'])

    good = True

    r = np.linalg.norm(to_vector(coordinates))
    n = 1 + (DtoR*r/10.).astype(int)
    ra = np.arctan(coordinates['y'][0]/coordinates['x'][0])*RtoD
    dec = np.arcsin(coordinates['z'][0]/r)*RtoD


    if (coordinates['x'] < 0) and (coordinates['y'] != 0):
        ra += 180.
    if ra < 0:
        ra += 360.
    
    if (survey_mask[n-1][(n*ra).astype(int)][(n*dec).astype(int)-n*dec_offset] == 0) or (r > r_limits[1]) or (r < r_limits[0]):
        good = False

    return good


def in_mask(coordinates, survey_mask, n, r_limits):
    '''
    Determine whether the specified coordinates are within the masked area.
    
    coordinates : astropy Table of shape (N,3), or Row
    '''

    # Convert coordinates to table if not already
    if isinstance(coordinates, Table):
        coordinates = to_array(coordinates)
    elif isinstance(coordinates, Row):
        coordinates = to_vector(coordinates)
        coordinates.shape = (1,3)
        
    #print(coordinates.shape)
    #exit()

    r = np.linalg.norm(coordinates, axis=1)
    ra = np.arctan(coordinates[:,1]/coordinates[:,0])*RtoD
    dec = np.arcsin(coordinates[:,2]/r)*RtoD


    boolean_ra180 = np.logical_and(coordinates[:,0] < 0, coordinates[:,1] != 0)
    ra[boolean_ra180] += 180.
    ra[ra < 0] += 360.

    angood = []
    
    for i in range(len(ra)):
        
        angood.append( survey_mask[ int(n*ra[i]), int(n*dec[i]) - n*dec_offset])
        
        
    #good = np.logical_and.reduce((np.array(angood), r <= r_limits[1], r >= r_limits[0]))
    good = np.logical_and.reduce((np.array(angood), r < r_limits[1], r > r_limits[0]))

    return good



def not_in_mask(coordinates, survey_mask_ra_dec, n, rmin, rmax):
    '''
    Determine whether a given set of coordinates falls within the survey.

    Parameters:
    ============

    coordinates : numpy.ndarray of shape (1,3), in x-y-z order and cartesian coordinates
        x,y, and z are measured in Mpc/h

    survey_mask_ra_dec : numpy.ndarray of shape (num_ra, num_dec) where 
        the element at [i,j] represents whether or not the ra corresponding to
        i and the dec corresponding to j fall within the mask.  ra and dec
        are both measured in degrees.

    n : integer
        Scale factor of coordinates in mask

    rmin, rmax : scalar, min and max values of survey distance in units of
        Mpc/h

    Returns:
    ========

    boolean : True if coordinates fall outside the survey_mask
    '''

    coords = coordinates[0]  # Convert shape from (1,3) to (3,)
    r = np.linalg.norm(coords)

    if (r < rmin) or (r > rmax) or (r == 0):
        return True

    ra = np.arctan(coords[1]/coords[0])*RtoD
    dec = np.arcsin(coords[2]/r)*RtoD

    if coords[0] < 0 and coords[1] != 0:
        ra += 180
    if ra < 0:
        ra += 360

    return not survey_mask_ra_dec[int(n*ra), int(n*dec) - n*dec_offset]






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


################################################################################
################################################################################

def xyz_to_radecz(sphere_table):
    '''
    Calculate the ra, dec coordinates for the centers of each sphere
    '''

    r = np.linalg.norm(to_array(sphere_table), axis=1)
    sphere_table['r'] = r.T
    sphere_table['r'].unit='Mpc/h'
    sphere_table['ra'] = np.arctan(sphere_table['y']/sphere_table['x'])*RtoD
    sphere_table['ra'].unit='deg'
    sphere_table['dec'] = np.arcsin(sphere_table['z']/sphere_table['r'])*RtoD
    sphere_table['dec'].unit='deg'

    # Adjust ra value as necessary
    boolean = np.logical_and(sphere_table['y'] != 0, sphere_table['x'] < 0)
    sphere_table['ra'][boolean] += 180.

    return sphere_table


################################################################################
################################################################################

def save_maximals(sphere_table, out1_filename, verbose=0):
    '''
    Calculate the ra, dec coordinates for the centers of each of the maximal spheres
    Save the maximal spheres to a text file
    '''
    
    if verbose > 0:
        print("Saving maximals to", out1_filename, flush=True)

    sphere_table = xyz_to_radecz(sphere_table)

    #print(sphere_table)

    sphere_table.write(out1_filename, format='ascii.commented_header',overwrite=True)
