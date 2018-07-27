#VoidFinder Function to do just about everything

import numpy as np
from sklearn import neighbors
from astropy.table import Table
import time

from hole_combine import combine_holes
from voidfinder_functions import mesh_galaxies, in_mask, in_survey, save_maximals, mesh_galaxies_dict
from table_functions import add_row, subtract_row, to_vector, to_array, table_dtype_cast, table_divide
from volume_cut import volume_cut

from avsepcalc import av_sep_calc
from mag_cutoff_function import mag_cut, field_gal_cut

maskra = 360
maskdec = 180
dec_offset = -90
min_dist = 0.
dr = 1. # Distance to shift the hole centers
frac = 0.1 # Overlap fraction for calculating maximal spheres

# Constants
c = 3e5
DtoR = np.pi/180.
RtoD = 180./np.pi

#def filter_galaxies(in_filename, mask_filename, ngrid, box, max_dist)
def filter_galaxies(infile, maskfile, dl, max_dist):
    
    ################################################################################
    #
    #   PRE-PROCESS DATA
    #
    ################################################################################
    print('Pre-processing data', flush=True)

    #infile = Table.read(in_filename, format='ascii.commented_header')

    # Remove faint galaxies
    infile = mag_cut(infile,-20)

    # Convert galaxy coordinates to Cartesian
    xin = infile['Rgal']*np.cos(infile['ra']*DtoR)*np.cos(infile['dec']*DtoR)
    yin = infile['Rgal']*np.sin(infile['ra']*DtoR)*np.cos(infile['dec']*DtoR)
    zin = infile['Rgal']*np.sin(infile['dec']*DtoR)
    coord_in_table = Table([xin, yin, zin], names=('x','y','z'))

    # Cartesian coordinate minima
    coord_min_x = [min(coord_in_table['x'])]
    coord_min_y = [min(coord_in_table['y'])]
    coord_min_z = [min(coord_in_table['z'])]
    coord_min_table = Table([coord_min_x, coord_min_y, coord_min_z], names=('x','y','z'))

    # Cartesian coordinate maxima
    coord_max_x = [max(coord_in_table['x'])]
    coord_max_y = [max(coord_in_table['y'])]
    coord_max_z = [max(coord_in_table['z'])]
    coord_max_table = Table([coord_max_x, coord_max_y, coord_max_z], names=('x','y','z'))

    # Number of galaxies
    N_gal = len(infile)

    print('x:', coord_min_table['x'][0], coord_max_table['x'][0], flush=True)
    print('y:', coord_min_table['y'][0], coord_max_table['y'][0], flush=True)
    print('z:', coord_min_table['z'][0], coord_max_table['z'][0], flush=True)
    print('There are', N_gal, 'galaxies in this simulation.', flush=True)

    # Convert coord_in, coord_min, coord_max tables to numpy arrays
    coord_in = to_array(coord_in_table)
    coord_min = to_vector(coord_min_table)
    coord_max = to_vector(coord_max_table)

    print('Reading mask',flush=True)

    #maskfile = Table.read(mask_filename, format='ascii.commented_header')
    mask = np.zeros((maskra, maskdec))
    mask[maskfile['ra'].astype(int), maskfile['dec'].astype(int) - dec_offset] = 1
    vol = len(maskfile)

    print('Read mask',flush=True)

    ################################################################################
    #
    #   PUT THE GALAXIES ON A CHAIN MESH
    #
    ################################################################################


    #dl = box/ngrid # length of each side of the box
    #print('Number of grid cells is', ngrid, dl, box)

    #print('Making the grid')

    # Array of size of survey in x, y, z directions [Mpc/h]
    box = np.array([coord_max_x[0] - coord_min_x[0], coord_max_y[0] - coord_min_y[0], coord_max_z[0] - coord_min_z[0]])

    # Array of number of cells in each direction
    ngrid = box/dl
    ngrid = np.ceil(ngrid).astype(int)

    print('Number of grid cells is', ngrid, 'with side lengths of', dl, flush=True)

    '''
    # Bin the galaxies onto a 3D grid
    #mesh_indices, ngal, chainlist, linklist = mesh_galaxies(coord_in_table, coord_min_table, dl, ngrid)
    #ngal, chainlist, linklist = mesh_galaxies(coord_in_table, coord_min_table, dl, tuple(ngrid))

    #print('Made the grid')

  
    print('Checking the grid')
    grid_good = True

    for i in range(ngrid[0]):
        for j in range(ngrid[1]):
            for k in range(ngrid[2]):
                count = 0
                igal = chainlist[i,j,k]
                while igal != -1:
                    count += 1
                    igal = linklist[igal]
                if count != ngal[i,j,k]:
                    print(i,j,k, count, ngal[i,j,k])
                    grid_good = False
    if grid_good:
        print('Grid construction was successful.')
    '''
    ################################################################################
    #
    #   SEPARATION
    #
    ################################################################################
  
    sep_start = time.time()

    print('Finding sep',flush=True)

    l, avsep, sd, dists3 = av_sep_calc(coord_in_table)

    print('Average separation of n3rd gal is', avsep, flush=True)
    print('The standard deviation is', sd,flush=True)

    # l = 5.81637  # s0 = 7.8, gamma = 1.2, void edge = -0.8
    # l = 7.36181  # s0 = 3.5, gamma = 1.4
    # or force l to have a fixed number by setting l = ****

    print('Going to build wall with search value', l, flush=True)

    sep_end = time.time()

    print('Time to find sep =',sep_end-sep_start, flush=True)

    fw_start = time.time()

    f_coord_table, w_coord_table = field_gal_cut(coord_in_table, dists3, l)

    f_coord_table.write('field_gal_file.txt', format='ascii.commented_header',overwrite=True)
    w_coord_table.write('wall_gal_file.txt', format='ascii.commented_header',overwrite=True)
    
    fw_end = time.time()

    print('Time to sort field and wall gals =',fw_end-fw_start, flush=True)

    nf =  len(f_coord_table)
    nwall = len(w_coord_table)
    print('Number of field gals:', nf,'Number of wall gals:', nwall, flush=True)

    return coord_min_table, mask, ngrid












#def find_voids(ngrid, box, max_dist, coord_min_table, mask, out1_filename, out2_filename):
def find_voids(ngrid, dl, max_dist, coord_min_table, mask, out1_filename, out2_filename):

    #dl = box/ngrid # length of each side of the box    
    
    w_coord_table = Table.read('wall_gal_file.txt', format='ascii.commented_header')
    w_coord = to_array(w_coord_table)


    ################################################################################
    #
    #   SET UP CELL GRID DISTRIBUTION
    #
    ################################################################################
    '''
    print('Setting up grid of wall galaxies')

    #wall_mesh_indices, ngal_wall, chainlist_wall, linklist_wall = mesh_galaxies(w_coord_table, coord_min_table, dl, ngrid)
    ngal_wall = mesh_galaxies(w_coord_table, coord_min_table, dl, tuple(ngrid))

    print('Wall galaxy grid set up')
    '''

    # Build a dictionary of all the cell IDs that have at least one galaxy in them
    cell_ID_dict = mesh_galaxies_dict(w_coord_table, coord_min_table, dl)


    print('Galaxy grid indices computed')
    ################################################################################
    #
    #   BUILD NEAREST-NEIGHBOR TREE
    #
    ################################################################################


    galaxy_tree = neighbors.KDTree(w_coord)


    ################################################################################
    #
    #   GROW HOLES
    #
    ################################################################################

    hole_times = []

    tot_hole_start = time.time()

    print('Growing holes', flush=True)

    # Center of the current cell
    hole_center_table = Table(np.zeros(6), names=('x', 'y', 'z', 'r', 'ra', 'dec'))

    # Initialize list of hole details
    myvoids_x = []
    myvoids_y = []
    myvoids_z = []
    myvoids_r = []

    # Number of holes found
    n_holes = 0

    # Find where all the empty cells are
    #empty_indices = np.where(ngal_wall == 0)

    # Go through each empty cell in the grid
    empty_cell = 0

    # Number of empty cells
    n_empty_cells = ngrid[0]*ngrid[1]*ngrid[2] - len(cell_ID_dict)

    #for empty_cell in range(len(empty_indices[0])):
    for i in range(ngrid[0]):
        for j in range(ngrid[1]):
            for k in range(ngrid[2]):

                check_bin_ID = (i,j,k)

                # Check if there are any galaxies in this grid in this cell
                if check_bin_ID not in cell_ID_dict:

                    hole_start = time.time()

                    '''
                    # Retrieve empty cell indices
                    i = empty_indices[0][empty_cell]
                    j = empty_indices[1][empty_cell]
                    k = empty_indices[2][empty_cell]
                    

                    if empty_cell%10000 == 0:
                        print('Looking in empty cell', empty_cell, 'of', len(empty_indices[0]))
                    '''
                    empty_cell += 1

                    if empty_cell%10000 == 0:
                        print('Looking in empty cell', empty_cell, 'of', n_empty_cells, flush=True)

                    # Calculate center coordinates of cell
                    hole_center_table['x'] = (i + 0.5)*dl + coord_min_table['x']
                    hole_center_table['y'] = (j + 0.5)*dl + coord_min_table['y']
                    hole_center_table['z'] = (k + 0.5)*dl + coord_min_table['z']
                    
                    hole_center = to_vector(hole_center_table)
                    hole_center = hole_center
                    
                    # Check to make sure that the hole center is still within the survey
                    if not in_mask(hole_center.T, mask, [min_dist, max_dist]):
                        continue

                    #print('______________________________________________')

                    # Find closest galaxy to cell center
                    modv1, k1g = galaxy_tree.query(hole_center.T, k=1)
                    modv1 = modv1[0][0]
                    k1g = k1g[0][0]

                    # Unit vector pointing from cell center to the closest galaxy
                    v1_unit = (w_coord[k1g] - hole_center.T)/modv1

                    #print('Hole radius', modv1, 'after finding 1st galaxy')

                    
                    ############################################################
                    # We are going to shift the center of the hole by dr along 
                    # the direction of the vector pointing from the nearest 
                    # galaxy to the center of the empty cell.  From there, we 
                    # will search within a radius of length the distance between 
                    # the center of the hole and the first galaxy from the 
                    # center of the hole to find the next nearest neighbors.  
                    # From there, we will minimize top/bottom to find which one 
                    # is the next nearest galaxy that bounds the hole.
                    ############################################################

                    galaxy_search = True

                    hole_center_2 = hole_center

                    while galaxy_search:

                        # Shift hole center away from first galaxy
                        hole_center_2 = hole_center_2 - dr*v1_unit.T
                        
                        # Distance between hole center and nearest galaxy
                        modv1 += dr
                        
                        # Search for nearest neighbors within modv1 of the hole center
                        i_nearest, dist_nearest = galaxy_tree.query_radius(hole_center_2.T, r=modv1, return_distance=True, sort_results=True)

                        i_nearest = i_nearest[0]
                        dist_nearest = dist_nearest[0]

            	        # Remove nearest galaxy from list
                        boolean_nearest = i_nearest != k1g
                        dist_nearest = dist_nearest[boolean_nearest]
                        i_nearest = i_nearest[boolean_nearest]

                        if len(i_nearest) > 0:
                            # Found at least one other nearest neighbor!

                            # Calculate vector pointing from next nearest galaxies to the nearest galaxy
                            BA = w_coord[k1g] - w_coord[i_nearest]
                            
                            bot = 2*np.dot(BA, v1_unit.T)
                            
                            top = np.sum(BA**2, axis=1)
                            
                            x2 = top/bot.T[0]

                            # Locate positive values of x2
                            valid_idx = np.where(x2 > 0)[0]
                            
                            if len(valid_idx) > 0:
                                # Find index of 2nd nearest galaxy
                                k2g_x2 = valid_idx[x2[valid_idx].argmin()]
                                k2g = i_nearest[k2g_x2]

                                minx2 = x2.T[k2g_x2]

                                galaxy_search = False
                            
                        elif not in_mask(hole_center_2.T, mask, [min_dist, max_dist]):
                            # Hole is no longer within survey limits
                            galaxy_search = False

                    # Check to make sure that the hole center is still within the survey
                    if not in_mask(hole_center_2.T, mask, [min_dist, max_dist]):
                        #print('hole not in survey')
                        continue

                    #print('Found 2nd galaxy')
                    '''
                    if k2g == i_nearest[0]:
                        print('2nd galaxy was the next nearest neighbor.')
                    else:
                        print('2nd galaxy was NOT the next nearest neighbor.')
                    '''

                    # Calculate new hole center
                    hole_radius = 0.5*np.sum(BA[k2g_x2]**2)/np.dot(BA[k2g_x2], v1_unit.T)
                    hole_center = w_coord[k1g] - hole_radius*v1_unit
                    #print('hole center',hole_center)
                    #print('Hole radius', hole_radius, 'after finding 2nd galaxy')

                    # Check to make sure that the hole center is still within the survey
                    if not in_mask(hole_center, mask, [min_dist, max_dist]):
                        #print('hole not in survey')
                        continue

                    '''
                    ########################################################################
                    # TEST BLOCK

                    # Make sure that there are no galaxies contained within hole
                    i_nearest, dist_nearest = galaxy_tree.query_radius(hole_center, r=hole_radius, return_distance=True, sort_results=True)
                    i_nearest = i_nearest[0]
                    dist_nearest = dist_nearest[0]

                    # Remove two nearest galaxies from list
                    boolean_nearest = np.logical_and(i_nearest != k1g, i_nearest != k2g)
                    dist_nearest = dist_nearest[boolean_nearest]
                    i_nearest = i_nearest[boolean_nearest]

                    if len(i_nearest) > 0:
                        print('2nd galaxy - There are galaxies inside the hole!', len(i_nearest))
                    ########################################################################
                    '''



                    ########################################################################
                    # Now find the third nearest galaxy.
                    ########################################################################

                    # Find the midpoint between the two nearest galaxies
                    midpoint = 0.5*(w_coord[k1g] + w_coord[k2g])                

                    # Define the unit vector along which to move the hole center
                    modv2 = np.linalg.norm(hole_center - midpoint)
                    v2_unit = (hole_center - midpoint)/modv2

                    # Same methodology as for finding the second galaxy

                    hole_center_3 = hole_center

                    galaxy_search = True

                    while galaxy_search:

                        # Shift hole center along unit vector
                        hole_center_3 = hole_center_3 + dr*v2_unit

                        # Calculate vector pointing from the hole center to the nearest galaxy
                        Acenter = w_coord[k1g] - hole_center
                        # Calculate vector pointing from the hole center to the second-nearest galaxy
                        Bcenter = w_coord[k2g] - hole_center

                        # New hole "radius"
                        search_radius = np.linalg.norm(w_coord[k1g] - hole_center_3)
                        
                        # Search for nearest neighbors within modv1 of the hole center
                        #i_nearest, dist_nearest = galaxy_tree.query_radius(hole_center, r=np.linalg.norm(Acenter), return_distance=True)
                        i_nearest, dist_nearest = galaxy_tree.query_radius(hole_center_3, r=search_radius, return_distance=True, sort_results=True)

                        i_nearest = i_nearest[0]
                        dist_nearest = dist_nearest[0]

            	        # Remove two nearest galaxies from list
                        boolean_nearest = np.logical_and(i_nearest != k1g, i_nearest != k2g)
                        dist_nearest = dist_nearest[boolean_nearest]
                        i_nearest = i_nearest[boolean_nearest]

                        if len(i_nearest) > 0:
                            # Found at least one other nearest neighbor!

                            # Calculate vector pointing from hole center to next nearest galaxies
                            Ccenter = w_coord[i_nearest] - hole_center
                            
                            bot = 2*np.dot((Ccenter - Acenter), v2_unit.T)
                            
                            top = np.sum(Ccenter**2, axis=1) - np.sum(Bcenter**2)
                            
                            x3 = top/bot.T[0]

                            # Locate positive values of x3
                            valid_idx = np.where(x3 > 0)[0]

                            if len(valid_idx) > 0:
                                # Find index of 3rd nearest galaxy
                                k3g_x3 = valid_idx[x3[valid_idx].argmin()]
                                k3g = i_nearest[k3g_x3]

                                minx3 = x3[k3g_x3]

                                galaxy_search = False

                        elif not in_mask(hole_center, mask, [min_dist, max_dist]):
                            # Hole is no longer within survey limits
                            galaxy_search = False

                    # Check to make sure that the hole center is still within the survey
                    if not in_mask(hole_center_3, mask, [min_dist, max_dist]):
                        #print('hole not in survey')
                        continue

                    #print('Found 3rd galaxy')
                    '''
                    if k3g == i_nearest[0]:
                        print('3rd galaxy was the next nearest neighbor.')
                    else:
                        print('3rd galaxy was NOT the next nearest neighbor.')
                    '''

                    # Calculate new hole center
                    hole_center = hole_center + minx3*v2_unit
                    hole_radius = np.linalg.norm(hole_center - w_coord[k1g])
                    #print('Hole radius', hole_radius, 'after finding 3rd galaxy')

                    # Check to make sure that the hole center is still within the survey
                    if not in_mask(hole_center, mask, [min_dist, max_dist]):
                        #print('hole not in survey')
                        continue

                    '''
                    ########################################################################
                    # TEST BLOCK

                    # Make sure that there are no galaxies contained within hole
                    i_nearest, dist_nearest = galaxy_tree.query_radius(hole_center, r=hole_radius, return_distance=True, sort_results=True)
                    i_nearest = i_nearest[0]
                    dist_nearest = dist_nearest[0]

                    # Remove two nearest galaxies from list
                    boolean_nearest = np.logical_and.reduce((i_nearest != k1g, i_nearest != k2g, i_nearest != k3g))
                    dist_nearest = dist_nearest[boolean_nearest]
                    i_nearest = i_nearest[boolean_nearest]

                    if len(i_nearest) > 0:
                        print('3rd galaxy - There are galaxies inside the hole!', len(i_nearest))
                    ########################################################################
                    '''




                    ########################################################################
                    # Now find the 4th nearest neighbor
                    #
                    # Process is very similar as before, except we do not know if we have to 
                    # move above or below the plane.  Therefore, we will find the next closest 
                    # if we move above the plane, and the next closest if we move below the 
                    # plane.
                    ########################################################################

                    # The vector along which to move the hole center is defined by the cross 
                    # product of the vectors pointing between the three nearest galaxies.
                    AB = w_coord[k1g] - w_coord[k2g]
                    BC = w_coord[k3g] - w_coord[k2g]
                    v3 = np.cross(AB,BC)

                    modv3 = np.linalg.norm(v3)
                    v3_unit = v3/modv3

                    # First move in the direction of the unit vector defined above
                    galaxy_search = True
                    
                    hole_center_41 = hole_center 

                    while galaxy_search:

                        # Shift hole center along unit vector
                        hole_center_41 = hole_center_41 + dr*v3_unit
                        #print('Shifted center to', hole_center_41)

                        # Calculate vector pointing from the hole center to the nearest galaxy
                        Acenter = w_coord[k1g] - hole_center
                        # Calculate vector pointing from the hole center to the second-nearest galaxy
                        Bcenter = w_coord[k2g] - hole_center

                        # New hole "radius"
                        search_radius = np.linalg.norm(w_coord[k1g] - hole_center_41)

                        # Search for nearest neighbors within R of the hole center
                        #i_nearest, dist_nearest = galaxy_tree.query_radius(hole_center_41, r=np.linalg.norm(Acenter),return_distance=True)
                        i_nearest, dist_nearest = galaxy_tree.query_radius(hole_center_41, r=search_radius, return_distance=True, sort_results=True)

                        i_nearest = i_nearest[0]
                        dist_nearest = dist_nearest[0]

            	        # Remove two nearest galaxies from list
                        boolean_nearest = np.logical_and.reduce((i_nearest != k1g, i_nearest != k2g, i_nearest != k3g))
                        dist_nearest = dist_nearest[boolean_nearest]
                        i_nearest = i_nearest[boolean_nearest]
                        #print('Number of nearby galaxies', len(i_nearest))

                        if len(i_nearest) > 0:
                            # Found at least one other nearest neighbor!

                            # Calculate vector pointing from hole center to next nearest galaxies
                            Dcenter = w_coord[i_nearest] - hole_center
                            
                            bot = 2*np.dot((Dcenter - Acenter), v3_unit.T) 
                            
                            top = np.sum(Dcenter**2, axis=1) - np.sum(Bcenter**2)
                            
                            x41 = top/bot.T[0]

                            # Locate positive values of x41
                            valid_idx = np.where(x41 > 0)[0]

                            if len(valid_idx) > 0:
                                # Find index of 4th nearest galaxy
                                k4g1_x41 = valid_idx[x41[valid_idx].argmin()]
                                k4g1 = i_nearest[k4g1_x41]

                                minx41 = x41[k4g1_x41]

                                galaxy_search = False

                        elif not in_mask(hole_center_41, mask, [min_dist, max_dist]):
                            # Hole is no longer within survey limits
                            galaxy_search = False

                    #print('Found first potential 4th galaxy')
                    '''
                    if k4g1 == i_nearest[0]:
                        print('First 4th galaxy was the next nearest neighbor.')
                    else:
                        print('First 4th galaxy was NOT the next nearest neighbor.')
                    '''

                    # Calculate potential new hole center
                    if in_mask(hole_center_41, mask, [min_dist, max_dist]):
                        hole_center_41 = hole_center + minx41*v3_unit
                        #print('______________________')
                        #print(hole_center_41, 'hc41')
                        #print('hole_radius_41', np.linalg.norm(hole_center_41 - w_coord[k1g]))
                   
                    # Repeat same search, but shift the hole center in the other direction this time
                    v3_unit = -v3_unit

                    # First move in the direction of the unit vector defined above
                    galaxy_search = True

                    # Initialize minx42 (in case it does not get created later)
                    minx42 = np.infty

                    hole_center_42 = hole_center
                    
                    minx42 = np.infty

                    while galaxy_search:

                        # Shift hole center along unit vector
                        hole_center_42 = hole_center_42 + dr*v3_unit

                        # Calculate vector pointing from the hole center to the nearest galaxy
                        Acenter = w_coord[k1g] - hole_center
                        # Calculate vector pointing from the hole center to the second-nearest galaxy
                        Bcenter = w_coord[k2g] - hole_center

                        # New hole "radius"
                        search_radius = np.linalg.norm(w_coord[k1g] - hole_center_42)

                        # Search for nearest neighbors within R of the hole center
                        #i_nearest, dist_nearest = galaxy_tree.query_radius(hole_center_42, r=np.linalg.norm(Acenter), return_distance=True)
                        i_nearest, dist_nearest = galaxy_tree.query_radius(hole_center_42, r=search_radius, return_distance=True, sort_results=True)

                        i_nearest = i_nearest[0]
                        dist_nearest = dist_nearest[0]

            	        # Remove three nearest galaxies from list
                        boolean_nearest = np.logical_and.reduce((i_nearest != k1g, i_nearest != k2g, i_nearest != k3g))
                        dist_nearest = dist_nearest[boolean_nearest]
                        i_nearest = i_nearest[boolean_nearest]

                        if len(i_nearest) > 0:
                            # Found at least one other nearest neighbor!

                            # Calculate vector pointing from hole center to next nearest galaxies
                            Dcenter = w_coord[i_nearest] - hole_center

                            bot = 2*np.dot((Dcenter - Acenter), v3_unit.T)

                            top = np.sum(Dcenter**2, axis=1) - np.sum(Bcenter**2)

                            x42 = top/bot.T[0]

                            # Locate positive values of x42
                            valid_idx = np.where(x42 > 0)[0]

                            if len(valid_idx) > 0:
                                # Find index of 3rd nearest galaxy
                                k4g2_x42 = valid_idx[x42[valid_idx].argmin()]
                                k4g2 = i_nearest[k4g2_x42]

                                minx42 = x42[k4g2_x42]

                                galaxy_search = False

                        elif not in_mask(hole_center_42, mask, [min_dist, max_dist]):
                          
                            # Hole is no longer within survey limits
                            galaxy_search = False

                    #print('Found second potential 4th galaxy')
                    '''
                    if k4g2 == i_nearest[0]:
                        print('Second 4th galaxy was the next nearest neighbor.')
                    else:
                        print('Second 4th galaxy was NOT the next nearest neighbor.')
                    '''

                    # Calculate potential new hole center
                    if in_mask(hole_center_42, mask, [min_dist, max_dist]):
                        hole_center_42 = hole_center + minx42*v3_unit
                        #print(hole_center_42, 'hc42')
                        #print('hole_radius_42', np.linalg.norm(hole_center_42 - w_coord[k1g]))
                        #print('minx41:', minx41, '   minx42:', minx42)
                    
                    '''if not in_mask(hole_center_41, mask, [min_dist, max_dist]):
                        print('hole_center_41 is NOT in the mask')
                    if not in_mask(hole_center_42, mask, [min_dist, max_dist]):
                        print('hole_center_42 is NOT in the mask')'''
                    
                    # Determine which is the 4th nearest galaxy
                    if in_mask(hole_center_41, mask, [min_dist, max_dist]) and minx41 <= minx42:
                        # The first 4th galaxy found is the next closest
                        hole_center = hole_center_41
                        k4g = k4g1
                    elif in_mask(hole_center_42, mask, [min_dist, max_dist]):
                        # The second 4th galaxy found is the next closest
                        hole_center = hole_center_42
                        k4g = k4g2
                    elif in_mask(hole_center_41, mask, [min_dist, max_dist]):
                        # The first 4th galaxy found is the next closest
                        hole_center = hole_center_41
                        k4g = k4g1
                    else:
                        # Neither hole center is within the mask - not a valid hole
                        continue

                    # Radius of the hole
                    hole_radius = np.linalg.norm(hole_center - w_coord[k1g])
                    hole_center = hole_center.T

                    '''
                    ########################################################################
                    # TEST BLOCK

                    # Make sure that there are no galaxies contained within hole
                    i_nearest, dist_nearest = galaxy_tree.query_radius(hole_center.T, r=hole_radius, return_distance=True, sort_results=True)
                    i_nearest = i_nearest[0]
                    dist_nearest = dist_nearest[0]

                    # Remove two nearest galaxies from list
                    boolean_nearest = np.logical_and.reduce((i_nearest != k1g, i_nearest != k2g, i_nearest != k3g, i_nearest != k4g))
                    dist_nearest = dist_nearest[boolean_nearest]
                    i_nearest = i_nearest[boolean_nearest]

                    if len(i_nearest) > 0:
                        print('________________________________________________')
                        print('There are galaxies inside the hole!', len(i_nearest))
                        print('Final radius:', hole_radius)
                    ########################################################################
                    '''
                    '''
                    if hole_radius > 23:
                        if len(i_nearest) == 0:
                            print('_______________________________________________')
                        print(hole_center_41, 'hc41')
                        print('hole_radius_41', np.linalg.norm(hole_center_41 - w_coord[k1g]))
                        print(hole_center_42, 'hc42')
                        print('hole_radius_42', np.linalg.norm(hole_center_42 - w_coord[k1g]))
                        print('Final hole radius:', hole_radius)
                    '''
                    
                    # Save hole
                    myvoids_x.append(hole_center[0,0])
                    myvoids_y.append(hole_center[1,0])
                    myvoids_z.append(hole_center[2,0])
                    myvoids_r.append(hole_radius)
                    hole_end = time.time()
                    n_holes += 1
                    hole_times.append(hole_end-hole_start)

                    '''
                    if n_holes%100 == 0:
                        print("number of holes=",n_holes)

                    print("number of holes=",n_holes)
                    '''


    print('Found a total of', n_holes, 'potential voids.',flush=True)

    tot_hole_end = time.time()

    print('Time to find all holes =', tot_hole_end - tot_hole_start,flush=True)
    print('AVG time to find each hole =', sum(hole_times)/len(hole_times),flush=True)

    ################################################################################
    #
    #   SORT HOLES BY SIZE
    #
    ################################################################################

    sort_start = time.time()

    print('Sorting holes by size',flush=True)

    potential_voids_table = Table([myvoids_x, myvoids_y, myvoids_z, myvoids_r], names=('x','y','z','radius'))

    # Need to sort the potential voids into size order
    potential_voids_table.sort('radius')
    potential_voids_table.reverse()

    '''
    potential_voids_file = open('potential_voids_list.txt', 'wb')
    pickle.dump(potential_voids_table, potential_voids_file)
    potential_voids_file.close()


    in_file = open('potential_voids_list.txt', 'rb')
    potential_voids_table = pickle.load(in_file)
    in_file.close()
    '''

    sort_end = time.time()

    print('Holes are sorted.',flush=True)
    print('Time to sort holes =', sort_end-sort_start,flush=True)

    ################################################################################
    #
    #   CHECK IF 90% OF VOID VOLUME IS WITHIN SURVEY LIMITS
    #
    ################################################################################

    print('Removing holes with at least 10% of their volume outside the mask',flush=True)

    potential_voids_table = volume_cut(potential_voids_table, mask, [min_dist, max_dist])

    potential_voids_table.write('potential_voids_list.txt', format='ascii.commented_header', overwrite=True)

    ################################################################################
    #
    #   FILTER AND SORT HOLES INTO UNIQUE VOIDS
    #
    ################################################################################

    combine_start = time.time()

    print('Combining holes into unique voids',flush=True)

    maximal_spheres_table, myvoids_table = combine_holes(potential_voids_table, frac)

    print('Number of unique voids is', len(maximal_spheres_table),flush=True)

    # Save list of all void holes
    myvoids_table.write(out2_filename, format='ascii.commented_header', overwrite=True)

    combine_end = time.time()

    print('Time to combine holes into voids =', combine_end-combine_start,flush=True)

    '''
    ################################################################################
    #
    #   COMPUTE VOLUME OF EACH VOID
    #
    ################################################################################
    print('Compute void volumes')

    # Initialize void volume array
    void_vol = np.zeros(void_count)

    nran = 10000

    for i in range(void_count):
        nsph = 0
        rad = 4*myvoids_table['radius'][v_index[i]]

        for j in range(nran):
            rand_pos = add_row(np.random.rand(3)*rad, myvoids_table['x','y','z'][v_index[i]]) - 0.5*rad
            
            for k in range(len(myvoids_table)):
                if myvoids_table['flag'][k]:
                    # Calculate difference between particle and sphere
                    sep = sum(to_vector(subtract_row(rand_pos, myvoids_table['x','y','z'][k])))
                    
                    if sep < myvoids_table['radius'][k]**2:
                        # This particle lies in at least one sphere
                        nsph += 1
                        break
        
        void_vol[i] = (rad**3)*nsph/nran
    
    
    ################################################################################
    #
    #   IDENTIFY VOID GALAXIES
    #
    ################################################################################
    print('Assign field galaxies to voids')

    # Count the number of galaxies in each void
    nfield = np.zeros(void_count)

    # Add void field to f_coord
    f_coord['vID'] = -99

    for i in range(nf): # Go through each void galaxy
        for j in range(len(myvoids_table)): # Go through each void
            if np.linalg.norm(to_vector(subtract_row(f_coord[i], myvoids_table['x','y','z'][j]))) < myvoids_table['radius'][j]:
                # Galaxy lives in the void
                nfield[myvoids_table['flag'][j]] += 1

                # Set void ID in f_coord to match void ID
                f_coord['vID'][i] = myvoids_table['flag'][j]

                break

    f_coord.write(voidgals_filename, format='ascii.commented_header')
    '''

    ################################################################################
    #
    #   MAXIMAL HOLE FOR EACH VOID
    #
    ################################################################################

    save_maximals(maximal_spheres_table, out1_filename)

    '''
    ################################################################################
    #
    #   VOID REGION SIZES
    #
    ################################################################################


    # Initialize
    void_regions = Table()

    void_regions['radius'] = myvoids_table['radius'][v_index]
    void_regions['effective_radius'] = (void_vol*0.75/np.pi)**(1./3.)
    void_regions['volume'] = void_vol
    void_regions['x'] = myvoids_table['x'][v_index]
    void_regions['y'] = myvoids_table['y'][v_index]
    void_regions['z'] = myvoids_table['z'][v_index]
    void_regions['deltap'] = (nfield - N_gal*void_vol/vol)/(N_gal*void_vol/vol)
    void_regions['n_gal'] = nfield
    void_regions['vol_maxHole'] = (4./3.)*np.pi*myvoids_table['radius'][v_index]**3/void_vol

    void_regions.write(out3_filename, format='ascii.commented_header')
    '''

