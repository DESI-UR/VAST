

from astropy.table import Table

import numpy as np

from .table_functions import to_array, to_vector

import time

#import warnings
#warnings.simplefilter('error')

from ._hole_combine_cython import remove_duplicates_2, \
                                  find_maximals_2, \
                                  find_maximals_3, \
                                  find_holes_2, \
                                  join_holes_to_maximals


################################################################################
################################################################################


def spherical_cap_volume(radius, height):
    '''Calculate the volume of a spherical cap'''

    volume = np.pi*(height**2)*(3*radius - height)/3.

    return volume


################################################################################
################################################################################


def cap_height(R, r, d):
    '''Calculate the height of a spherical cap.

    Parameters
    __________

    R : radius of sphere

    r : radius of other sphere

    d : distance between sphere centers


    Output
    ______

    h : height of cap
    '''
    '''for elem in d:
        if np.isnan(elem) or elem == 0:
            print(d)
            print(':(')'''
    h = (r - R + d)*(r + R - d)/(2*d)
    

    return h


################################################################################
################################################################################


def combine_holes(spheres_table, frac=0.1):
    '''
    Combines the potential void spheres into voids.


    Parameters:
    ===========

    spheres_table : astropy table of length N
        Table of holes found, sorted by radius.  Required columns: radius, x, y, 
        z (in units of Mpc/h)

    frac : float
        Fraction of hole volume.  If a hole overlaps a maximal sphere by more 
        than this fraction, it is not part of a unique void.  Default is 10%.


    Returns:
    ========

    maximal_spheres_table : astropy table of length M
        Table of maximal spheres (largest hole in each void).  Minimum radius is 
        10 Mpc/h.  Columns: x, y, z, radius (all in units of Mpc/h), flag 
        (unique void number).

    holes_table : astropy table of length P
        Table of all holes that belong to voids.  Columns: x, y, z, radius (all 
        in units of Mpc/h), flag (void number identifier).
    '''


    print('Starting hole combine', flush=True)

    time_start = time.time()

    ############################################################################
    # Remove duplicate holes
    #---------------------------------------------------------------------------
    unique_spheres_table = remove_duplicates(spheres_table)
    ############################################################################

    print("Remove duplicate spheres:", time.time() - time_start, flush=True)


    ############################################################################
    # Maximal spheres
    #---------------------------------------------------------------------------
    time_start = time.time()
    
    maximal_spheres_table, maximal_spheres_indices = find_maximals(unique_spheres_table, frac)

    print("Find maximal holes:", time.time() - time_start, flush=True)

    # Array of coordinates for maximal spheres
    maximal_spheres_coordinates = to_array(maximal_spheres_table)

    # Array of radii for maximal spheres
    maximal_spheres_radii = np.array(maximal_spheres_table['radius'])
    ############################################################################



    ############################################################################
    # Assign spheres to voids
    #---------------------------------------------------------------------------
    time_start = time.time()

    holes_table = find_holes(unique_spheres_table, maximal_spheres_table, maximal_spheres_indices)

    print("Merge holes into voids:", time.time() - time_start, flush=True)
    ############################################################################

    return maximal_spheres_table, holes_table







def combine_holes_2(x_y_z_r_array,
                    boundary_holes, 
                    mask_checker,
                    dup_tol=0.1,
                    maximal_overlap_frac=0.1,
                    min_maximal_radius=10.0,
                    hole_join_frac=0.5,
                    verbose=0,
                    ):
    """
    Perform 3 steps to unionize the found holes into Voids.
    
    1).  Remove specific duplicate holes
    2).  Find holes which qualify as a Maximal
    3).  Join all other holes to the Maximals to form Voids.
    

    PARAMETERS
    ==========
    
    x_y_z_r_array : ndarray of shape (N,4)
        the xyz coordinates of the holes and their radii r

    boundary_holes : ndarray of shape (N,)
        whether or not the hole is on the survey boundary

    mask_checker : 
        
    dup_tol : float 
        the tolerance in units of distance to check 2 hole centers
        against in the remove duplicates step
        
    maximal_overlap_frac : float in [0.0, 1.0)
        in the find maximals step, any 2 potential maximals which overlap
        by more than this percentage means the smaller hole will not be
        considered a maximal
        
    min_maximal_radius : float
        the minimum radius in units of distance for a hole to be considered
        for maximal status
        
    hole_join_frac : float in [0,1.0)
        the fraction of the hole's volume required to overlap for a maximal
        for the hole to join that maximal's Void    
    

    RETURNS
    =======

    maximals_table : astropy table of length M
        Table containing the holes from x_y_z_r_array that are identified 
        maximal spheres

    holes_table : astropy table of length P
        Table containing the holes from x_y_z_r_array that are part of a void, 
        including the maximal spheres.
    """
    
    if verbose > 0:
        print("Starting hole combine", flush=True)
    
    ############################################################################
    # Remove holes who are nearly identical
    #---------------------------------------------------------------------------
    remove_dup_start = time.time()
    
    unique_index = remove_duplicates_2(x_y_z_r_array, dup_tol)
    
    if verbose > 0:
        print("Removing duplicate spheres:", time.time() - remove_dup_start, 
              flush=True)
    
    x_y_z_r_array = x_y_z_r_array[unique_index]
    boundary_holes = boundary_holes[unique_index]
    ############################################################################


    
    ############################################################################
    # Iterate through all the holes to find the maximal spheres, return an array
    # of the indices which correspond to the holes which are maximals
    #---------------------------------------------------------------------------
    maximals_start = time.time()
    '''
    maximal_spheres_indices = find_maximals_2(x_y_z_r_array, 
                                              maximal_overlap_frac, 
                                              min_maximal_radius)
    '''
    maximal_spheres_indices, maximal_grid_info = find_maximals_3(x_y_z_r_array, 
                                                                 maximal_overlap_frac, 
                                                                 min_maximal_radius)
    
    if verbose > 0:
        print("Find maximal holes:", time.time() - maximals_start, flush=True)
    ############################################################################

    
    
    ############################################################################
    # Using the list of maximals, build a group of holes (A void, finally!) 
    # around each maximal based on percent intersection
    #---------------------------------------------------------------------------
    hole_merge_start = time.time()
    '''
    hole_flag_array = find_holes_2(x_y_z_r_array, 
                                   maximal_spheres_indices, 
                                   hole_join_frac)
    '''
    hole_flag_array = join_holes_to_maximals(x_y_z_r_array, 
                                             maximal_spheres_indices, 
                                             hole_join_frac, 
                                             maximal_grid_info)
    
    if verbose > 0:
        print("Merging holes into voids:", time.time() - hole_merge_start, 
              flush=True)
    
    holes_index = hole_flag_array[:,0]
    
    holes_flag_index = hole_flag_array[:,1]
    
    maximals_xyzr = x_y_z_r_array[maximal_spheres_indices]
    
    holes_xyzr = x_y_z_r_array[holes_index]

    boundary_voids = boundary_holes[holes_index]
    ############################################################################


    
    ############################################################################
    # Format the results as astropy tables
    #---------------------------------------------------------------------------
    maximals_table = Table(maximals_xyzr, names=('x','y','z','radius'))
    maximals_table["flag"] = np.arange(maximals_xyzr.shape[0])

    holes_table =  Table(holes_xyzr, names=('x','y','z','radius'))
    holes_table["flag"] = holes_flag_index
    ############################################################################



    ############################################################################
    # Mark boundary voids
    #---------------------------------------------------------------------------
    maximals_table["edge"] = 0

    for i,void in enumerate(maximals_table["flag"]):

        # Find all holes associated with this void
        void_hole_indices = holes_table["flag"] == void

        # Check to see if any of the holes are on the boundary
        if np.any(boundary_voids[void_hole_indices]):

            maximals_table["edge"][i] = 1

        else:
            #-------------------------------------------------------------------
            # Also mark those voids with at least one hole within 10 Mpc/h of 
            # the survey boundary
            #-------------------------------------------------------------------
            void_holes = holes_table[void_hole_indices]

            for j in range(np.sum(void_hole_indices)):

                # Find the points which are 10 Mpc/h in each direction from the 
                # center
                hole_x_min = void_holes['x'][j] - 10.
                hole_y_min = void_holes['y'][j] - 10.
                hole_z_min = void_holes['z'][j] - 10.
                hole_x_max = void_holes['x'][j] + 10.
                hole_y_max = void_holes['y'][j] + 10.
                hole_z_max = void_holes['z'][j] + 10.

                # Coordinates to check
                x_coords = [hole_x_min, 
                            hole_x_max, 
                            void_holes['x'][j], 
                            void_holes['x'][j], 
                            void_holes['x'][j], 
                            void_holes['x'][j]]

                y_coords = [void_holes['y'][j], 
                            void_holes['y'][j], 
                            hole_y_min, 
                            hole_y_max, 
                            void_holes['y'][j], 
                            void_holes['y'][j]]

                z_coords = [void_holes['z'][j], 
                            void_holes['z'][j], 
                            void_holes['z'][j], 
                            void_holes['z'][j], 
                            hole_z_min, 
                            hole_z_max]

                extreme_coords = np.array([x_coords, y_coords, z_coords])
                
                # Check to see if any of these are outside the survey
                for k in range(6):

                    if mask_checker.not_in_mask(extreme_coords[:,k]):

                        # Hole center is within 10 Mpc/h of the survey edge
                        maximals_table["edge"][i] = 2

                        break
            #-------------------------------------------------------------------
    ############################################################################
    
    return maximals_table, holes_table





################################################################################
################################################################################
################################################################################


def remove_duplicates(spheres_table, tol=0.1):
    '''
    Remove all duplicate spheres


    Parameters:
    ===========

    spheres_table : astropy table of length N
        Table of spheres found by VoidFinder.  Columns must include x, y, z, 
        radius (all in units of Mpc/h)

    tol : float
        Tolerence within which to consider two spheres to be identical.


    Returns:
    ========

    unique_spheres_table : astropy table of length Q
        Table of unique spheres.  Columns include x, y, z, radius (all in units 
        of Mpc/h)
    '''



    '''
    time_1 = time.time()

    # At least the first sphere will remain
    unique_spheres_indices = [0]

    
    for i in range(1, len(spheres_table)):

        # Coordinates of sphere i
        sphere_i_coordinates = to_vector(spheres_table[i])

        # Radius of sphere i
        sphere_i_radius = spheres_table['radius'][i]


        ########################################################################
        # COMPARE AGAINST LAST UNIQUE SPHERE
        #
        # Since spheres_table is sorted by radius, identical spheres should be 
        # next to each other in the table.
        ########################################################################

        # Array of coordinates for previously identified unique spheres
        unique_sphere_coordinates = to_array(spheres_table[unique_spheres_indices[-1]])

        # Array of radii for previously identified unique spheres
        unique_sphere_radius = np.array(spheres_table['radius'][unique_spheres_indices[-1]])

        # Distance between sphere i's center and the center of the last unique sphere
        separation = np.linalg.norm(unique_sphere_coordinates - sphere_i_coordinates)

        if (separation > tol) or (unique_sphere_radius - sphere_i_radius > tol):
            # Sphere i is a unique sphere
            unique_spheres_indices.append(i)
    
        
        #else:
        #    # Sphere i is the same as the last unique sphere
        #    print('Sphere i is not a unique sphere')
    '''
    
    array = np.array([spheres_table['x'], 
                      spheres_table['y'], 
                      spheres_table['z'], 
                      spheres_table['radius']])
    x_y_z_r_array = array.T

    unique_index = remove_duplicates_2(x_y_z_r_array, tol)

    # Build unique_spheres_table
    #unique_spheres_table = spheres_table[unique_spheres_indices]
    unique_spheres_table = spheres_table[unique_index]

    return unique_spheres_table



################################################################################
################################################################################
################################################################################


def find_maximals(spheres_table, frac):
    '''
    IDENTIFY MAXIMAL SPHERES
    
    We only consider holes with radii greater than 10 Mpc/h as seeds for a void.  
    If two holes of this size overlap by more than X% of their volume, then they 
    are considered part of the same void.  Otherwise, they are independent 
    voids.  The search runs from the largest hole to the smallest.
    

    Parameters:
    ===========

    spheres_table : astropy table of length N
        Table of holes found, sorted by radius.  Required columns: radius, x, y, 
        z (in units of Mpc/h)

    frac : float
        Fraction of hole volume.  If a hole overlaps a maximal sphere by more 
        than this fraction, it is not part of a unique void.  Default is 10%.
    

    Returns:
    ========

    maximal_spheres_table : astropy table of length M
        Table of maximal spheres.  Columns: x, y, z, radius (all in units of 
        Mpc/h), flag (unique void identifier)

    maximal_spheres_indices : numpy array of shape (M,)
        Integer numpy array of the indices in spheres_table corresponding to 
        each maximal sphere
    '''



    array = np.array([spheres_table['x'], 
                      spheres_table['y'], 
                      spheres_table['z'], 
                      spheres_table['radius']])
    x_y_z_r_array = array.T
    
    #time_1 = time.time()
    
    maximal_spheres_indices_2 = find_maximals_2(x_y_z_r_array, frac, 10.0)
    
    #print("New FIND_MAXIMALS, TIME_1: ", time.time() - time_1, flush=True)

    #time_2 = time.time()

    large_spheres_indices = np.nonzero(spheres_table['radius'] > 10)[0]
    
    all_sphere_coords = to_array(spheres_table)
    all_sphere_radii = spheres_table['radius'].data
    
    maximal_spheres_coordinates = np.empty((large_spheres_indices.shape[0], 3))
    maximal_spheres_radii = np.empty(large_spheres_indices.shape[0])


    maximal_spheres_coordinates[0,:] = to_vector(spheres_table[0])
    maximal_spheres_radii[0] = spheres_table['radius'][0]
    
    out_idx = 1


    # The largest hole is a void
    N_voids = 1
    maximal_spheres_indices = [0]

    for i in large_spheres_indices[1:]:

        # Coordinates of sphere i
        #sphere_i_coordinates = to_vector(spheres_table[i])

        # Radius of sphere i
        #sphere_i_radius = spheres_table['radius'][i]
        
        sphere_i_coordinates = all_sphere_coords[i,:]
        sphere_i_radius = all_sphere_radii[i]

        ########################################################################
        #
        # COMPARE AGAINST MAXIMAL SPHERES
        #
        ########################################################################

        # Array of coordinates for previously identified maximal spheres
        #maximal_spheres_coordinates = to_array(spheres_table[maximal_spheres_indices])

        # Array of radii for previously identified maximal spheres
        #maximal_spheres_radii = np.array(spheres_table['radius'][maximal_spheres_indices])

        # Distance between sphere i's center and the centers of the other maximal spheres
        separation = np.linalg.norm((maximal_spheres_coordinates[0:out_idx,:] - sphere_i_coordinates), axis=1)

        ########################################################################
        # Does sphere i live completely inside another maximal sphere?
        #-----------------------------------------------------------------------
        if any((maximal_spheres_radii[0:out_idx] - sphere_i_radius) >= separation):
            # Sphere i is completely inside another sphere --- sphere i is not a maximal sphere
            #print('Sphere i is completely inside another sphere')
            continue
        ########################################################################


        ########################################################################
        # Does sphere i overlap by less than x% with another maximal sphere?
        #-----------------------------------------------------------------------
        # First - determine which maximal spheres overlap with sphere i
        overlap_boolean =  separation <= (sphere_i_radius + maximal_spheres_radii[0:out_idx])

        if any(overlap_boolean):
            # Sphere i overlaps at least one maximal sphere by some amount.
            # Check to see by how much.

            # Heights of the spherical caps
            height_i = cap_height(sphere_i_radius, 
                                  maximal_spheres_radii[0:out_idx][overlap_boolean], 
                                  separation[overlap_boolean])

            height_maximal = cap_height(maximal_spheres_radii[0:out_idx][overlap_boolean], 
                                        sphere_i_radius, separation[overlap_boolean])

            # Overlap volume
            overlap_volume = spherical_cap_volume(sphere_i_radius, height_i) \
                             + spherical_cap_volume(maximal_spheres_radii[0:out_idx][overlap_boolean], height_maximal)

            # Volume of sphere i
            volume_i = (4./3.)*np.pi*sphere_i_radius**3

            if all(overlap_volume <= frac*volume_i):
                # Sphere i does not overlap by more than x% with any of the other known maximal spheres.
                # Sphere i is therefore a maximal sphere.
                #print('Overlap by less than x%: maximal sphere')
                N_voids += 1
                maximal_spheres_indices.append(i)
                maximal_spheres_coordinates[out_idx,:] = sphere_i_coordinates
                maximal_spheres_radii[out_idx] = sphere_i_radius
                out_idx += 1

        else:
            # No overlap!  Sphere i is a maximal sphere
            #print('No overlap: maximal sphere')
            N_voids += 1
            maximal_spheres_indices.append(i)
            maximal_spheres_coordinates[out_idx,:] = sphere_i_coordinates
            maximal_spheres_radii[out_idx] = sphere_i_radius
            out_idx += 1
        ########################################################################


    # Extract table of maximal spheres
    #maximal_spheres_table = spheres_table[maximal_spheres_indices]
    maximal_spheres_table = Table(maximal_spheres_coordinates[0:out_idx], names=['x','y','z'])
    maximal_spheres_table['radius'] = maximal_spheres_radii[0:out_idx]

    # Add void flag identifier to maximal spheres
    maximal_spheres_table['flag'] = np.arange(N_voids) + 1

    # Convert maximal_spheres_indices to numpy array of type int
    maximal_spheres_indices = np.array(maximal_spheres_indices, dtype=int)
    
    #print("New FIND_MAXIMALS, TIME_2: ", time.time() - time_2)
    
    #print(maximal_spheres_indices_2[0:10])
    #print(maximal_spheres_indices[0:10])
    #print(np.all(maximal_spheres_indices==maximal_spheres_indices_2))

    return maximal_spheres_table, maximal_spheres_indices



################################################################################
################################################################################
################################################################################


def find_holes(spheres_table, maximal_spheres_table, maximal_spheres_indices):
    '''
    ASSIGN SPHERES TO VOIDS
    
    A sphere is part of a void if it overlaps one maximal sphere by at least 50% 
    of the smaller sphere's volume.


    Parameters:
    ===========

    spheres_table : astropy table of length N
        Table of holes found, sorted by radius.  Required columns: radius, x, y, 
        z (in units of Mpc/h)

    maximal_spheres_table : astropy table of length M
        Table of maximal spheres.  Columns: x, y, z, radius (all in units of 
        Mpc/h), flag (unique void identifier)

    maximal_spheres_indices : numpy array of shape (M,)
        Integer numpy array of the indices in spheres_table corresponding to 
        each maximal sphere


    Returns:
    ========

    hole_table : astropy table of length P
        Table of void holes.  Columns are x, y, z, radius (all in units of 
        Mpc/h), flag (void identifier)
    '''


    array = np.array([spheres_table['x'], 
                      spheres_table['y'], 
                      spheres_table['z'], 
                      spheres_table['radius']])
    x_y_z_r_array = array.T
    
    #time_1 = time.time()

    out = find_holes_2(x_y_z_r_array, maximal_spheres_indices, 0.5)
    
    new_holes_index = out[:,0]
    new_flag_col = out[:,1]
    
    #print("NEW FIND HOLES TIME: ", time.time() - time_1)



    #time_2 = time.time()


    # Initialize void flag identifier
    spheres_table['flag'] = -1

    # Number of holes
    N_spheres = len(spheres_table)

    # Initialize index array for holes
    holes_indices = []

    # Number of spheres which are assigned to a void (holes)
    N_holes = 0

    # Coordinates of maximal spheres
    maximal_spheres_coordinates = to_array(maximal_spheres_table)

    # Radii of maximal spheres
    maximal_spheres_radii = np.array(maximal_spheres_table['radius'])

    maximal_indices = np.arange(len(maximal_spheres_indices))
    
    
    all_sphere_coords = to_array(spheres_table)
    all_sphere_radii = spheres_table['radius'].data
    
    
    
    
    maximal_sphere_index = {}
    
    for idx,element in enumerate(maximal_spheres_indices):
        
        maximal_sphere_index[element] = idx
    
    

    for i in range(N_spheres):

        ########################################################################
        # First - check if i is a maximal sphere
        #-----------------------------------------------------------------------
        if i in maximal_sphere_index:
            N_holes += 1
            holes_indices.append(i)
            #spheres_table['flag'][i] = maximal_spheres_table['flag'][maximal_spheres_indices == i]
            spheres_table['flag'][i] = maximal_spheres_table['flag'][maximal_sphere_index[i]]
            #print('sphere i is a maximal sphere')
            continue
        ########################################################################


        # Coordinates of sphere i
        #sphere_i_coordinates = to_vector(spheres_table[i])
        sphere_i_coordinates = all_sphere_coords[i,:]

        # Radius of sphere i
        #sphere_i_radius = spheres_table['radius'][i]
        sphere_i_radius = all_sphere_radii[i]


        ########################################################################
        #
        # COMPARE AGAINST MAXIMAL SPHERES
        #
        ########################################################################

        # Distance between sphere i's center and the centers of the maximal spheres
        separation = np.linalg.norm((maximal_spheres_coordinates - sphere_i_coordinates), axis=1)
        '''
        if any(separation == 0):
            print(i)
            print(np.where(separation==0))
            print(maximal_spheres_coordinates[np.where(separation==0)])
            print(sphere_i_coordinates)
        '''

        ########################################################################
        # Does sphere i live completely inside a maximal sphere?
        #-----------------------------------------------------------------------
        if any((maximal_spheres_radii - sphere_i_radius) >= separation):
            # Sphere i is completely inside another sphere --- sphere i should not be saved
            #print('Sphere completely inside another sphere', sphere_i_radius)
            continue
        ########################################################################


        ########################################################################
        # Does sphere i overlap by more than 50% with a maximal sphere?
        #-----------------------------------------------------------------------
        # First - determine which maximal spheres sphere i overlaps with
        overlap_boolean =  separation <= (sphere_i_radius + maximal_spheres_radii)
        
        if any(overlap_boolean):
            # Sphere i overlaps at least one maximal sphere by some amount.
            # Check to see by how much.

            maximal_overlap_indices = maximal_indices[overlap_boolean]

            # Heights of the spherical caps
            height_i = cap_height(sphere_i_radius, 
                                  maximal_spheres_radii[overlap_boolean], 
                                  separation[overlap_boolean])
            
            height_maximal = cap_height(maximal_spheres_radii[overlap_boolean], 
                                        sphere_i_radius, 
                                        separation[overlap_boolean])

            # Overlap volume
            overlap_volume = spherical_cap_volume(sphere_i_radius, height_i) \
                             + spherical_cap_volume(maximal_spheres_radii[overlap_boolean], height_maximal)

            # Volume of sphere i
            volume_i = (4./3.)*np.pi*sphere_i_radius**3

            # Does sphere i overlap by at least 50% of its volume with a maximal sphere?
            overlap2_boolean = overlap_volume > 0.5*volume_i
            
            if sum(overlap2_boolean) == 1:
                # Sphere i overlaps by more than 50% with one maximal sphere
                # Sphere i is therefore a hole in that void

                #print('Hole inside void', sphere_i_radius)

                N_holes += 1

                holes_indices.append(i)

                spheres_table['flag'][i] = maximal_spheres_table['flag'][maximal_overlap_indices[overlap2_boolean]]
            '''
            else:
                print('Hole overlaps void, but not part of one', sphere_i_radius)
            '''
        ########################################################################


    ############################################################################
    #
    #   OUTPUT TABLES
    #
    ############################################################################

    holes_table = spheres_table[holes_indices]
    
    #print("OLD FIND HOLES TIME: ", time.time() - time_2)
    #print(new_holes_index[0:10])
    #print(np.array(holes_indices)[0:10])
    
    matches = new_holes_index==np.array(holes_indices)
    #print(np.array(holes_indices).shape[0])
    #print(new_holes_index.shape[0])
    #print(np.sum(matches))
    
    n_match = 0
    for elem in np.array(holes_indices):
        
        if elem in new_holes_index:
            n_match += 1
            
    #print("N_match: ", n_match)
    
    
    #print(match_pct)
    #print(np.all(new_holes_index==np.array(holes_indices)))
    

    return holes_table



################################################################################
#
#   TEST SCRIPT
#
################################################################################


if __name__ == '__main__':

    #import pickle
    from astropy.table import Table
    import time

    from voidfinder_functions import save_maximals

    '''
    in_file = open('potential_voids_list.txt', 'rb')
    potential_voids_table = pickle.load(in_file)
    in_file.close()

    potential_voids_table.reverse()
    '''
    survey_name = 'DESI_void_flatmock_1_'
    out_directory = 'DESI/mocks/'
    galaxies_filename = 'void_flatmock_1.fits'

    potential_voids_table = Table.read(survey_name + 'potential_voids_list.txt', format = 'ascii.commented_header')

    combine_start = time.time()

    maximal_spheres_table, myvoids_table = combine_holes(potential_voids_table, 0.1)

    print('Number of unique voids is', len(maximal_spheres_table))
    print('Number of void holes is', len(myvoids_table))

    # Save list of all void holes
    out2_filename = out_directory + galaxies_filename[:-5] + '_holes.txt'
    myvoids_table.write(out2_filename, format='ascii.commented_header', overwrite=True)

    combine_end = time.time()

    print('Time to combine holes into voids =', combine_end-combine_start,flush=True)

    # Save list of maximal spheres
    out1_filename = out_directory + galaxies_filename[:-5] + '_maximal.txt'
    save_maximals(maximal_spheres_table, out1_filename)
    
    '''
    fake_x = [0, 1, 0, 30, 55, -18, 72, 0]
    fake_y = [0, 0, -18, 0, 0, 0, 0, 100]
    fake_radius = [20, 11, 15, 16, 18, 9, 8, 7]
    fake_table = Table([fake_x, fake_y, fake_radius], names=('x','y','radius'))
    fake_table['z'] = 0
    fake_table.sort('radius')
    fake_table.reverse()

    maximal_spheres_table, myvoids_table = combine_holes(fake_table, 0.1)

    maximal_spheres_table.pprint()
    myvoids_table.pprint()
    '''

