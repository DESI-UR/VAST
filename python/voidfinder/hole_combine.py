from astropy.table import Table

import numpy as np

from .table_functions import to_array,to_vector

#import warnings
#warnings.simplefilter('error')


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


    ############################################################################
    # Remove duplicate holes
    #---------------------------------------------------------------------------
    unique_spheres_table = remove_duplicates(spheres_table)
    ############################################################################



    ############################################################################
    # Maximal spheres
    #---------------------------------------------------------------------------
    maximal_spheres_table, maximal_spheres_indices = find_maximals(unique_spheres_table, frac)


    # Array of coordinates for maximal spheres
    maximal_spheres_coordinates = to_array(maximal_spheres_table)

    # Array of radii for maximal spheres
    maximal_spheres_radii = np.array(maximal_spheres_table['radius'])

    print('Maximal spheres identified')
    ############################################################################



    ############################################################################
    # Assign spheres to voids
    #---------------------------------------------------------------------------
    holes_table = find_holes(unique_spheres_table, maximal_spheres_table, maximal_spheres_indices)
    ############################################################################


    return maximal_spheres_table, holes_table



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


    # At least the first sphere will remain
    unique_spheres_indices = [0]


    for i in range(len(spheres_table))[1:]:

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
        '''
        else:
            # Sphere i is the same as the last unique sphere
            print('Sphere i is not a unique sphere')
        '''


    # Build unique_spheres_table
    unique_spheres_table = spheres_table[unique_spheres_indices]

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


    large_spheres_indices = np.nonzero(spheres_table['radius'] > 10)

    # The largest hole is a void
    N_voids = 1
    maximal_spheres_indices = [0]

    for i in large_spheres_indices[0][1:]:

        # Coordinates of sphere i
        sphere_i_coordinates = to_vector(spheres_table[i])

        # Radius of sphere i
        sphere_i_radius = spheres_table['radius'][i]

        ########################################################################
        #
        # COMPARE AGAINST MAXIMAL SPHERES
        #
        ########################################################################

        # Array of coordinates for previously identified maximal spheres
        maximal_spheres_coordinates = to_array(spheres_table[maximal_spheres_indices])

        # Array of radii for previously identified maximal spheres
        maximal_spheres_radii = np.array(spheres_table['radius'][maximal_spheres_indices])

        # Distance between sphere i's center and the centers of the other maximal spheres
        separation = np.linalg.norm((maximal_spheres_coordinates - sphere_i_coordinates), axis=1)

        ########################################################################
        # Does sphere i live completely inside another maximal sphere?
        #-----------------------------------------------------------------------
        if any((maximal_spheres_radii - sphere_i_radius) >= separation):
            # Sphere i is completely inside another sphere --- sphere i is not a maximal sphere
            #print('Sphere i is completely inside another sphere')
            continue
        ########################################################################


        ########################################################################
        # Does sphere i overlap by less than x% with another maximal sphere?
        #-----------------------------------------------------------------------
        # First - determine which maximal spheres overlap with sphere i
        overlap_boolean =  separation <= (sphere_i_radius + maximal_spheres_radii)

        if any(overlap_boolean):
            # Sphere i overlaps at least one maximal sphere by some amount.
            # Check to see by how much.

            # Heights of the spherical caps
            height_i = cap_height(sphere_i_radius, 
                                  maximal_spheres_radii[overlap_boolean], 
                                  separation[overlap_boolean])

            height_maximal = cap_height(maximal_spheres_radii[overlap_boolean], 
                                        sphere_i_radius, separation[overlap_boolean])

            # Overlap volume
            overlap_volume = spherical_cap_volume(sphere_i_radius, height_i) \
                             + spherical_cap_volume(maximal_spheres_radii[overlap_boolean], height_maximal)

            # Volume of sphere i
            volume_i = (4./3.)*np.pi*sphere_i_radius**3

            if all(overlap_volume <= frac*volume_i):
                # Sphere i does not overlap by more than x% with any of the other known maximal spheres.
                # Sphere i is therefore a maximal sphere.
                #print('Overlap by less than x%: maximal sphere')
                N_voids += 1
                maximal_spheres_indices.append(i)

        else:
            # No overlap!  Sphere i is a maximal sphere
            #print('No overlap: maximal sphere')
            N_voids += 1
            maximal_spheres_indices.append(i)
        ########################################################################


    # Extract table of maximal spheres
    maximal_spheres_table = spheres_table[maximal_spheres_indices]

    # Add void flag identifier to maximal spheres
    maximal_spheres_table['flag'] = np.arange(N_voids) + 1

    # Convert maximal_spheres_indices to numpy array of type int
    maximal_spheres_indices = np.array(maximal_spheres_indices, dtype=int)

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

    for i in range(N_spheres):

        ########################################################################
        # First - check if i is a maximal sphere
        #-----------------------------------------------------------------------
        if i in maximal_spheres_indices:
            N_holes += 1
            holes_indices.append(i)
            spheres_table['flag'][i] = maximal_spheres_table['flag'][maximal_spheres_indices == i]
            #print('sphere i is a maximal sphere')
            continue
        ########################################################################


        # Coordinates of sphere i
        sphere_i_coordinates = to_vector(spheres_table[i])

        # Radius of sphere i
        sphere_i_radius = spheres_table['radius'][i]


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

