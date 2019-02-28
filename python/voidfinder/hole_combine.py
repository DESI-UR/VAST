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


def combine_holes(spheres_table, frac):
    '''
    Combines the potential void spheres into voids.

    Assumes that spheres_table is sorted by radius.
    '''

    ############################################################################
    # IDENTIFY MAXIMAL SPHERES
    #
    # We only consider holes with radii greater than 10 Mpc/h as seeds for a 
    # void.  If two holes of this size overlap by more than X% of their volume, 
    # then they are considered part of the same void.  Otherwise, they are 
    # independent voids.  The search runs from the largest hole to the smallest.
    ############################################################################


    large_spheres_indices = np.nonzero(spheres_table['radius'] > 10)

    # The largest hole is a void
    N_voids = 1
    maximal_spheres_indices = [0]

    for i in large_spheres_indices[0][1:]:

        #print('__________________________________________')
        #print('Looking at large sphere', i, 'of', len(maximal_spheres_indices))

        # Coordinates of sphere i
        sphere_i_coordinates = to_vector(spheres_table[i])
        #print(sphere_i_coordinates.shape)

        # Radius of sphere i
        sphere_i_radius = spheres_table['radius'][i]

        #print(sphere_i_coordinates)

        ########################################################################
        #
        # COMPARE AGAINST MAXIMAL SPHERES
        #
        ########################################################################

        # Array of coordinates for previously identified maximal spheres
        maximal_spheres_coordinates = to_array(spheres_table[maximal_spheres_indices])
        #print(maximal_spheres_coordinates)

        # Array of radii for previously identified maximal spheres
        maximal_spheres_radii = np.array(spheres_table['radius'][maximal_spheres_indices])
        #print(maximal_spheres_radii)

        # Distance between sphere i's center and the centers of the other maximal spheres
        separation = np.linalg.norm((maximal_spheres_coordinates - sphere_i_coordinates), axis=1)
        #print(separation)
        #print('max spheres',maximal_spheres_coordinates[0][:5])
        #print('subtraction',(maximal_spheres_coordinates[0] - sphere_i_coordinates)[:5])
        #print(separation.shape)

        ########################################################################
        # Does sphere i live completely inside another maximal sphere?
        ########################################################################

        if any((maximal_spheres_radii - sphere_i_radius) >= separation):
            # Sphere i is completely inside another sphere --- sphere i is not a maximal sphere
            #print('Sphere i is completely inside another sphere')
            continue

        ########################################################################
        # Does sphere i overlap by less than x% with another maximal sphere?
        ########################################################################

        # First - determine which maximal spheres overlap with sphere i
        overlap_boolean =  separation <= (sphere_i_radius + maximal_spheres_radii)
        #print(sphere_i_radius)
        #print(maximal_spheres_radii[:5])
        #print((sphere_i_radius + maximal_spheres_radii)[:5])

        if any(overlap_boolean):
            #print('overlap true: maximal')
            # Sphere i overlaps at least one maximal sphere by some amount.
            # Check to see by how much.

            # Heights of the spherical caps
            height_i = cap_height(sphere_i_radius, maximal_spheres_radii[overlap_boolean], separation[overlap_boolean])

            height_maximal = cap_height(maximal_spheres_radii[overlap_boolean], sphere_i_radius, separation[overlap_boolean])

            # Overlap volume
            overlap_volume = spherical_cap_volume(sphere_i_radius, height_i) + spherical_cap_volume(maximal_spheres_radii[overlap_boolean], height_maximal)
            '''for elem in height_maximal:
                if np.isnan(elem):
                    print('first max height', height_maximal)'''

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


    # Extract table of maximal spheres
    maximal_spheres_table = spheres_table[maximal_spheres_indices]

    # Convert maximal_spheres_indices to numpy array of type int
    maximal_spheres_indices = np.array(maximal_spheres_indices, dtype=int)

    # Add void flag identifier to maximal spheres
    maximal_spheres_table['flag'] = np.arange(N_voids) + 1

    # Array of coordinates for maximal spheres
    maximal_spheres_coordinates = to_array(maximal_spheres_table)

    # Array of radii for maximal spheres
    maximal_spheres_radii = np.array(maximal_spheres_table['radius'])

    print('Maximal spheres identified')


    ############################################################################
    # ASSIGN SPHERES TO VOIDS
    #
    # A sphere is part of a void if it overlaps one maximal sphere by at least 
    # 50% of the smaller sphere's volume.
    ############################################################################


    # Initialize void flag identifier
    spheres_table['flag'] = -1

    # Number of holes
    N_spheres = len(spheres_table)

    # Initialize index array for holes
    holes_indices = []

    # Number of spheres which are assigned to a void (holes)
    N_holes = 0

    maximal_indices = np.arange(N_voids)

    for i in range(N_spheres):

        # First - check if i is a maximal sphere
        if i in maximal_spheres_indices:
            #print('Maximal sphere', spheres_table['radius'][i])
            N_holes += 1
            holes_indices.append(i)
            spheres_table['flag'][i] = maximal_spheres_table['flag'][maximal_spheres_indices == i]
            #print('sphere i is a maximal sphere')
            continue

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
        if any(separation == 0):
            '''print(i)
            print(np.where(separation==0))
            print(maximal_spheres_coordinates[np.where(separation==0)])
            print(sphere_i_coordinates)'''

        ########################################################################
        # Does sphere i live completely inside a maximal sphere?
        ########################################################################

        if any((maximal_spheres_radii - sphere_i_radius) >= separation):
            # Sphere i is completely inside another sphere --- sphere i should not be saved
            #print('Sphere completely inside another sphere', sphere_i_radius)
            continue

        ########################################################################
        # Does sphere i overlap by more than 50% with a maximal sphere?
        ########################################################################

        # First - determine which maximal spheres sphere i overlaps with
        overlap_boolean =  separation <= (sphere_i_radius + maximal_spheres_radii)
        #print('ob', overlap_boolean)
        if any(overlap_boolean):
            # Sphere i overlaps at least one maximal sphere by some amount.
            # Check to see by how much.
            maximal_overlap_indices = maximal_indices[overlap_boolean]
            # Heights of the spherical caps
            height_i = cap_height(sphere_i_radius, maximal_spheres_radii[overlap_boolean], separation[overlap_boolean])
            height_maximal = cap_height(maximal_spheres_radii[overlap_boolean], sphere_i_radius, separation[overlap_boolean])

            # Overlap volume
            overlap_volume = spherical_cap_volume(sphere_i_radius, height_i) + spherical_cap_volume(maximal_spheres_radii[overlap_boolean], height_maximal)
            '''for elem in height_maximal:
                if np.isnan(elem):
                    print('second max height', height_maximal)'''

            # Volume of sphere i
            volume_i = (4./3.)*np.pi*sphere_i_radius**3
            # Does sphere i overlap by at least 50% of its volume with a maximal sphere?
            #print(overlap_volume/volume_i)
            overlap2_boolean = overlap_volume > 0.5*volume_i
            #print(overlap_volume)
            #print('new', overlap_boolean)
            if sum(overlap2_boolean) == 1:
                # Sphere i overlaps by more than 50% with one maximal sphere
                # Sphere i is therefore a hole in that void
                #print('Hole inside void', sphere_i_radius)
                N_holes += 1
                holes_indices.append(i)
                #print(maximal_spheres_table['flag'].shape)
                #print(overlap_boolean.shape)
                spheres_table['flag'][i] = maximal_spheres_table['flag'][maximal_overlap_indices[overlap2_boolean]]
            #else:
                #print('Hole overlaps void, but not part of one', sphere_i_radius)
            


    ############################################################################
    #
    #   OUTPUT TABLES
    #
    ############################################################################

    holes_table = spheres_table[holes_indices]

    return maximal_spheres_table, holes_table


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

