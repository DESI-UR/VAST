#imports 

import numpy as np

from .voidfinder_functions import in_mask, not_in_mask
from .hole_combine import spherical_cap_volume
from astropy.table import Table
from ._voidfinder_cython_find_next import not_in_mask as nim_cython


# function to find which spheres stick out of the mask
def max_range_check(spheres_table, direction, sign, survey_mask, mask_resolution, r_limits):
    '''
    Given the list of potential hole locations and their radii in spheres_table,
    and an axes x,y,z and direction +/-, add the radii of each hole to the hole
    location and check if that location is within the mask.
    
    Returns a boolean array of length N where True indicates the location is valid.
    '''

    #print("Max Range Check", direction, sign, "hole_table ID: ", id(spheres_table))
    #print(spheres_table['x'][0])



    if sign == '+':
       spheres_table[direction] += spheres_table['radius']
    else:
       spheres_table[direction] -= spheres_table['radius']
       
       
       
    #print(spheres_table['x'][0])
    #print(spheres_table)

    boolean = in_mask(spheres_table, survey_mask, mask_resolution, r_limits)

    return boolean


def check_coordinates(coord, direction, sign, survey_mask, mask_resolution, r_limits):

    dr = 0
    check_coord = coord
    #mask_check = True
    mask_check2 = False
    #mask_check3 = False
    
    #print(id(check_coord), id(coord))
    
    np_check_coord = np.empty((1,3), dtype=np.float64)
    np_check_coord[0,0] = coord['x']
    np_check_coord[0,1] = coord['y']
    np_check_coord[0,2] = coord['z']
    
    if direction == 'x':
        np_dir = 0
    elif direction == 'y':
        np_dir = 1
    elif direction == 'z':
        np_dir = 2
    
    #out_log = open("VF_DEBUG_volume_cut.txt", 'a')

    #while dr < coord['radius'] and mask_check:
    while dr < coord['radius'] and not mask_check2:

        dr += 1

        if sign == '+':
        #    check_coord[direction] = coord[direction] + dr
            np_check_coord[0,np_dir] = np_check_coord[0,np_dir] + dr
        else:
        #    check_coord[direction] = coord[direction] - dr
            np_check_coord[0,np_dir] = np_check_coord[0,np_dir] - dr

        #mask_check = in_mask(check_coord, survey_mask, mask_resolution, r_limits)
        
        mask_check2 = nim_cython(np_check_coord, survey_mask, mask_resolution, r_limits[0], r_limits[1])
        
        #mask_check3 = not_in_mask(np_check_coord, survey_mask, mask_resolution, r_limits[0], r_limits[1])
        
        #if mask_check == mask_check3: # or \
        #   mask_check != mask_check3 or \
        #if mask_check2 != mask_check3:
            #out_log.write(str(check_coord)+"\n")
            #out_log.write(str(np_check_coord)+","+str(mask_check)+","+str(mask_check2)+","+str(mask_check3)+"\n")
            
    #out_log.close()
        

    height_i = check_coord['radius'] - dr
    cap_volume_i = spherical_cap_volume(check_coord['radius'], height_i)
    sphere_volume = np.pi*(4/3)*(check_coord['radius']**3)
    
    return cap_volume_i, sphere_volume



def volume_cut(hole_table, survey_mask, mask_resolution, r_limits):
    
    #print("Vol cut hole_table ID: ", id(hole_table))
    #print(hole_table['x'][0])
    
    
    
    # xpos, xneg, etc are True when the hole center + hole_radius in that direction
    # is within the mask
    xpos = max_range_check(Table(hole_table), 'x', '+', survey_mask, mask_resolution, r_limits)
    xneg = max_range_check(Table(hole_table), 'x', '-', survey_mask, mask_resolution, r_limits)

    ypos = max_range_check(Table(hole_table), 'y', '+', survey_mask, mask_resolution, r_limits)
    yneg = max_range_check(Table(hole_table), 'y', '-', survey_mask, mask_resolution, r_limits)

    zpos = max_range_check(Table(hole_table), 'z', '+', survey_mask, mask_resolution, r_limits)
    zneg = max_range_check(Table(hole_table), 'z', '-', survey_mask, mask_resolution, r_limits)


    comb_bool = np.logical_and.reduce((xpos, xneg, ypos, yneg, zpos, zneg))
    
    
    
    #print("Comb bool: ", np.sum(comb_bool))
    
    

    false_indices = np.where(comb_bool == False)

    out_spheres_indices = []

    for i in false_indices[0]:

        not_removed = True

        coord = hole_table[i]

        # Check x-direction 

        if not xpos[i]:

            cap_volume, sphere_volume = check_coordinates(Table(coord), 'x', '+', survey_mask, mask_resolution, r_limits)

            if cap_volume > 0.1*sphere_volume:
                out_spheres_indices.append(i)
                not_removed = False

        elif xneg[i] == False and not_removed:

            cap_volume, sphere_volume = check_coordinates(Table(coord), 'x', '-', survey_mask, mask_resolution, r_limits)

            if cap_volume > 0.1*sphere_volume:
                out_spheres_indices.append(i)
                not_removed = False

        # Check y-direction

        if ypos[i] == False and not_removed:

            cap_volume, sphere_volume = check_coordinates(Table(coord), 'y', '+', survey_mask, mask_resolution, r_limits)

            if cap_volume > 0.1*sphere_volume:
                out_spheres_indices.append(i)
                not_removed = False


        elif yneg[i] == False and not_removed:

            cap_volume, sphere_volume = check_coordinates(Table(coord), 'y', '-', survey_mask, mask_resolution, r_limits)

            if cap_volume > 0.1*sphere_volume:
                out_spheres_indices.append(i)
                not_removed = False


        # Check z-direction

        if zpos[i] == False and not_removed:
            cap_volume, sphere_volume = check_coordinates(Table(coord), 'z', '+', survey_mask, mask_resolution, r_limits)

            if cap_volume > 0.1*sphere_volume:
                out_spheres_indices.append(i)
                not_removed = False

        elif zneg[i] == False and not_removed:

            cap_volume, sphere_volume = check_coordinates(Table(coord), 'z', '-', survey_mask, mask_resolution, r_limits)

            if cap_volume > 0.1*sphere_volume:
                out_spheres_indices.append(i)
                not_removed = False
    
    out_spheres_indices = np.unique(out_spheres_indices)


    if len(out_spheres_indices) > 0:
    
        hole_table.remove_rows(out_spheres_indices)

    return hole_table


def check_hole_bounds(x_y_z_r_array, 
                             mask, 
                             mask_resolution, 
                             r_limits,
                             cut_pct=0.1,
                             num_surf_pts=14,
                             num_cpus=1):
    """
    Description
    ===========
    
    Remove holes from the output of _hole_finder() whose volume falls outside of the mask
    by X % or more.  
    
    This is accomplished by a 2-phase approach, first, N points are
    distributed on the surface of each sphere, and those N points are checked against
    the mask.  If any of those N points fall outside the mask, the percentage of the
    volume of the sphere which falls outside the mask is calculated by using a
    monte-carlo-esque method whereby the hole in question is filled with points
    corresponding to some minimum density, and each of those points is checked.
    The percentage of volume outside the mask is then approximated as the percentage
    of those points which fall outside the mask.
    
    Parameters
    ==========
    
    x_y_z_r_array : numpy.ndarray of shape (N,4)
        x,y,z locations of the holes, and radius, in that order
        
    mask : numpy.ndarray of shape (K,L) dtype np.uint8
        the mask used, mask[ra_integer,dec_integer] returns True if that ra,dec position is
        within the survey, and false if it is not.  Note ra,dec must be converted into integer
        values depending on the mask_resolution.  For mask_resolution of 1, ra is in [0,359]
        and dec in [-90,90], for mask_resolution of 2, ra is in [0,719], dec in [-180,180] etc.
        
    mask_resolution : int
        value of 1 indicates each entry in the mask accounts for 1 degree, value of 2
        means half-degree, 4 means quarter-degree increments, etc
        
    r_limits : 2-tuple (min_r, max_r)
        min and max radius limits of the survey
        
    cut_pct : float in [0,1)
        if this fraction of a hole volume overlaps with the mask, discard that hole
        
    num_surf_pts : int
        distribute this many points on the surface of each sphere and check them against
        the mask before doing the monte-carlo volume calculation.
        
    num_cpus : int
        number of processes to use
        
        
    Returns
    =======
    
    valid_index : numpy.ndarray shape (N,)
        boolean array of length corresponding to input x_y_z_r_array
        True if hole is within bounds, False is hole falls outside
        the mask too far based on the cut_pct criteria
    """

   

    if num_cpus == 1:
        
        oob_cut_single()
        
    else:
        oob_cut_multi()
        
        
def oob_cut_single(x_y_z_r_array, 
                     mask, 
                     mask_resolution, 
                     r_limits,
                     cut_pct=0.1,
                     num_surf_pts=14):


    valid_index = np.ones(x_y_z_r_array.shape[0], dtype=np.bool)

    












