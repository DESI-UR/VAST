#imports 

import numpy as np

from .voidfinder_functions import in_mask
from .hole_combine import spherical_cap_volume
from astropy.table import Table



# function to find which spheres stick out of the mask
def max_range_check(spheres_table, direction, sign, survey_mask, mask_resolution, r_limits):

    if sign == '+':
       spheres_table[direction] += spheres_table['radius']
    else:
       spheres_table[direction] -= spheres_table['radius']

    boolean = in_mask(spheres_table, survey_mask, mask_resolution, r_limits)

    return boolean


def check_coordinates(coord, direction, sign, survey_mask, mask_resolution, r_limits):

    dr = 0
    check_coord = coord
    mask_check = True
    

    while dr < coord['radius'] and mask_check:

        dr += 1

        if sign == '+':
            check_coord[direction] = coord[direction] + dr
        else:
            check_coord[direction] = coord[direction] - dr 

        mask_check = in_mask(check_coord, survey_mask, mask_resolution, r_limits)

    height_i = check_coord['radius'] - dr
    cap_volume_i = spherical_cap_volume(check_coord['radius'], height_i)
    sphere_volume = np.pi*(4/3)*(check_coord['radius']**3)
    
    return cap_volume_i, sphere_volume



def volume_cut(hole_table, survey_mask, mask_resolution, r_limits):

    xpos = max_range_check(Table(hole_table), 'x', '+', survey_mask, mask_resolution, r_limits)
    xneg = max_range_check(Table(hole_table), 'x', '-', survey_mask, mask_resolution, r_limits)

    ypos = max_range_check(Table(hole_table), 'y', '+', survey_mask, mask_resolution, r_limits)
    yneg = max_range_check(Table(hole_table), 'y', '-', survey_mask, mask_resolution, r_limits)

    zpos = max_range_check(Table(hole_table), 'z', '+', survey_mask, mask_resolution, r_limits)
    zneg = max_range_check(Table(hole_table), 'z', '-', survey_mask, mask_resolution, r_limits)


    comb_bool = np.logical_and.reduce((xpos, xneg, ypos, yneg, zpos, zneg))

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

    hole_table.remove_rows(out_spheres_indices)

    return hole_table

   

