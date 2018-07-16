#6 calls to in mask (fix the coordinates to actually function, these are just so you know what's happening)
#fix all the in mask inputs actually

import numpy as np
from voidfinder_functions import in_mask


def max_range_check(spheres_table, direction, sign):

    if sign == '+':
       spheres_table[direction] += spheres_table['radius']
   else:
       spheres_table[direction] -= spheres_table['radius']

   boolean = in_mask(table, survey_mask, r_limits)

   return boolean






def volume_cut(hole_table,survey_mask,r_limits):

    xpos = max_range_check(hole_table, 'x', '+')
    xneg = max_range_check(hole_table, 'x', '-')

    ypos = max_range_check(hole_table, 'y', '+')
    yneg = max_range_check(hole_table, 'y', '-')

    zpos = max_range_check(hole_table, 'z', '+')
    zneg = max_range_check(hole_table, 'z', '-')

    '''
    xpos = in_mask(X+R, survey_mask, r_limits)
    xneg = in_mask(X-R, survey_mask, r_limits)

    ypos = in_mask(Y+R, survey_mask, r_limits)
    yneg = in_mask(Y-R, survey_mask, r_limits)

    zpos = in_mask(Z+R, survey_mask, r_limits)
    zneg = in_mask(Z-R, survey_mask, r_limits)
    '''

    comb_bool = np.logical_and.reduce(xpos, xneg, ypos, yneg, zpos, zneg)

    false_indices = np.where(comb_bool == False)

    out_spheres_indices = []

    not_removed = True

    for i in false_indices:
        coord = hole_table[i]
        if xpos[i] == False:
            dr = 0
            check_coord = coord
            mask_check = True
            while dr < coord['radius'] and mask_check:
                dr += 1
                check_coord['x'] = coord['x'] + dr
                mask_check = in_mask(check_coord, survey_mask, r_limits)
            height_i = check_coord['radius'] - dr
            cap_volume_i = spherical_cap_volume(check_coord['radius'], height_i)
            sphere_volume = np.pi*(4/3)*(check_coord['radius']**3)
            #in_volume = sphere_volume-cap_volume
            if cap_volume > 0.1*sphere_volume:
                #MAKE NOTE TO TAKE SPHERE OUT?
                out_spheres_indices.append(i)
                not_removed = False


        elif xneg[i] == False and not_removed:
            dr = 0
            check_coord = coord
            mask_check = True
            while dr < coord['radius'] and mask_check:
                dr += 1
                check_coord = coord['x']-dr
                mask_check = in_mask(check_coord, survey_mask, r_limits)
            height_i = check_coord['radius']-dr
            cap_volume_i = spherical_cap_volume(check_coord['radius'], height_i)
            sphere_volume = np.pi*(4/3)*(check_coord['radius']**3)
            in_volume = sphere_volume-cap_volume
            if in_volume/sphere_volume <= 0.9:
                #MAKE NOTE TO TAKE SPHERE OUT?
                out_spheres_indices.append(i)


        if ypos[i] == False:
            dr = 0
            check_coord = coord
            mask_check = True
            while dr < coord['radius'] and mask_check:
                dr += 1
                check_coord = coord['y']+dr
                mask_check = in_mask(check_coord, survey_mask, r_limits)
            height_i = check_coord['radius']-dr
            cap_volume_i = spherical_cap_volume(check_coord['radius'], height_i)
            sphere_volume = np.pi*(4/3)*(check_coord['radius']**3)
            in_volume = sphere_volume-cap_volume
            if in_volume/sphere_volume <= 0.9:
                #MAKE NOTE TO TAKE SPHERE OUT?
                out_spheres_indices.append(i)


        if yneg[i] == False:
            dr = 0
            check_coord = coord
            mask_check = True
            while dr < coord['radius'] and mask_check:
                dr += 1
                check_coord = coord['y']-dr
                mask_check = in_mask(check_coord, survey_mask, r_limits)
            height_i = check_coord['radius']-dr
            cap_volume_i = spherical_cap_volume(check_coord['radius'], height_i)
            sphere_volume = np.pi*(4/3)*(check_coord['radius']**3)
            in_volume = sphere_volume-cap_volume
            if in_volume/sphere_volume <= 0.9:
                #MAKE NOTE TO TAKE SPHERE OUT?
                out_spheres_indices.append(i)


        if zpos[i] == False:
            dr = 0
            check_coord = coord
            mask_check = True
            while dr < coord['radius'] and mask_check:
                dr += 1
                check_coord = coord['z']+dr
                mask_check = in_mask(check_coord, survey_mask, r_limits)
            height_i = check_coord['radius']-dr
            cap_volume_i = spherical_cap_volume(check_coord['radius'], height_i)
            sphere_volume = np.pi*(4/3)*(check_coord['radius']**3)
            in_volume = sphere_volume-cap_volume
            if in_volume/sphere_volume <= 0.9:
                #MAKE NOTE TO TAKE SPHERE OUT?
                out_spheres_indices.append(i)


        if zneg[i] == False:
            dr = 0
            check_coord = coord
            mask_check = True
            while dr < coord['radius'] and mask_check:
                dr += 1
                check_coord = coord['z']-dr
                mask_check = in_mask(check_coord, survey_mask, r_limits)
            height_i = check_coord['radius']-dr
            cap_volume_i = spherical_cap_volume(check_coord['radius'], height_i)
            sphere_volume = np.pi*(4/3)*(check_coord['radius']**3)
            in_volume = sphere_volume-cap_volume
            if in_volume/sphere_volume <= 0.9:
                #MAKE NOTE TO TAKE SPHERE OUT?
                out_spheres_indices.append(i)

    for i in out_spheres_indices:
        hole_table.remove_row(i)
