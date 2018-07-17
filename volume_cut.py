#imports 

import numpy as np
from voidfinder_functions import in_mask
from hole_combine import spherical_cap_volume
from astropy.table import Table


# function to find which spheres stick out of the mask
def max_range_check(spheres_table, direction, sign, survey_mask, r_limits):

    if sign == '+':
       spheres_table[direction] += spheres_table['radius']
    else:
       spheres_table[direction] -= spheres_table['radius']

    boolean = in_mask(spheres_table, survey_mask, r_limits)

    return boolean


def check_coordinates(coord, direction, sign, survey_mask, r_limits):

    dr = 0
    check_coord = coord
    mask_check = True

    while dr < coord['radius'] and mask_check:

        dr += 1

        if sign == '+':
            check_coord[direction] = coord[direction] + dr
        else:
            check_coord[direction] = coord[direction] - dr 

        mask_check = in_mask(check_coord, survey_mask, r_limits)

    height_i = check_coord['radius'] - dr
    cap_volume_i = spherical_cap_volume(check_coord['radius'], height_i)
    sphere_volume = np.pi*(4/3)*(check_coord['radius']**3)
    
    return cap_volume_i, sphere_volume



def volume_cut(hole_table, survey_mask, r_limits):
    '''# so these can be used in other subfunctions    
    survey_mask = survey_mask
    r_limits = r_limits'''

    xpos = max_range_check(Table(hole_table), 'x', '+', survey_mask, r_limits)
    xneg = max_range_check(Table(hole_table), 'x', '-', survey_mask, r_limits)

    ypos = max_range_check(Table(hole_table), 'y', '+', survey_mask, r_limits)
    yneg = max_range_check(Table(hole_table), 'y', '-', survey_mask, r_limits)

    zpos = max_range_check(Table(hole_table), 'z', '+', survey_mask, r_limits)
    zneg = max_range_check(Table(hole_table), 'z', '-', survey_mask, r_limits)

    comb_bool = np.logical_and.reduce((xpos, xneg, ypos, yneg, zpos, zneg))

    false_indices = np.where(comb_bool == False)

    out_spheres_indices = []

    for i in false_indices[0]:

        coord = hole_table[i]

        not_removed = True

        # Check x-direction 

        if not xpos[i]:
            cap_volume, sphere_volume = check_coordinates(Table(coord), 'x', '+', survey_mask, r_limits)
            if cap_volume > 0.1*sphere_volume:
                #MAKE NOTE TO TAKE SPHERE OUT?
                out_spheres_indices.append(i)
                not_removed = False

        elif xneg[i] == False and not_removed:
            cap_volume, sphere_volume = check_coordinates(Table(coord), 'x', '-', survey_mask, r_limits)
            if cap_volume > 0.1*sphere_volume:
                #MAKE NOTE TO TAKE SPHERE OUT?
                out_spheres_indices.append(i)
                not_removed = False


        # Check y-direction

        if ypos[i] == False and not_removed:
            cap_volume, sphere_volume = check_coordinates(Table(coord), 'y', '+', survey_mask, r_limits)
            if cap_volume > 0.1*sphere_volume:
                #MAKE NOTE TO TAKE SPHERE OUT?
                out_spheres_indices.append(i)
                not_removed = False


        elif yneg[i] == False and not_removed:
            cap_volume, sphere_volume = check_coordinates(Table(coord), 'y', '-', survey_mask, r_limits)
            if cap_volume > 0.1*sphere_volume:
                #MAKE NOTE TO TAKE SPHERE OUT?
                out_spheres_indices.append(i)
                not_removed = False


        # Check z-direction

        if zpos[i] == False and not_removed:
            cap_volume, sphere_volume = check_coordinates(Table(coord), 'z', '+', survey_mask, r_limits)
            if cap_volume > 0.1*sphere_volume:
                #MAKE NOTE TO TAKE SPHERE OUT?
                out_spheres_indices.append(i)
                not_removed = False

        elif zneg[i] == False and not_removed:
            cap_volume, sphere_volume = check_coordinates(Table(coord), 'z', '-', survey_mask, r_limits)
            if cap_volume > 0.1*sphere_volume:
                #MAKE NOTE TO TAKE SPHERE OUT?
                out_spheres_indices.append(i)
                not_removed = False

    hole_table.remove_rows(out_spheres_indices)

    return hole_table

if __name__ == '__main__':

    from astropy.table import Table

    from voidfinder_functions import save_maximals
    from hole_combine import combine_holes

    maskra = 360
    maskdec = 180
    min_dist = 0.
    max_dist = 300.
    dec_offset = -90
    
    maskfile = Table.read('SDSSdr7/cbpdr7mask.dat', format='ascii.commented_header')
    mask = np.zeros((maskra, maskdec))
    mask[maskfile['ra'].astype(int), maskfile['dec'].astype(int) - dec_offset] = 1

    holes_table = Table.read('potential_voids_list.txt', format='ascii.commented_header')
    potential_voids_table = volume_cut(holes_table, mask, [min_dist, max_dist])

    maximal_spheres_table, myvoids_table = combine_holes(potential_voids_table, 0.1)

    print('Number of unique voids is', len(maximal_spheres_table))
    print('Number of void holes is', len(myvoids_table))

    #save_maximals(maximal_spheres_table, 'maximal_spheres_test.txt')
    
   

