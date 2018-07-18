#imports 

import numpy as np
from voidfinder_functions import in_mask
from hole_combine import spherical_cap_volume, combine_holes


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

    xpos = max_range_check(Table(hole_table), 'x', '+', survey_mask, r_limits)
    #print(xpos[:50])
    xneg = max_range_check(Table(hole_table), 'x', '-', survey_mask, r_limits)
    #print(xneg[:10])

    ypos = max_range_check(Table(hole_table), 'y', '+', survey_mask, r_limits)
    #print(ypos[:10])
    yneg = max_range_check(Table(hole_table), 'y', '-', survey_mask, r_limits)
    #print(yneg[:10])

    zpos = max_range_check(Table(hole_table), 'z', '+', survey_mask, r_limits)
    #print(zpos[:10])
    zneg = max_range_check(Table(hole_table), 'z', '-', survey_mask, r_limits)
    #print(zneg[:10])

    comb_bool = np.logical_and.reduce((xpos, xneg, ypos, yneg, zpos, zneg))

    false_indices = np.where(comb_bool == False)

    out_spheres_indices = []

    out_volumes = []

    
    for i in false_indices[0]:

        not_removed = True

        coord = hole_table[i]
        
        #print('___________________________')

        if xpos[i] == False:

            cap_volume, sphere_volume = check_coordinates(Table(coord), 'x', '+', survey_mask, r_limits)

            #print('xpos')
            #out_volumes.append(cap_volume[0]/sphere_volume[0])

            if cap_volume[0] > 0.1*sphere_volume[0]:
                out_spheres_indices.append(i)
                #print('out of mask')
                not_removed = False


        elif xneg[i] == False and not_removed:
            
            cap_volume, sphere_volume = check_coordinates(Table(coord), 'x', '-', survey_mask, r_limits)
            #print('xneg')

            #out_volumes.append(cap_volume[0]/sphere_volume[0])

            if cap_volume[0] > 0.1*sphere_volume[0]:
                out_spheres_indices.append(i)
                #print('out of mask')
                not_removed = False


        if ypos[i] == False and not_removed:
            
            cap_volume, sphere_volume = check_coordinates(Table(coord), 'y', '+', survey_mask, r_limits)
            #print('ypos')
            #out_volumes.append(cap_volume[0]/sphere_volume[0])

            if cap_volume[0] > 0.1*sphere_volume[0]:
                out_spheres_indices.append(i)
                #print('out of mask')
                not_removed = False


        elif yneg[i] == False and not_removed:
            
            cap_volume, sphere_volume = check_coordinates(Table(coord), 'y', '-', survey_mask, r_limits)
            #print('yneg')

            #out_volumes.append(cap_volume[0]/sphere_volume[0])

            if cap_volume[0] > 0.1*sphere_volume[0]:
                out_spheres_indices.append(i)
                #print('out of mask')
                not_removed = False


        if zpos[i] == False and not_removed:
            
            cap_volume, sphere_volume = check_coordinates(Table(coord), 'z', '+', survey_mask, r_limits)

            #print('zpos')
            #out_volumes.append(cap_volume[0]/sphere_volume[0])

            if cap_volume[0] > 0.1*sphere_volume[0]:
                out_spheres_indices.append(i)
                #print('out of mask')
                not_removed = False

        elif zneg[i] == False and not_removed:
            
            cap_volume, sphere_volume = check_coordinates(Table(coord), 'z', '-', survey_mask, r_limits)

            #print('zneg')
            #out_volumes.append(cap_volume[0]/sphere_volume[0])

            if cap_volume[0] > 0.1*sphere_volume[0]:
                out_spheres_indices.append(i)
                #print('out of mask')
                not_removed = False
    
    print(len(out_spheres_indices))

    out_spheres_indices = np.unique(out_spheres_indices)

    print(len(out_spheres_indices))

    hole_table.remove_rows(out_spheres_indices)

    return hole_table, out_volumes

if __name__ == '__main__':

    from astropy.table import Table

    from voidfinder_functions import save_maximals

    maskra = 360
    maskdec = 180
    min_dist = 0.
    max_dist = 300.
    dec_offset = -90
    
    maskfile = Table.read('cbpdr7mask.dat', format='ascii.commented_header')
    mask = np.zeros((maskra, maskdec))
    mask[maskfile['ra'].astype(int), maskfile['dec'].astype(int) - dec_offset] = 1

    holes_table = Table.read('potential_voids_list.txt', format='ascii.commented_header')
    potential_voids_table, volumes_list = volume_cut(holes_table, mask, [min_dist, max_dist])

    maximal_spheres_table, myvoids_table = combine_holes(potential_voids_table, 0.1)

    print('Number of unique voids is', len(maximal_spheres_table))
    print('Number of void holes is', len(myvoids_table))

    save_maximals(maximal_spheres_table, 'maximal_spheres_test.txt')



    '''import numpy as np
    import matplotlib.mlab as mlab
    import matplotlib.pyplot as plt
 
    num_bins = 20
    n, bins, patches = plt.hist(volumes_list, num_bins, facecolor='mediumvioletred', histtype='stepfilled')
    plt.xlabel('Percent of Volume Outside of Mask')
    plt.ylabel('Number of Holes')
    plt.title('Histogram of Volume Percentages of Holes Around Border of Survey')
    plt.show()
    '''
   

