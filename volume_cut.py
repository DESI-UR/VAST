#6 calls to in mask (fix the coordinates to actually function, these are just so you know what's happening)
#fix all the in mask inputs actually

import numpy as np

xpos = in_mask(X+R, survey_mask, r_limits)
xneg = in_mask(X-R, survey_mask, r_limits)

ypos = in_mask(Y+R, survey_mask, r_limits)
yneg = in_mask(Y-R, survey_mask, r_limits)

zpos = in_mask(Z+R, survey_mask, r_limits)
zneg = in_mask(Z-R, survey_mask, r_limits)

comb_bool = np.logical_and.reduce(xpos, xneg, ypos, yneg, zpos, zneg)

false_indices = np.where(comb_bool == False)

out_spheres_indices = []

for i in false_indices:
    coord = potential_voids_table(i)
    if xpos[i] == false:
        dr = 0
        check_coord = coord
        mask_check = True
        while dr < R[i] and mask_check:
            dr += 1
            check_coord = coord['x']+dr
            mask_check = in_mask(check_coord, survey_mask, r_limits)
        height_i = R[i]-dr
        cap_volume_i = spherical_cap_volume(R[i], height_i)
        sphere_volume = np.pi*(4/3)*(R[i]**3)
        in_volume = sphere_volume-cap_volume
        if in_volume/sphere_volume <= 0.9:
            #MAKE NOTE TO TAKE SPHERE OUT?
            out_spheres_indices.append(i)


    if xneg[i] == false:
        dr = 0
        check_coord = coord
        mask_check = True
        while dr < R and mask_check:
            dr += 1
            check_coord = coord['x']-dr
            mask_check = in_mask(check_coord, survey_mask, r_limits)
        height_i = R[i]-dr
        cap_volume_i = spherical_cap_volume(R[i], height_i)
        sphere_volume = np.pi*(4/3)*(R[i]**3)
        in_volume = sphere_volume-cap_volume
        if in_volume/sphere_volume <= 0.9:
            #MAKE NOTE TO TAKE SPHERE OUT?
            out_spheres_indices.append(i)


    if ypos[i] == false:
        dr = 0
        check_coord = coord
        mask_check = True
        while dr < R and mask_check:
            dr += 1
            check_coord = coord['y']+dr
            mask_check = in_mask(check_coord, survey_mask, r_limits)
        height_i = R[i]-dr
        cap_volume_i = spherical_cap_volume(R[i], height_i)
        sphere_volume = np.pi*(4/3)*(R[i]**3)
        in_volume = sphere_volume-cap_volume
        if in_volume/sphere_volume <= 0.9:
            #MAKE NOTE TO TAKE SPHERE OUT?
            out_spheres_indices.append(i)


    if yneg[i] == false:
        dr = 0
        check_coord = coord
        mask_check = True
        while dr < R and mask_check:
            dr += 1
            check_coord = coord['y']-dr
            mask_check = in_mask(check_coord, survey_mask, r_limits)
        height_i = R[i]-dr
        cap_volume_i = spherical_cap_volume(R[i], height_i)
        sphere_volume = np.pi*(4/3)*(R[i]**3)
        in_volume = sphere_volume-cap_volume
        if in_volume/sphere_volume <= 0.9:
            #MAKE NOTE TO TAKE SPHERE OUT?
            out_spheres_indices.append(i)


    if zpos[i] == false:
        dr = 0
        check_coord = coord
        mask_check = True
        while dr < R and mask_check:
            dr += 1
            check_coord = coord['z']+dr
            mask_check = in_mask(check_coord, survey_mask, r_limits)
        height_i = R[i]-dr
        cap_volume_i = spherical_cap_volume(R[i], height_i)
        sphere_volume = np.pi*(4/3)*(R[i]**3)
        in_volume = sphere_volume-cap_volume
        if in_volume/sphere_volume <= 0.9:
            #MAKE NOTE TO TAKE SPHERE OUT?
            out_spheres_indices.append(i)


    if zneg[i] == false:
        dr = 0
        check_coord = coord
        mask_check = True
        while dr < R and mask_check:
            dr += 1
            check_coord = coord['z']-dr
            mask_check = in_mask(check_coord, survey_mask, r_limits)
        height_i = R[i]-dr
        cap_volume_i = spherical_cap_volume(R[i], height_i)
        sphere_volume = np.pi*(4/3)*(R[i]**3)
        in_volume = sphere_volume-cap_volume
        if in_volume/sphere_volume <= 0.9:
            #MAKE NOTE TO TAKE SPHERE OUT?
            out_spheres_indices.append(i)

for i in out_spheres_indices:
    ORIGINALSPHERETABLE.remove_row(i)
