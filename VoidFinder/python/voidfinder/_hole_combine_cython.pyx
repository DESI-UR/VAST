
from __future__ import print_function

cimport cython

import numpy as np

cimport numpy as np

np.import_array()  # required in order to use C-API


from typedefs cimport DTYPE_CP128_t, \
                      DTYPE_CP64_t, \
                      DTYPE_F64_t, \
                      DTYPE_F32_t, \
                      DTYPE_B_t, \
                      ITYPE_t, \
                      DTYPE_INT32_t, \
                      DTYPE_INT64_t, \
                      DTYPE_INT8_t

from numpy.math cimport NAN, INFINITY

from libc.math cimport fabs, sqrt, asin, atan, ceil#, exp, pow, cos, sin, asin




@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cpdef np.ndarray remove_duplicates_2(DTYPE_F64_t[:,:] x_y_z_r_array,
                                     DTYPE_F64_t tolerance):
    """
    Description
    ===========
    
    This function is not intuitive - it serves the unique purpose of removing
    only a small subset of duplicate hole candidates.  It is assumes the
    x_y_z_r_array is sorted such that the largest hole is at index 0, and we
    compare pairwise adjacent holes to remove duplicates only from that 
    selection.
    
    """
    
    cdef ITYPE_t num_holes = x_y_z_r_array.shape[0]
    
    cdef ITYPE_t idx, last_idx
    
    cdef DTYPE_F64_t[3] diff
    cdef DTYPE_F64_t curr_radius, last_radius, separation
    
    out_index = np.zeros(num_holes, dtype=np.uint8)
    
    cdef DTYPE_B_t[:] out_index_memview = out_index
    
    out_index[0] = True
    
    last_idx = 0


    for idx in range(1, num_holes):
        
        curr_radius = x_y_z_r_array[idx, 3]
        
        last_radius = x_y_z_r_array[last_idx, 3]
        
        diff[0] = x_y_z_r_array[idx, 0] - x_y_z_r_array[last_idx, 0]
        diff[1] = x_y_z_r_array[idx, 1] - x_y_z_r_array[last_idx, 1]
        diff[2] = x_y_z_r_array[idx, 2] - x_y_z_r_array[last_idx, 2]
        
        separation = sqrt(diff[0]*diff[0] + diff[1]*diff[1] + diff[2]*diff[2])
        
        if (separation > tolerance) or (last_radius - curr_radius > tolerance):
            
            out_index_memview[idx] = 1
            
            last_idx = idx
            
    return out_index.astype(np.bool)








@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef DTYPE_F64_t cap_height(DTYPE_F64_t R, 
                            DTYPE_F64_t r, 
                            DTYPE_F64_t d):
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
            
    cdef DTYPE_F64_t h
    
    h = (r - R + d)*(r + R - d)/(2*d)
    
    return h



@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef DTYPE_F64_t spherical_cap_volume(DTYPE_F64_t radius, 
                                      DTYPE_F64_t height):
    '''Calculate the volume of a spherical cap'''

    cdef DTYPE_F64_t volume
    
    volume = np.pi*(height**2)*(3*radius - height)/3.

    return volume



@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cpdef np.ndarray find_maximals_2(DTYPE_F64_t[:,:] x_y_z_r_array,
                                               DTYPE_F64_t frac,
                                               DTYPE_F64_t min_radius):
                                               
                                               
    #out_maximals_xyzr = np.empty(x_y_z_r_array.shape, dtype=np.float64)
    out_maximals_index = np.empty(x_y_z_r_array.shape[0], dtype=np.int64)
                                 
    #cdef DTYPE_F64_t[:,:] out_xyzr_memview = out_maximals_xyzr                
    cdef DTYPE_INT64_t[:] out_index_memview = out_maximals_index
    
    
    cdef DTYPE_F64_t[3] diffs
    cdef DTYPE_F64_t separation, curr_radius, curr_maximal_radius
    cdef DTYPE_F64_t curr_cap_height, maximal_cap_height, overlap_volume, curr_sphere_volume_thresh
    cdef ITYPE_t N_voids = 1
    cdef ITYPE_t idx, jdx, maximal_idx, num_holes
    cdef DTYPE_B_t is_maximal
    
    #out_xyzr_memview[0,:] = x_y_z_r_array[0,:]
    out_maximals_index[0] = 0
    
    num_holes = x_y_z_r_array.shape[0]
    
    for idx in range(1, num_holes):
        
        if x_y_z_r_array[idx,3] <= min_radius: #Only holes which are large enough can become maximals
            
            continue
        
        #Assume true to start, set to False and break if conditions met
        is_maximal = True
        
        curr_radius = x_y_z_r_array[idx,3]
        
        curr_sphere_volume_thresh = frac*(4.0/3.0)*np.pi*curr_radius*curr_radius*curr_radius
        
        for jdx in range(N_voids):
            
            maximal_idx = out_maximals_index[jdx]
            
            diffs[0] = x_y_z_r_array[idx,0] - x_y_z_r_array[maximal_idx,0]
            diffs[1] = x_y_z_r_array[idx,1] - x_y_z_r_array[maximal_idx,1]
            diffs[2] = x_y_z_r_array[idx,2] - x_y_z_r_array[maximal_idx,2]
            
            separation = sqrt(diffs[0]*diffs[0] + diffs[1]*diffs[1] + diffs[2]*diffs[2])
            
            curr_maximal_radius = x_y_z_r_array[maximal_idx,3]
            
            if curr_maximal_radius - curr_radius >= separation:
                #hole is fully contained in another hole, it is not a maximal
                is_maximal = False
                break
            
            elif curr_maximal_radius + curr_radius >= separation:
                
                curr_cap_height = cap_height(curr_radius, curr_maximal_radius, separation)
                
                maximal_cap_height = cap_height(curr_maximal_radius, curr_radius, separation)
            
                overlap_volume = spherical_cap_volume(curr_radius, curr_cap_height) + spherical_cap_volume(curr_maximal_radius, maximal_cap_height)
            
                if overlap_volume > curr_sphere_volume_thresh:
                    #Overlaps too much with another maximal, is not a maximal
                    is_maximal = False
                    break
                else:
                    # Didn't overlap too much with this maximal, but still
                    # need to check all the others so keep going
                    pass
            
            else:
                # Didn't overlap with this maximal at all, but still need to check
                # all the others
                pass
            
        if is_maximal:
            
            #out_xyzr_memview[N_voids,:] = x_y_z_r_array[idx,:]
            out_maximals_index[N_voids] = idx
            N_voids += 1
            
            
    return out_maximals_index[0:N_voids]
        






@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cpdef np.ndarray find_holes_2(DTYPE_F64_t[:,:] x_y_z_r_array,
                                            DTYPE_INT64_t[:] maximals_index,
                                            DTYPE_F64_t frac,
                                            ):
    """
    We have found the maximals, so intuitively one might think the 'holes' would
    just be all the remaining rows in the table.  However, some may be discarded 
    still because they are completely contained within another hole.  Also,
    the maximals themselves are also holes, so they get included in the 
    holes index.
    """

    cdef ITYPE_t idx, jdx, maximal_idx, num_holes, num_maximals, num_out_holes, num_matches, last_maximal_idx
    
    cdef DTYPE_F64_t[3] diffs
    cdef DTYPE_F64_t separation, curr_radius, curr_maximal_radius, curr_sphere_volume_thresh
    cdef DTYPE_F64_t curr_cap_height, maximal_cap_height, overlap_volume
    
    cdef DTYPE_B_t is_hole
    
    num_holes = x_y_z_r_array.shape[0]
    
    num_maximals = maximals_index.shape[0]

    holes_index = np.zeros(num_holes, dtype=np.int64)
    
    flag_column = np.zeros(num_holes, dtype=np.int64)
    
    cdef DTYPE_INT64_t[:] holes_index_memview = holes_index
    
    cdef DTYPE_INT64_t[:] flag_column_memview = flag_column
    
    ################################################################################
    # Set up some mappings for convenience
    ################################################################################
    cdef DTYPE_B_t[:] is_maximal_col = np.zeros(num_holes, dtype=np.uint8)
    
    cdef DTYPE_INT64_t[:] maximal_IDs = np.zeros(num_holes, dtype=np.int64)

    for idx in range(num_maximals):

        is_maximal_col[maximals_index[idx]] = 1
        
        maximal_IDs[maximals_index[idx]] = idx
    
    
    ################################################################################
    # Iterate through all the holes
    ################################################################################
    num_out_holes = 0
    
    for idx in range(num_holes):
        
        if is_maximal_col[idx]:
            
            holes_index_memview[num_out_holes] = idx

            flag_column_memview[num_out_holes] = maximal_IDs[idx]
            
            num_out_holes += 1
            
            continue


        curr_radius = x_y_z_r_array[idx,3]
        
        curr_sphere_volume_thresh = frac*(4.0/3.0)*np.pi*curr_radius*curr_radius*curr_radius

        is_hole = False
        
        num_matches = 0
        
        for jdx in range(num_maximals):
            
            maximal_idx = maximals_index[jdx]

            diffs[0] = x_y_z_r_array[idx,0] - x_y_z_r_array[maximal_idx,0]
            diffs[1] = x_y_z_r_array[idx,1] - x_y_z_r_array[maximal_idx,1]
            diffs[2] = x_y_z_r_array[idx,2] - x_y_z_r_array[maximal_idx,2]
            
            curr_maximal_radius = x_y_z_r_array[maximal_idx,3]


            separation = sqrt(diffs[0]*diffs[0] + diffs[1]*diffs[1] + diffs[2]*diffs[2])
            
            
            if (curr_maximal_radius - curr_radius) >= separation:
                #current sphere is completely contained within another,
                #throw it away
                break
            
            elif (curr_maximal_radius + curr_radius) >= separation:

                curr_cap_height = cap_height(curr_radius, curr_maximal_radius, separation)
                
                maximal_cap_height = cap_height(curr_maximal_radius, curr_radius, separation)
            
                overlap_volume = spherical_cap_volume(curr_radius, curr_cap_height) + spherical_cap_volume(curr_maximal_radius, maximal_cap_height)
            

                if overlap_volume > curr_sphere_volume_thresh:
                    
                    num_matches += 1
                    
                    last_maximal_idx = maximal_idx
                    
                    
                    
        ################################################################################
        # To match reference implementation, we need to only attach holes who
        # match up with exactly 1 maximal.  Future schemes may include assigning a hole
        # who matches more than 1 maximal based on the larger volume overlap
        ################################################################################
        if num_matches == 1:
            
            holes_index_memview[num_out_holes] = idx

            flag_column_memview[num_out_holes] = maximal_IDs[last_maximal_idx]
            
            num_out_holes += 1
    
    ################################################################################
    # Cython complained about returning a type (np.ndarray, np.ndarray). instead
    # return an (N,2) array since we're lucky that both return items have the
    # same dtype.
    ################################################################################
    out = np.zeros((num_out_holes,2), dtype=np.int64)
    
    out[:,0] = holes_index[0:num_out_holes]
    
    out[:,1] = flag_column[0:num_out_holes]
    
    return out











