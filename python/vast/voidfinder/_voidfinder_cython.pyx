#cython: language_level=3
#cython: initializedcheck=True
#cython: boundscheck=True
#cython: wraparound=False
#cython: cdivision=True
#cython: nonecheck=True
#cython: profile=False

from __future__ import print_function

cimport cython

import numpy as np

cimport numpy as np

np.import_array()  # required in order to use C-API

from .typedefs cimport DTYPE_CP128_t, \
                      DTYPE_CP64_t, \
                      DTYPE_F64_t, \
                      DTYPE_F32_t, \
                      DTYPE_B_t, \
                      ITYPE_t, \
                      DTYPE_INT32_t, \
                      DTYPE_INT64_t, \
                      DTYPE_INT8_t, \
                      CELL_ID_t

from numpy.math cimport NAN, INFINITY

from libc.math cimport fabs, sqrt, asin, atan#, exp, pow, cos, sin, asin

from ._voidfinder_cython_find_next cimport not_in_mask, \
                                           DistIdxPair, \
                                           Cell_ID_Memory, \
                                           GalaxyMapCustomDict, \
                                           HoleGridCustomDict, \
                                           FindNextReturnVal, \
                                           NeighborMemory, \
                                           MaskChecker, \
                                           SphereGrower, \
                                           SpatialMap
                                           #_query_first

#from ._voidfinder_cython_find_next import _query_first


import time

#Debugging
from .viz import VoidRender




cpdef DTYPE_INT64_t fill_ijk_zig_zag(DTYPE_INT64_t[:,:] i_j_k_array,
                                     DTYPE_INT64_t start_idx,
                                     DTYPE_INT64_t batch_size,
                                     DTYPE_INT64_t i_lim,
                                     DTYPE_INT64_t j_lim,
                                     DTYPE_INT64_t k_lim,
                                     HoleGridCustomDict cell_ID_dict
                                     ):
    """
    Description
    ===========
    
    Given a starting index and a batch size, generate and fill in the (i,j,k) coordinates
    into the provided i_j_k_array.
    
    This version of fill_ijk is preferred because it goes in zigzag/snake order in an 
    attempt to preserve the spatial locality of the array data while processing.  That 
    is to say, on a 3x3 grid, this functions goes:
    (0,0,0), (0,0,1), (0,0,2), (0,1,2), (0,1,1), (0,1,0), (0,2,0), ...
    whaeras a normal ordering would be:
    (0,0,0), (0,0,1), (0,0,2), (0,1,0), (0,1,1), (0,1,2), (0,2,0)
    so that the next cell grid returned is always adjacent to both its predecessor and
    its antecedent.
    
    Also filter out cell IDs which we don't want to grow a hole in (since there are galaxies
    there) using the cell_ID_dict.
    
    
    
    Parameters
    ==========
    
    i_j_k_array : memoryview onto a numpy array of shape (N, 3) where N is >= batch_size
        the array to fill in result values into
        
    start_idx : int
        the sequential index into the grid to start at
        
    batch_size : int
        the maximum number of (i,j,k) triplets to write
    
    i_lim, j_lim, k_lim : int
        the number of cells in each i,j,k direction of the grid we're searching
        
    cell_ID_dict : custom dictionary/hash table
        provides a contains() method that can check if a grid cell is in the
        galaxy map
        
    Output
    ======
    
    Writes i-j-k triplets into the i_j_k_array memory
    
    
    Notes
    =====
    We are given a block of batch_size indices to attempt to put
    into the ijk array.  In order to stay synced with the main
    process-shared counter, we can ONLY return indices from this block
    of batch_size.  This means we have 'batch_size' chances to generate
    a cell ID, NOT that we have to generate until we get 'batch_size'
    results.
    
    
    """
    
    
    cdef DTYPE_INT64_t i, j, k, remainder
    
    cdef DTYPE_INT8_t i_mod, j_mod
    
    cdef DTYPE_INT64_t num_out = 0
    
    cdef DTYPE_INT64_t num_attempt = 0
    
    cdef DTYPE_B_t skip_this_index
    
    ################################################################################
    # Make batch_size attempts, filtered by the cell_ID_dict.  Given the sequential
    # 'start_idx', convert into an i,j,k location.  To zig-zag, we
    # use the modulus of the 3 ijk values in order to convert from moving in a 
    # positive direction to a negative direction.  For example, if i=0 and j=0,
    # k moves in the "positive" direction, but if i=0 and j=1, k moves in the
    # "negative" direction.  But if i=1 and j=1, k moves in the "positive" direction
    # again.  So k depends on the modulus of i and j, j depends on the modulus of i,
    # and i always runs positive.
    ################################################################################
    for num_attempt in range(batch_size):
        
        if start_idx >= i_lim*j_lim*k_lim:
            
            return num_out
        
        i = start_idx/(j_lim*k_lim)
    
        remainder = start_idx % (j_lim*k_lim)
        
        i_mod = i % 2
            
        if i_mod == 0:
            
            j = remainder/k_lim
            
            j_mod = j % 2
            
            if j_mod == 0:
        
                k = remainder % k_lim
                
            else:
                
                k = k_lim - 1 - (remainder % k_lim)
                
        else:
            
            j = j_lim - 1 - (remainder/k_lim)
            
            j_mod = j % 2
            
            if j_mod == 0:
                
                k = k_lim - 1 - (remainder % k_lim)
                
            else:
        
                k = remainder % k_lim
                
        skip_this_index = cell_ID_dict.contains(i,j,k)
        
        #skip_this_index = False
        
        if not skip_this_index:
        
            i_j_k_array[num_out, 0] = i
            i_j_k_array[num_out, 1] = j
            i_j_k_array[num_out, 2] = k
            
            num_out += 1
            
        start_idx += 1
        
        #num_attempt += 1
        
    return num_out   





cpdef DTYPE_INT64_t fill_ijk(DTYPE_INT64_t[:,:] i_j_k_array,
                             DTYPE_INT64_t i,
                             DTYPE_INT64_t j,
                             DTYPE_INT64_t k,
                             DTYPE_INT64_t batch_size,
                             DTYPE_INT64_t i_lim,
                             DTYPE_INT64_t j_lim,
                             DTYPE_INT64_t k_lim,
                             HoleGridCustomDict cell_ID_dict
                             ):
    
    """
    POSSIBLE PROBLEM - SEE DOCSTRING FOR fill_ijk_zig_zag
    
    This fill_ijk goes in natural grid order, not zig-zag order.
    """
    
    
    if i >= i_lim or j >= j_lim or k >= k_lim:
        return 0
        
    cdef DTYPE_INT64_t num_attempt = 0
    
    cdef DTYPE_INT64_t num_out = 0
    
    cdef DTYPE_B_t inc_j
    
    cdef DTYPE_B_t inc_i
    
    cdef DTYPE_B_t skip_this_index
    
    #while num_attempt < batch_size:
    for num_attempt in range(batch_size):
            
        skip_this_index = cell_ID_dict.contains(i,j,k)
        
        if not skip_this_index:
        
            i_j_k_array[num_out, 0] = i
            i_j_k_array[num_out, 1] = j
            i_j_k_array[num_out, 2] = k
            
            num_out += 1
        
        inc_j = False
        
        if k >= (k_lim - 1) or k < 0:
            
            inc_j = True
        
        inc_i = False
        
        if (inc_j and j >= (j_lim - 1)) or j < 0:
            
            inc_i = True
            
        k = (k + 1) % k_lim
        
        if inc_j:
            
            j = (j + 1) % j_lim
            
        if inc_i:
            
            i += 1
            
            if i >= i_lim:
                
                break
            
        #num_attempt += 1
            
    return num_out   



#cpdef void grow_spheres(DTYPE_INT64_t[:,:] ijk_array,
def grow_spheres(DTYPE_INT64_t[:,:] ijk_array,
                        DTYPE_INT64_t batch_size,
                        DTYPE_F64_t[:,:] return_array,
                        SpatialMap galaxy_map,
                        SphereGrower sphere_grower,
                        MaskChecker mask_checker):
    """
    Description
    ===========
    Given a set of locations in the cubic-grid ijk paradigm, attempt to grow
    a sphere at each location and write the results to output memory.
    
    
    Parameters
    ==========
    
    ijk_array : array of shape (N, 3)
        ijk descriptors of grid cell locations in which to grow a sphere
        
    batch_size : int
        number of grid cell locations to attempt sphere growing
        
    return_array : array of shape (N, 4)
        memory in which to write results of either [NAN, ...] for invalid
        spheres or [x, y , z, radius] for valid spheres
        
    galaxy_map : _voidfinder_cython_find_next.SpatialMap
        class which handles the operations relating to finding neighbors
        and translating quickly between a spatial location and the tracers in 
        that region
    
    sphere_grower : _voidfinder_cython_find_next.SphereGrower
        helper class which has methods to grow a single sphere by tracking
        the center of that sphere, updating that center location, and calculating
        the new unit vector to search along at each stage of the growing process
    
    mask_checker : _voidfinder_cython_find_next.MaskChecker
        class which can be used to determine if a spatial location has left
        the valid areas of space
    
    
    Returns
    =======
    
    Writes x,y,z,radius values into the return_array value.
    Will write a value of NAN into the x-location of a row if that starting cell
    resulted in an invalid sphere (aka leaving the valid mask area).
    
    """
    
    
    cdef ITYPE_t working_idx, idx
    
    cdef ITYPE_t k1g, k2g, k3g, k4g, k4g1, k4g2
    
    cdef DTYPE_F64_t min_x_2, min_x_3, min_x_4, minx41, minx42
    
    cdef DTYPE_B_t failed_2, failed_3, failed_4, failed_41, failed_42
    
    cdef FindNextReturnVal result
    
    cdef DTYPE_F64_t temp_f64_accum, temp_f64_val
    
    cdef DTYPE_B_t k4g1_is_valid, k4g2_is_valid
    
    
    start_time = time.time()
    
    for working_idx in range(batch_size):
        
        print("Working index: ", working_idx, time.time() - start_time, flush=True)
        
        ################################################################################
        # Initial Setup
        #-------------------------------------------------------------------------------
        
        galaxy_map.ijk_to_xyz(ijk_array[working_idx, :], sphere_grower.sphere_center_xyz)
    
    
        #DEBUGGING
        #if np.isnan(sphere_grower.sphere_center_xyz[0]):
        #    print("CHECKPOINT-1", flush=True)
    
    
        if mask_checker.not_in_mask(sphere_grower.sphere_center_xyz):
            
            return_array[working_idx, 0] = NAN
            
            continue
        
        ################################################################################
        # Find first bounding point (k1g) and setup to find second
        #
        # Finding k1g is easy and unique - we literally just do a k=1 neighbor search
        # on the starting hole center.  After we find k1g, we don't need to update
        # the hole center because it doesn't move based on the new neighbor, and since
        # the hole center didn't move, we don't need to check the mask again since we
        # already did that above.
        #-------------------------------------------------------------------------------
        k1g = galaxy_map.find_first_neighbor(sphere_grower.sphere_center_xyz)

        # This function accounts for the Zero Vector/point on top of each other case
        sphere_grower.calculate_search_unit_vector_after_k1g(galaxy_map.points_xyz[k1g])
        
        sphere_grower.existing_bounding_idxs[0] = k1g
        
        ################################################################################
        # Find second bounding point (k2g) and setup to find 3rd
        #
        # Since find_next_bounding_point steps the hole center in steps 
        # of dl, the final step could result in a hole which actually has
        # a valid center within the mask based on the value of minx2, but
        # will fail the final mask check because we overstepped due to 
        # the jumps of size dl, so we need find_next_bounding_point to
        # check the real result at the final stage, and if it does, then 
        # the 2nd check below on the real updated hole center below will
        # become unnecessary since find_next_bounding_point has already
        # checked it.
        #
        # Sike - find_next now only checks the mask when it HASNT found
        # a valid result at all as a way to not grow to infinity, so we 
        # need to both check that the result was valid and then update
        # the hole center and check that location as well
        #
        # This check is still necessary with the new iterative version of
        # find_next_bounding_point to communicate that it failed by way of
        # leaving the mask
        #-------------------------------------------------------------------------------
        result = galaxy_map.find_next_bounding_point(sphere_grower.sphere_center_xyz,
                                                     sphere_grower.search_unit_vector,
                                                     sphere_grower.existing_bounding_idxs,
                                                     1,
                                                     mask_checker)
        
        k2g = result.nearest_neighbor_index
        min_x_2 = result.min_x_val
        failed_2 = result.failed
        
        if failed_2:
            
            return_array[working_idx, 0] = NAN
            
            continue
        
        sphere_grower.update_hole_center(min_x_2)
        
        
        #DEBUGGING
        #if np.isnan(sphere_grower.sphere_center_xyz[0]):
        #    print("CHECKPOINT-2", flush=True)
        
        if mask_checker.not_in_mask(sphere_grower.sphere_center_xyz):
            
            return_array[working_idx, 0] = NAN
            
            continue
        
        # This function accounts for the Co-linear case
        sphere_grower.calculate_search_unit_vector_after_k2g(galaxy_map.points_xyz[k1g],
                                                             galaxy_map.points_xyz[k2g])

        sphere_grower.existing_bounding_idxs[1] = k2g


        ################################################################################
        # Find third bounding point (k3g) and setup to find 4th
        #-------------------------------------------------------------------------------
        
        result = galaxy_map.find_next_bounding_point(sphere_grower.sphere_center_xyz,
                                                     sphere_grower.search_unit_vector,
                                                     sphere_grower.existing_bounding_idxs,
                                                     2,
                                                     mask_checker)
        
        k3g = result.nearest_neighbor_index
        min_x_3 = result.min_x_val
        failed_3 = result.failed
        
        if failed_3:
            
            return_array[working_idx, 0] = NAN
            
            continue
        
        sphere_grower.update_hole_center(min_x_3)
        
        
        
        #DEBUGGING
        #if np.isnan(sphere_grower.sphere_center_xyz[0]):
        #    print("CHECKPOINT-3", flush=True)
            
        if mask_checker.not_in_mask(sphere_grower.sphere_center_xyz):
            
            return_array[working_idx, 0] = NAN
            
            continue
        
        # This function accounts for the Co-Planar case
        is_coplanar = sphere_grower.calculate_search_unit_vector_after_k3g(galaxy_map.points_xyz[k1g],
                                                                           galaxy_map.points_xyz[k2g],
                                                                           galaxy_map.points_xyz[k3g])
        
        sphere_grower.existing_bounding_idxs[2] = k3g
        
        
        ################################################################################
        # Find 4th and final bounding point
        #-------------------------------------------------------------------------------
        if is_coplanar:
            
            k4g1_is_valid = True
            k4g2_is_valid = True
            
            
            ################################################################################
            # Start on the side we calculated with the search unit vector
            #-------------------------------------------------------------------------------
            
            result = galaxy_map.find_next_bounding_point(sphere_grower.sphere_center_xyz,
                                                         sphere_grower.search_unit_vector,
                                                         sphere_grower.existing_bounding_idxs,
                                                         3,
                                                         mask_checker)
            
            k4g1 = result.nearest_neighbor_index
            minx41 = result.min_x_val
            failed_41 = result.failed
            
            if failed_41:
                k4g1_is_valid = False
            else:
                
                for idx in range(3):
                    sphere_grower.hole_center_k4g1[idx] = sphere_grower.sphere_center_xyz[idx] + minx41*sphere_grower.search_unit_vector[idx]
                
                
                
                #DEBUGGING
                #if np.isnan(sphere_grower.hole_center_k4g1[0]):
                #    print("CHECKPOINT-4", flush=True)
                
                not_in_mask_k4g1 = mask_checker.not_in_mask(sphere_grower.hole_center_k4g1)
                
                if not_in_mask_k4g1:
                    k4g1_is_valid = False
                
            ################################################################################
            # Flip the unit vector and search other side
            #-------------------------------------------------------------------------------
            for idx in range(3):
                sphere_grower.search_unit_vector[idx] *= -1.0
            
            result = galaxy_map.find_next_bounding_point(sphere_grower.sphere_center_xyz,
                                                         sphere_grower.search_unit_vector,
                                                         sphere_grower.existing_bounding_idxs,
                                                         3,
                                                         mask_checker)
            
            k4g2 = result.nearest_neighbor_index
            minx42 = result.min_x_val
            failed_42 = result.failed
            
            if failed_42:
                k4g2_is_valid = False
            else:
                
                for idx in range(3):
                    sphere_grower.hole_center_k4g2[idx] = sphere_grower.sphere_center_xyz[idx] + minx42*sphere_grower.search_unit_vector[idx]
                
                
                
                #DEBUGGING
                #if np.isnan(sphere_grower.hole_center_k4g2[0]):
                #    print("CHECKPOINT-5", flush=True)
                
                not_in_mask_k4g2 = mask_checker.not_in_mask(sphere_grower.hole_center_k4g2)
                
                if not_in_mask_k4g2:
                    k4g2_is_valid = False
                
            ################################################################################
            # Select one of the two possible holes we have grown
            #-------------------------------------------------------------------------------
            if k4g1_is_valid and k4g2_is_valid:
                #If both valid, pick the smaller one
                if minx41 <= minx42:
                    #select k4g1
                    
                    k4g = k4g1
                    min_x_4 = minx41
                    #flip unit vector back
                    for idx in range(3):
                        sphere_grower.search_unit_vector[idx] *= -1.0
                else:
                    k4g = k4g2
                    min_x_4 = minx42
                
                
            elif k4g1_is_valid:
                #select k4g2
                
                k4g = k4g1
                min_x_4 = minx41
                #flip unit vector back
                for idx in range(3):
                    sphere_grower.search_unit_vector[idx] *= -1.0
                
            elif k4g2_is_valid:
                #select k4g2
                k4g = k4g2
                min_x_4 = minx42
                
                    
            else:
                
                return_array[working_idx, 0] = NAN
                
                continue
            
        ########################################################################
        # In the non-coplanar majority case, just do the same process to get
        # k4g
        ########################################################################
        else:
        
            result = galaxy_map.find_next_bounding_point(sphere_grower.sphere_center_xyz,
                                                         sphere_grower.search_unit_vector,
                                                         sphere_grower.existing_bounding_idxs,
                                                         3,
                                                         mask_checker)
            
            k4g = result.nearest_neighbor_index
            min_x_4 = result.min_x_val
            failed_4 = result.failed
            
            if failed_4:
                
                return_array[working_idx, 0] = NAN
            
                continue
        
        sphere_grower.update_hole_center(min_x_4)
        
        
        #DEBUGGING
        #if np.isnan(sphere_grower.sphere_center_xyz[0]):
        #    print("CHECKPOINT-6", flush=True)
        
        #Check against mask
        if mask_checker.not_in_mask(sphere_grower.sphere_center_xyz):
            
            return_array[working_idx, 0] = NAN
            
            continue

        ########################################################################
        # Passed all checks, calculate sphere radius and write the valid 
        # (x,y,z,r) values!
        #-----------------------------------------------------------------------
        
        temp_f64_accum = 0.0
        
        for idx in range(3):
            
            temp_f64_val = sphere_grower.sphere_center_xyz[idx] - galaxy_map.points_xyz[k1g, idx]
            
            temp_f64_accum += temp_f64_val*temp_f64_val
        
        
        return_array[working_idx, 0] = sphere_grower.sphere_center_xyz[0]
        
        return_array[working_idx, 1] = sphere_grower.sphere_center_xyz[1]
        
        return_array[working_idx, 2] = sphere_grower.sphere_center_xyz[2]
        
        return_array[working_idx, 3] = sqrt(temp_f64_accum)
        ########################################################################


    return




def placeholder():
    """
    Literally a placeholder for the Eclipse Outline
    """
    pass














