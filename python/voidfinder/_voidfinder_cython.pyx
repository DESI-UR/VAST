



cimport cython
import numpy as np
cimport numpy as np
np.import_array()  # required in order to use C-API


from typedefs cimport DTYPE_CP128_t, DTYPE_CP64_t, DTYPE_F64_t, DTYPE_F32_t, DTYPE_B_t, ITYPE_t, DTYPE_INT32_t, DTYPE_INT64_t

from numpy.math cimport NAN, INFINITY

from libc.math cimport fabs, sqrt, asin, atan#, exp, pow, cos, sin, asin

from _voidfinder_cython_find_next cimport find_next_galaxy, not_in_mask



import time


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cpdef void main_algorithm(DTYPE_INT64_t[:,:] i_j_k_array,
                          galaxy_tree,
                          DTYPE_F64_t[:,:] w_coord,
                          DTYPE_F64_t dl, 
                          DTYPE_F64_t dr,
                          DTYPE_F64_t[:,:] coord_min, 
                          DTYPE_B_t[:,:] mask,
                          DTYPE_INT32_t mask_resolution,
                          DTYPE_F64_t min_dist,
                          DTYPE_F64_t max_dist,
                          DTYPE_F64_t[:,:] return_array,
                          int verbose,
                          DTYPE_F32_t[:,:] PROFILE_ARRAY
                          ) except *:
    '''
    
    Description:
    ============
    Given a potential void cell center denoted by i,j,k, find the 4 bounding galaxies
    that maximize the interior dimensions of the void sphere at this location.
    
    There are some really weird particulars to this algorithm that need to be laid out
    in better detail.

    The code below will utilize the naming scheme galaxy A, B, C, and D to denote the
    1st, 2nd, 3rd, and 4th "neighbor" bounding galaxies found during the running of this algorithm.
    I tried to use 1/A, 2/B, 3/C and 4/D to be clear on the numbers and letters are together.
    
    The distance metrics are somewhat special.  Galaxy A is found by normal minimzation of euclidean
    distance between the cell center and itself and the other neighbors.  Galaxies B, C, and D are 
    found by propogating a hole center in specific directions and minimizing a ratio of two other
    distance-metric-like values.  This needs more detail on how and why.
    
    Parameters:
    ===========
    
    Fill in later
    
    Returns:
    ========
    
    NAN or (x,y,z,r) values filled into the return_array parameter.
    
    '''
    
    
    
    
    
    ############################################################
    # Declarations - Definitely needed
    ############################################################
    
    ############################################################
    # re-used helper index variables
    # re-used helper computation variables
    ############################################################
    
    
    cdef ITYPE_t working_idx
    
    cdef ITYPE_t idx
    
    cdef ITYPE_t jdx
    
    cdef ITYPE_t temp_idx
    
    
    cdef DTYPE_F64_t temp_f64_accum
    
    cdef DTYPE_F64_t temp_f64_accum2
    
    cdef DTYPE_F64_t temp_f64_val
    
    
    ############################################################
    #hole center vector and propogate hole center memory.  We can re-use the same
    #memory for finding galaxies 2/B and 3/C but not for galaxy 4/D
    ############################################################
    cdef DTYPE_F64_t[:,:] hole_center_memview = np.empty((1,3), dtype=np.float64, order='C')
    
    cdef DTYPE_F64_t[:,:] hole_center_2_3_memview = np.empty((1,3), dtype=np.float64, order='C')
    
    cdef DTYPE_F64_t[:,:] hole_center_41_memview = np.empty((1,3), dtype=np.float64, order='C')
    
    cdef DTYPE_F64_t[:,:] hole_center_42_memview = np.empty((1,3), dtype=np.float64, order='C')
    
    ############################################################
    # the nearest_gal_index_list variable stores 2 things -
    # the sentinel value -1 meaning 'no value here' or the 
    # index of the found galaxies A,B,C (but not D since its 
    # the last one)
    ############################################################
    cdef DTYPE_INT64_t[:] nearest_gal_index_list = np.empty(3, dtype=np.int64, order='C')
    
    
    
    ############################################################
    #memory for the return variables when calling find_next_galaxy
    ############################################################
    cdef ITYPE_t[:] x_ratio_index_return_mem = np.zeros(1, dtype=np.int64, order='C')
    
    cdef ITYPE_t[:] gal_idx_return_mem = np.zeros(1, dtype=np.int64, order='C')
    
    cdef DTYPE_F64_t[:] min_x_return_mem = np.zeros(1, dtype=np.float64, order='C')
    
    cdef DTYPE_B_t[:] in_mask_return_mem = np.ones(1, dtype=np.uint8)

    ############################################################
    # vector_modulus is re-used each time the new unit vector is calculated
    # unit_vector_memview is re-used each time the new unit vector is calculated
    # v3_memview is used in calculating the cross product for galaxy 4/D
    # hole_radius is used in some hole update calculations
    ############################################################
    cdef DTYPE_F64_t vector_modulus
    
    cdef DTYPE_F64_t[:] unit_vector_memview = np.empty(3, dtype=np.float64, order='C')
    
    cdef DTYPE_F64_t[:] v3_memview = np.empty(3, dtype=np.float64, order='C')
    
    cdef DTYPE_F64_t hole_radius
    
    
    
    ############################################################
    #variables for the 4 galaxy seaarches whether the hole center is in the mask
    ############################################################
    cdef DTYPE_B_t in_mask_2
    
    cdef DTYPE_B_t in_mask_3
    
    cdef DTYPE_B_t in_mask_41
    
    cdef DTYPE_B_t in_mask_42
    
    
    ############################################################
    #variables for the 4 bounding galaxies (gal 4/D gets 3 variables because it uses 2 searches
    #and then picks one of the two based on some criteria)
    ############################################################
    cdef ITYPE_t k1g
    
    cdef ITYPE_t k2g
    
    cdef ITYPE_t k3g
    
    cdef ITYPE_t k4g1
    
    cdef ITYPE_t k4g2
    
    cdef ITYPE_t k4g
    
    ############################################################
    #minx3 is used in updating a hole center and minx41-42 are used in comparing to find
    #galaxy 4/D
    ############################################################
    cdef DTYPE_F64_t minx3
    
    cdef DTYPE_F64_t minx41
    
    cdef DTYPE_F64_t minx42
    
    ############################################################
    #these are used in calcuating the unit vector for galaxy 4/D
    ############################################################
    cdef DTYPE_F64_t[:] midpoint_memview = np.empty(3, dtype=np.float64, order='C')
    
    #cdef DTYPE_F64_t[:] Acenter_memview = np.empty(3, dtype=np.float64, order='C')
    
    cdef DTYPE_F64_t[:] AB_memview = np.empty(3, dtype=np.float64, order='C')
    
    cdef DTYPE_F64_t[:] BC_memview = np.empty(3, dtype=np.float64, order='C')
    
    
    
    
    
    
    ############################################################
    # Computation memory to pass into find_next_galaxy()
    #
    # NOTE: This memory is currently unused because of 
    # I need to figure out how to reallocate it if it isnt
    # big enough without leaking memory and causing a segfault
    #
    #
    ############################################################
    
    cdef DTYPE_F64_t[:] Bcenter_memview = np.empty(3, dtype=np.float64, order='C')
    
    cdef ITYPE_t[:] MAX_NEAREST = np.empty(1, dtype=np.int64)
    
    MAX_NEAREST[0] = 16 #guess at the max number of results returned by kdtree
    
    cdef ITYPE_t[:] i_nearest_reduced_memview = np.empty(MAX_NEAREST[0], dtype=np.int64)
                
    cdef DTYPE_F64_t[:,:] candidate_minus_A_memview = np.empty((MAX_NEAREST[0], 3), dtype=np.float64, order='C')

    cdef DTYPE_F64_t[:,:] candidate_minus_center_memview = np.empty((MAX_NEAREST[0], 3), dtype=np.float64, order='C')
    
    cdef DTYPE_F64_t[:] bot_memview = np.empty(MAX_NEAREST[0], dtype=np.float64, order='C')
    
    cdef DTYPE_F64_t[:] top_memview = np.empty(MAX_NEAREST[0], dtype=np.float64, order='C')
    
    cdef DTYPE_F64_t[:] x_ratio_memview = np.empty(MAX_NEAREST[0], dtype=np.float64, order='C')
    
    
    
    
    
    
    
    
    
    
    ############################################################
    # Declarations - possibly remove
    ############################################################
    
    #these are never actually used
    #cdef ITYPE_t k2g_x2
    #cdef ITYPE_t k3g_x3
    #cdef ITYPE_t k4g1_x41
    #cdef ITYPE_t k4g2_x42

    
    cdef ITYPE_t num_cells = i_j_k_array.shape[0]
    
    
    
    cdef DTYPE_F64_t PROFILE_start_time
    cdef DTYPE_F64_t[:] PROFILE_kdtree_time = np.zeros(1, dtype=np.float64)
    PROFILE_kdtree_time[0] = 0.0
    cdef DTYPE_F64_t PROFILE_kdtree_time_collect = 0.0
    #cdef DTYPE_F64_t PROFILE_kdtree_time_collect_after = 0.0
    
    for working_idx in range(num_cells):
        
        
        PROFILE_kdtree_time[0] = 0.0
        PROFILE_start_time = time.time()
        
        
        #Don't do the print statements in the main algorithm
        #Let the caller handle how many have been processed
        #if (verbose > 0 and working_idx % 10000 == 0):
        
        #    print("Processing cell "+str(working_idx)+" of "+str(num_cells))
        
        
        #re-init nearest gal index list on new cell
        for idx in range(3):
        
            nearest_gal_index_list[idx] = -1
    
        
        ############################################################
        # initialize the starting hole center based on the given
        # i,j,k and grid spacing parameters, and then check to
        # make sure it's in the survey
        ############################################################
        hole_center_memview[0,0] = i_j_k_array[working_idx, 0]
        
        hole_center_memview[0,1] = i_j_k_array[working_idx, 1]
        
        hole_center_memview[0,2] = i_j_k_array[working_idx, 2]
    
        
        for idx in range(3):
            
            hole_center_memview[0,idx] = (hole_center_memview[0,idx] + 0.5)*dl + coord_min[0,idx]
            
            
        if not_in_mask(hole_center_memview, mask, mask_resolution, min_dist, max_dist):
            
            return_array[working_idx, 0] = NAN
            
            return_array[working_idx, 1] = NAN
            
            return_array[working_idx, 2] = NAN
            
            return_array[working_idx, 3] = NAN
            
            
            PROFILE_ARRAY[working_idx, 0] = time.time() - PROFILE_start_time
            PROFILE_ARRAY[working_idx, 1] = 0
            PROFILE_ARRAY[working_idx, 2] = <DTYPE_F32_t>PROFILE_kdtree_time[0]
            
            #return 
            continue
        
        
        ############################################################
        #
        # Find Galaxy 1/A - super easy using KDTree
        #
        ############################################################
        
        
        PROFILE_kdtree_time_collect = time.time()
        
        neighbor_1_dists, neighbor_1_idxs = galaxy_tree.query(hole_center_memview, k=1)
        
        #PROFILE_kdtree_time_collect_after = time.time()
        
        PROFILE_kdtree_time[0] += time.time() - PROFILE_kdtree_time_collect
        
        #print("KDTREE1_TIME: ", PROFILE_kdtree_time_collect_after - PROFILE_kdtree_time_collect, flush=True)
        
        k1g = neighbor_1_idxs[0][0]
        
        ############################################################################
        #
        # Start Galaxy 2/B
        #
        # v1_unit_memview - unit vector hole propagation direction
        #
        # modv1 - l2 norm modulus of v1_unit
        #
        ############################################################################
        vector_modulus = neighbor_1_dists[0][0] #float64
        
        for idx in range(3):
            
            unit_vector_memview[idx] = (w_coord[k1g,idx] - hole_center_memview[0,idx])/vector_modulus
            
        hole_radius = vector_modulus
        
        ############################################################################
        # Make a copy of the hole center for propogation during the galaxy finding
        # set the nearest_gal_index_list first neighbor index since we have a neighbor
        # now
        #
        # Also fill in neighbor 1/A into the existing neighbor idx list
        ############################################################################
        for idx in range(3):
            
            hole_center_2_3_memview[0,idx] = hole_center_memview[0,idx]
    
        ############################################################################
        #
        # Find galaxy 2/B
        #
        # set the nearest neighbor 1 index in the list and init the in_mask to 1
        ############################################################################
        nearest_gal_index_list[0] = k1g
        
        in_mask_return_mem[0] = 1
        
        find_next_galaxy(hole_center_memview,
                         hole_center_2_3_memview, 
                         hole_radius, 
                         dr, 
                         -1.0,
                         unit_vector_memview, 
                         galaxy_tree, 
                         nearest_gal_index_list, 
                         1,
                         w_coord, 
                         mask, 
                         mask_resolution,
                         min_dist, 
                         max_dist,
                         Bcenter_memview,
                         MAX_NEAREST,
                         i_nearest_reduced_memview,
                         candidate_minus_A_memview,
                         candidate_minus_center_memview,
                         bot_memview,
                         top_memview,
                         x_ratio_memview,
                         #x_ratio_index_return_mem,  #return variable
                         gal_idx_return_mem,        #return variable
                         min_x_return_mem,          #return variable
                         in_mask_return_mem,        #return variable
                         PROFILE_kdtree_time)
    
    
        #k2g_x2 = x_ratio_index_return_mem[0]
        
        k2g = gal_idx_return_mem[0]
        
        in_mask_2 = in_mask_return_mem[0]
        
        #minx2 = min_x_return_mem[0]
        
        if not in_mask_2:
        
            return_array[working_idx, 0] = NAN
            
            return_array[working_idx, 1] = NAN
            
            return_array[working_idx, 2] = NAN
            
            return_array[working_idx, 3] = NAN
            
            
            PROFILE_ARRAY[working_idx, 0] = time.time() - PROFILE_start_time
            PROFILE_ARRAY[working_idx, 1] = 1
            PROFILE_ARRAY[working_idx, 2] = <DTYPE_F32_t>PROFILE_kdtree_time[0]
            
            #return 
            continue
        
        
        ###################################################################################
        #
        # Start Galaxy 3/C
        #
        # Calculate the new starting hole radius, and move the hole center where it 
        # already was?
        # then make sure that location is still in the mask
        ##################################################################################
        
        temp_f64_accum = 0.0
        
        temp_f64_accum2 = 0.0
        
        for idx in range(3):
        
            temp_f64_val = w_coord[k1g,idx] - w_coord[k2g, idx]
            
            temp_f64_accum += temp_f64_val*temp_f64_val
            
            temp_f64_accum2 += temp_f64_val*unit_vector_memview[idx]
        
        hole_radius = 0.5*temp_f64_accum/temp_f64_accum2
    
        
        for idx in range(3):
            
            hole_center_memview[0,idx] = w_coord[k1g,idx] - hole_radius*unit_vector_memview[idx]
        
        
        if not_in_mask(hole_center_memview, mask, mask_resolution, min_dist, max_dist):
            
            return_array[working_idx, 0] = NAN
            
            return_array[working_idx, 1] = NAN
            
            return_array[working_idx, 2] = NAN
            
            return_array[working_idx, 3] = NAN
            
            
            PROFILE_ARRAY[working_idx, 0] = time.time() - PROFILE_start_time
            PROFILE_ARRAY[working_idx, 1] = 2
            PROFILE_ARRAY[working_idx, 2] = <DTYPE_F32_t>PROFILE_kdtree_time[0]
            
            #return 
            continue
    
        ############################################################################
        # Find the midpoint between the two nearest galaxies
        # calculate the new modulus between the hole center and midpoint spot
        # Define the new unit vector along which to move the hole center
        ############################################################################
        
        for idx in range(3):
            
            midpoint_memview[idx] = 0.5*(w_coord[k1g,idx] + w_coord[k2g,idx])
        
        
        temp_f64_accum = 0.0
        
        for idx in range(3):
            
            temp_f64_val = hole_center_memview[0,idx] - midpoint_memview[idx]
            
            temp_f64_accum += temp_f64_val*temp_f64_val
        
        vector_modulus = sqrt(temp_f64_accum)
        
        
        for idx in range(3):
        
            unit_vector_memview[idx] = (hole_center_memview[0,idx] - midpoint_memview[idx])/vector_modulus
        
        
        ############################################################################
        # Calculate vector pointing from the hole center to the nearest galaxy
        # Calculate vector pointing from the hole center to the second-nearest galaxy
        # Initialize moving hole center
        ############################################################################
        '''
        for idx in range(3):
            
            Acenter_memview[idx] = w_coord[k1g, idx] - hole_center_memview[0,idx]
        
        for idx in range(3):
            
            Bcenter_memview[idx] = w_coord[k2g, idx] - hole_center_memview[0,idx]
        '''
        for idx in range(3):
            
            hole_center_2_3_memview[0,idx] = hole_center_memview[0,idx]
        
    
        ############################################################################
        #
        # Find galaxy 3/C
        #
        # set neighbors 1 and 2 in the list, and re-init the in_mask variable to 1
        ############################################################################
        nearest_gal_index_list[0] = k1g
        
        nearest_gal_index_list[1] = k2g
    
        in_mask_return_mem[0] = 1
    
        find_next_galaxy(hole_center_memview,
                         hole_center_2_3_memview, 
                         hole_radius, 
                         dr, 
                         1.0,
                         unit_vector_memview, 
                         galaxy_tree, 
                         nearest_gal_index_list, 
                         2,
                         w_coord, 
                         mask, 
                         mask_resolution, 
                         min_dist, 
                         max_dist,
                         Bcenter_memview,
                         MAX_NEAREST,
                         i_nearest_reduced_memview,
                         candidate_minus_A_memview,
                         candidate_minus_center_memview,
                         bot_memview,
                         top_memview,
                         x_ratio_memview,
                         #x_ratio_index_return_mem,  #return variable
                         gal_idx_return_mem,        #return variable
                         min_x_return_mem,          #return variable
                         in_mask_return_mem,        #return variable
                         PROFILE_kdtree_time)
    
        #k3g_x3 = x_ratio_index_return_mem[0]
        
        k3g = gal_idx_return_mem[0]
        
        minx3 = min_x_return_mem[0]
        
        in_mask_3 = in_mask_return_mem[0]
        
        if not in_mask_3:
        
            return_array[working_idx, 0] = NAN
            
            return_array[working_idx, 1] = NAN
            
            return_array[working_idx, 2] = NAN
            
            return_array[working_idx, 3] = NAN
            
            
            PROFILE_ARRAY[working_idx, 0] = time.time() - PROFILE_start_time
            PROFILE_ARRAY[working_idx, 1] = 3
            PROFILE_ARRAY[working_idx, 2] = <DTYPE_F32_t>PROFILE_kdtree_time[0]
            
            
            #return 
            continue
    
        ###########################################################################
        #
        # Start Galaxy 4/D-1 (galaxy 4/D takes 2 attempts)
        #
        # Process is very similar as before, except we do not know if we have to 
        # move above or below the plane.  Therefore, we will find the next closest 
        # if we move above the plane, and the next closest if we move below the 
        # plane.
        #
        # Update hole center
        # update hole radius
        #
        ###########################################################################
        for idx in range(3):
        
            hole_center_memview[0,idx] += minx3*unit_vector_memview[idx]
    
    
        temp_f64_accum = 0.0
    
        for idx in range(3):
    
            temp_f64_val = hole_center_memview[0,idx] - w_coord[k1g,idx]
    
            temp_f64_accum += temp_f64_val*temp_f64_val
    
        hole_radius = sqrt(temp_f64_accum)
        
    
        if not_in_mask(hole_center_memview, mask, mask_resolution, min_dist, max_dist):
        
            return_array[working_idx, 0] = NAN
            
            return_array[working_idx, 1] = NAN
            
            return_array[working_idx, 2] = NAN
            
            return_array[working_idx, 3] = NAN
            
            
            PROFILE_ARRAY[working_idx, 0] = time.time() - PROFILE_start_time
            PROFILE_ARRAY[working_idx, 1] = 4
            PROFILE_ARRAY[working_idx, 2] = <DTYPE_F32_t>PROFILE_kdtree_time[0]
            
            #return 
            continue
    
        ############################################################################
        #
        # The vector along which to move the hole center is defined by the cross 
        # product of the vectors pointing between the three nearest galaxies.
        #
        # Calculate the cross product of the difference vectors, calculate
        # the modulus of that vector and normalize to a unit vector
        # 
        ############################################################################
    
        for idx in range(3):
    
            AB_memview[idx] = w_coord[k1g, idx] - w_coord[k2g, idx]
    
            BC_memview[idx] = w_coord[k3g, idx] - w_coord[k2g, idx]
    
    
        v3_memview[0] = AB_memview[1]*BC_memview[2] - AB_memview[2]*BC_memview[1]
    
        v3_memview[1] = AB_memview[2]*BC_memview[0] - AB_memview[0]*BC_memview[2]
    
        v3_memview[2] = AB_memview[0]*BC_memview[1] - AB_memview[1]*BC_memview[0]
    
    
        temp_f64_accum = 0.0
    
        for idx in range(3):
    
            temp_f64_accum += v3_memview[idx]*v3_memview[idx]
    
        vector_modulus = sqrt(temp_f64_accum)
    
    
        for idx in range(3):
    
            unit_vector_memview[idx] = v3_memview[idx]/vector_modulus
    
        ############################################################################
        # Calculate vector pointing from the hole center to the neighbor 1/A
        # Calculate vector pointing from the hole center to the neighbor 2/B
        # Update new hole center for propagation
        ############################################################################
        '''
        for idx in range(3):
    
            Acenter_memview[idx] = w_coord[k1g, idx] - hole_center_memview[0, idx]
    
    
        for idx in range(3):
    
            Bcenter_memview[idx] = w_coord[k2g, idx] - hole_center_memview[0, idx]
        '''
    
        for idx in range(3):
    
            hole_center_41_memview[0, idx] = hole_center_memview[0, idx]
    
        ############################################################################
        #
        # Find galaxy 4/D-1
        #
        # update the exiting neighbors 1/A, 2/B and 3/C in the 
        # nearest_gal_index_list and re-init in_mask to 1
        ############################################################################
        nearest_gal_index_list[0] = k1g
        
        nearest_gal_index_list[1] = k2g
        
        nearest_gal_index_list[2] = k3g
    
        in_mask_return_mem[0] = 1
    
        find_next_galaxy(hole_center_memview,
                         hole_center_41_memview, 
                         hole_radius, 
                         dr, 
                         1.0,
                         unit_vector_memview, 
                         galaxy_tree, 
                         nearest_gal_index_list, 
                         3,
                         w_coord, 
                         mask, 
                         mask_resolution,
                         min_dist, 
                         max_dist,
                         Bcenter_memview,
                         MAX_NEAREST,
                         i_nearest_reduced_memview,
                         candidate_minus_A_memview,
                         candidate_minus_center_memview,
                         bot_memview,
                         top_memview,
                         x_ratio_memview,
                         #x_ratio_index_return_mem,  #return variable
                         gal_idx_return_mem,        #return variable
                         min_x_return_mem,          #return variable
                         in_mask_return_mem,        #return variable
                         PROFILE_kdtree_time)
    
        #k4g1_x41 = x_ratio_index_return_mem[0]
                        
        k4g1 = gal_idx_return_mem[0]
        
        minx41 = min_x_return_mem[0]
    
        in_mask_41 = in_mask_return_mem[0]
    
    
        # Calculate potential new hole center
        if in_mask_41:
    
            for idx in range(3):
    
                hole_center_41_memview[0, idx] = hole_center_memview[0, idx] + minx41*unit_vector_memview[idx]
    
    
    
            
       
        ############################################################################
        #
        # Start galaxy 4/D-2
        #
        # Repeat same search, but shift the hole center in the other direction 
        # this time, so flip the v3_unit_memview in other direction
        ############################################################################
        for idx in range(3):
    
            unit_vector_memview[idx] *= -1.0
    
    
        minx42 = INFINITY
    
    
        for idx in range(3):
    
            hole_center_42_memview[0, idx] = hole_center_memview[0, idx]
    
        
        ############################################################################
        #
        # Find galaxy 4/D-2
        #
        # nearest_neighbor_gal_list already updated from galaxy 4/D-1
        # re-init in_mask to 1
        ############################################################################
        in_mask_return_mem[0] = 1
    
        find_next_galaxy(hole_center_memview,
                         hole_center_42_memview, 
                         hole_radius, 
                         dr, 
                         1.0,
                         unit_vector_memview, 
                         galaxy_tree, 
                         nearest_gal_index_list, 
                         3,
                         w_coord, 
                         mask, 
                         mask_resolution,
                         min_dist, 
                         max_dist,
                         Bcenter_memview,
                         MAX_NEAREST,
                         i_nearest_reduced_memview,
                         candidate_minus_A_memview,
                         candidate_minus_center_memview,
                         bot_memview,
                         top_memview,
                         x_ratio_memview,
                         #x_ratio_index_return_mem,  #return variable
                         gal_idx_return_mem,        #return variable
                         min_x_return_mem,          #return variable
                         in_mask_return_mem,        #return variable
                         PROFILE_kdtree_time)
    
        #k4g2_x42 = x_ratio_index_return_mem[0]
                        
        k4g2 = gal_idx_return_mem[0]
        
        minx42 = min_x_return_mem[0]
    
        in_mask_42 = in_mask_return_mem[0]
        
        # Calculate potential new hole center
        if in_mask_42:
    
            for idx in range(3):
    
                hole_center_42_memview[0, idx] = hole_center_memview[0, idx] + minx42*unit_vector_memview[idx]
    
        
        
        ############################################################################
        # Figure out whether galaxy 4/D is 4/D-1 or 4/D-2
        # use the minx41 and minx42 variables to figure out which one is 
        # closer? then set the 4th galaxy index based on that and update the
        # output hole center based on that.  Or, if the conditions aren't filled
        # because we left the survey, return NAN output
        ############################################################################
        
        not_in_mask_41 = not_in_mask(hole_center_41_memview, mask, mask_resolution, min_dist, max_dist)
    
        if not not_in_mask_41 and minx41 <= minx42:
    
            for idx in range(3):
    
                hole_center_memview[0, idx] = hole_center_41_memview[0, idx]
    
            k4g = k4g1
    
        elif not not_in_mask(hole_center_42_memview, mask, mask_resolution, min_dist, max_dist):
            
            for idx in range(3):
    
                hole_center_memview[0, idx] = hole_center_42_memview[0, idx]
    
            k4g = k4g2
    
        elif not not_in_mask_41:
    
            for idx in range(3):
    
                hole_center_memview[0, idx] = hole_center_41_memview[0, idx]
    
            k4g = k4g1
            
        else:
            
            return_array[working_idx, 0] = NAN
            
            return_array[working_idx, 1] = NAN
            
            return_array[working_idx, 2] = NAN
            
            return_array[working_idx, 3] = NAN
            
            
            
            PROFILE_ARRAY[working_idx, 0] = time.time() - PROFILE_start_time
            PROFILE_ARRAY[working_idx, 1] = 5
            PROFILE_ARRAY[working_idx, 2] = <DTYPE_F32_t>PROFILE_kdtree_time[0]
            
            
            
            #return 
            continue
    
    
        ############################################################################
        # Now that we have all 4 bounding galaxies, calculate the hole radius
        # and write the valid (x,y,z,r) values!
        ############################################################################
    
        temp_f64_accum = 0.0
    
        for idx in range(3):
    
            temp_f64_val = hole_center_memview[0, idx] - w_coord[k1g, idx]
    
            temp_f64_accum += temp_f64_val*temp_f64_val
    
        hole_radius = sqrt(temp_f64_accum)
    
    
        return_array[working_idx, 0] = hole_center_memview[0, 0]
        
        return_array[working_idx, 1] = hole_center_memview[0, 1]
        
        return_array[working_idx, 2] = hole_center_memview[0, 2]
        
        return_array[working_idx, 3] = hole_radius
        
        
        PROFILE_ARRAY[working_idx, 0] = time.time() - PROFILE_start_time
        PROFILE_ARRAY[working_idx, 1] = 6
        PROFILE_ARRAY[working_idx, 2] = <DTYPE_F32_t>PROFILE_kdtree_time[0]
    
    
    #print("Finished main loop")
    
    #return (x_val, y_val, z_val, r_val)
    return


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cpdef void main_algorithm_geometric(DTYPE_INT64_t[:,:] i_j_k_array,
                                    galaxy_tree,
                                    DTYPE_F64_t[:,:] w_coord,
                                    DTYPE_F64_t dl, 
                                    DTYPE_F64_t dr,
                                    DTYPE_F64_t[:,:] coord_min, 
                                    DTYPE_B_t[:,:] mask,
                                    DTYPE_INT32_t mask_resolution,
                                    DTYPE_F64_t min_dist,
                                    DTYPE_F64_t max_dist,
                                    DTYPE_F64_t[:,:] return_array,
                                    int verbose,
                                    DTYPE_F32_t[:,:] PROFILE_ARRAY
                                    ) except *:

    """
    Like to attempt writing a new main_algorithm which is based around finding
    the neighbors without a bunch of successive radius queries
    
    Find neighbor 1 same way - nearest neighbor query.  Now you have a
    Line of Motion from neighbor 1, through the cell center with an orientation
    
    Find 2 arbitrary vectors in the plane to define the plane, then find the first
    neighbor on the cell center's side of the the plane (hard part, maybe a bunch of k=50 queries)
     now do 1 radius query using the calculated center on line of motion so that the sphere
     passes through both the Line of Motion and the neighbor 1 and the new potential neighbor 2
     to verify that it actually is neighbor 2, and repeat the plane-search process to finde neighbor 3 and 4
     
    One way to find which side of a plane a point is on is here:
    https://math.stackexchange.com/questions/214187/point-on-the-left-or-right-side-of-a-plane-in-3d-space
    """                                    
    pass





