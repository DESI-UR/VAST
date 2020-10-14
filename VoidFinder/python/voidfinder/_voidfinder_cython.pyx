




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

from libc.math cimport fabs, sqrt, asin, atan#, exp, pow, cos, sin, asin

from _voidfinder_cython_find_next cimport find_next_galaxy, \
                                          not_in_mask, \
                                          _query_first, \
                                          DistIdxPair, \
                                          Cell_ID_Memory, \
                                          GalaxyMapCustomDict, \
                                          HoleGridCustomDict, \
                                          FindNextReturnVal, \
                                          NeighborMemory

import time






@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
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
    
    This version of fill_ijk goes in zigzag/snake order in an attempt to preserve spatial locality of
    the array data while processing.  That is to say, on a 3x3 grid, this functions goes:
    (0,0,0), (0,0,1), (0,0,2), (0,1,2), (0,1,1), (0,1,0), (0,2,0), ...
    whaeras a normal ordering would be:
    (0,0,0), (0,0,1), (0,0,2), (0,1,0), (0,1,1), (0,1,2), (0,2,0)
    so that the next cell grid returned is always adjacent to both its predecessor and
    its antecedent
    
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






@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
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









@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
@cython.profile(True)
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
                          #DTYPE_F64_t hole_radial_mask_check_dist,
                          DTYPE_F64_t[:,:] return_array,
                          Cell_ID_Memory cell_ID_mem,
                          NeighborMemory neighbor_mem,
                          int verbose,
                          ):
    '''
    Description
    ===========
    
    Given a potential void cell center denoted by i,j,k, find the 4 bounding 
    galaxies that maximize the interior dimensions of the hole sphere at this 
    location.
    
    There are some really weird particulars to this algorithm that need to be 
    laid out in better detail.

    The code below will utilize the naming scheme galaxy A, B, C, and D to 
    denote the 1st, 2nd, 3rd, and 4th "neighbor" bounding galaxies found during 
    the running of this algorithm.  I tried to use 1/A, 2/B, 3/C and 4/D to be 
    clear on the numbers and letters are together.
    
    The distance metrics are somewhat special.  Galaxy A is found by normal 
    minimzation of euclidean distance between the cell center and itself and the 
    other neighbors.  Galaxies B, C, and D are found by propogating a hole 
    center in specific directions and minimizing a ratio of two other 
    distance-metric-like values.  This needs more detail on how and why.
    
    
    Parameters:
    ===========
    
    i_j_k_array : memoryview of (N,3) numpy array
        a batch of N cell IDs on which to run the VoidFinder hole-growing 
        algorithm
      
    galaxy_tree : python object
        a glorified wrapper around a couple of the data structures used by 
        VoidFinder
        
    w_coord : memoryview to (K,3) numpy array
        xyz space galaxy coordinates
        
    dl : float
        edge length in Mpc/h of the hole grid cell size
        
    dr : float
        Step-size for moving the hole center used while searching for the next 
        nearest galaxy. 
        
    coord_min : memoryview to (1,3) numpy array
        xyz-space coordinates of the (0,0,0) origin location of the hole grid 
        and galaxy map grids 
    
    mask : memoryview into (?,?) numpy array
        True when a location is inside the survey, False when a location is 
        outside the survey
        
    mask_resolution : int
        The length of the mask binning used to generate the survey mask, in 
        units of degrees. 
        
    min_dist : float
        Minimum distance of the galaxy distribution.
        
    max_dist : float
        Maximum distance of the galaxy distribution.
        
    DEPRECATED hole_radial_mask_check_dist : float
        fraction of hole radius at which to check the 6 directions +/- x,y,z to see
        if too much of the hole is outside the mask, and if so, discard the hole
        
    return_array : memoryview into (N,4) numpy array
        memory for results, each row is (x,y,z,r)
        
    cell_ID_mem : _voidfinder_cython_find_next.Cell_ID_Memory
        object which coordinates access to cell ID grid locations when seaching 
        for neighbor galaxies
        
    neighbor_mem : _voidfinder_cython_find_next.NeighborMemory
        object which coordinates access to a handful of arrays which may need to 
        be dynamically resized during nearest neighbor calculations
        
    verbose : int
        values >= 1 indicate to make a lot more print statements
    
    Returns:
    ========
    
    NAN or (x,y,z,r) rows filled into the return_array parameter.
    
    '''
    
    
    ############################################################################
    # re-used helper index variables
    # re-used helper computation variables
    ############################################################################
    
    
    #DEBUG_OUTFILE = open("VF_DEBUG.txt", 'a')
    
    
    cdef ITYPE_t working_idx
    
    cdef ITYPE_t idx
    
    cdef ITYPE_t jdx
    
    cdef ITYPE_t temp_idx
    
    
    cdef DTYPE_F64_t temp_f64_accum
    
    cdef DTYPE_F64_t temp_f64_accum2
    
    cdef DTYPE_F64_t temp_f64_val
    
    ############################################################################
    # Hole center vector and propogate hole center memory.  We can re-use the 
    # same memory for finding galaxies 2/B and 3/C but not for galaxy 4/D
    ############################################################################
    cdef DTYPE_F64_t[:,:] hole_center_memview = np.empty((1,3), dtype=np.float64, order='C')
    
    cdef DTYPE_F64_t[:,:] hole_center_2_3_memview = np.empty((1,3), dtype=np.float64, order='C')
    
    cdef DTYPE_F64_t[:,:] hole_center_41_memview = np.empty((1,3), dtype=np.float64, order='C')
    
    cdef DTYPE_F64_t[:,:] hole_center_42_memview = np.empty((1,3), dtype=np.float64, order='C')
    
    ############################################################################
    # The nearest_gal_index_list variable stores 2 things - the sentinel value 
    # -1 meaning 'no value here' or the index of the found galaxies A,B,C (but 
    # not D since it is the last one)
    ############################################################################
    cdef DTYPE_INT64_t[:] nearest_gal_index_list = np.empty(3, dtype=np.int64, order='C')
    

    ############################################################################
    # vector_modulus is re-used each time the new unit vector is calculated
    # unit_vector_memview is re-used each time the new unit vector is calculated
    # v3_memview is used in calculating the cross product for galaxy 4/D
    # hole_radius is used in some hole update calculations
    ############################################################################
    cdef DTYPE_F64_t vector_modulus
    
    cdef DTYPE_F64_t[:] unit_vector_memview = np.empty(3, dtype=np.float64, order='C')
    
    cdef DTYPE_F64_t[:] v3_memview = np.empty(3, dtype=np.float64, order='C')
    
    cdef DTYPE_F64_t hole_radius
    
    
    
    ############################################################################
    # Variables for the 4 galaxy seaarches whether the hole center is in the 
    # mask
    ############################################################################
    cdef DTYPE_B_t in_mask_2
    
    cdef DTYPE_B_t in_mask_3
    
    cdef DTYPE_B_t in_mask_41
    
    cdef DTYPE_B_t in_mask_42
    
    cdef DTYPE_B_t discard_mask_overlap
    
    
    ############################################################################
    # variables for the 4 bounding galaxies (gal 4/D gets 3 variables because it 
    # uses 2 searches and then picks one of the two based on some criteria)
    ############################################################################
    cdef ITYPE_t k1g
    
    cdef ITYPE_t k2g
    
    cdef ITYPE_t k3g
    
    cdef ITYPE_t k4g1
    
    cdef ITYPE_t k4g2
    
    cdef ITYPE_t k4g
    
    ############################################################################
    # minx3 is used in updating a hole center and minx41-42 are used in 
    # comparing to find galaxy 4/D
    ############################################################################
    cdef DTYPE_F64_t minx3
    
    cdef DTYPE_F64_t minx41
    
    cdef DTYPE_F64_t minx42
    
    ############################################################################
    # These are used in calcuating the unit vector for galaxy 4/D
    ############################################################################
    cdef DTYPE_F64_t[:] midpoint_memview = np.empty(3, dtype=np.float64, order='C')
    
    #cdef DTYPE_F64_t[:] Acenter_memview = np.empty(3, dtype=np.float64, order='C')
    
    cdef DTYPE_F64_t[:] AB_memview = np.empty(3, dtype=np.float64, order='C')
    
    cdef DTYPE_F64_t[:] BC_memview = np.empty(3, dtype=np.float64, order='C')
    
    
    ############################################################################
    # Computation memory to pass into find_next_galaxy()
    #
    # NOTE: This memory is currently unused because I need to figure out how to 
    #       reallocate it if it is not big enough without leaking memory and 
    #       causing a segfault
    ############################################################################
    cdef DTYPE_F64_t[:] Bcenter_memview = np.empty(3, dtype=np.float64, order='C')
    
    
    ############################################################################
    #
    ############################################################################
    cdef DistIdxPair query_vals
    
    cdef ITYPE_t num_cells = i_j_k_array.shape[0]
    
    cdef FindNextReturnVal find_next_retval
    
    
    ############################################################################
    # Main Loop - grow a hole in each of the N hole grid cell locations
    ############################################################################
    for working_idx in range(num_cells):
        
        
        ########################################################################
        # Re-init nearest gal index list on new cell
        ########################################################################
        for idx in range(3):
        
            nearest_gal_index_list[idx] = -1
    
        
        ########################################################################
        # Initialize the starting hole center based on the given i,j,k and grid 
        # spacing parameters, and then check to make sure it is in the survey
        ########################################################################
        hole_center_memview[0,0] = i_j_k_array[working_idx, 0]
        
        hole_center_memview[0,1] = i_j_k_array[working_idx, 1]
        
        hole_center_memview[0,2] = i_j_k_array[working_idx, 2]
        
        #print("Working ijk: "+str(i_j_k_array[working_idx, 0])+","+str(i_j_k_array[working_idx, 1])+","+str(i_j_k_array[working_idx, 2]), flush=True)
    
        
        for idx in range(3):
            
            hole_center_memview[0,idx] = (hole_center_memview[0,idx] + 0.5)*dl + coord_min[0,idx]
            
            
        if not_in_mask(hole_center_memview, mask, mask_resolution, min_dist, max_dist):
            
            '''
            out_str = ""
            out_str += str(i_j_k_array[working_idx, 0])
            out_str += ","
            out_str += str(i_j_k_array[working_idx, 1])
            out_str += ","
            out_str += str(i_j_k_array[working_idx, 2])
            out_str += " "
            out_str += "nim1"
            out_str += "\n"
            DEBUG_OUTFILE.write(out_str)
            '''
            
            return_array[working_idx, 0] = NAN
            
            return_array[working_idx, 1] = NAN
            
            return_array[working_idx, 2] = NAN
            
            return_array[working_idx, 3] = NAN
            
            continue
        
        
        ########################################################################
        #
        # Find Galaxy 1/A - super easy using _query_first
        #
        ########################################################################
        
        query_vals = _query_first(galaxy_tree.reference_point_ijk,
                                  galaxy_tree.coord_min,
                                  galaxy_tree.dl,
                                  galaxy_tree.shell_boundaries_xyz,
                                  galaxy_tree.cell_center_xyz,
                                  galaxy_tree.galaxy_map,
                                  galaxy_tree.galaxy_map_array,
                                  galaxy_tree.w_coord,
                                  cell_ID_mem,
                                  hole_center_memview)
        
        k1g = query_vals.idx
        
        vector_modulus = query_vals.dist
        
        ########################################################################
        #
        # Start Galaxy 2/B
        #
        # unit_vector_memview - unit vector hole propagation direction.  It 
        #     actually points TOWARDS the k1g galaxy right now since we are 
        #     doing k1g - hole_center in the calculation below
        #
        # modv1 - l2 norm modulus of v1_unit
        #
        ########################################################################
        
        for idx in range(3):
            
            unit_vector_memview[idx] = (w_coord[k1g,idx] - hole_center_memview[0,idx])/vector_modulus
            
        hole_radius = vector_modulus
        
        ########################################################################
        # Make a copy of the hole center for propagation during the galaxy 
        # finding set the nearest_gal_index_list first neighbor index since we 
        # have a neighbor now
        #
        # Also fill in neighbor 1/A into the existing neighbor idx list
        ########################################################################
        for idx in range(3):
            
            hole_center_2_3_memview[0,idx] = hole_center_memview[0,idx]
    
        ########################################################################
        #
        # Find galaxy 2/B
        #
        # Set the nearest neighbor 1 index in the list and init the in_mask to 1
        # Using direction_mod = -1.0 since the unit_vector points TOWARDS k1g
        # as per the above calculation
        #
        ########################################################################
        nearest_gal_index_list[0] = k1g
        
        find_next_retval = find_next_galaxy(hole_center_memview,
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
                                             cell_ID_mem,
                                             neighbor_mem,
                                             )
    
        k2g = find_next_retval.nearest_neighbor_index
        
        in_mask_2 = find_next_retval.in_mask
        
        if not in_mask_2:
            
            '''
            out_str = ""
            out_str += str(i_j_k_array[working_idx, 0])
            out_str += ","
            out_str += str(i_j_k_array[working_idx, 1])
            out_str += ","
            out_str += str(i_j_k_array[working_idx, 2])
            out_str += " "
            out_str += "nim2"
            out_str += "\n"
            DEBUG_OUTFILE.write(out_str)
            '''
            
        
            return_array[working_idx, 0] = NAN
            
            return_array[working_idx, 1] = NAN
            
            return_array[working_idx, 2] = NAN
            
            return_array[working_idx, 3] = NAN
            
            continue
        
        
        ########################################################################
        #
        # Start Galaxy 3/C
        #
        # When we found Galaxy 2/B, we only found the next closest neighbor as 
        # we iterated away from Galaxy 1/A.  Since we iterated by jumps of dr, 
        # we may not have found the exact center of the circle formed by the 
        # points k1g, k2g and the unit_vector
        #
        # Before finding Galaxy 3/C, we need to reinitialize our hole center by 
        # calculating the exact position of the center of the circle, and then 
        # check that that location is still within the mask.
        #
        # Since we know k1g and k2g are both distance R from the center of this 
        # circle, we can use the triangle formed by k1g, k2g and the center C of 
        # the circle, to first determine the value of R.  Then, we know the hole 
        # center C is going to be along the unit_vector direction a distance of 
        # R plus the k1g location (it is actually going to be 
        # k1g - R*unit_vector below because unit_vector is pointing in the 
        # opposite direction than we want).
        #
        # The triangle formed by k1g, k2g and C has sides, R, R, and |k1g-k2g|.  
        # If we allow the side of length R formed by k1g and C to be divided 
        # into 2 parts, we end up with 2 triangles:  (k1g, k2g, x) and (C,k2g,x) 
        # who share the edge (k2g, x) which is perpendicular to the line 
        # (k1g, C).  In triangle 1, call the line (k1g,x) to have length b, and 
        # b = the projection of k1g-k2g in the unit vector direction, or 
        # b = dot((k1g-k2g), unit_vector)
        #
        #          k2g
        #          /|-__
        # k1g-k2g / |   -__R
        #        /  |      -__
        #       /___|_________-_ C
        #   k1g   b
        #             R
        # By pythagorean theorem:  |k1g-k2g|^2 - b^2 = R^2 - (R-b)^2
        # We get:
        # 
        #     R = (|k1g-k2g|^2)/(2b) = (|k1g-k2g|^2)/(2(k1g-k2g).u)
        #
        # temp_f64_accum calculates |k1g-k2g|^2
        # temp_f64_accum2 calculates dot((k1g-k2g), unit_vector)
        #########################################################################
        
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
            
            '''
            out_str = ""
            out_str += str(i_j_k_array[working_idx, 0])
            out_str += ","
            out_str += str(i_j_k_array[working_idx, 1])
            out_str += ","
            out_str += str(i_j_k_array[working_idx, 2])
            out_str += " "
            out_str += "nim3"
            out_str += "\n"
            DEBUG_OUTFILE.write(out_str)
            '''
            
            
            
            return_array[working_idx, 0] = NAN
            
            return_array[working_idx, 1] = NAN
            
            return_array[working_idx, 2] = NAN
            
            return_array[working_idx, 3] = NAN
            
            continue
    
        ########################################################################
        # Define the new unit vector along which to move the hole center
        #
        # Use the line which runs through the midpoint of k1g and k2g, and the
        # center of the new circle we defined to define the new unit vector
        # direction.
        #
        # This time, the unit vector is pointing AWAY from the existing 
        # neighbors.
        ########################################################################
        
        for idx in range(3):
            
            midpoint_memview[idx] = 0.5*(w_coord[k1g,idx] + w_coord[k2g,idx])
        
        
        temp_f64_accum = 0.0
        
        for idx in range(3):
            
            temp_f64_val = hole_center_memview[0,idx] - midpoint_memview[idx]
            
            temp_f64_accum += temp_f64_val*temp_f64_val
        
        vector_modulus = sqrt(temp_f64_accum)
        
        
        for idx in range(3):
        
            unit_vector_memview[idx] = (hole_center_memview[0,idx] - midpoint_memview[idx])/vector_modulus
        
        
        ########################################################################
        # Copy the center of the circle formed by k1g and k2g that we found just 
        # a bit ago into the other memory, this will be our starting location 
        # for finding galaxy 3/C
        ########################################################################
        
        for idx in range(3):
            
            hole_center_2_3_memview[0,idx] = hole_center_memview[0,idx]
        
    
        ########################################################################
        #
        # Find galaxy 3/C
        #
        # Set neighbors 1 and 2 in the list
        # 
        ########################################################################
        nearest_gal_index_list[0] = k1g
        
        nearest_gal_index_list[1] = k2g
    
        find_next_retval = find_next_galaxy(hole_center_memview,
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
                                             cell_ID_mem,
                                             neighbor_mem,
                                             )
    
        k3g = find_next_retval.nearest_neighbor_index
        
        minx3 = find_next_retval.min_x_ratio
        
        in_mask_3 = find_next_retval.in_mask
        
        if not in_mask_3:
            
            
            '''
            out_str = ""
            out_str += str(i_j_k_array[working_idx, 0])
            out_str += ","
            out_str += str(i_j_k_array[working_idx, 1])
            out_str += ","
            out_str += str(i_j_k_array[working_idx, 2])
            out_str += " "
            out_str += "nim4"
            out_str += "\n"
            DEBUG_OUTFILE.write(out_str)
            '''
            
        
            return_array[working_idx, 0] = NAN
            
            return_array[working_idx, 1] = NAN
            
            return_array[working_idx, 2] = NAN
            
            return_array[working_idx, 3] = NAN
            
            continue
    
        ########################################################################
        #
        # Start Galaxy 4/D-1 (galaxy 4/D takes 2 attempts)
        #
        # Process is very similar as before, except we do not know if we have to 
        # move above or below the plane.  Therefore, we will find the next 
        # closest if we move above the plane, and the next closest if we move 
        # below the plane.
        #
        # Update hole center
        # update hole radius
        #
        ########################################################################
        for idx in range(3):
        
            hole_center_memview[0,idx] += minx3*unit_vector_memview[idx]
    
    
        temp_f64_accum = 0.0
    
        for idx in range(3):
    
            temp_f64_val = hole_center_memview[0,idx] - w_coord[k1g,idx]
    
            temp_f64_accum += temp_f64_val*temp_f64_val
    
        hole_radius = sqrt(temp_f64_accum)
        
    
        if not_in_mask(hole_center_memview, mask, mask_resolution, min_dist, max_dist):
        
            '''
            out_str = ""
            out_str += str(i_j_k_array[working_idx, 0])
            out_str += ","
            out_str += str(i_j_k_array[working_idx, 1])
            out_str += ","
            out_str += str(i_j_k_array[working_idx, 2])
            out_str += " "
            out_str += "nim5"
            out_str += "\n"
            DEBUG_OUTFILE.write(out_str)
            '''
        
        
        
            return_array[working_idx, 0] = NAN
            
            return_array[working_idx, 1] = NAN
            
            return_array[working_idx, 2] = NAN
            
            return_array[working_idx, 3] = NAN
            
            continue
    
        ########################################################################
        #
        # The vector along which to move the hole center is defined by the cross 
        # product of the vectors pointing between the three nearest galaxies.
        #
        # Calculate the cross product of the difference vectors, calculate the 
        # modulus of that vector and normalize to a unit vector.
        # 
        ########################################################################
    
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
    
        ########################################################################
        # Calculate vector pointing from the hole center to the neighbor 1/A
        # Calculate vector pointing from the hole center to the neighbor 2/B
        # Update new hole center for propagation
        ########################################################################
        for idx in range(3):
    
            hole_center_41_memview[0, idx] = hole_center_memview[0, idx]
            
    
        ########################################################################
        #
        # Find galaxy 4/D-1
        #
        # update the exiting neighbors 1/A, 2/B and 3/C in the 
        # nearest_gal_index_list
        #
        ########################################################################
        nearest_gal_index_list[0] = k1g
        
        nearest_gal_index_list[1] = k2g
        
        nearest_gal_index_list[2] = k3g
    
        find_next_retval = find_next_galaxy(hole_center_memview,
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
                                             cell_ID_mem,
                                             neighbor_mem,
                                             )
    
        k4g1 = find_next_retval.nearest_neighbor_index
        
        minx41 = find_next_retval.min_x_ratio
        
        in_mask_41 = find_next_retval.in_mask
    
        # Calculate potential new hole center
        if in_mask_41:
    
            for idx in range(3):
    
                hole_center_41_memview[0, idx] = hole_center_memview[0, idx] + minx41*unit_vector_memview[idx]

        ########################################################################
        #
        # Start galaxy 4/D-2
        #
        # Repeat same search, but shift the hole center in the other direction 
        # this time, so flip the v3_unit_memview in other direction.
        #
        ########################################################################
        for idx in range(3):
    
            unit_vector_memview[idx] *= -1.0
    
    
        minx42 = INFINITY
    
    
        for idx in range(3):
    
            hole_center_42_memview[0, idx] = hole_center_memview[0, idx]
    
        
        ########################################################################
        #
        # Find galaxy 4/D-2
        #
        # nearest_neighbor_gal_list already updated from galaxy 4/D-1
        # 
        ########################################################################
        find_next_retval = find_next_galaxy(hole_center_memview,
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
                                             cell_ID_mem,
                                             neighbor_mem,
                                             )
    
        k4g2 = find_next_retval.nearest_neighbor_index
        
        minx42 = find_next_retval.min_x_ratio
        
        in_mask_42 = find_next_retval.in_mask
        
        # Calculate potential new hole center
        if in_mask_42:
    
            for idx in range(3):
    
                hole_center_42_memview[0, idx] = hole_center_memview[0, idx] + minx42*unit_vector_memview[idx]
    
        
        
        ########################################################################
        # Figure out whether galaxy 4/D is 4/D-1 or 4/D-2
        #
        # Use the minx41 and minx42 variables to figure out which one is closer?
        # Then set the 4th galaxy index based on that and update the output hole 
        # center based on that.  Or, if the conditions are not filled because we 
        # left the survey, return NAN output
        ########################################################################
        
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
            
            
            
            '''
            out_str = ""
            out_str += str(i_j_k_array[working_idx, 0])
            out_str += ","
            out_str += str(i_j_k_array[working_idx, 1])
            out_str += ","
            out_str += str(i_j_k_array[working_idx, 2])
            out_str += " "
            out_str += "nim6"
            out_str += "\n"
            DEBUG_OUTFILE.write(out_str)
            '''
            
            
            
            return_array[working_idx, 0] = NAN
            
            return_array[working_idx, 1] = NAN
            
            return_array[working_idx, 2] = NAN
            
            return_array[working_idx, 3] = NAN
            
            continue
    
        
        ########################################################################
        # Now that we have all 4 bounding galaxies, calculate the hole radius
        ########################################################################
    
        temp_f64_accum = 0.0
    
        for idx in range(3):
    
            temp_f64_val = hole_center_memview[0, idx] - w_coord[k1g, idx]
    
            temp_f64_accum += temp_f64_val*temp_f64_val
    
        hole_radius = sqrt(temp_f64_accum)
    
    
    
    
        ########################################################################
        # 
        # DEPRECATED
        #
        # Finally, check 6 directions: +/- x,y,z with a proportion of the
        # radius to see if too much of this hole falls outside the mask
        # (This implements the "If 10% falls outside the mask, discard hole"
        # check)
        #
        # re-using hole_center_42_memview as empty memory
        #
        # We are removing this function and making a new implementation
        # outside of the _voidfinder_cython.main_algorithm 
        #
        ########################################################################
        '''
        discard_mask_overlap = check_mask_overlap(hole_center_memview,
                                                  hole_center_42_memview,
                                                   hole_radial_mask_check_dist,
                                                   hole_radius,
                                                   mask, 
                                                   mask_resolution, 
                                                   min_dist, 
                                                   max_dist)
        
        if discard_mask_overlap:
            
            return_array[working_idx, 0] = NAN
            
            return_array[working_idx, 1] = NAN
            
            return_array[working_idx, 2] = NAN
            
            return_array[working_idx, 3] = NAN
            
            continue
        '''
        
    
        ########################################################################
        # Passed all checks, write the valid (x,y,z,r) values!
        ########################################################################
    
    
    
        return_array[working_idx, 0] = hole_center_memview[0, 0]
        
        return_array[working_idx, 1] = hole_center_memview[0, 1]
        
        return_array[working_idx, 2] = hole_center_memview[0, 2]
        
        return_array[working_idx, 3] = hole_radius
        
        
        '''
        out_str = ""
        out_str += str(i_j_k_array[working_idx, 0])
        out_str += ","
        out_str += str(i_j_k_array[working_idx, 1])
        out_str += ","
        out_str += str(i_j_k_array[working_idx, 2])
        out_str += " "
        out_str += "hole"
        out_str += " "
        out_str += str(hole_center_memview[0, 0])
        out_str += ","
        out_str += str(hole_center_memview[0, 1])
        out_str += ","
        out_str += str(hole_center_memview[0, 2])
        out_str += ","
        out_str += str(hole_radius)
        out_str += "\n"
        DEBUG_OUTFILE.write(out_str)
        '''
        
    #DEBUG_OUTFILE.close()
        
    return




@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
@cython.profile(True)
cpdef DTYPE_B_t check_mask_overlap(DTYPE_F64_t[:,:] coordinates,
                                  DTYPE_F64_t[:,:] temp_coordinates,
                                  DTYPE_F64_t hole_radial_mask_check_dist,
                                  DTYPE_F64_t hole_radius,
                                  DTYPE_B_t[:,:] mask, 
                                  DTYPE_INT32_t mask_resolution,
                                  DTYPE_F64_t min_dist, 
                                  DTYPE_F64_t max_dist):
                                   
    cdef DTYPE_B_t discard
    cdef DTYPE_F64_t check_dist = hole_radius*hole_radial_mask_check_dist
    
    #Positive X direction
    temp_coordinates[0,0] = coordinates[0,0] + check_dist
    temp_coordinates[0,1] = coordinates[0,1]
    temp_coordinates[0,2] = coordinates[0,2]

    discard = not_in_mask(temp_coordinates, mask, mask_resolution, min_dist, max_dist)
    
    if discard:
        
        return discard
    
    #Negative X direction
    temp_coordinates[0,0] = coordinates[0,0] - check_dist

    discard = not_in_mask(temp_coordinates, mask, mask_resolution, min_dist, max_dist)
    
    if discard:
        
        return discard
    
    #Positive Y direction
    temp_coordinates[0,0] = coordinates[0,0] #reset X
    temp_coordinates[0,1] = coordinates[0,1] + check_dist

    discard = not_in_mask(temp_coordinates, mask, mask_resolution, min_dist, max_dist)
    
    if discard:
        
        return discard
    
    #Negative Y direction
    temp_coordinates[0,1] = coordinates[0,1] - check_dist

    discard = not_in_mask(temp_coordinates, mask, mask_resolution, min_dist, max_dist)
    
    if discard:
        
        return discard
    
    #Positive Z direction
    temp_coordinates[0,1] = coordinates[0,1] #reset Y
    temp_coordinates[0,2] = coordinates[0,2] + check_dist

    discard = not_in_mask(temp_coordinates, mask, mask_resolution, min_dist, max_dist)
    
    if discard:
        
        return discard
    
    #Negative Z direction
    temp_coordinates[0,2] = coordinates[0,2] - check_dist

    discard = not_in_mask(temp_coordinates, mask, mask_resolution, min_dist, max_dist)
    
    if discard:
        
        return discard
    
    return False
    


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
                                    ):

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





