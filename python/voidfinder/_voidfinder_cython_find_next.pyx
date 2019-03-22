







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
                      DTYPE_INT64_t

from numpy.math cimport NAN, INFINITY

from libc.math cimport fabs, sqrt, asin, atan#, exp, pow, cos, sin, asin

#from libc.stdlib cimport malloc, free










@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef void find_next_galaxy(DTYPE_F64_t[:,:] hole_center_memview, 
                           DTYPE_F64_t[:,:] temp_hole_center_memview,
                           DTYPE_F64_t search_radius, 
                           DTYPE_F64_t dr, 
                           DTYPE_F64_t direction_mod,
                           DTYPE_F64_t[:] unit_vector_memview, 
                           galaxy_tree, 
                           DTYPE_INT64_t[:] nearest_gal_index_list, 
                           ITYPE_t num_neighbors,
                           DTYPE_F64_t[:,:] w_coord, 
                           DTYPE_B_t[:,:,:] mask, 
                           DTYPE_F64_t min_dist, 
                           DTYPE_F64_t max_dist, 
                           
                           
                           DTYPE_F64_t[:] Bcenter_memview,
                           
                           ITYPE_t[:] MAX_NEAREST,
                           ITYPE_t[:] i_nearest_reduced_memview_z,
                           DTYPE_F64_t[:,:] candidate_minus_A_memview_z,
                           DTYPE_F64_t[:,:] candidate_minus_center_memview_z,
                           DTYPE_F64_t[:] bot_memview_z,
                           DTYPE_F64_t[:] top_memview_z,
                           DTYPE_F64_t[:] x_ratio_memview_z,
                           
                           
                           ITYPE_t[:] nearest_neighbor_index,           #return variable
                           DTYPE_F64_t[:] min_x_ratio,                  #return variable
                           DTYPE_B_t[:] in_mask                         #return variable
                           
                           ): 
                           #) except *:              

    '''
    Description:
    ============
    Function to locate the next nearest galaxy during hole center propagation 
    along direction defined by unit_vector_memview.

    The algorithm needs to find 4 bounding galaxies per cell.  The 
    first galaxy is found as the minimum of regular euclidean distance to the hole center.  The
    second galaxy is determined using a ratio of distance with respect to projected distance
    along the hole center search path.  To find galaxy 2, the hole center search path and 
    distance ratios are calculated with respect to neighbor 1.  

    Implementation Detail:
    For galaxies 3 and 4, the 
    distance ratios are calculated with respect to the previous galaxies, so this code uses
    an if-block to detect whether we're calculating for galaxy 2/B, 3/C, or 4/D, and runs slightly
    different distance ratio calculations.  The if block looks for the number of elements
    in the input nearest_gal_index_list, when its 1 it assumes we're finding for galaxy 2
    and when its not 1 it assumes we have the first 2 galaxies and we're looking for galaxy
    3 or 4.



    Parameters:
    ===========

    hole_center_memview : memview of shape (1,3)
        x,y,z coordinate of current center of hole in units of Mpc/h

    search_radius : float
        Radius of hole in units of Mpc/h

    dr : float
        Incrememt value for hole propagation

    unit_vector_memview : memview of shape (3)
        Unit vector indicating direction hole center will shift

    galaxy_tree : sklearn KDTree
        Tree to query for nearest-neighbor results

    nearest_gal_index_list : memview of shape (N)
        List of row indices in w_coord for existing bounding galaxies
        
    num_neighbors : int
        number of valid neighbor indices in the nearest_gal_index_list object    

    w_coord : memview of shape (N_galaxies, 3)
        x,y,z coordinates of all galaxies in sample in units of Mpc/h

    mask : memview of shape (z_dim, ra_dim, dec_dim)
        uint8 array of whether location is within survey footprint

    min_dist : float
        minimum distance (redshift) in survey in units of Mpc/h

    max_dist : float
        maximum distance (redshift) in survey in units of Mpc/h


    Returns:
    ========

    nearest_neighbor_x_ratio_index : index
        Index value to nearest_neighbor_x_ratio array of next nearest neighbor

    nearest_neighbor_index : index
        Index value to w_coord array of next nearest neighbor

    min_x_ratio : float
        ???

    in_mask : boolean
        Flag indicating whether or not the temporary hole center is within the 
        survey footprint.
    '''



    # We are going to shift the center of the hole by dr along 
    # the direction of the vector pointing from the nearest 
    # galaxy to the center of the empty cell.  From there, we 
    # will search within a radius of length the distance between 
    # the center of the hole and the first galaxy from the 
    # center of the hole to find the next nearest neighbors.  
    # From there, we will minimize top/bottom to find which one 
    # is the next nearest galaxy that bounds the hole.



    '''
    i_nearest_reduced_memview = np.empty(num_nearest, dtype=np.int64)
                
    candidate_minus_A_memview = np.empty((num_nearest, 3), dtype=np.float64, order='C')

    candidate_minus_center_memview = np.empty((num_nearest, 3), dtype=np.float64, order='C')
    
    bot_memview = np.empty(num_nearest, dtype=np.float64, order='C')
    
    top_memview = np.empty(num_nearest, dtype=np.float64, order='C')
    
    x_ratio_memview = np.empty(num_nearest, dtype=np.float64, order='C')
    '''

    
    cdef ITYPE_t[:] i_nearest_reduced_memview
    
    cdef DTYPE_F64_t[:,:] candidate_minus_A_memview
    
    cdef DTYPE_F64_t[:,:] candidate_minus_center_memview
    
    cdef DTYPE_F64_t[:] bot_memview
    
    cdef DTYPE_F64_t[:] top_memview
    
    cdef DTYPE_F64_t[:] x_ratio_memview
    


    ############################################################################
    #
    #   DECLARATIONS
    #
    ############################################################################

    
    ############################################################################
    # helper index and calculation variables, re-useable
    ############################################################################
    cdef ITYPE_t idx
    
    cdef ITYPE_t jdx
    
    cdef ITYPE_t temp_idx
    
    cdef DTYPE_F64_t temp_f64_accum
    
    cdef DTYPE_F64_t temp_f64_val

    ############################################################################
    # Used in filtering exiting neighbors out of results
    ############################################################################
    
    cdef ITYPE_t[:] i_nearest_memview
    
    cdef ITYPE_t num_results
    
    cdef ITYPE_t num_nearest
    
    cdef DTYPE_B_t[:] boolean_nearest_memview
    
    ############################################################################
    #
    ############################################################################
    
    
    ############################################################################
    # Used in finding valid x ratio values
    ############################################################################
    cdef DTYPE_B_t any_valid = 0
            
    cdef ITYPE_t valid_min_idx
    
    cdef DTYPE_F64_t valid_min_val
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    

    ############################################################################
    #
    # Main galaxy finding loop.  Contains internal logic to use slightly different
    # processes depending on whether we're finding galaxy 2/B, 3/C, or 4/D.
    #
    # The general gist is that we take a hole center, move it in the direction
    # of the unit vector by steps of dr, then look at all the galaxies in the
    # sphere centered at the new hole center of larger radius until we find one
    # that bounds the sphere
    #
    ############################################################################
    
    cdef DTYPE_B_t galaxy_search = True

    while galaxy_search:


        ############################################################################
        # shift hole center
        ############################################################################
        for idx in range(3):

            temp_hole_center_memview[0, idx] = temp_hole_center_memview[0, idx] + direction_mod*dr*unit_vector_memview[idx]

        
        ############################################################################
        # calculate new search radius
        ############################################################################
        if num_neighbors == 1:
            
            search_radius += dr
        
        elif num_neighbors > 1:
            
            temp_f64_accum = 0.0
            
            for idx in range(3):
                
                temp_f64_val = w_coord[nearest_gal_index_list[0],idx] - temp_hole_center_memview[0,idx]
                
                temp_f64_accum += temp_f64_val*temp_f64_val
                
            search_radius = sqrt(temp_f64_accum)
            
        
        
        
        
        ############################################################################
        # use KDtree to find the galaxies within our target sphere
        ############################################################################

        i_nearest = galaxy_tree.query_radius(temp_hole_center_memview, r=search_radius)

        i_nearest = i_nearest[0]

        i_nearest_memview = i_nearest


        ############################################################################
        # The resulting galaxies may include galaxies we already found in previous
        # steps, so build a boolean index representing whether a resultant galaxy
        # is valid or not, and track how many valid result galaxies we actually have
        # for the next step.
        ############################################################################

        num_results = i_nearest_memview.shape[0]

        boolean_nearest_memview = np.ones(num_results, dtype=np.uint8)
        
        num_nearest = num_results

        for idx in range(num_results):

            for jdx in range(num_neighbors):
                
                if i_nearest_memview[idx] == nearest_gal_index_list[jdx]:

                    boolean_nearest_memview[idx] = 0
                    
                    num_nearest -= 1
                    
                    break
                
        ############################################################################
        # If we have any valid result galaxies, use the special x ratio distance
        # metric on them.  Note that metric is dependent on whether we are
        # searching for galaxy 2/B, 3/C or 4/D, so this code uses the
        # num_neighbors input parameter to tell which one we are looking for.
        # num_neighbors == 1 means we have 1 bounding galaxy so we're looking
        # for galaxy 2/B, if num_neighbors > 1 it means we have at least 2 galaxies
        # so we're looking for 3/C or 4/D which use the same process.
        ############################################################################
        if num_nearest > 0:
            
            
            '''
            if num_nearest > MAX_NEAREST[0]:
                
                #reallocate the memory for computation if not big enough
                
                i_nearest_reduced_memview = np.empty(num_nearest, dtype=np.int64)
                
                candidate_minus_A_memview = np.empty((num_nearest, 3), dtype=np.float64, order='C')

                candidate_minus_center_memview = np.empty((num_nearest, 3), dtype=np.float64, order='C')
                
                bot_memview = np.empty(num_nearest, dtype=np.float64, order='C')
                
                top_memview = np.empty(num_nearest, dtype=np.float64, order='C')
                
                x_ratio_memview = np.empty(num_nearest, dtype=np.float64, order='C')
                
                #print("REALLOCATE", num_nearest)
                
                MAX_NEAREST[0] = num_nearest
            '''
            i_nearest_reduced_memview = np.empty(num_nearest, dtype=np.int64)
                
            candidate_minus_A_memview = np.empty((num_nearest, 3), dtype=np.float64, order='C')

            candidate_minus_center_memview = np.empty((num_nearest, 3), dtype=np.float64, order='C')
            
            bot_memview = np.empty(num_nearest, dtype=np.float64, order='C')
            
            top_memview = np.empty(num_nearest, dtype=np.float64, order='C')
            
            x_ratio_memview = np.empty(num_nearest, dtype=np.float64, order='C')
            ############################################################################
            # copy the valid galaxy indicies into the i_nearest_reduced memory
            ############################################################################
            
            #i_nearest_reduced_memview = np.empty(num_nearest, dtype=np.int64)
            
            jdx = 0
            
            for idx in range(num_results):
                
                if boolean_nearest_memview[idx]:
                    
                    i_nearest_reduced_memview[jdx] = i_nearest_memview[idx]
                    
                    jdx += 1
            
            #i_nearest_reduced_memview = i_nearest_reduced
            
            
            ############################################################################
            # Calculate vectors pointing from hole center and galaxy 1/A to next 
            # nearest candidate galaxy
            ############################################################################
            #candidate_minus_A_memview = np.empty((num_nearest, 3), dtype=np.float64, order='C')

            #candidate_minus_center_memview = np.empty((num_nearest, 3), dtype=np.float64, order='C')
            
            for idx in range(num_nearest):

                temp_idx = i_nearest_reduced_memview[idx]

                for jdx in range(3):
                    
                    if num_neighbors == 1:
                        
                        candidate_minus_A_memview[idx, jdx] = w_coord[nearest_gal_index_list[0], jdx] - w_coord[temp_idx, jdx]
                        
                    else:

                        candidate_minus_A_memview[idx, jdx] = w_coord[temp_idx, jdx] - w_coord[nearest_gal_index_list[0], jdx]

                    candidate_minus_center_memview[idx, jdx] = w_coord[temp_idx, jdx] - hole_center_memview[0, jdx]


            ############################################################################
            # Calculate bottom of ratio to be minimized
            ############################################################################
            #bot_memview = np.empty(num_nearest, dtype=np.float64, order='C')
            
            for idx in range(num_nearest):
                
                temp_f64_accum = 0.0
                
                for jdx in range(3):
                    
                    temp_f64_accum += candidate_minus_A_memview[idx,jdx]*unit_vector_memview[jdx]
                    
                bot_memview[idx] = 2*temp_f64_accum
            
            
            ############################################################################
            # Calculate top of ratio to be minimized
            ############################################################################
            #top_memview = np.empty(num_nearest, dtype=np.float64, order='C')

            if num_neighbors == 1:

                for idx in range(num_nearest):
                
                    temp_f64_accum = 0.0
                    
                    for jdx in range(3):
                        
                        temp_f64_accum += candidate_minus_A_memview[idx,jdx]*candidate_minus_A_memview[idx,jdx]
                        
                    top_memview[idx] = temp_f64_accum

            else:

                for idx in range(3):

                    Bcenter_memview[idx] = w_coord[nearest_gal_index_list[1], idx] - hole_center_memview[0, idx]


                temp_f64_accum = 0.0
                
                for idx in range(3):
                    
                    temp_f64_accum += Bcenter_memview[idx]*Bcenter_memview[idx]
                    
                temp_f64_val = temp_f64_accum

                
                for idx in range(num_nearest):
                    
                    temp_f64_accum = 0.0
                    
                    for jdx in range(3):
                        
                        temp_f64_accum += candidate_minus_center_memview[idx, jdx]*candidate_minus_center_memview[idx, jdx]
                        
                    top_memview[idx] = temp_f64_accum - temp_f64_val



            ############################################################################
            # Calculate the minimization ratios
            ############################################################################
            #x_ratio_memview = np.empty(num_nearest, dtype=np.float64, order='C')

            for idx in range(num_nearest):

                x_ratio_memview[idx] = top_memview[idx]/bot_memview[idx]

            ############################################################################
            # Locate positive values of x_ratio
            ############################################################################
            any_valid = 0
            
            valid_min_idx = 0
            
            valid_min_val = INFINITY
            
            for idx in range(num_nearest):
                
                temp_f64_val = x_ratio_memview[idx]
                
                if temp_f64_val > 0.0:
                    
                    any_valid = 1
                    
                    if temp_f64_val < valid_min_val:
                        
                        valid_min_idx = idx
                        
                        valid_min_val = temp_f64_val
                        

            ############################################################################
            # If we found any positive values, we have a result, end the while loop
            # using galaxy_search = False
            ############################################################################
            if any_valid:
                
                nearest_neighbor_index[0] = i_nearest_reduced_memview[valid_min_idx]

                min_x_ratio[0] = x_ratio_memview[valid_min_idx]
                
                galaxy_search = False
            

        elif not_in_mask(temp_hole_center_memview, mask, min_dist, max_dist):
            
            galaxy_search = False

            in_mask[0] = False


    return









cdef DTYPE_F64_t RtoD = 180./np.pi
cdef DTYPE_F64_t DtoR = np.pi/180.
cdef DTYPE_F64_t dec_offset = -90


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef DTYPE_B_t not_in_mask(DTYPE_F64_t[:,:] coordinates, 
                  DTYPE_B_t[:,:,:] survey_mask_ra_dec, 
                  DTYPE_F64_t rmin, 
                  DTYPE_F64_t rmax):
    '''
    Determine whether a given set of coordinates falls within the survey.

    Parameters:
    ============

    coordinates : numpy.ndarray of shape (3,), in x-y-z order and cartesian coordinates
        x,y, and z are measured in Mpc/h

    survey_mask_ra_dec : numpy.ndarray of shape (num_ra, num_dec) where 
        the element at [i,j] represents whether or not the ra corresponding to
        i and the dec corresponding to j fall within the mask.  ra and dec
        are both measured in degrees.

    rmin, rmax : scalar, min and max values of survey distance in units of
        Mpc/h

    Returns:
    ========

    boolean : True if coordinates fall outside the survey_mask
    '''

    
    cdef DTYPE_F64_t r_sq
    cdef DTYPE_F64_t r
    
    
    cdef DTYPE_F64_t ra
    cdef DTYPE_F64_t dec
    
    cdef ITYPE_t n
    cdef DTYPE_F64_t n_float
    
    
    cdef ITYPE_t idx1
    cdef ITYPE_t idx2
    
    cdef DTYPE_F64_t coord_x
    cdef DTYPE_F64_t coord_y
    cdef DTYPE_F64_t coord_z
    
    cdef DTYPE_B_t return_mask_value

    
    coord_x = coordinates[0,0]
    coord_y = coordinates[0,1]
    coord_z = coordinates[0,2]
    
    
    #r = sqrt(coord_x*coord_x + coord_y*coord_y + coord_z*coord_z)
    
    r_sq = coord_x*coord_x + coord_y*coord_y + coord_z*coord_z

    if r_sq < rmin*rmin or r_sq > rmax*rmax:
        
        return True
    
    r = sqrt(r_sq)

    n = 1 + <ITYPE_t>(DtoR*r/10.)
    
    n_float = <DTYPE_F64_t>n
    
    #ra = np.arctan(coordinates[0,1]/coordinates[0,0])*RtoD
    
    ra = atan(coord_y/coord_x)*RtoD
    
    #dec = np.arcsin(coordinates[0,2]/r)*RtoD
    
    dec = asin(coord_z/r)*RtoD
    

    if coord_x < 0.0 and coord_y != 0.0:
        
        ra += 180.0
        
    if ra < 0:
        
        ra += 360.0
        
        
    idx1 = <ITYPE_t>(n_float*ra)
    
    idx2 = <ITYPE_t>(n_float*dec) - <ITYPE_t>(n_float*dec_offset)
    
    
    #return_mask_value = survey_mask_ra_dec[n-1][idx1][idx2]
    
    return_mask_value = survey_mask_ra_dec[n-1, idx1, idx2]
    
    
    
    if return_mask_value == 1:
        return_mask_value = 0
    elif return_mask_value == 0:
        return_mask_value = 1

    return return_mask_value






