







cimport cython
import numpy as np
cimport numpy as np
np.import_array()  # required in order to use C-API

'''
cdef extern from "complex.h" nogil:
    float crealf(float complex)
    double creal(double complex)
    long double creall(long double complex)


ctypedef np.complex128_t DTYPE_CP128_t
ctypedef np.complex64_t DTYPE_CP64_t
ctypedef np.float64_t DTYPE_F64_t  
ctypedef np.float32_t DTYPE_F32_t
ctypedef np.uint8_t DTYPE_B_t
ctypedef np.intp_t ITYPE_t  
ctypedef np.int32_t DTYPE_INT32_t
'''




from typedefs cimport DTYPE_CP128_t, DTYPE_CP64_t, DTYPE_F64_t, DTYPE_F32_t, DTYPE_B_t, ITYPE_t, DTYPE_INT32_t

from numpy.math cimport NAN, INFINITY


from libc.math cimport fabs, sqrt, asin, atan#, exp, pow, cos, sin, asin











#@cython.boundscheck(False)
#@cython.wraparound(False)

@cython.cdivision(True)
cdef void find_next_galaxy(DTYPE_F64_t[:,:] hole_center_memview, 
                           DTYPE_F64_t[:,:] moving_hole_center_memview,
                            DTYPE_F64_t hole_radius, 
                            DTYPE_F64_t dr, 
                            DTYPE_F64_t direction_mod,
                            DTYPE_F64_t[:] unit_vector_memview, 
                            galaxy_tree, 
                            ITYPE_t[:] nearest_gal_index_list, 
                            DTYPE_F64_t[:,:] w_coord, 
                            DTYPE_B_t[:,:,:] mask, 
                            DTYPE_F64_t min_dist, 
                            DTYPE_F64_t max_dist, \
                            ITYPE_t[:] nearest_neighbor_x_ratio_index, \
                            ITYPE_t[:] nearest_neighbor_index, \
                            DTYPE_F64_t[:] min_x_ratio, \
                            DTYPE_B_t[:] in_mask) except *:

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
    an if-block to detect whether we're calculating for galaxy 2, 3, or 4, and runs slightly
    different distance ratio calculations.  The if block looks for the number of elements
    in the input nearest_gal_index_list, when its 1 it assumes we're finding for galaxy 2
    and when its not 1 it assumes we have the first 2 galaxies and we're looking for galaxy
    3 or 4.



    Parameters:
    ===========

    hole_center_memview : memview of shape (1,3)
        x,y,z coordinate of current center of hole in units of Mpc/h

    hole_radius : float
        Radius of hole in units of Mpc/h

    dr : float
        Incrememt value for hole propagation

    unit_vector_memview : memview of shape (3)
        Unit vector indicating direction hole center will shift

    galaxy_tree : sklearn KDTree
        Tree to query for nearest-neighbor results

    nearest_gal_index_list : memview of shape (N)
        List of row indices in w_coord for existing bounding galaxies

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


    ############################################################################
    #
    #   DECLARATIONS
    #
    ############################################################################

    cdef DTYPE_B_t galaxy_search = True

    #cdef DTYPE_B_t in_mask = True

    cdef DTYPE_F64_t[:,:] temp_hole_center_memview = np.empty((1,3), dtype=np.float64, order='C')
    cdef DTYPE_F64_t[:,:] candidate_minus_center_memview
    cdef DTYPE_F64_t[:,:] candidate_minus_A_memview

    cdef DTYPE_F64_t[:] Bcenter_memview = np.empty(3, dtype=np.float64, order='C')

    cdef DTYPE_F64_t[:] x_ratio_memview

    cdef ITYPE_t[:] i_nearest_memview
    cdef ITYPE_t[:] i_nearest_reduced_memview

    cdef DTYPE_F64_t temp_f64_accum
    cdef DTYPE_F64_t temp_f64_val

    cdef DTYPE_F64_t search_radius

    cdef ITYPE_t idx
    cdef ITYPE_t jdx
    cdef ITYPE_t temp_idx

    cdef ITYPE_t num_results
    cdef ITYPE_t num_neighbors
    cdef ITYPE_t num_nearest

    #cdef ITYPE_t nearest_neighbor_x_ratio_index
    #cdef ITYPE_t nearest_neighbor_index

    #cdef DTYPE_F64_t min_x_ratio

    ############################################################################
    ############################################################################


    #print("NN idxs: ", np.asarray(nearest_gal_index_list,dtype=np.int64, order='C'))



    # Initialize temp_hole_center_memview

    for idx in range(3):

        temp_hole_center_memview[0, idx] = moving_hole_center_memview[0, idx]



    # Initialize search radius

    search_radius = hole_radius




    # First move in the direction of the unit vector

    while galaxy_search:


        #-----------------------------------------------------------------------
        # Shift hole center along unit vector

        for idx in range(3):

            temp_hole_center_memview[0, idx] = temp_hole_center_memview[0, idx] + direction_mod*dr*unit_vector_memview[idx]

        #-----------------------------------------------------------------------
        ########################################################################



        #-----------------------------------------------------------------------
        # New hole "radius"

        #search_radius += dr
        
        
        
        num_neighbors = nearest_gal_index_list.shape[0]
        
        if num_neighbors == 1:
            
            search_radius += dr
        
        elif num_neighbors > 1:
            
            temp_f64_accum = 0.0
            
            for idx in range(3):
                
                temp_f64_val = w_coord[nearest_gal_index_list[0],idx] - temp_hole_center_memview[0,idx]
                
                temp_f64_accum += temp_f64_val*temp_f64_val
                
            search_radius = sqrt(temp_f64_accum)
            
        
        
        

        #-----------------------------------------------------------------------
        ########################################################################



        #-----------------------------------------------------------------------
        # Search for nearest neighbors within R of the hole center
        i_nearest = galaxy_tree.query_radius(temp_hole_center_memview, r=search_radius)

        i_nearest = i_nearest[0]

        i_nearest_memview = i_nearest

        #-----------------------------------------------------------------------
        ########################################################################



        #-----------------------------------------------------------------------
        # Remove nearest galaxies from list

        num_results = i_nearest_memview.shape[0]

        num_neighbors = nearest_gal_index_list.shape[0]


        boolean_nearest = np.ones(num_results, dtype=np.uint8)
        
        
        num_nearest = num_results
        
        #print("Num results: ", num_results)
        #print(np.asarray(i_nearest_memview, dtype=np.int64, order='C'))

        for idx in range(num_results):

            for jdx in range(num_neighbors):
                
                
                #print("NN idx check: ", i_nearest_memview[idx], nearest_gal_index_list[jdx])

                if i_nearest_memview[idx] == nearest_gal_index_list[jdx]:

                    boolean_nearest[idx] = False
                    
                    num_nearest -= 1
                    
                    #print("fail", num_nearest)
                    
                    break
                
                #print("succeed")

        
        #i_nearest = i_nearest[boolean_nearest]
        #-----------------------------------------------------------------------
        ########################################################################



        #num_nearest is int of ITYPE_t
        #num_nearest = i_nearest_reduced_memview.shape[0]

        if num_nearest > 0:
            # Found at least one other nearest neighbor!
            
            i_nearest_reduced = np.empty(num_nearest, dtype=np.int64)
            
            jdx = 0
            
            #print("building output: ")
            #print(np.asarray(boolean_nearest, dtype=np.uint8, order='C'))
            
            for idx in range(num_results):
                
                
                #print("idx: ", idx, boolean_nearest[idx])
                
                if boolean_nearest[idx]:
                    
                    
                    #print("succeed: ", i_nearest_memview[idx])
                    
                    i_nearest_reduced[jdx] = i_nearest_memview[idx]
                    
                    jdx += 1
                    
                    #print("succeed")
                    
                    
            #print("i_nearest_reduced")
            #print(i_nearest_reduced)
            
            i_nearest_reduced_memview = i_nearest_reduced
            
            #print(np.asarray(i_nearest_reduced_memview, dtype=np.int64, order='C'))


            #-------------------------------------------------------------------
            # Calculate vectors pointing from hole center and galaxy 1/A to next nearest candidate galaxy

            candidate_minus_A_memview = np.empty((num_nearest, 3), dtype=np.float64, order='C')

            candidate_minus_center_memview = np.empty((num_nearest, 3), dtype=np.float64, order='C')
            
            for idx in range(num_nearest):

                temp_idx = i_nearest_reduced_memview[idx]

                for jdx in range(3):
                    
                    
                    if num_neighbors == 1:
                        
                        
                        candidate_minus_A_memview[idx, jdx] = w_coord[nearest_gal_index_list[0], jdx] - w_coord[temp_idx, jdx]
                        
                    else:

                        candidate_minus_A_memview[idx, jdx] = w_coord[temp_idx, jdx] - w_coord[nearest_gal_index_list[0], jdx]

                    candidate_minus_center_memview[idx, jdx] = w_coord[temp_idx, jdx] - hole_center_memview[0, jdx]

            #-------------------------------------------------------------------
            ####################################################################
            
            


            #-------------------------------------------------------------------
            # Calculate bottom of ratio to be minimized

            bot_memview = np.empty(num_nearest, dtype=np.float64, order='C')
            
            for idx in range(num_nearest):
                
                temp_f64_accum = 0.0
                
                for jdx in range(3):
                    
                    temp_f64_accum += candidate_minus_A_memview[idx,jdx]*unit_vector_memview[jdx]
                    
                bot_memview[idx] = 2*temp_f64_accum
            
            #-------------------------------------------------------------------
            ####################################################################





            #-------------------------------------------------------------------
            # Calculate top of ratio to be minimized
            
            top_memview = np.empty(num_nearest, dtype=np.float64, order='C')


            if num_neighbors == 1:

                for idx in range(num_nearest):
                
                    temp_f64_accum = 0.0
                    
                    for jdx in range(3):
                        
                        temp_f64_accum += candidate_minus_A_memview[idx,jdx]*candidate_minus_A_memview[idx,jdx]
                        
                    top_memview[idx] = temp_f64_accum


            else:

                #---------------------------------------------------------------

                for idx in range(3):

                    Bcenter_memview[idx] = w_coord[nearest_gal_index_list[1], idx] - hole_center_memview[0, idx]

                #---------------------------------------------------------------



                #---------------------------------------------------------------

                temp_f64_accum = 0.0
                
                for idx in range(3):
                    
                    temp_f64_accum += Bcenter_memview[idx]*Bcenter_memview[idx]
                    
                temp_f64_val = temp_f64_accum

                #---------------------------------------------------------------



                #---------------------------------------------------------------
                
                for idx in range(num_nearest):
                    
                    temp_f64_accum = 0.0
                    
                    for jdx in range(3):
                        
                        temp_f64_accum += candidate_minus_center_memview[idx, jdx]*candidate_minus_center_memview[idx, jdx]
                        
                    top_memview[idx] = temp_f64_accum - temp_f64_val

                #---------------------------------------------------------------
            
            #-------------------------------------------------------------------
            ####################################################################




            #-------------------------------------------------------------------

            x_ratio_memview = np.empty(num_nearest, dtype=np.float64, order='C')

            for idx in range(num_nearest):

                x_ratio_memview[idx] = top_memview[idx]/bot_memview[idx]

            #-------------------------------------------------------------------
            ####################################################################




            #-------------------------------------------------------------------
            # Locate positive values of x_ratio
            
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

            #-------------------------------------------------------------------
            ####################################################################




            #-------------------------------------------------------------------
            
            if any_valid:
                
                # used to index into the x_ratio distance array
                nearest_neighbor_x_ratio_index[0] = valid_min_idx
                
                # used to index into the w_coord array
                #print("result: ")
                #print(valid_min_idx)
                #print(np.asarray(i_nearest_reduced_memview, dtype=np.int64, order='C'))
                
                
                nearest_neighbor_index[0] = i_nearest_reduced_memview[valid_min_idx]

                # ???????
                min_x_ratio[0] = x_ratio_memview[nearest_neighbor_x_ratio_index[0]]
                
                galaxy_search = False
            
            #-------------------------------------------------------------------
            ####################################################################



        
        #-----------------------------------------------------------------------

        elif not_in_mask(temp_hole_center_memview, mask, min_dist, max_dist):
            # Hole is no longer within survey limits

            galaxy_search = False

            in_mask[0] = False

        #-----------------------------------------------------------------------
        ########################################################################


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



    #print("Coordinates")
    #print(coordinates)
    #print(type(coordinates))
    #print(coordinates.shape)
    
    
    
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

    #coords = coordinates[0]  # Convert shape from (1,3) to (3,)
    
    #r = np.linalg.norm(coordinates[0,:])
    
    coord_x = coordinates[0,0]
    coord_y = coordinates[0,1]
    coord_z = coordinates[0,2]
    
    
    r = sqrt(coord_x*coord_x + coord_y*coord_y + coord_z*coord_z)
    
    
    
    
    

    if r < rmin or r > rmax:
        
        return True


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






