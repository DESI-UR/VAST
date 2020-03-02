



from __future__ import print_function




cimport cython

from cpython.mem cimport PyMem_Malloc, PyMem_Realloc, PyMem_Free

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
                      DTYPE_UINT16_t, \
                      CELL_ID_t

from numpy.math cimport NAN, INFINITY

from libc.math cimport fabs, sqrt, asin, atan, ceil#, exp, pow, cos, sin, asin

#from libc.stdlib cimport malloc, free


import time







@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
@cython.profile(True)
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
                           DTYPE_B_t[:,:] mask, 
                           DTYPE_INT32_t mask_resolution,
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
                           Cell_ID_Memory cell_ID_mem,
                           
                           
                           ITYPE_t[:] nearest_neighbor_index,           #return variable
                           DTYPE_F64_t[:] min_x_ratio,                  #return variable
                           DTYPE_B_t[:] in_mask,                         #return variable
                           #DTYPE_F64_t[:] PROFILE_kdtree_time
                           
                           ) except *: 
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
        
    NEEDS DEFINITION temp_hole_center_memview :

    search_radius : float
        Radius of hole in units of Mpc/h

    dr : float
        Incrememt value for hole propagation
        
    direction_mod : -1.0 or 1.0
        basically a switch to go in the opposite direction of vector propagation for
        finding galaxies 4a and 4b

    unit_vector_memview : memview of shape (3)
        Unit vector indicating direction hole center will shift

    galaxy_tree : custom thingy
        Data structure to query for nearest-neighbor results

    nearest_gal_index_list : memview of shape (N)
        List of row indices in w_coord for existing bounding galaxies
        
    num_neighbors : int
        number of valid neighbor indices in the nearest_gal_index_list object    

    w_coord : memview of shape (N_galaxies, 3)
        x,y,z coordinates of all galaxies in sample in units of Mpc/h

    mask : memview of shape (ra_dim, dec_dim)
        uint8 array of whether location is within survey footprint

    mask_resolution : integer
        Scale factor of coordinates used to index mask

    min_dist : float
        minimum distance (redshift) in survey in units of Mpc/h

    max_dist : float
        maximum distance (redshift) in survey in units of Mpc/h
        
    NEEDS DEFINITION Bcenter_memview : 
        some memory for ???
                           
    DEPRECATED  MAX_NEAREST : int
        represented number of slots for memory for holding nearest neighbor
        indices
    
    NEEDS DEFINITION i_nearest_reduced_memview_z :
    
    NEEDS DEFINITION candidate_minus_A_memview_z : 
    
    NEEDS DEFINITION candidate_minus_center_memview_z :
     
    NEEDS DEFINITION bot_memview_z :
    
    NEEDS DEFINITION top_memview_z :
    
    NEEDS DEFINITION x_ratio_memview_z :
    
    Cell_ID_Memory cell_ID_mem,
   
   
    ITYPE_t[:] nearest_neighbor_index,           #return variable
    DTYPE_F64_t[:] min_x_ratio,                  #return variable
    DTYPE_B_t[:] in_mask,      
        


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
    # Used in finding valid x ratio values
    ############################################################################
    cdef DTYPE_B_t any_valid = 0
            
    cdef ITYPE_t valid_min_idx
    
    cdef DTYPE_F64_t valid_min_val
    
    
    
    
    
    
    
    
    ############################################################################
    # PROFILING VARIABLES
    ############################################################################
    
    
    #cdef DTYPE_F64_t PROFILE_kdtree_time_collect
    
    
    
    
    
    
    
    

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

        dr *= 1.1
        
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
            
        
        
        
        #print("Check1")
        ############################################################################
        # use KDtree to find the galaxies within our target sphere
        ############################################################################

        #PROFILE_kdtree_time_collect = time.time()

        #i_nearest = galaxy_tree.query_radius(temp_hole_center_memview, r=search_radius) #sklearn
        #i_nearest = galaxy_tree.query_ball_point(temp_hole_center_memview, search_radius) #scipy
        i_nearest_memview = _query_shell_radius(galaxy_tree.reference_point_ijk,
                                                galaxy_tree.w_coord,
                                                galaxy_tree.coord_min,
                                                galaxy_tree.dl, 
                                                galaxy_tree.galaxy_map,
                                                galaxy_tree.galaxy_map_array,
                                                cell_ID_mem,
                                                temp_hole_center_memview, 
                                                search_radius) #custom
        
        
        #i_nearest = galaxy_kdtree.query_radius(temp_hole_center_memview, r=search_radius)
        
        
        #for idx, element in enumerate(i_nearest[0]):
            
        #    if i_nearest[idx] != i_nearest_memview[idx]:
                
        #        print("Radius query mismatch")
        
        
        #PROFILE_kdtree_time[0] += time.time() - PROFILE_kdtree_time_collect
        
        
        
        #i_nearest = i_nearest[0] #sklearn kdtree and scipy kdtree not custom
        
        
        #print("I NEAREST: ", len(i_nearest), flush=True)
        
        
        #if i_nearest: #added for scipy KDTree

                
    
        #i_nearest_memview = i_nearest #sklearn kdtree
        #i_nearest_memview = np.array(i_nearest) #added cast to numpy.array for scipy KDTree

        
        #print("Check2")
        
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
                
        #else:#added if-else for scipy KDTree
        #    num_nearest = 0 #added for scipy KDTree
        #print("Check3")
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
            

        elif not_in_mask(temp_hole_center_memview, mask, mask_resolution, min_dist, max_dist):
            
            galaxy_search = False

            in_mask[0] = False
            
        #print("Check4")


    return









cdef DTYPE_F64_t RtoD = 180./np.pi
cdef DTYPE_F64_t DtoR = np.pi/180.
cdef DTYPE_F64_t dec_offset = -90


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
@cython.profile(True)
cdef DTYPE_B_t not_in_mask(DTYPE_F64_t[:,:] coordinates, 
                  DTYPE_B_t[:,:] survey_mask_ra_dec, 
                  DTYPE_INT32_t n,
                  DTYPE_F64_t rmin, 
                  DTYPE_F64_t rmax) except *:
    '''
    Description
    ===========
    
    Determine whether a given set of coordinates falls within the survey.  Since 
    survey_mask_ra_dec is only used to index the ra (Right Ascension) and dec (Declination)
    of the survey, rmin and rmax are provided as proxies to the minimum and maximum
    Redshift (commonly denoted by z).  In the following code, the radius of the xyz position
    is calculated and compared to rmin and rmax to check against redshift, then the
    ra and dec of the xyz position are calculated.  The ra and dec of the target
    position are then scaled by n, to account for upsampling or downsampling in the
    survey mask, and the integer part of those ra and dec values form the index
    into the survey mask.

    Parameters
    ==========

    coordinates : numpy.ndarray of shape (1,3), 
        Coordinates of a point to check in x,y,z cartesian coordinates
        x, y, and z are measured in Mpc/h

    survey_mask_ra_dec : numpy.ndarray of shape (num_ra, num_dec) 
        the element at [i,j] represents whether or not the scaled ra corresponding to
        i and the scaled dec corresponding to j fall within the survey mask.  RA and dec
        are both measured in degrees and scaled by n.

    n : integer
        Scale factor of coordinates used to index survey_mask_ra_dec

    rmin, rmax : scalar
        min and max values of the survey in x,y,z units of Mpc/h
        Note these values form a proxy for redshift

    Returns
    =======

    return_mask_value : bool
        True if coordinates fall outside the survey mask "not in mask"
        and False if the coordinates lie inside the mask
    '''

    ######################################################################
    # Initialize some cdef memory for our variables
    ######################################################################
    cdef DTYPE_F64_t r_sq
    cdef DTYPE_F64_t r
    
    
    cdef DTYPE_F64_t ra
    cdef DTYPE_F64_t dec
    
    cdef DTYPE_F64_t n_float
    
    
    cdef ITYPE_t idx1
    cdef ITYPE_t idx2
    
    cdef DTYPE_F64_t coord_x
    cdef DTYPE_F64_t coord_y
    cdef DTYPE_F64_t coord_z
    
    cdef DTYPE_B_t return_mask_value

    ######################################################################
    # Unpack our target coordinate into the cdef variables and check
    # in x,y,z space if we've exceeded rmin or rmax
    ######################################################################
    coord_x = coordinates[0,0]
    coord_y = coordinates[0,1]
    coord_z = coordinates[0,2]
    
    r_sq = coord_x*coord_x + coord_y*coord_y + coord_z*coord_z

    if r_sq < rmin*rmin or r_sq > rmax*rmax:
        
        return True

    ######################################################################
    # We'll need to scale by n later, so create a float version of it
    ######################################################################
    n_float = <DTYPE_F64_t>n

    ######################################################################
    # Now calculate the ra and dec of the current point, and convert
    # them into 
    #
    # Double check with Kelly - what is the allowed range of ra and dec 
    # here?
    #
    ######################################################################
    r = sqrt(r_sq)
    
    ra = atan(coord_y/coord_x)*RtoD
    
    dec = asin(coord_z/r)*RtoD
    
    if coord_x < 0.0 and coord_y != 0.0:
        
        ra += 180.0
        
    if ra < 0:
        
        ra += 360.0
        
    ######################################################################
    # Index into the mask by taking the integer part of the scaled
    # ra and dec
    ######################################################################
    idx1 = <ITYPE_t>(n_float*ra)
    
    idx2 = <ITYPE_t>(n_float*dec) - <ITYPE_t>(n_float*dec_offset)
    
    return_mask_value = survey_mask_ra_dec[idx1, idx2]
    
    ######################################################################
    # Since the mask tells us if we're "in" the mask, and we want to
    # return whether we're "not in" the mask, flip the value we got,
    # then return.
    ######################################################################
    
    if return_mask_value == 1:
        return_mask_value = 0
    elif return_mask_value == 0:
        return_mask_value = 1

    return return_mask_value

'''
put this in pxd file

cdef packed struct LOOKUPMEM_t
    DTYPE_B_t filled_flag
    CELL_ID_t key_i, key_j, key_k
    DTYPE_INT64_t offset, num_elements
    
          
cdef struct OffsetNumPair:
    DTYPE_INT64_t offset, num_elements   
    
'''

cdef class GalaxyMapCustomDict:
    """
    
    layout of lookup_memory:
    
    lookup_memory = numpy.zeros(next_prime, dtype=[("filled_flag", numpy.uint8, 1),
                                                   ("ijk", numpy.uint16, 1),
                                                   ("j", numpy.uint16, 1),
                                                   ("k", numpy.uint16, 1),
                                                   ("offset", numpy.int64, 1),
                                                   ("num_elements", numpy.int64, 1])
                                                   
    23 bytes per element - 1 + 2*3 + 8*2 = 23
    
    Since ijk are limited to uint16 right now that means a maximum grid of
    shape (65536, 65536, 65536), or 2.8*10^14 (2^48) grid locations.  
                                            
    cdef DTYPE_INT64_t i_dim, j_dim, k_dim, jk_mod
    cdef LOOKUPMEM_t[:] lookup_memory
    cdef DTYPE_INT64_t mem_length
    """

    def __init__(self, 
                 grid_dimensions, 
                 lookup_memory):
            
            self.i_dim = grid_dimensions[0]
            self.j_dim = grid_dimensions[1]
            self.k_dim = grid_dimensions[2]
            
            self.jk_mod = self.j_dim*self.k_dim
            
            #print("GALAXYMAPCUSTOMDICT: ", np.asarray(self.lookup_memory))
            #print("GALAXYMAPCUSTOMDICT: ", lookup_memory.shape, lookup_memory.dtype)
            
            self.lookup_memory = lookup_memory
            
            
            #self.lookup_memory = np.zeros(5000, dtype=[("filled_flag", np.uint8, 1),
            #                                              ("ijk", np.uint16, 3),
            #                                              ("offset_and_num", np.int64, 2)])
            
            self.mem_length = lookup_memory.shape[0]
            
            self.num_collisions = 0
            
            
    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.cdivision(True)
    cdef DTYPE_INT64_t custom_hash(self, 
                                   CELL_ID_t i, 
                                   CELL_ID_t j, 
                                   CELL_ID_t k):
        """
        Given a cell ID (i,j,k), calculate its hash address in our
        memory array.  Uses the natural grid cell ordering
        0->(0,0,0), 1->(0,0,1), 2->(0,0,2) etc, to calculate the sequential
        grid cell index, then takes that value modulus of the number of slots
        in the memory array.
        """
        
        cdef DTYPE_INT64_t index
        cdef DTYPE_INT64_t hash_addr
        
        index = self.jk_mod * <DTYPE_INT64_t>i + \
                self.k_dim * <DTYPE_INT64_t>j + \
                <DTYPE_INT64_t>k
        
        hash_addr = index % self.mem_length
        
        return hash_addr
    
    
    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.cdivision(True)
    cpdef DTYPE_B_t contains(self,
                                 CELL_ID_t i, 
                                 CELL_ID_t j, 
                                 CELL_ID_t k):
        
        cdef DTYPE_INT64_t hash_addr
        
        cdef DTYPE_INT64_t hash_offset
        
        cdef DTYPE_INT64_t curr_hash_addr
        
        cdef DTYPE_B_t mem_flag
        
        cdef LOOKUPMEM_t curr_element
        
        hash_addr = self.custom_hash(i, j, k)
        
        for hash_offset in range(self.mem_length):
            
            curr_hash_addr = (hash_addr + hash_offset) % self.mem_length
            
            curr_element = self.lookup_memory[curr_hash_addr]
            
            #mem_flag = curr_element.filled_flag
            
            if not curr_element.filled_flag:
                
                return False
            
            else:
                
                #mem_key = self.lookup_memory[curr_hash_addr][1]
                
                if curr_element.key_i == i and \
                   curr_element.key_j == j and \
                   curr_element.key_k == k:
                    
                    return True
                        
        return False
    
    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.cdivision(True)
    cpdef OffsetNumPair getitem(self,
                                CELL_ID_t i, 
                                CELL_ID_t j, 
                                CELL_ID_t k):
        
        cdef DTYPE_INT64_t hash_addr
        
        cdef DTYPE_INT64_t hash_offset
        
        cdef DTYPE_INT64_t curr_hash_addr
        
        cdef DTYPE_B_t mem_flag
        
        cdef LOOKUPMEM_t curr_element
        
        #cdef DTYPE_INT64_t return_offset
        
        #cdef DTYPE_INT64_t return_num
        
        cdef OffsetNumPair out
        
        hash_addr = self.custom_hash(i, j, k)
        
        for hash_offset in range(self.mem_length):
            
            curr_hash_addr = (hash_addr + hash_offset) % self.mem_length
            
            curr_element = self.lookup_memory[curr_hash_addr]
            
            #mem_flag = self.lookup_memory[curr_hash_addr][0]
            
            if not curr_element.filled_flag:
                
                raise KeyError("key: ", i, j, k, " not in dictionary")
            
            else:
                
                #mem_key = self.lookup_memory[curr_hash_addr][1]
                
                #if numpy.all(mem_key == numpy.array(ijk).astype(numpy.uint16)):
                if curr_element.key_i == i and \
                   curr_element.key_j == j and \
                   curr_element.key_k == k:
                    
                    out.offset = curr_element.offset
                    out.num_elements = curr_element.num_elements
                    
                    #return_offset = curr_element.offset
                    #return_num = curr_element.num_elements
                    
                    return out
                
        raise KeyError("key: ", i, j, k, " not in dictionary")
    
    
    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.cdivision(True)
    cpdef void setitem(self, 
                           CELL_ID_t i,
                           CELL_ID_t j,
                           CELL_ID_t k, 
                           DTYPE_INT64_t offset,
                           DTYPE_INT64_t num_elements):
        """
        Will always succeed since we initialize the length of
        self.lookup_memory to be longer than the number of items
        """
        
        cdef DTYPE_INT64_t hash_addr
        
        cdef DTYPE_INT64_t hash_offset
        
        cdef DTYPE_INT64_t curr_hash_addr
        
        cdef DTYPE_B_t mem_flag
        
        cdef LOOKUPMEM_t curr_element
        
        cdef LOOKUPMEM_t out_element
        
        cdef DTYPE_B_t first_try = True
        
        out_element.filled_flag = 1
        out_element.key_i = i
        out_element.key_j = j
        out_element.key_k = k
        out_element.offset = offset
        out_element.num_elements = num_elements
                
        
        hash_addr = self.custom_hash(i, j, k)
        
        
        
        for hash_offset in range(self.mem_length):
            
            curr_hash_addr = (hash_addr + hash_offset) % self.mem_length
            
            curr_element = self.lookup_memory[curr_hash_addr]
            
            #mem_flag = self.lookup_memory[curr_hash_addr][0]
            
            #if not mem_flag:
            if not curr_element.filled_flag:
                
                if not first_try:
                    
                    self.num_collisions += 1
                
                self.lookup_memory[curr_hash_addr] = out_element
                
                return
            
            else:
                
                #mem_key = self.lookup_memory[curr_hash_addr][1]
                
                #if numpy.all(mem_key == numpy.array(ijk).astype(numpy.uint16)):
                if curr_element.key_i == i and \
                   curr_element.key_j == j and \
                   curr_element.key_k == k:
                    
                    if not first_try:
                    
                        self.num_collisions += 1
                    
                    self.lookup_memory[curr_hash_addr] = out_element
                
                    return
                
            first_try = False
    
    
    
    
    


cdef class GalaxyMap:
    
    
    '''
    cdef DTYPE_F64_t[:,:] w_coord
    
    cdef DTYPE_F64_t[:,:] coord_min
    
    cdef DTYPE_F64_t dl
    
    cdef DTYPE_INT64_t[:,:] reference_point_ijk
    
    cdef DTYPE_F64_t[:,:] shell_boundaries_xyz
    
    #cdef DTYPE_F64_t[:,:] min_containing_dist_mem
    
    cdef DTYPE_F64_t[:,:] cell_center_xyz
    
    cdef DTYPE_F64_t temp1
    
    cdef DTYPE_F64_t temp2
    
    cdef public dict nonvoid_cell_ID_dict
    
    cdef public dict galaxy_map
    '''
    
    def __init__(self, 
                 w_coord, 
                 coord_min, 
                 dl,
                 galaxy_map,
                 galaxy_map_array):
        """
        w_coord is the 3d positions of all galaxies
        
        coord_min is some magic normalizer
        
        dl is the grid side length
        
        
        The logic here is going to supercede the 
        "mesh_galaxies_dict" function from voidfinder_functions.py
        
        
        This class is going to use names suffixed with _ijk and
        with _xyz to make note of the two working spaces, ijk refers to
        the cell ID space, and xyz refers to the euclidean space that
        the galaxies live in
        
        
        
        TODO:  Appropriately address the existing reliance on casting to
               an integer. I believe python always rounds towards 0, and we
               are implicitly relying on this behavior.  Let's make this explicit
               so I can do it correctly in cython
        """
        
        #Intentionally using bad naming scheme here to match
        #other voidfinder functions
        self.w_coord = w_coord
        
        self.coord_min = coord_min
        
        self.dl = dl
        
        self.reference_point_ijk = np.empty((1,3), dtype=np.int16)
        
        self.galaxy_map = galaxy_map
        
        self.galaxy_map_array = galaxy_map_array
        
        '''
        mesh_indices = ((w_coord - coord_min)/dl).astype(np.int64)
        
        # Initialize dictionary of cell IDs with at least one galaxy in them
        self.nonvoid_cell_ID_dict = {}
        
        self.galaxy_map = {}
    
        for idx in range(mesh_indices.shape[0]):
    
            bin_ID = tuple(mesh_indices[idx])
    
            self.nonvoid_cell_ID_dict[bin_ID] = 1
            
            if bin_ID not in self.galaxy_map:
                
                self.galaxy_map[bin_ID] = []
            
            self.galaxy_map[bin_ID].append(idx)
            
        #Convert lists to numpy arrays
        for key in self.galaxy_map:
            
            indices = self.galaxy_map[key]
            
            self.galaxy_map[key] = np.array(indices, dtype=np.int64)
        '''
            
        self.shell_boundaries_xyz = np.empty((2,3), dtype=np.float64)
        #self.min_containing_dist_mem = np.empty((2,3), dtype=np.float64)
        self.cell_center_xyz = np.empty((1,3), dtype=np.float64)
        
        #self.cell_ID_mem = np.empty((1000,3), dtype=np.int64)
            
            
    def query_first(self, reference_point_xyz):
        """
        Description
        -----------
        
        Finds first nearest neighbor for the given reference point
        
        NOTE:  This function is OK as a "find first only" setup because
        we're only ever going to give it data points which are Cell centers
        and not data points from w_coord, if we gave it a point from w_coord
        it would always just return that same point which would be dumb, but
        we're ok cause we're not gonna do that.
        
        
        Parameters
        ----------
        
        reference_point_xyz : ndarray of shape (1,3)
            the point in xyz coordinates of whom we would like to find
            the nearest neighbors for
            
        num_next_neighbors : integer
            number of neighbors to find
        """
        
        self.reference_point_ijk[:] = ((reference_point_xyz - self.coord_min)/self.dl).astype(np.int64)
        
        current_shell = -1
        
        check_next_shell = True
        
        neighbor_idx = 0
        neighbor_dist_xyz_sq = np.inf
        neighbor_dist_xyz = np.inf
        
        while check_next_shell:
            
            current_shell += 1
            
            #get bounding information for current shell
            boundary_maxes_xyz, boundary_mins_xyz = self.gen_shell_boundaries(self.reference_point_ijk, current_shell)
            
            min1 = np.min(np.abs(boundary_maxes_xyz - reference_point_xyz))
            min2 = np.min(np.abs(boundary_mins_xyz - reference_point_xyz))
            
            min_containing_radius_xyz = min(min1, min2)
            
            
            
            #search_current_shell()
            shell_cell_IDs = self.gen_shell(self.reference_point_ijk, current_shell)
            
            for cell_ID in shell_cell_IDs:
                
                if tuple(cell_ID) in self.galaxy_map:
                    
                    potential_neighbor_idxs = self.galaxy_map[tuple(cell_ID)]
                    
                    #
                    #This loop and logic could be vectorized
                    #
                    for potential_neighbor_idx in potential_neighbor_idxs:
                        
                        potential_neighbor_xyz = self.w_coord[<ITYPE_t>potential_neighbor_idx]
                        
                        dist_sq = np.sum((potential_neighbor_xyz - reference_point_xyz)**2)
                        
                        if dist_sq < neighbor_dist_xyz_sq:
                            
                            neighbor_idx = potential_neighbor_idx
                            
                            neighbor_dist_xyz_sq = dist_sq
                            
                            neighbor_dist_xyz = np.sqrt(dist_sq)
                            
                            #Don't need to check against the min_containing_radius here
                            #because we want to check everybody in this shell
                            
            if neighbor_dist_xyz < min_containing_radius_xyz:
                
                check_next_shell = False
                                
                                #Don't break loop cause there could still be someone
                                #closer in this shell
                        
                        
        #return np.array([neighbor_dist_xyz],dtype=np.float64), np.array([neighbor_idx], dtype=np.int64)
        return neighbor_dist_xyz, neighbor_idx
    
                        
                        
                        
    def query_shell_radius(self, reference_point_xyz, search_radius_xyz):
        """
        Find all the neighbors within a given radius of a reference point.
        """
        
        self.reference_point_ijk[:] = ((reference_point_xyz - self.coord_min)/self.dl).astype(np.int64)
                        
        output = []
        
        if search_radius_xyz < 0.5*self.dl:
            
            max_shell = 0
            
        else:
            
            max_shell = int(np.ceil((search_radius_xyz - 0.5*self.dl)/self.dl))
            
            
        for current_shell in range(max_shell+1): #+1 to include max_shell
            
            shell_cell_IDs = self.gen_shell(self.reference_point_ijk, current_shell)
            
            for cell_ID in shell_cell_IDs:
                
                if tuple(cell_ID) in self.galaxy_map:
                    
                    output.append(self.galaxy_map[tuple(cell_ID)])
        
        
        if output:
        
            return np.concatenate(output)
        
        else:
            
            return np.array([], dtype=np.int64)
        
        
    
                    
    def gen_shell_boundaries(self, center_ijk, level):
        
        cell_center_xyz = (center_ijk + 0.5)*self.dl + self.coord_min
        
        boundary_maxes_xyz = cell_center_xyz + 0.5*self.dl + level*self.dl
        
        boundary_mins_xyz = cell_center_xyz - 0.5*self.dl - level*self.dl
        
        return boundary_maxes_xyz, boundary_mins_xyz
    
        
        
        

    def gen_shell(self, center_ijk, level):
        
        
        if level == 0:
            
            return center_ijk
        
        
        num_return = (2*level + 1)**3 - (2*level - 1)**3
        
        return_array = np.empty((num_return, 3), dtype=np.int64)
        
        out_idx = 0
        #i first
        
        #i_0 = level
        #i_1 = -level
        
        #i first
        for j in range(-level, level+1):
            for k in range(-level, level+1):
                
                return_array[out_idx, 0] = level
                return_array[out_idx, 1] = j
                return_array[out_idx, 2] = k
                
                return_array[out_idx+1, 0] = -level
                return_array[out_idx+1, 1] = j
                return_array[out_idx+1, 2] = k
                
                out_idx += 2
                
        
        #do j
        for i in range(-level+1, level):
            for k in range(-level, level+1):
                
                return_array[out_idx, 0] = i
                return_array[out_idx, 1] = level
                return_array[out_idx, 2] = k
                
                return_array[out_idx+1, 0] = i
                return_array[out_idx+1, 1] = -level
                return_array[out_idx+1, 2] = k
                
                out_idx += 2
                
        #do k
        for i in range(-level+1, level):
            for j in range(-level+1, level):
                
                return_array[out_idx, 0] = i
                return_array[out_idx, 1] = j
                return_array[out_idx, 2] = level
                
                return_array[out_idx+1, 0] = i
                return_array[out_idx+1, 1] = j
                return_array[out_idx+1, 2] = -level
                
                out_idx += 2
                
        return return_array + center_ijk
    



    
cdef class Cell_ID_Memory:
    """
    Description
    ===========
    
    This class is essentially a wrapper to the memory that holds the 
    (i,j,k) cell IDs that need to be searched.
    
    Through the course of running VoidFinder on an algorithm, searching adjacent
    grid cells at the various (i,j,k) locations will require unknown amounts of memory,
    so this class comes with a resize() method.  During a run, the memory held by this class
    will grow larger and larger, but ultimately there has to be 1 galaxy in the survey whose
    grid search for the next nearest neighbor is the farthest.  If this distance possible to 
    compute ahead of time, we would just pre-allocate that many rows, but that would be a 
    very expensive overhead so instead we provide the resize() method.  In practice the number
    of resizes is very small and quickly converges to a reasonable limit.
    
    For example, if the largest grid size that needs to be searched is 50 cells wide we need
    50^3 = 125,000 slots (times 3 for ijk and times sizeof(CELL_ID_t) total bytes, so 750kb for
    a 50x50x50 grid search)
    """

    #cdef DTYPE_INT64_t* data

    def __cinit__(self, size_t level):
        
        
        num_rows = (2*level + 1)**3
        
        self.max_level_mem = level
        self.max_level_available = 0
        
        
        # allocate some memory (uninitialised, may contain arbitrary data)
        self.data = <CELL_ID_t*> PyMem_Malloc(num_rows * 3 * sizeof(CELL_ID_t))
        
        if not self.data:
            
            raise MemoryError()
        
        self.level_start_idx = <DTYPE_INT64_t*> PyMem_Malloc((level+1)*sizeof(DTYPE_INT64_t))
        
        if not self.level_start_idx:
            
            raise MemoryError()
        
        self.level_stop_idx = <DTYPE_INT64_t*> PyMem_Malloc((level+1)*sizeof(DTYPE_INT64_t))
        
        if not self.level_stop_idx:
            
            raise MemoryError()
        
        
        self.total_num_rows = num_rows
        self.num_available_rows = num_rows
        self.next_unused_row_idx = 0
        
        self.curr_ijk = <CELL_ID_t*> PyMem_Malloc(3 * sizeof(CELL_ID_t))
        
        if not self.curr_ijk:
            
            raise MemoryError()
        
        #a 65536^3 grid is 2.8*10^14 grid cells, so initializing
        #to 65535 in the hopes we will never have a grid with
        #a side of length 65536
        self.curr_ijk[0] = 65535
        self.curr_ijk[1] = 65535
        self.curr_ijk[2] = 65535
        
        
        #print("Initialized!")

    def resize(self, size_t level):
        """
        Allocates num_rows * 3 * sizeof(CELL_ID_t) bytes,
        -PRESERVING THE CURRENT CONTENT- and making a best-effort to
        re-use the original data location.
        """
        
        print("Cell ID Mem resizing: ", self.max_level_mem, level, flush=True)

        
        if level > <size_t>self.max_level_mem:
            
            num_rows = (2*level+1)**3
            
            mem = <CELL_ID_t*> PyMem_Realloc(self.data, num_rows * 3 * sizeof(CELL_ID_t))
            
            if not mem:
                
                raise MemoryError()
            
            # Only overwrite the pointer if the memory was really reallocated.
            # On error (mem is NULL), the original memory has not been freed.
            self.data = mem
            
            
            mem2 = <DTYPE_INT64_t*> PyMem_Realloc(self.level_start_idx, (level+1)*sizeof(DTYPE_INT64_t))
            
            if not mem2:
                
                raise MemoryError()
            
            self.level_start_idx = mem2
            
            
            mem3 = <DTYPE_INT64_t*> PyMem_Realloc(self.level_stop_idx, (level+1)*sizeof(DTYPE_INT64_t))
            
            if not mem3:
                
                raise MemoryError()
            
            self.level_stop_idx = mem3
            
            
            
            
            self.total_num_rows = num_rows
            
            self.max_level_mem = level

    def __dealloc__(self):
        
        PyMem_Free(self.data)  # no-op if self.data is NULL







@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
@cython.profile(True)
cdef DistIdxPair _query_first(CELL_ID_t[:,:] reference_point_ijk,
                              DTYPE_F64_t[:,:] coord_min,
                              DTYPE_F64_t dl,
                              DTYPE_F64_t[:,:] shell_boundaries_xyz,
                              DTYPE_F64_t[:,:] cell_center_xyz,
                              GalaxyMapCustomDict galaxy_map,
                              DTYPE_INT64_t[:] galaxy_map_array,
                              DTYPE_F64_t[:,:] w_coord,
                              Cell_ID_Memory cell_ID_mem,
                              DTYPE_F64_t[:,:] reference_point_xyz
                              ) except *:
    """
    Description
    ===========
    
    Only called once in _voidfinder_cython.main_algorithm()
    
    Finds first nearest neighbor for the given reference point
    
    NOTE:  This function is OK as a "find first only" setup because
    we're only ever going to give it data points which are Cell centers
    and not data points from w_coord, if we gave it a point from w_coord
    it would always just return that same point which would be dumb, but
    we're ok cause we're not gonna do that.
    
    
    Parameters
    ==========
    
    reference_point_xyz : ndarray of shape (1,3)
        the point in xyz coordinates of whom we would like to find
        the nearest neighbors for
        
    """
    
    
    ################################################################################
    # Convert our query point from xyz to ijk space
    ################################################################################
    
    reference_point_ijk[0,0] = <CELL_ID_t>((reference_point_xyz[0,0] - coord_min[0,0])/dl)
    reference_point_ijk[0,1] = <CELL_ID_t>((reference_point_xyz[0,1] - coord_min[0,1])/dl)
    reference_point_ijk[0,2] = <CELL_ID_t>((reference_point_xyz[0,2] - coord_min[0,2])/dl)
    
    ################################################################################
    # All the variables we need cdef'd
    ################################################################################
    cdef DTYPE_INT32_t current_shell = -1
    
    cdef DTYPE_B_t check_next_shell = True
    
    cdef ITYPE_t neighbor_idx = 0
    
    cdef DTYPE_F64_t neighbor_dist_xyz_sq = INFINITY
    
    cdef DTYPE_F64_t neighbor_dist_xyz = INFINITY
    
    cdef DTYPE_F64_t min_containing_radius_xyz
    
    cdef ITYPE_t cell_ID_idx
    
    cdef ITYPE_t offset, num_elements
    
    cdef OffsetNumPair curr_offset_num_pair
    
    cdef DistIdxPair return_vals
    
    
    cdef ITYPE_t idx
    cdef DTYPE_F64_t dist_sq
    cdef DTYPE_F64_t temp1
    cdef DTYPE_F64_t temp2
    cdef DTYPE_F64_t temp3
    
    cdef ITYPE_t potential_neighbor_idx
    
    #cdef DTYPE_INT64_t[:,:] shell_cell_IDs
    
    cdef DTYPE_F64_t[:] potential_neighbor_xyz
    
    cdef DTYPE_INT64_t num_cell_IDs
    
    cdef DTYPE_INT64_t cell_start_row, cell_end_row
    
    cdef CELL_ID_t id1, id2, id3
    
    ################################################################################
    # Iterate through the grid cells shell-by-shell growing outwards until we find
    # the nearest neighbor to our reference_point_xyz
    ################################################################################
    while check_next_shell:
        
        current_shell += 1
        
        _gen_shell_boundaries(shell_boundaries_xyz,
                              cell_center_xyz,
                              coord_min,
                              dl,
                              reference_point_ijk,
                              current_shell)
        
        min_containing_radius_xyz = _min_contain_radius(shell_boundaries_xyz, 
                                                        reference_point_xyz)
        
        
        
        
        #print("Gen Shell: (", reference_point_ijk[0,0], reference_point_ijk[0,1], reference_point_ijk[0,2], ") level: ", current_shell, flush=True)
    
        
        
        
        cell_start_row, cell_end_row = _gen_shell(reference_point_ijk, 
                                                  current_shell,
                                                  cell_ID_mem,
                                                  galaxy_map)
        
        #print("Shell start/end: ", cell_start_row, cell_end_row, flush=True)
        
        for cell_ID_idx in range(<ITYPE_t>cell_start_row, <ITYPE_t>cell_end_row):
            
            id1 = cell_ID_mem.data[3*cell_ID_idx]
            id2 = cell_ID_mem.data[3*cell_ID_idx+1]
            id3 = cell_ID_mem.data[3*cell_ID_idx+2]
            
            #print("Working Cell ID: "+str(id1)+","+str(id2)+","+str(id3), flush=True)
            
            
            # I think we can remove this "if galaxy_map.contains()" 
            # because the cell_ID_mem has already checked all its cell
            # IDs against the galaxy map
            #if galaxy_map.contains(id1, id2, id3):
            
            #print("Gwababrara", flush=True)
            
            curr_offset_num_pair = galaxy_map.getitem(id1, id2, id3)
            
            offset = curr_offset_num_pair.offset
            
            num_elements = curr_offset_num_pair.num_elements
            
            for idx in range(num_elements):
                
                potential_neighbor_idx = <ITYPE_t>galaxy_map_array[offset+idx]
                
                potential_neighbor_xyz = w_coord[potential_neighbor_idx]
                
                temp1 = potential_neighbor_xyz[0] - reference_point_xyz[0,0]
                temp2 = potential_neighbor_xyz[1] - reference_point_xyz[0,1]
                temp3 = potential_neighbor_xyz[2] - reference_point_xyz[0,2]
                
                dist_sq = temp1*temp1 + temp2*temp2 + temp3*temp3
                
                
                if dist_sq < neighbor_dist_xyz_sq:
                    
                    neighbor_idx = potential_neighbor_idx
                    
                    neighbor_dist_xyz_sq = dist_sq
                    
                    neighbor_dist_xyz = sqrt(dist_sq)
                    
                    #Don't need to check against the min_containing_radius here
                    #because we want to check everybody in this shell, since
                    #even if this guy matches our criteria, someone could be closer
                            
        if neighbor_dist_xyz < min_containing_radius_xyz:
            
            check_next_shell = False

                    
    return_vals.idx = neighbor_idx
    
    return_vals.dist = neighbor_dist_xyz
    
    return return_vals
                    





@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
@cython.profile(True)
cdef ITYPE_t[:] _query_shell_radius(CELL_ID_t[:,:] reference_point_ijk,
                                    DTYPE_F64_t[:,:] w_coord, 
                                    DTYPE_F64_t[:,:] coord_min, 
                                    DTYPE_F64_t dl,
                                    GalaxyMapCustomDict galaxy_map,
                                    DTYPE_INT64_t[:] galaxy_map_array,
                                    Cell_ID_Memory cell_ID_mem,
                                    DTYPE_F64_t[:,:] reference_point_xyz, 
                                    DTYPE_F64_t search_radius_xyz
                                    ) except *:
    """
    Description
    ===========
    
    Only called once in find_next_galaxy()
    
    Find all the neighbors within a given radius of a reference point.
    
    
    """
    
    
    #print("query_shell_radius start")
    
    #print("Reference point xyz: ", reference_point_xyz[0,0], reference_point_xyz[0,1], reference_point_xyz[0,2], flush=True)
    
    reference_point_ijk[0,0] = <CELL_ID_t>((reference_point_xyz[0,0] - coord_min[0,0])/dl)
    reference_point_ijk[0,1] = <CELL_ID_t>((reference_point_xyz[0,1] - coord_min[0,1])/dl)
    reference_point_ijk[0,2] = <CELL_ID_t>((reference_point_xyz[0,2] - coord_min[0,2])/dl)
         
    #print("Reference point ijk: ", reference_point_ijk[0,0], reference_point_ijk[0,1], reference_point_ijk[0,2], flush=True)
    
                    
    output = []
    
    cdef DTYPE_INT32_t max_shell
    
    #cdef DTYPE_INT32_t current_shell
    
    cdef ITYPE_t cell_ID_idx
    
    cdef ITYPE_t idx
    
    cdef ITYPE_t curr_galaxy_idx
    
    cdef ITYPE_t offset, num_elements
    
    cdef OffsetNumPair curr_offset_num_pair
    
    #cdef DTYPE_INT64_t[:,:] shell_cell_IDs
    
    cdef DTYPE_INT64_t num_cell_IDs
    
    cdef CELL_ID_t id1, id2, id3
    
    cdef DTYPE_F64_t search_radius_xyz_sq = search_radius_xyz*search_radius_xyz
    
    cdef DTYPE_INT64_t[:] curr_galaxies_idxs
    
    cdef DTYPE_F64_t[:] galaxy_xyz
    
    cdef DTYPE_F64_t temp1, temp2, temp3, temp4
    
    cdef DTYPE_F64_t dist_sq
    
    cdef DTYPE_F64_t[:,:] cell_ijk_in_xyz = np.empty((1,3), dtype=np.float64)
    
    #if search_radius_xyz < 0.5*dl:
        
    #    max_shell = 0
        
    #else:
    
    ################################################################################
    # Calculate the max shell needed to search
    # fill in implementation details here, using component-wise max for something
    ################################################################################
        
    cell_ijk_in_xyz[0,0] = (<DTYPE_F64_t>reference_point_ijk[0,0] + 0.5)*dl + coord_min[0,0]
    cell_ijk_in_xyz[0,1] = (<DTYPE_F64_t>reference_point_ijk[0,1] + 0.5)*dl + coord_min[0,1]
    cell_ijk_in_xyz[0,2] = (<DTYPE_F64_t>reference_point_ijk[0,2] + 0.5)*dl + coord_min[0,2]
    
    temp1 = fabs((cell_ijk_in_xyz[0,0] - reference_point_xyz[0,0])/dl)
    temp4 = temp1
    temp2 = fabs((cell_ijk_in_xyz[0,1] - reference_point_xyz[0,1])/dl)
    if temp2 > temp4:
        temp4 = temp2
    temp3 = fabs((cell_ijk_in_xyz[0,2] - reference_point_xyz[0,2])/dl)
    if temp3 > temp4:
        temp4 = temp3
    
    max_shell = <DTYPE_INT32_t>ceil((search_radius_xyz - (0.5-temp4)*dl)/dl)
        

    ################################################################################
    # Since we are querying based on radius, we can calculate the maximum grid
    # shape we are going to search this time, so use _gen_cube to generate all
    # the grid cells (filtered by galaxy_map.contains()) that we need to search
    ################################################################################
    
    #print("Gen Cube: ", "(", reference_point_ijk[0,0], reference_point_ijk[0,1], reference_point_ijk[0,2], ") level:", max_shell, flush=True)
    
    num_cell_IDs = _gen_cube(reference_point_ijk, 
                             max_shell,
                             cell_ID_mem,
                             galaxy_map)
    
    #print("query shell radius mid", num_cell_IDs)
    
    for cell_ID_idx in range(<ITYPE_t>num_cell_IDs):
        
        id1 = cell_ID_mem.data[3*cell_ID_idx]
        id2 = cell_ID_mem.data[3*cell_ID_idx+1]
        id3 = cell_ID_mem.data[3*cell_ID_idx+2]
        
        
        #print("Working Cell ID: "+str(id1)+","+str(id2)+","+str(id3), flush=True)
        
    
        # Similar to the usage above in _query_first(),
        # I think we can remove this "if galaxy_map.contains()" 
        # because the cell_ID_mem has already checked all its cell
        # IDs against the galaxy map
        #if galaxy_map.contains(id1, id2, id3):
        
        curr_offset_num_pair = galaxy_map.getitem(id1, id2, id3)
            
        offset = curr_offset_num_pair.offset
        
        num_elements = curr_offset_num_pair.num_elements
        
        #print("bloop", num_elements, flush=True)
        
        for idx in range(num_elements):
            
            curr_galaxy_idx = <ITYPE_t>galaxy_map_array[offset+idx]
            
            galaxy_xyz = w_coord[curr_galaxy_idx]
            
            temp1 = galaxy_xyz[0] - reference_point_xyz[0,0]
            temp2 = galaxy_xyz[1] - reference_point_xyz[0,1]
            temp3 = galaxy_xyz[2] - reference_point_xyz[0,2]
            
            dist_sq = temp1*temp1 + temp2*temp2 + temp3*temp3
            
            #print("Doop", idx, flush=True)
            
            if dist_sq < search_radius_xyz_sq:
                
                output.append(curr_galaxy_idx)
                
            #print("Poop", flush=True)

    
    if output:
        
        return np.array(output).astype(np.intp)
    
    else:
        
        return np.array([], dtype=np.intp)
                


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
@cython.profile(True)
cdef void _gen_shell_boundaries(DTYPE_F64_t[:,:] shell_boundaries_xyz, 
                                DTYPE_F64_t[:,:] cell_center_xyz,
                                DTYPE_F64_t[:,:] coord_min,
                                DTYPE_F64_t dl,
                                CELL_ID_t[:,:] center_ijk, 
                                DTYPE_INT64_t level
                                ) except *:
    """
    Calculate the xyz center of the cell given the ijk center, calculate the xyz arm length
    of the distance from the center of shell 0 to the edge of shell 'level', then add
    that arm length to the xyz in the first row and subtract it in the 2nd row to get
    a pair of maximal and minimal bounding points for comparison
    """
    
    cell_center_xyz[0,0] = (<DTYPE_F64_t>center_ijk[0,0] + 0.5)*dl + coord_min[0,0]
    cell_center_xyz[0,1] = (<DTYPE_F64_t>center_ijk[0,1] + 0.5)*dl + coord_min[0,1]
    cell_center_xyz[0,2] = (<DTYPE_F64_t>center_ijk[0,2] + 0.5)*dl + coord_min[0,2]
    
    cdef DTYPE_F64_t temp1 = 0.5*dl + <DTYPE_F64_t>level*dl
    
    shell_boundaries_xyz[0,0] = cell_center_xyz[0,0] + temp1
    shell_boundaries_xyz[0,1] = cell_center_xyz[0,1] + temp1
    shell_boundaries_xyz[0,2] = cell_center_xyz[0,2] + temp1
    shell_boundaries_xyz[1,0] = cell_center_xyz[0,0] - temp1
    shell_boundaries_xyz[1,1] = cell_center_xyz[0,1] - temp1
    shell_boundaries_xyz[1,2] = cell_center_xyz[0,2] - temp1
        
        
        
        

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
@cython.profile(True)
cdef DTYPE_F64_t _min_contain_radius(DTYPE_F64_t[:,:] shell_boundaries_xyz, 
                                     DTYPE_F64_t[:,:] reference_point_xyz
                                     ) except *:
    """
    Find the minimum distance from our reference point to the existing shell boundary
    given by self.shell_boundaries_xyz
    """
    
    cdef DTYPE_F64_t running_min = INFINITY
    cdef DTYPE_F64_t temp
    
    running_min = fabs(shell_boundaries_xyz[0,0] - reference_point_xyz[0,0])
    
    temp = fabs(shell_boundaries_xyz[0,1] - reference_point_xyz[0,1])
    if temp < running_min:
        running_min = temp
        
    temp = fabs(shell_boundaries_xyz[0,2] - reference_point_xyz[0,2])
    if temp < running_min:
        running_min = temp
        
    temp = fabs(shell_boundaries_xyz[1,0] - reference_point_xyz[0,0])
    if temp < running_min:
        running_min = temp
        
    temp = fabs(shell_boundaries_xyz[1,1] - reference_point_xyz[0,1])
    if temp < running_min:
        running_min = temp
        
    temp = fabs(shell_boundaries_xyz[1,2] - reference_point_xyz[0,2])
    if temp < running_min:
        running_min = temp
        
    return running_min
    
    
    
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
@cython.profile(True)
cdef (DTYPE_INT64_t, DTYPE_INT64_t) _gen_shell(CELL_ID_t[:,:] center_ijk, 
                                               DTYPE_INT32_t level,
                                               Cell_ID_Memory cell_ID_mem,
                                               GalaxyMapCustomDict galaxy_map,
                                               ) except *:
    """
    Description
    ===========
    
    
    
    Generate all the possible locations in the "shell" defined by the level parameter.
    
    Given a cubic grid structure, level 0 is a single cell, level 1 is a shell of
    3x3x3 cells minus the 1 interior, level 2 is 5x5x5 minus the 3x3x3 interior, etc.
    
    
    Notes
    =====
    
    Only called once in _query_first() (_voidfinder_cython_find_next) in main_algorithm()
    (_voidfinder_cython)
    
    This means in main_algorithm, we come upon an ijk cell.  We call _query_first on that ijk
    cell, and within _query_first we stay on that ijk cell and grow outward shells until we find
    the first neighbor.  Then _gen_shell won't be called again until the next _ijk shell, and instead
    we will use _gen_cube to find the 2nd 3rd and 4a/b-th neighbors
    
    ASSUMES THIS WILL ONLY BE CALLED SEQUENTIALLY AND IN-ORDER FOR A SINGLE IJK, WHICH BASED ON ABOVE
    USAGE WOULD SEEM TO BE TRUE (except for level 0, which should never be called, level 1, then 2, 3, 
    4,5, etc).  VIOLATING THIS WILL RESULT IN THE cell_ID_mem.max_level_available possibly being 
    set incorrectly
    
    """
    
    
    #print("Entered _gen_shell() function", flush=True)
    
    #print("PRE-SETTING-1 level_stop_idx: ", cell_ID_mem.level_stop_idx[1], flush=True)
    
    
    cdef ITYPE_t out_idx = 0
    
    cdef DTYPE_INT32_t i, j, k, temp
    
    cdef CELL_ID_t out_i, out_j, out_k
    
    cdef DTYPE_INT32_t i_lower, i_upper, j_lower, j_upper, k_lower, k_upper
    
    cdef CELL_ID_t center_i, center_j, center_k
    
    #cdef DTYPE_INT64_t num_return_rows = (2*level + 1)**3 - (2*level - 1)**3
    
    cdef DTYPE_INT64_t num_written = 0
    
    
    ################################################################################
    # We can precompute the maximum number of rows we might need for this shell 
    # generation to be successful, so if necessary resize the cell_ID_mem to be big 
    # enough to hold the data.
    #
    # Note that we may use less than this number of rows due to the
    # galaxy_map.contains() filtering out some grid cells
    #
    # Also - cool trick, turns out PyMem_Realloc PRESERVES THE EXISTING DATA!
    # So gonna use this to optimize the cell_ID_mem cell generation
    ################################################################################
    
    if cell_ID_mem.max_level_mem < <DTYPE_INT64_t>level:
        
        #print("CELL ID RESIZE: ", cell_ID_mem.max_level_mem, level, flush=True)
        
        cell_ID_mem.resize(<size_t>level)
    
    
    if center_ijk[0,0] == cell_ID_mem.curr_ijk[0] and \
       center_ijk[0,1] == cell_ID_mem.curr_ijk[1] and \
       center_ijk[0,2] == cell_ID_mem.curr_ijk[2]:
        
        #we're working from the same ijk as last time this was called, so
        #we might already have the ijk values stored
        if <DTYPE_INT64_t>level <= cell_ID_mem.max_level_available:
            
            #print("Gen shell0-E stop idx: ", cell_ID_mem.level_stop_idx[level], level, flush=True)
           
            return cell_ID_mem.level_start_idx[level], cell_ID_mem.level_stop_idx[level]
    
        #we don't have the current level stored, so we need to calculate it, but
        #we may have other valuable level information stored, so use the unused
        #rows in cell_ID_mem
        else:
            
            out_idx = 3*cell_ID_mem.next_unused_row_idx
            
            cell_ID_mem.level_start_idx[level] = cell_ID_mem.next_unused_row_idx
            
            cell_ID_mem.max_level_available = <DTYPE_INT64_t>level
            
    else:
        
        #New ijk, so have to start over 
        
        cell_ID_mem.max_level_available = <DTYPE_INT64_t>level
        
        cell_ID_mem.next_unused_row_idx = 0
        
        cell_ID_mem.curr_ijk[0] = center_ijk[0,0]
        cell_ID_mem.curr_ijk[1] = center_ijk[0,1]
        cell_ID_mem.curr_ijk[2] = center_ijk[0,2]
        
        cell_ID_mem.level_start_idx[level] = 0
        
        
    ################################################################################
    # For level 0, the algorithm below actually would write out the original
    # cell ijk twice, but we only want to write it once so have a special block
    # here to handle that single special case and return early
    ################################################################################
    if level == 0:
        
        if galaxy_map.contains(center_ijk[0,0], center_ijk[0,1], center_ijk[0,2]):
            
            cell_ID_mem.data[out_idx] = center_ijk[0,0]
            cell_ID_mem.data[out_idx+1] = center_ijk[0,1]
            cell_ID_mem.data[out_idx+2] = center_ijk[0,2]
            
            #print("Cell ID Writing0: "+str(center_ijk[0,0])+","+str(center_ijk[0,1])+","+str(center_ijk[0,2]), flush=True)
            
            num_written = 1
            
        else:
            
            num_written = 0
        
        cell_ID_mem.next_unused_row_idx += num_written
    
        cell_ID_mem.level_stop_idx[level] = cell_ID_mem.next_unused_row_idx
        
        return cell_ID_mem.level_start_idx[level], cell_ID_mem.level_stop_idx[level]
    
    
    ################################################################################
    # Technically not necessary but made the code below look a tad cleaner
    ################################################################################
    
    center_i = center_ijk[0,0]
    center_j = center_ijk[0,1]
    center_k = center_ijk[0,2]
    
    
    
    ################################################################################
    # i dimension first
    # Iterate through the possible shell locations, starting with the i dimension
    # this iteration does all the cells in the 2 "planes" at the "i +/- level"
    # grid coordinate 
    ################################################################################
    for j in range(-level, level+1):
        
        for k in range(-level, level+1):
            
            out_i = <CELL_ID_t>level + center_i
            out_j = <CELL_ID_t>j + center_j
            out_k = <CELL_ID_t>k + center_k
            
            if galaxy_map.contains(out_i, out_j, out_k):
            
                cell_ID_mem.data[out_idx] = out_i
                cell_ID_mem.data[out_idx+1] = out_j
                cell_ID_mem.data[out_idx+2] = out_k
                
                #print("Cell ID Writing: "+str(out_i)+","+str(out_j)+","+str(out_k), flush=True)
                
                out_idx += 3
                num_written += 1
                
            out_i = <CELL_ID_t>(-level) + center_i
            
            if out_i >= 0 and galaxy_map.contains(out_i, out_j, out_k):
            
                cell_ID_mem.data[out_idx] = out_i
                cell_ID_mem.data[out_idx+1] = out_j
                cell_ID_mem.data[out_idx+2] = out_k
                
                #print("Cell ID Writing: "+str(out_i)+","+str(out_j)+","+str(out_k), flush=True)
                
                out_idx += 3
                num_written += 1
                
                
    ################################################################################
    # j dimension
    # Next do the 2 "planes" at the "j +/- level" coordinates, except for the edges
    # which have already been done by doing the "i +/- level" planes, so the i
    # parameter below runs from (-level+1, level) instead of (-level, level+1)
    ################################################################################
    for i in range(-level+1, level):
        for k in range(-level, level+1):
            
            out_i = <CELL_ID_t>i + center_i
            out_j = <CELL_ID_t>level + center_j
            out_k = <CELL_ID_t>k + center_k
            
            if galaxy_map.contains(out_i, out_j, out_k):
            
                cell_ID_mem.data[out_idx] = out_i
                cell_ID_mem.data[out_idx+1] = out_j
                cell_ID_mem.data[out_idx+2] = out_k
                
                #print("Cell ID Writing: "+str(out_i)+","+str(out_j)+","+str(out_k), flush=True)
                
                out_idx += 3
                num_written += 1
                
            out_j = <CELL_ID_t>(-level) + center_j
        
            if out_j >= 0 and galaxy_map.contains(out_i, out_j, out_k):
            
                cell_ID_mem.data[out_idx] = out_i
                cell_ID_mem.data[out_idx+1] = out_j
                cell_ID_mem.data[out_idx+2] = out_k
                
                #print("Cell ID Writing: "+str(out_i)+","+str(out_j)+","+str(out_k), flush=True)
                
                out_idx += 3
                num_written += 1
            
    ################################################################################
    # k dimension
    # Lastly do the 2 "planes" at the "k +/- level" coordinates, noting that since
    # we are in 3D and have already done i and j, the border cells around the
    # positive and negative k planes have already been checked, so both i and j
    # run from (-level+1, level) instead of (-level, level+1)
    ################################################################################
    for i in range(-level+1, level):
        for j in range(-level+1, level):
            
            out_i = <CELL_ID_t>i + center_i
            out_j = <CELL_ID_t>j + center_j
            out_k = <CELL_ID_t>level + center_k
            
            if galaxy_map.contains(out_i, out_j, out_k):
            
                cell_ID_mem.data[out_idx] = out_i
                cell_ID_mem.data[out_idx+1] = out_j
                cell_ID_mem.data[out_idx+2] = out_k
                
                #print("Cell ID Writing: "+str(out_i)+","+str(out_j)+","+str(out_k), flush=True)
                
                out_idx += 3
                num_written += 1
                
            out_k = <CELL_ID_t>(-level) + center_k
        
            if out_k >= 0 and galaxy_map.contains(out_i, out_j, out_k):
            
                cell_ID_mem.data[out_idx] = out_i
                cell_ID_mem.data[out_idx+1] = out_j
                cell_ID_mem.data[out_idx+2] = out_k
                
                #print("Cell ID Writing: "+str(out_i)+","+str(out_j)+","+str(out_k), flush=True)
                
                out_idx += 3
                num_written += 1
            
            
    cell_ID_mem.next_unused_row_idx += num_written
    
    cell_ID_mem.level_stop_idx[level] = cell_ID_mem.next_unused_row_idx
    
    return cell_ID_mem.level_start_idx[level], cell_ID_mem.level_stop_idx[level]
    
        
        



@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
@cython.profile(True)
cdef DTYPE_INT64_t _gen_cube(CELL_ID_t[:,:] center_ijk, 
                             DTYPE_INT32_t level,
                             Cell_ID_Memory cell_ID_mem,
                             GalaxyMapCustomDict galaxy_map
                             ) except *:
    """
    Description
    ===========
    
    Only called once in _query_shell_radius() (this file) which is only called once in
    find_next_galaxy() (also this file)
    
    Take advantage of the optimizations in _gen_shell()
    
    """
    
    
    cdef DTYPE_INT32_t curr_level
    
    cdef DTYPE_INT64_t row_start, row_stop
    
    for curr_level in range(level+1): #level 5 means include 0,1,2,3,4,5 so do level+1 for range
        
        row_start, row_stop = _gen_shell(center_ijk,
                                         curr_level,
                                         cell_ID_mem,
                                         galaxy_map)
                
    return cell_ID_mem.next_unused_row_idx
    
        
        





