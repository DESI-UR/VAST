



cimport cython
import numpy as np
cimport numpy as np
np.import_array()  # required in order to use C-API


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

from numpy.math cimport NAN, INFINITY


from libc.math cimport fabs, sqrt, asin, atan#, exp, pow, cos, sin, asin




@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cpdef void main_algorithm(int i, 
                          int j, 
                          int k,
                          galaxy_tree,
                          DTYPE_F64_t[:,:] w_coord,
                          DTYPE_F64_t dl, 
                          DTYPE_F64_t dr,
                          DTYPE_F64_t[:,:] coord_min, 
                          DTYPE_B_t[:,:,:] mask,
                          DTYPE_F64_t min_dist,
                          DTYPE_F64_t max_dist,
                          DTYPE_F64_t[:] return_array
                          ) except *:
    '''
    hole_center variables need to be shape (1,3) for KDTree queries
    everything else can be shape (3,)
    '''
    
   
    #i, j, k = hole_center_coords
    
    
    '''
    print(i,j,k)
    print(galaxy_tree)
    print(w_coord)
    print(dl)
    print(dr)
    print(coord_min)
    print(mask)
    print(min_dist)
    print(max_dist)
    print(return_array)
    '''
    
    
    #print(mask)
    #print(type(mask))
    #print(mask.dtype)
    #print(mask.shape)
    
    
    
    cdef DTYPE_B_t galaxy_search
    cdef DTYPE_B_t in_mask_2
    cdef DTYPE_B_t in_mask_3
    
    cdef DTYPE_F64_t[:,:] hole_center_memview = np.empty((1,3), dtype=np.float64, order='C')
    cdef DTYPE_F64_t[:,:] hole_center_2_memview = np.empty((1,3), dtype=np.float64, order='C')
    cdef DTYPE_F64_t[:,:] hole_center_3_memview = np.empty((1,3), dtype=np.float64, order='C')
    
    
    cdef DTYPE_F64_t[:] neighbor_1_w_coords_memview = np.empty(3, dtype=np.float64, order='C')
    
    
    cdef DTYPE_F64_t[:] v1_unit_memview = np.empty(3, dtype=np.float64, order='C')
    cdef DTYPE_F64_t[:] v2_unit_memview = np.empty(3, dtype=np.float64, order='C')
    
    cdef DTYPE_F64_t modv1
    cdef DTYPE_F64_t modv2
    
    cdef ITYPE_t k1g
    cdef ITYPE_t k2g
    
    cdef ITYPE_t k2g_x2
    
    
    
    
    
    
    
    cdef ITYPE_t[:] i_nearest_memview
    cdef ITYPE_t num_nearest
    cdef DTYPE_F64_t[:,:] BA_memview
    cdef DTYPE_F64_t[:] bot_memview
    cdef DTYPE_F64_t[:] top_memview
    cdef DTYPE_F64_t[:] x2_memview
    cdef DTYPE_B_t[:] valid_idx_memview
    cdef DTYPE_B_t any_valid
    cdef ITYPE_t valid_min_idx
    cdef DTYPE_F64_t valid_min_val
    
    
    
    
    cdef DTYPE_F64_t temp_f64_accum
    cdef DTYPE_F64_t temp_f64_accum2
    cdef DTYPE_F64_t temp_f64_val
    
    
    
    
    cdef DTYPE_F64_t hole_radius
    
    
    cdef DTYPE_F64_t[:] midpoint_memview = np.empty(3, dtype=np.float64, order='C')
    
    cdef DTYPE_F64_t[:] Acenter_memview = np.empty(3, dtype=np.float64, order='C')
    
    cdef DTYPE_F64_t[:] Bcenter_memview = np.empty(3, dtype=np.float64, order='C')
    
    cdef DTYPE_F64_t[:,:] Ccenter_memview
    
    cdef DTYPE_F64_t[:,:] C_minus_A_center_memview
    
    
    cdef DTYPE_F64_t search_radius
    
    

    hole_center = (np.array([[i, j, k]], dtype=np.float64) + 0.5)*dl + coord_min  # Purposefully making hole_center have shape (1,3) for KDtree queries
    
    hole_center_memview[0,0] = i
    hole_center_memview[0,1] = j
    hole_center_memview[0,2] = k
    
    
    cdef ITYPE_t idx
    cdef ITYPE_t jdx
    cdef ITYPE_t temp_idx
    
    for idx in range(3):
        
        hole_center_memview[0,idx] = (hole_center_memview[0,idx] + 0.5)*dl + coord_min[0,idx]
        
    
    
    
    #np.add(hole_center_memview, 0.5, out=hole_center_memview)
    #np.multiply(hole_center_memview, dl, out=hole_center_memview)
    #np.add(hole_center_memview, coord_min, out=hole_center_memview)
    
    
    #hole_center_memview = hole_center
                    
    '''
    print("Hole center")
    print(hole_center)
    print(type(hole_center))
    print(hole_center.dtype)
    '''
    #exit()
    
                    
    # Check to make sure that the hole center is still within the survey
    if not_in_mask(hole_center_memview, mask, min_dist, max_dist):
        
        
        return_array[0] = NAN
        return_array[1] = NAN
        return_array[2] = NAN
        return_array[3] = NAN
        
        return 
    
    
    ############################################################
    #
    # Find Galaxy 1 (closest to cell center)
    #
    # and calculate Unit vector pointing from cell 
    # center to the closest galaxy
    #
    # After [0][0] indexing, modv1 is a float scalar and
    # k1g is an integer scalar
    # The first [0] index gives us the 'array' corresponding to
    # query sample 0, and the second [0] index gives us the value
    # of the neighbor for that query sample, and since we set k=1
    # we don't have to worry about sorting the results
    #
    #
    ############################################################
    neighbor_1_dists, neighbor_1_idxs = galaxy_tree.query(hole_center_memview, k=1)
    
    #print(modv1.shape)
    #print(k1g.shape)
    
    
    
    #modulus of vector 1, dist from galaxy 1 to cell center
    modv1 = neighbor_1_dists[0][0] #float64
    #neighbor_1_dist = modv1
    
    
    #neighbor 1 galaxy index
    k1g = neighbor_1_idxs[0][0] #integer index
    #neighbor_1_idx = k1g
    
    
    
    for idx in range(3):
        
        neighbor_1_w_coords_memview[idx] = w_coord[k1g,idx]
        
        v1_unit_memview[idx] = (neighbor_1_w_coords_memview[idx] - hole_center_memview[0,idx])/modv1
    
    
    #galaxy 1 unit vector
    v1_unit = (w_coord[k1g] - hole_center)/modv1 #np.ndarray shape (1,3)
    
    
    
    
    
    #print("V1_unit shape: ", type(v1_unit), v1_unit.shape) 
    
    #print(modv1)
    #print(k1g)
    #print(v1_unit)

    ############################################################
    #
    # Find Galaxy 2 
    #
    # We are going to shift the center of the hole by dr along 
    # the direction of the vector pointing from the nearest 
    # galaxy to the center of the empty cell.  From there, we 
    # will search within a radius of length the distance between 
    # the center of the hole and the first galaxy from the 
    # center of the hole to find the next nearest neighbors.  
    # From there, we will minimize top/bottom to find which one 
    # is the next nearest galaxy that bounds the hole.
    ############################################################

    galaxy_search = True
    
    
    ############################################################
    # Update hole center 2
    ############################################################

    hole_center_2 = hole_center
    
    for idx in range(3):
        hole_center_2_memview[0,idx] = hole_center_memview[0,idx]

    in_mask_2 = True

    while galaxy_search:

        # Shift hole center away from first galaxy
        hole_center_2 = hole_center_2 - dr*v1_unit
        
        for idx in range(3):
            
            hole_center_2_memview[0,idx] = hole_center_2_memview[0,idx] - dr*v1_unit_memview[idx]
        
        
        
        # Distance between hole center and nearest galaxy
        modv1 += dr
        
        ############################################################
        # Search for nearest neighbors within modv1 of the hole center
        #
        # given data.shape = (N,M)
        # output = tree.query_radius(data, r=radius)
        # output.shape will be (N,), and len(output[i]) == K where
        # K values are returned for the ith of N samples.
        #
        # Since below, N always == 1, we can just use the 0 index
        # to get the K results for our single query point
        # 
        ############################################################
        i_nearest = galaxy_tree.query_radius(hole_center_2_memview, r=modv1)

        i_nearest = i_nearest[0] 
        
        #i_nearest is now an array of shape (K,) where K represents number
        #of galaxies returned
        
        boolean_nearest = i_nearest != k1g
        
        i_nearest = i_nearest[boolean_nearest]
        
        
        #print(type(w_coord))
        #print(i_nearest.dtype)
        #print(type(k1g))
        
        
        #num_nearest is int of ITYPE_t
        num_nearest = i_nearest.shape[0]

        if num_nearest > 0:
            # Found at least one other nearest neighbor!

            # Calculate vector pointing from next nearest galaxies to the nearest galaxy
            
            #print(w_coord.shape)
            #print(i_nearest)
            
            i_nearest_memview = i_nearest
            
            temp1 = w_coord[k1g]
            
            temp2 = np.take(w_coord, i_nearest, axis=0)
            
            
            
            
            
            #elementwise distances between galaxy B and A
            BA = np.subtract(temp1, temp2)  # shape (N,3)
            
            BA_memview = np.empty((num_nearest, 3), dtype=np.float64, order='C')
            
            for idx in range(num_nearest):
                
                for jdx in range(3):
                    
                    temp_idx = i_nearest_memview[idx]
                    
                    BA_memview[idx,jdx] = neighbor_1_w_coords_memview[jdx] - w_coord[temp_idx, jdx]
            
            
            
            
            
            bot = 2*np.dot(BA, v1_unit.T)  # shape (N,1)
            
            bot_memview = np.empty(num_nearest, dtype=np.float64, order='C')
            
            for idx in range(num_nearest):
                
                temp_f64_accum = 0.0
                
                for jdx in range(3):
                    
                    temp_f64_accum += BA_memview[idx,jdx]*v1_unit_memview[jdx]
                    
                bot_memview[idx] = temp_f64_accum
                    
                    
            
            
            
            
            
            top = np.sum(BA**2, axis=1)  # shape (N,)
            
            top_memview = np.empty(num_nearest, dtype=np.float64, order='C')
            
            for idx in range(num_nearest):
                
                temp_f64_accum = 0.0
                
                for jdx in range(3):
                    
                    temp_f64_accum += BA_memview[idx,jdx]*BA_memview[idx,jdx]
                    
                top_memview[idx] = temp_f64_accum
            
            
            
            
            #x2 = temp name
            x2 = top/bot.T[0]  # shape (N,) instead of (1,N)
            
            
            x2_memview = np.empty(num_nearest, dtype=np.float64, order='C')
            
            for idx in range(num_nearest):
                
                x2_memview[idx] = top_memview[idx]/bot_memview[idx]
            
            
            
            
            

            # Locate positive values of x2
            #note np.where returns a list of integer indices of locations where
            #the condition is true, , not a boolean array
            valid_idx = np.where(x2 > 0)[0]  # shape (n,)
            
            
            
            any_valid = 0
            
            valid_min_idx = 0
            
            valid_min_val = INFINITY
            
            valid_idx_memview = np.empty(num_nearest, dtype=np.uint8, order='C')
            
            for idx in range(num_nearest):
                
                temp_f64_val = x2_memview[idx]
                
                if temp_f64_val > 0.0:
                    
                    valid_idx_memview[idx] = 1
                    
                    any_valid = 1
                    
                    if temp_f64_val < valid_min_val:
                        
                        valid_min_idx = idx
                        
                        valid_min_val = temp_f64_val
                    
            
            '''
            if len(valid_idx) > 0:
                # Find index of 2nd nearest galaxy
                k2g_x2 = valid_idx[x2[valid_idx].argmin()]
                
                k2g = i_nearest[k2g_x2]

                #minx2 = x2[k2g_x2]  # Eliminated transpose on x2

                galaxy_search = False
            '''
                
            if any_valid:
                pass
            
                #used to index into the BA distance array
                k2g_x2 = valid_min_idx
                
                #used to index into the w_coord array
                k2g = i_nearest_memview[valid_min_idx]
                
                galaxy_search = False
                
                
                
                
                
                
                
            
        elif not_in_mask(hole_center_2, mask, min_dist, max_dist):
            # Hole is no longer within survey limits
            galaxy_search = False
            
            in_mask_2 = False

    # Check to make sure that the hole center is still within the survey
    if not in_mask_2:
        #print('hole not in survey')
        
        #put_start = time.time()
        
        #return_queue.put(("data", (None, None, None, None)))
        
        #time_returning += time.time() - put_start
        
        #continue
    
        return_array[0] = NAN
        return_array[1] = NAN
        return_array[2] = NAN
        return_array[3] = NAN
        
        return 

    #print('Found 2nd galaxy')

    ############################################################
    # Update hole center 3
    ############################################################
    
    # Calculate new hole center
    hole_radius = 0.5*np.sum(BA[k2g_x2]**2)/np.dot(BA[k2g_x2], v1_unit.T)  # shape (1,)
    
    
    temp_f64_accum = 0.0
    
    temp_f64_accum2 = 0.0
    
    for idx in range(3):
        
        temp_f64_val = BA_memview[k2g_x2, idx]
        
        temp_f64_accum += temp_f64_val*temp_f64_val
        
        temp_f64_accum2 += BA_memview[k2g_x2, idx]*v1_unit_memview[idx]
        
    hole_radius = 0.5*temp_f64_accum/temp_f64_accum2
        
        
    
    
    
    
    
    
    hole_center = w_coord[k1g] - hole_radius*v1_unit  # shape (1,3)
    
    for idx in range(3):
        
        hole_center_memview[0,idx] = neighbor_1_w_coords_memview[idx] - hole_radius*v1_unit_memview[idx]
    
   
   
   
   
   
   
   
    # Check to make sure that the hole center is still within the survey
    if not_in_mask(hole_center_memview, mask, min_dist, max_dist):
        #print('hole not in survey')
        
        #put_start = time.time()
        
        #return_queue.put(("data", (None, None, None, None)))
        
        #time_returning += time.time() - put_start
        
        #continue
    
        return_array[0] = NAN
        return_array[1] = NAN
        return_array[2] = NAN
        return_array[3] = NAN
        
        return 

    ########################################################################
    # Find Galaxy 3 (closest to cell center)
    #
    # (Same methodology as for finding the second galaxy)
    ########################################################################
    

    # Find the midpoint between the two nearest galaxies
    midpoint = 0.5*(np.add(w_coord[k1g], w_coord[k2g]))  # shape (3,)
    #print('midpoint shape:', midpoint.shape)           
    
    for idx in range(3):
        
        midpoint_memview[idx] = 0.5*(w_coord[k1g,idx]+ w_coord[k2g,idx])
    
    
    
    
    
    
    
    

    # Define the unit vector along which to move the hole center
    # modulus of v2
    modv2 = np.linalg.norm(hole_center - midpoint)
    
    for idx in range(3):
        
        temp_f64_val = hole_center_memview[0,idx] - midpoint_memview[idx]
        
        temp_f64_accum += temp_f64_val*temp_f64_val
    
    modv2 = sqrt(temp_f64_accum)
    
    
    
    
    
    
    
    
    
    
    v2_unit = (hole_center - midpoint)/modv2  # shape (1,3)
    #print('v2_unit shape', v2_unit.shape)
    
    for idx in range(3):
    
        v2_unit_memview[idx] = (hole_center_memview[0,idx] - midpoint_memview[idx])/modv2
    
    
    
    
    
    
    

    # Calculate vector pointing from the hole center to the nearest galaxy
    Acenter = w_coord[k1g] - hole_center  # shape (1,3)
    
    for idx in range(3):
        
        Acenter_memview[idx] = w_coord[k1g, idx] - hole_center_memview[0,idx]
    
    
    
    
    
    
    
    # Calculate vector pointing from the hole center to the second-nearest galaxy
    Bcenter = w_coord[k2g] - hole_center  # shape (1,3)
    
    for idx in range(3):
        
        Bcenter_memview[idx] = w_coord[k2g, idx] - hole_center_memview[0,idx]
    
    
    

    # Initialize moving hole center
    hole_center_3 = hole_center  # shape (1,3)
    
    for idx in range(3):
        
        hole_center_3_memview[0,idx] = hole_center_memview[0,idx]
    
    
    
    
    
    
    

    galaxy_search = True

    in_mask_3 = True

    while galaxy_search:

        # Shift hole center along unit vector
        hole_center_3 = hole_center_3 + dr*v2_unit
        
        for idx in range(3):
            
            
            hole_center_3_memview[0,idx] = hole_center_3_memview[0,idx] + dr*v2_unit_memview[idx]
        
        
        
        
        
        
        
        
        

        # New hole "radius"
        search_radius = np.linalg.norm(w_coord[k1g] - hole_center_3)
        
        temp_f64_accum = 0.0
        
        for idx in range(3):
            
            temp_f64_val = w_coord[k1g,idx] - hole_center_3_memview[0,idx]
            
            temp_f64_accum += temp_f64_val*temp_f64_val
            
        search_radius = sqrt(temp_f64_accum)
        
        
        
        
        
        
        
        
        
        
        
        
        
        # Search for nearest neighbors within modv1 of the hole center
        # i_nearest, dist_nearest = galaxy_tree.query_radius(hole_center, r=np.linalg.norm(Acenter), return_distance=True, sort_results=True)
        i_nearest = galaxy_tree.query_radius(hole_center_3_memview, r=search_radius)

        i_nearest = i_nearest[0]

        # Remove two nearest galaxies from list
        boolean_nearest = np.logical_and(i_nearest != k1g, i_nearest != k2g)
        
        i_nearest = i_nearest[boolean_nearest]
        #dist_nearest = dist_nearest[boolean_nearest]







        #num_nearest is int of ITYPE_t
        num_nearest = i_nearest.shape[0]

        if num_nearest > 0:
            # Found at least one other nearest neighbor!
            
            
            
            
            i_nearest_memview = i_nearest
            
            # Calculate vector pointing from hole center to next nearest galaxies
            
            temp_1 = np.take(w_coord, i_nearest, axis=0)
            
            Ccenter = np.subtract(temp_1, hole_center)  # shape (N,3)
            
            Ccenter_memview = np.empty((num_nearest, 3), dtype=np.float64, order='C')
            
            for idx in range(num_nearest):
                
                for jdx in range(3):
                    
                    temp_idx = i_nearest_memview[idx]
                    
                    Ccenter_memview[idx,jdx] = w_coord[temp_idx, jdx] - hole_center_memview[0,jdx]
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            C_minus_A_center_memview = np.empty((num_nearest, 3), dtype=np.float64, order='C')
            
            for idx in range(num_nearest):
                
                for jdx in range(3):
                    
                    C_minus_A_center_memview[idx, jdx] = Ccenter_memview[idx, jdx] - Acenter_memview[jdx]
            
            
            
            
            
            
            
            
            
            
            bot = 2*np.dot((Ccenter - Acenter), v2_unit.T)  # shape (N,1)
            
            bot_memview = np.empty(num_nearest, dtype=np.float64, order='C')
            
            for idx in range(num_nearest):
                
                temp_f64_accum = 0.0
                
                for jdx in range(3):
                    
                    temp_f64_accum += C_minus_A_center_memview[idx,jdx]*v2_unit_memview[jdx]
                    
                bot_memview[idx] = temp_f64_accum
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            top = np.sum(Ccenter**2, axis=1) - np.sum(Bcenter**2)  # shape (N,)
            
            top_memview = np.empty(num_nearest, dtype=np.float64, order='C')
            
            temp_f64_accum = 0.0
            
            for idx in range(3):
                
                temp_f64_accum += Bcenter_memview[idx]*Bcenter_memview[idx]
                
            temp_f64_val = temp_f64_accum
            
            for idx in range(num_nearest):
                
                temp_f64_accum = 0.0
                
                for jdx in range(3):
                    
                    temp_f64_accum += Ccenter_memview[idx,jdx]*Ccenter_memview[idx,jdx]
                    
                top_memview[idx] = temp_f64_accum - temp_f64_val
            
            
            
            
            
            
            
            
            
            
            
            
            
            x3 = top/bot.T[0]  # shape (N,)

            # Locate positive values of x3
            valid_idx = np.where(x3 > 0)[0]  # shape (N,)

            if len(valid_idx) > 0:
                # Find index of 3rd nearest galaxy
                k3g_x3 = valid_idx[x3[valid_idx].argmin()]
                k3g = i_nearest[k3g_x3]

                minx3 = x3[k3g_x3]

                galaxy_search = False

        #elif not in_mask(hole_center_3, mask, [min_dist, max_dist]):
        elif not_in_mask(hole_center_3, mask, min_dist, max_dist):
            # Hole is no longer within survey limits
            galaxy_search = False
            in_mask_3 = False

    # Check to make sure that the hole center is still within the survey
    #if not in_mask(hole_center_3, mask, [min_dist, max_dist]):
    #if not_in_mask(hole_center_3, mask, min_dist, max_dist):
    if not in_mask_3:
        #print('hole not in survey')
        
        #put_start = time.time()
        
        #return_queue.put(("data", (None, None, None, None)))
        
        #time_returning += time.time() - put_start
        
        #continue
    
        return_array[0] = NAN
        return_array[1] = NAN
        return_array[2] = NAN
        return_array[3] = NAN
        
        return 

    #print('Found 3rd galaxy')
    
    ############################################################
    # Update hole center 4
    ############################################################
    hole_center = hole_center + minx3*v2_unit  # shape (1,3)
    
    hole_radius = np.linalg.norm(hole_center - w_coord[k1g])  # shape ()

    # Check to make sure that the hole center is still within the survey
    if not_in_mask(hole_center, mask, min_dist, max_dist):
        #print('hole not in survey')
        
        #put_start = time.time()
        
        #return_queue.put(("data", (None, None, None, None)))
        
        #time_returning += time.time() - put_start
        
        #continue
    
        return_array[0] = NAN
        return_array[1] = NAN
        return_array[2] = NAN
        return_array[3] = NAN
        
        return 


    ########################################################################
    #
    # Find Galaxy 4 
    #
    # Process is very similar as before, except we do not know if we have to 
    # move above or below the plane.  Therefore, we will find the next closest 
    # if we move above the plane, and the next closest if we move below the 
    # plane.
    ########################################################################

    # The vector along which to move the hole center is defined by the cross 
    # product of the vectors pointing between the three nearest galaxies.
    AB = np.subtract(w_coord[k1g], w_coord[k2g])  # shape (3,)
    BC = np.subtract(w_coord[k3g], w_coord[k2g])  # shape (3,)
    v3 = np.cross(AB,BC)  # shape (3,)
    
    
    modv3 = np.linalg.norm(v3)
    v3_unit = v3/modv3  # shape (3,)

    # Calculate vector pointing from the hole center to the nearest galaxy
    Acenter = np.subtract(w_coord[k1g], hole_center)  # shape (1,3)
    # Calculate vector pointing from the hole center to the second-nearest galaxy
    Bcenter = np.subtract(w_coord[k2g], hole_center)  # shape (1,3)


    # First move in the direction of the unit vector defined above

    galaxy_search = True
    
    hole_center_41 = hole_center 

    in_mask_41 = True

    while galaxy_search:

        # Shift hole center along unit vector
        hole_center_41 = hole_center_41 + dr*v3_unit
        #print('Shifted center to', hole_center_41)

        # New hole "radius"
        search_radius = np.linalg.norm(w_coord[k1g] - hole_center_41)

        # Search for nearest neighbors within R of the hole center
        #i_nearest, dist_nearest = galaxy_tree.query_radius(hole_center_41, r=np.linalg.norm(Acenter), return_distance=True, sort_results=True)
        i_nearest = galaxy_tree.query_radius(hole_center_41, r=search_radius)

        i_nearest = i_nearest[0]
        #dist_nearest = dist_nearest[0]

        # Remove two nearest galaxies from list
        boolean_nearest = np.logical_and.reduce((i_nearest != k1g, i_nearest != k2g, i_nearest != k3g))
        i_nearest = i_nearest[boolean_nearest]
        #dist_nearest = dist_nearest[boolean_nearest]
        #print('Number of nearby galaxies', len(i_nearest))

        #if i_nearest.shape[0] > 0:
        if len(i_nearest) > 0:
            # Found at least one other nearest neighbor!

            # Calculate vector pointing from hole center to next nearest galaxies
            
            temp_1 = np.take(w_coord, i_nearest, axis=0)
            
            Dcenter = np.subtract(temp_1, hole_center)  # shape (N,3)
            #print('Dcenter shape:', Dcenter.shape)
            
            bot = 2*np.dot((Dcenter - Acenter), v3_unit.T)  # shape (N,)
            #print('bot shape:', bot.shape)
            
            top = np.sum(Dcenter**2, axis=1) - np.sum(Bcenter**2)  # shape (N,)
            #print('top shape:', top.shape)
            
            x41 = top/bot.T[0]  # shape (N,)
            #print('x41 shape:', x41.shape)

            # Locate positive values of x41
            valid_idx = np.where(x41 > 0)[0]  # shape (n,)
            #print('valid_idx shape:', valid_idx.shape)

            #if valid_idx.shape[0] == 1:
            #    k4g1 = i_nearest[valid_idx[0]]
            #    minx41 = x41[valid_idx[0]]
            #    galaxy_search = False
            #    
            #elif valid_idx.shape[0] > 1:
            if len(valid_idx) > 0:
                # Find index of 4th nearest galaxy
                k4g1_x41 = valid_idx[x41[valid_idx].argmin()]
                k4g1 = i_nearest[k4g1_x41]

                minx41 = x41[k4g1_x41]

                galaxy_search = False


        #elif not in_mask(hole_center_41, mask, [min_dist, max_dist]):
        elif not_in_mask(hole_center_41, mask, min_dist, max_dist):
            # Hole is no longer within survey limits
            galaxy_search = False
            in_mask_41 = False

    #print('Found first potential 4th galaxy')
    '''
    if k4g1 == i_nearest[0]:
        print('First 4th galaxy was the next nearest neighbor.')
    else:
        print('First 4th galaxy was NOT the next nearest neighbor.')
    '''

    # Calculate potential new hole center
    #if in_mask(hole_center_41, mask, [min_dist, max_dist]):
    #if not not_in_mask(hole_center_41, mask, min_dist, max_dist):
    if in_mask_41:
        hole_center_41 = hole_center + minx41*v3_unit  # shape (1,3)
        #print('______________________')
        #print(hole_center_41, 'hc41')
        #print('hole_radius_41', np.linalg.norm(hole_center_41 - w_coord[k1g]))
   
    ########################################################################
    # Repeat same search, but shift the hole center in the other direction 
    # this time
    ########################################################################
    v3_unit = -v3_unit

    # First move in the direction of the unit vector defined above
    galaxy_search = True

    # Initialize minx42 (in case it does not get created later)
    minx42 = np.infty

    hole_center_42 = hole_center
    
    minx42 = np.infty

    in_mask_42 = True

    while galaxy_search:

        # Shift hole center along unit vector
        hole_center_42 = hole_center_42 + dr*v3_unit

        # New hole "radius"
        search_radius = np.linalg.norm(w_coord[k1g] - hole_center_42)

        # Search for nearest neighbors within R of the hole center
        #i_nearest, dist_nearest = galaxy_tree.query_radius(hole_center_42, r=np.linalg.norm(Acenter), return_distance=True, sort_results=True)
        i_nearest = galaxy_tree.query_radius(hole_center_42, r=search_radius)

        i_nearest = i_nearest[0]
        #dist_nearest = dist_nearest[0]

        # Remove three nearest galaxies from list
        boolean_nearest = np.logical_and.reduce((i_nearest != k1g, i_nearest != k2g, i_nearest != k3g))
        i_nearest = i_nearest[boolean_nearest]
        #dist_nearest = dist_nearest[boolean_nearest]

        if len(i_nearest) > 0:
            # Found at least one other nearest neighbor!

            # Calculate vector pointing from hole center to next nearest galaxies
            
            temp_1 = np.take(w_coord, i_nearest, axis=0)
            
            Dcenter = np.subtract(temp_1, hole_center)  # shape (N,3)

            bot = 2*np.dot((Dcenter - Acenter), v3_unit.T)  # shape (N,)

            top = np.sum(Dcenter**2, axis=1) - np.sum(Bcenter**2)  # shape (N,)

            x42 = top/bot.T[0]  # shape (N,)

            # Locate positive values of x42
            valid_idx = np.where(x42 > 0)[0]  # shape (n,)

            if len(valid_idx) > 0:
                # Find index of 3rd nearest galaxy
                k4g2_x42 = valid_idx[x42[valid_idx].argmin()]
                k4g2 = i_nearest[k4g2_x42]

                minx42 = x42[k4g2_x42]

                galaxy_search = False

        #elif not in_mask(hole_center_42, mask, [min_dist, max_dist]):
        elif not_in_mask(hole_center_42, mask, min_dist, max_dist):
            # Hole is no longer within survey limits
            galaxy_search = False
            in_mask_42 = False

    #print('Found second potential 4th galaxy')
    '''
    if k4g2 == i_nearest[0]:
        print('Second 4th galaxy was the next nearest neighbor.')
    else:
        print('Second 4th galaxy was NOT the next nearest neighbor.')
    '''

    # Calculate potential new hole center
    #if in_mask(hole_center_42, mask, [min_dist, max_dist]):
    #if not not_in_mask(hole_center_42, mask, min_dist, max_dist):
    if in_mask_42:
        hole_center_42 = hole_center + minx42*v3_unit  # shape (1,3)
        #print(hole_center_42, 'hc42')
        #print('hole_radius_42', np.linalg.norm(hole_center_42 - w_coord[k1g]))
        #print('minx41:', minx41, '   minx42:', minx42)
    
    
    ########################################################################
    # Figure out which is the real galaxy 4
    ########################################################################
    '''
    if not in_mask(hole_center_41, mask, [min_dist, max_dist]):
        print('hole_center_41 is NOT in the mask')
    if not in_mask(hole_center_42, mask, [min_dist, max_dist]):
        print('hole_center_42 is NOT in the mask')
    '''
    
    # Determine which is the 4th nearest galaxy
    #if in_mask(hole_center_41, mask, [min_dist, max_dist]) and minx41 <= minx42:
    not_in_mask_41 = not_in_mask(hole_center_41, mask, min_dist, max_dist)
    if not not_in_mask_41 and minx41 <= minx42:
        # The first 4th galaxy found is the next closest
        hole_center = hole_center_41
        k4g = k4g1
    #elif in_mask(hole_center_42, mask, [min_dist, max_dist]):
    elif not not_in_mask(hole_center_42, mask, min_dist, max_dist):
        # The second 4th galaxy found is the next closest
        hole_center = hole_center_42
        k4g = k4g2
    #elif in_mask(hole_center_41, mask, [min_dist, max_dist]):
    elif not not_in_mask_41:
        # The first 4th galaxy found is the next closest
        hole_center = hole_center_41
        k4g = k4g1
    else:
        # Neither hole center is within the mask - not a valid hole
        
        #put_start = time.time()
        
        #return_queue.put(("data", (None, None, None, None)))
        
        #time_returning += time.time() - put_start
        
        #continue
    
        return_array[0] = NAN
        return_array[1] = NAN
        return_array[2] = NAN
        return_array[3] = NAN
        
        return 

    ########################################################################
    # Calculate Radius of the hole
    ########################################################################
    hole_radius = np.linalg.norm(hole_center - w_coord[k1g])

    ########################################################################
    # Save hole
    ########################################################################
    #myvoids_x.append(hole_center[0,0])
    x_val = hole_center[0,0]
    
    #myvoids_y.append(hole_center[0,1])
    y_val = hole_center[0,1]
    
    #myvoids_z.append(hole_center[0,2])
    z_val = hole_center[0,2]
    
    #myvoids_r.append(hole_radius)
    r_val = hole_radius
    
    #hole_times.append(time.time() - hole_start)
    
    #print(hole_times[n_holes], i,j,k)
    
    #n_holes += 1
    
    #put_start = time.time()
    
    #return_queue.put(("data", (x_val, y_val, z_val, r_val)))
    
    #time_returning += time.time() - put_start

    return_array[0] = x_val
    return_array[1] = y_val
    return_array[2] = z_val
    return_array[3] = r_val
    
    return 


    #return (x_val, y_val, z_val, r_val)






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






################################################################################
################################################################################




