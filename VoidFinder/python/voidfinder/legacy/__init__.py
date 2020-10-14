






        

    
def run_single_process_legacy(cell_ID_list, 
                       ngrid, 
                       dl, 
                       dr,
                       coord_min, 
                       mask,
                       mask_resolution,
                       min_dist,
                       max_dist,
                       w_coord,
                       batch_size=1000,
                       verbose=0,
                       print_after=10000,
                       num_cpus=None):
    
    """
    
    TODO:  Significant work has been done on updating the run_single and run_multi Cython
           methods in this module, and for instance they now return an (N,4) numpy
           array instead of 4 lists of x,y,z,r.  This method has not been updated
           appropriately to the new standard yet.
    
    """
    
    
    
    from sklearn import neighbors
    
    
    ################################################################################
    #
    # Profiling parameters
    #
    ################################################################################
    
    #PROFILE_total_query_time = 0.0
    
    PROFILE_total_start = time.time()
    
    #PROFILE_mask_checking_time = 0.0
    
    PROFILE_mask_times = []
    
    PROFILE_loop_times = []
    
    PROFILE_query_times = []
    
    PROFILE_section_1_times = []
    PROFILE_section_2_times = []
    PROFILE_section_3_times = []
    PROFILE_section_4_times = []
    PROFILE_void_times = []
    
    
    
    
    ################################################################################
    #
    # Initialize some output containers and counter variables
    #
    ################################################################################
    
    print("Running single-process mode")
    
    #hole_times = []
    
    # Initialize list of hole details
    myvoids_x = []
    
    myvoids_y = []
    
    myvoids_z = []
    
    myvoids_r = []
    
    # Number of holes found
    n_holes = 0

    # Counter for the number of empty cells
    empty_cell_counter = 0
    
    # Number of empty cells
    #n_empty_cells = ngrid[0]*ngrid[1]*ngrid[2] - len(cell_ID_dict)
    n_empty_cells = len(cell_ID_list)
    
    ################################################################################
    #
    #   BUILD NEAREST-NEIGHBOR TREE
    #   galaxy_tree : sklearn.neighbors/scipy KDTree or similar implementing sklearn interface
    #   nearest neighbor finder for the galaxies in x,y,z space
    #
    ################################################################################
    
    if verbose:
        
        kdtree_start_time = time.time()

    galaxy_tree = neighbors.KDTree(w_coord)
    
    if verbose:
        
        print('KDTree creation time:', time.time() - kdtree_start_time)
    
    
    ################################################################################
    # Main loop
    ################################################################################
    num_cells_processed = 0
    
    for hole_center_coords in cell_ID_list:
                    
        num_cells_processed += 1
        
        #print(hole_center_coords)
        
        
        PROFILE_loop_start_time = time.time()
        
        
        if verbose:
            
            if num_cells_processed % 10000 == 0:
                
               print('Processed', num_cells_processed, 'cells of', n_empty_cells)
        
        
        #if num_cells_processed % 10000 == 0:
        #    print("Processed: ", num_cells_processed, "time: ", time.time() - worker_lifetime_start, "main: ", time_main, "empty: ", time_empty)
        
            
        i, j, k = hole_center_coords
        

        hole_center = (np.array([[i, j, k]]) + 0.5)*dl + coord_min  # Purposefully making hole_center have shape (1,3) for KDtree queries
                        
        #hole_center = to_vector(hole_center_table)
        
        
        ############################################################
        # Check to make sure in mask
        ############################################################
        
        timer1 = time.time()
        temp = not_in_mask(hole_center, mask, mask_resolution, min_dist, max_dist)
        timer2 = time.time()
        PROFILE_mask_times.append(timer2- timer1)
        # Check to make sure that the hole center is still within the survey
        if temp:
            
            PROFILE_loop_times.append(time.time() - PROFILE_loop_start_time)
            
            continue
        
        
        ############################################################
        #
        # Find Galaxy 1 (closest to cell center)
        #
        # and calculate Unit vector pointing from cell 
        # center to the closest galaxy
        #
        ############################################################
        PROFILE_1_start = time.time()
        
        
        timer1 = time.time()
        modv1, k1g = galaxy_tree.query(hole_center, k=1)
        timer2 = time.time()
        PROFILE_query_times.append(timer2-timer1)
        
        modv1 = modv1[0][0]
        
        k1g = k1g[0][0]
    
        v1_unit = (w_coord[k1g] - hole_center)/modv1
        
        
        PROFILE_1_time = time.time() - PROFILE_1_start
    
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
        
        PROFILE_2_start = time.time()
    
        galaxy_search = True
    
        hole_center_2 = hole_center
    
        in_mask_2 = True
    
        while galaxy_search:
    
            # Shift hole center away from first galaxy
            hole_center_2 = hole_center_2 - dr*v1_unit
            
            # Distance between hole center and nearest galaxy
            modv1 += dr
            
            # Search for nearest neighbors within modv1 of the hole center
            timer1 = time.time()
            i_nearest = galaxy_tree.query_radius(hole_center_2, r=modv1)
            timer2 = time.time()
            PROFILE_query_times.append(timer2-timer1)
    
            i_nearest = i_nearest[0]
            #dist_nearest = dist_nearest[0]
    
            # Remove nearest galaxy from list
            boolean_nearest = i_nearest != k1g
            
            i_nearest = i_nearest[boolean_nearest]
            
            if len(i_nearest) <= 0:
                timer1 = time.time()
                PROFILE_TEMP = not_in_mask(hole_center_2, mask, mask_resolution, min_dist, max_dist)
                timer2 = time.time()
                PROFILE_mask_times.append(timer2- timer1)
                
    
            if len(i_nearest) > 0:
                # Found at least one other nearest neighbor!
    
                # Calculate vector pointing from next nearest galaxies to the nearest galaxy
                BA = w_coord[k1g] - w_coord[i_nearest]  # shape (N,3)
                
                bot = 2*np.dot(BA, v1_unit.T)  # shape (N,1)
                
                top = np.sum(BA**2, axis=1)  # shape (N,)
                
                x2 = top/bot.T[0]  # shape (N,)
    
                # Locate positive values of x2
                valid_idx = np.where(x2 > 0)[0]  # shape (n,)
                
                if len(valid_idx) > 0:
                    # Find index of 2nd nearest galaxy
                    k2g_x2 = valid_idx[x2[valid_idx].argmin()]
                    
                    k2g = i_nearest[k2g_x2]
    
                    minx2 = x2[k2g_x2]  # Eliminated transpose on x2
    
                    galaxy_search = False
                
            elif PROFILE_TEMP:
                # Hole is no longer within survey limits
                galaxy_search = False
                
                in_mask_2 = False
    
        # Check to make sure that the hole center is still within the survey
        if not in_mask_2:
            #print('hole not in survey')
            PROFILE_loop_times.append(time.time() - PROFILE_loop_start_time)
            
            continue
    
        #print('Found 2nd galaxy')
    
    
        PROFILE_2_time = time.time() - PROFILE_2_start
    
        ############################################################
        # Update hole center
        ############################################################
        
        # Calculate new hole center
        hole_radius = 0.5*np.sum(BA[k2g_x2]**2)/np.dot(BA[k2g_x2], v1_unit.T)  # shape (1,)
        
        hole_center = w_coord[k1g] - hole_radius*v1_unit  # shape (1,3)
       
        # Check to make sure that the hole center is still within the survey
        
        timer1 = time.time()
        temp = not_in_mask(hole_center, mask, mask_resolution, min_dist, max_dist)
        timer2 = time.time()
        PROFILE_mask_times.append(timer2- timer1)
        
        if temp:
            #print('hole not in survey')
            PROFILE_loop_times.append(time.time() - PROFILE_loop_start_time)
            
            continue
    
        ########################################################################
        # Find Galaxy 3 (closest to cell center)
        #
        # (Same methodology as for finding the second galaxy)
        ########################################################################
        
        PROFILE_3_start = time.time()
        
    
        # Find the midpoint between the two nearest galaxies
        midpoint = 0.5*(w_coord[k1g] + w_coord[k2g])  # shape (3,)
        #print('midpoint shape:', midpoint.shape)           
    
        # Define the unit vector along which to move the hole center
        modv2 = np.linalg.norm(hole_center - midpoint)
        v2_unit = (hole_center - midpoint)/modv2  # shape (1,3)
        #print('v2_unit shape', v2_unit.shape)
    
        # Calculate vector pointing from the hole center to the nearest galaxy
        Acenter = w_coord[k1g] - hole_center  # shape (1,3)
        # Calculate vector pointing from the hole center to the second-nearest galaxy
        Bcenter = w_coord[k2g] - hole_center  # shape (1,3)
    
        # Initialize moving hole center
        hole_center_3 = hole_center  # shape (1,3)
    
        galaxy_search = True
    
        in_mask_3 = True
    
        while galaxy_search:
    
            # Shift hole center along unit vector
            hole_center_3 = hole_center_3 + dr*v2_unit
    
            # New hole "radius"
            search_radius = np.linalg.norm(w_coord[k1g] - hole_center_3)
            
            # Search for nearest neighbors within modv1 of the hole center
            #i_nearest, dist_nearest = galaxy_tree.query_radius(hole_center, r=np.linalg.norm(Acenter), return_distance=True, sort_results=True)
            timer1 = time.time()
            i_nearest = galaxy_tree.query_radius(hole_center_3, r=search_radius)
            timer2 = time.time()
            PROFILE_query_times.append(timer2 - timer1)
    
            i_nearest = i_nearest[0]
            #dist_nearest = dist_nearest[0]
    
            # Remove two nearest galaxies from list
            boolean_nearest = np.logical_and(i_nearest != k1g, i_nearest != k2g)
            i_nearest = i_nearest[boolean_nearest]
            #dist_nearest = dist_nearest[boolean_nearest]
    
    
            if len(i_nearest) <= 0:
                timer1 = time.time()
                PROFILE_TEMP = not_in_mask(hole_center_3, mask, mask_resolution, min_dist, max_dist)
                timer2 = time.time()
                PROFILE_mask_times.append(timer2- timer1)
    
    
            if len(i_nearest) > 0:
                # Found at least one other nearest neighbor!
    
                # Calculate vector pointing from hole center to next nearest galaxies
                Ccenter = w_coord[i_nearest] - hole_center  # shape (N,3)
                
                bot = 2*np.dot((Ccenter - Acenter), v2_unit.T)  # shape (N,1)
                
                top = np.sum(Ccenter**2, axis=1) - np.sum(Bcenter**2)  # shape (N,)
                
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
            elif PROFILE_TEMP:
                # Hole is no longer within survey limits
                galaxy_search = False
                in_mask_3 = False
    
        # Check to make sure that the hole center is still within the survey
        #if not in_mask(hole_center_3, mask, [min_dist, max_dist]):
        #if not_in_mask(hole_center_3, mask, min_dist, max_dist):
        if not in_mask_3:
            #print('hole not in survey')
            PROFILE_loop_times.append(time.time() - PROFILE_loop_start_time)
            
            continue
    
        #print('Found 3rd galaxy')
        
        PROFILE_3_time = time.time() - PROFILE_3_start
        
        
        ############################################################
        # Update hole center
        ############################################################
        hole_center = hole_center + minx3*v2_unit  # shape (1,3)
        
        hole_radius = np.linalg.norm(hole_center - w_coord[k1g])  # shape ()
    
        # Check to make sure that the hole center is still within the survey
        timer1 = time.time()
        temp = not_in_mask(hole_center, mask, mask_resolution, min_dist, max_dist)
        timer2 = time.time()
        PROFILE_mask_times.append(timer2- timer1)
        if temp:
            #print('hole not in survey')
            PROFILE_loop_times.append(time.time() - PROFILE_loop_start_time)
            
            continue
    
    
        ########################################################################
        #
        # Find Galaxy 4 
        #
        # Process is very similar as before, except we do not know if we have to 
        # move above or below the plane.  Therefore, we will find the next closest 
        # if we move above the plane, and the next closest if we move below the 
        # plane.
        ########################################################################
        
        PROFILE_4_start = time.time()
        
    
        # The vector along which to move the hole center is defined by the cross 
        # product of the vectors pointing between the three nearest galaxies.
        AB = w_coord[k1g] - w_coord[k2g]  # shape (3,)
        BC = w_coord[k3g] - w_coord[k2g]  # shape (3,)
        v3 = np.cross(AB,BC)  # shape (3,)
        
        
        modv3 = np.linalg.norm(v3)
        v3_unit = v3/modv3  # shape (3,)
    
        # Calculate vector pointing from the hole center to the nearest galaxy
        Acenter = w_coord[k1g] - hole_center  # shape (1,3)
        # Calculate vector pointing from the hole center to the second-nearest galaxy
        Bcenter = w_coord[k2g] - hole_center  # shape (1,3)
    
    
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
            timer1 = time.time()
            i_nearest = galaxy_tree.query_radius(hole_center_41, r=search_radius)
            timer2 = time.time()
            PROFILE_query_times.append(timer2 - timer1)
    
            i_nearest = i_nearest[0]
            #dist_nearest = dist_nearest[0]
    
            # Remove two nearest galaxies from list
            boolean_nearest = np.logical_and.reduce((i_nearest != k1g, i_nearest != k2g, i_nearest != k3g))
            i_nearest = i_nearest[boolean_nearest]
            #dist_nearest = dist_nearest[boolean_nearest]
            #print('Number of nearby galaxies', len(i_nearest))
    
    
            if len(i_nearest) <= 0:
                timer1 = time.time()
                PROFILE_TEMP = not_in_mask(hole_center_41, mask, mask_resolution, min_dist, max_dist)
                timer2 = time.time()
                PROFILE_mask_times.append(timer2- timer1)
    
    
    
            #if i_nearest.shape[0] > 0:
            if len(i_nearest) > 0:
                # Found at least one other nearest neighbor!
    
                # Calculate vector pointing from hole center to next nearest galaxies
                Dcenter = w_coord[i_nearest] - hole_center  # shape (N,3)
                #print('Dcenter shape:', Dcenter.shape)
                
                bot = 2*np.dot((Dcenter - Acenter), v3_unit.T)  # shape (N,)
                #print('bot shape:', bot.shape)
                
                top = np.sum(Dcenter**2, axis=1) - np.sum(Bcenter**2)  # shape (N,)
                #print('top shape:', top.shape)
                
                x41 = top/bot  # shape (N,)
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
    
    
            #elif not in_mask(hole_center_41, mask, mask_resolution, [min_dist, max_dist]):
            elif PROFILE_TEMP:
                # Hole is no longer within survey limits
                galaxy_search = False
                in_mask_41 = False
    
        #print('Found first potential 4th galaxy')
        
    
        # Calculate potential new hole center
        #if not not_in_mask(hole_center_41, mask, mask_resolution, min_dist, max_dist):
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
            timer1 = time.time()
            i_nearest = galaxy_tree.query_radius(hole_center_42, r=search_radius)
            timer2 = time.time()
            PROFILE_query_times.append(timer2 - timer1)
    
            i_nearest = i_nearest[0]
            #dist_nearest = dist_nearest[0]
    
            # Remove three nearest galaxies from list
            boolean_nearest = np.logical_and.reduce((i_nearest != k1g, i_nearest != k2g, i_nearest != k3g))
            i_nearest = i_nearest[boolean_nearest]
            #dist_nearest = dist_nearest[boolean_nearest]
    
            if len(i_nearest) <= 0:
                timer1 = time.time()
                PROFILE_TEMP = not_in_mask(hole_center_42, mask, mask_resolution, min_dist, max_dist)
                timer2 = time.time()
                PROFILE_mask_times.append(timer2- timer1)
    
            if len(i_nearest) > 0:
                # Found at least one other nearest neighbor!
    
                # Calculate vector pointing from hole center to next nearest galaxies
                Dcenter = w_coord[i_nearest] - hole_center  # shape (N,3)
    
                bot = 2*np.dot((Dcenter - Acenter), v3_unit.T)  # shape (N,)
    
                top = np.sum(Dcenter**2, axis=1) - np.sum(Bcenter**2)  # shape (N,)
    
                x42 = top/bot  # shape (N,)
    
                # Locate positive values of x42
                valid_idx = np.where(x42 > 0)[0]  # shape (n,)
    
                if len(valid_idx) > 0:
                    # Find index of 3rd nearest galaxy
                    k4g2_x42 = valid_idx[x42[valid_idx].argmin()]
                    k4g2 = i_nearest[k4g2_x42]
    
                    minx42 = x42[k4g2_x42]
    
                    galaxy_search = False
    
            #elif not in_mask(hole_center_42, mask, mask_resolution, [min_dist, max_dist]):
            elif PROFILE_TEMP:
                # Hole is no longer within survey limits
                galaxy_search = False
                in_mask_42 = False
    
        #print('Found second potential 4th galaxy')
        
    
        # Calculate potential new hole center
        #if not not_in_mask(hole_center_42, mask, mask_resolution, min_dist, max_dist):
        if in_mask_42:
            hole_center_42 = hole_center + minx42*v3_unit  # shape (1,3)
            #print(hole_center_42, 'hc42')
            #print('hole_radius_42', np.linalg.norm(hole_center_42 - w_coord[k1g]))
            #print('minx41:', minx41, '   minx42:', minx42)
        
        
        ########################################################################
        # Figure out which is the real galaxy 4
        ########################################################################
        
        
        # Determine which is the 4th nearest galaxy
        #if in_mask(hole_center_41, mask, mask_resolution, [min_dist, max_dist]) and minx41 <= minx42:
        timer1 = time.time()
        not_in_mask_41 = not_in_mask(hole_center_41, mask, mask_resolution, min_dist, max_dist)
        timer2 = time.time()
        PROFILE_mask_times.append(timer2- timer1)
        
        
        if not not_in_mask_41 and minx41 <= minx42:
            # The first 4th galaxy found is the next closest
            hole_center = hole_center_41
            k4g = k4g1
        elif not not_in_mask(hole_center_42, mask, mask_resolution, min_dist, max_dist):
            # The second 4th galaxy found is the next closest
            
            timer1 = time.time()
            not_in_mask(hole_center_42, mask, mask_resolution, min_dist, max_dist)
            timer2 = time.time()
            PROFILE_mask_times.append(timer2- timer1)
            
            
            
            
            hole_center = hole_center_42
            k4g = k4g2
        elif not not_in_mask_41:
            # The first 4th galaxy found is the next closest
            hole_center = hole_center_41
            k4g = k4g1
        else:
            # Neither hole center is within the mask - not a valid hole
            PROFILE_loop_times.append(time.time() - PROFILE_loop_start_time)
            
            continue
    
    
        PROFILE_4_time = time.time() - PROFILE_4_start
    
        ########################################################################
        # Calculate Radius of the hole
        ########################################################################
        hole_radius = np.linalg.norm(hole_center - w_coord[k1g])
    
        
        ########################################################################
        # Save hole
        ########################################################################
        
        
        PROFILE_section_1_times.append(PROFILE_1_time)
        PROFILE_section_2_times.append(PROFILE_2_time)
        PROFILE_section_3_times.append(PROFILE_3_time)
        PROFILE_section_4_times.append(PROFILE_4_time)
        
        
        
        myvoids_x.append(hole_center[0,0])
        #x_val = hole_center[0,0]
        
        myvoids_y.append(hole_center[0,1])
        #y_val = hole_center[0,1]
        
        myvoids_z.append(hole_center[0,2])
        #z_val = hole_center[0,2]
        
        myvoids_r.append(hole_radius)
        #r_val = hole_radius
        
        #hole_times.append(time.time() - hole_start)
        
        #print(hole_times[n_holes], i,j,k)
        
        n_holes += 1
        
        PROFILE_loop_end_time = time.time()
        
        PROFILE_void_times.append(PROFILE_loop_end_time - PROFILE_loop_start_time)
        
        PROFILE_loop_times.append(PROFILE_loop_end_time - PROFILE_loop_start_time)
    
    
    ########################################################################
    # Profiling statistics
    ########################################################################
    
    
    total_time = time.time() - PROFILE_total_start
    
    print("Total time: ", total_time)
    print("Loop time: ", np.sum(PROFILE_loop_times))
    print("Query time: ", np.sum(PROFILE_query_times))
    print("Mask time: ", np.sum(PROFILE_mask_times))
    print("Total loops: ", len(PROFILE_loop_times))
    print("Total queries: ", len(PROFILE_query_times))
    print("Total masks: ", len(PROFILE_mask_times))
    print("Total (void-cell) time: ", np.sum(PROFILE_void_times))
    print("Section 1 (void) time: ", np.sum(PROFILE_section_1_times))
    print("Section 2 (void) time: ", np.sum(PROFILE_section_2_times))
    print("Section 3 (void) time: ", np.sum(PROFILE_section_3_times))
    print("Section 4 (void) time: ", np.sum(PROFILE_section_4_times))
    
    
    
    fig = plt.figure(figsize=(14,10))
    plt.hist(PROFILE_loop_times, bins=50)
    plt.title("All Single Cell processing times (sec)")
    #plt.show()
    plt.savefig("Cell_time_dist.png")
    plt.close()
    
    fig = plt.figure(figsize=(14,10))
    plt.hist(PROFILE_query_times, bins=50)
    plt.title("All Query KDTree times (sec)")
    #plt.show()
    plt.savefig("Query_time_dist.png")
    plt.close()
    
    fig = plt.figure(figsize=(14,10))
    plt.hist(PROFILE_mask_times, bins=50)
    plt.title("All calls to not_in_mask times (sec)")
    #plt.show()
    plt.savefig("not_in_mask_time_dist.png")
    plt.close()
    
    fig = plt.figure(figsize=(19.2,12))
    top_left = plt.subplot(221)
    top_right = plt.subplot(222)
    bot_left = plt.subplot(223)
    bot_right = plt.subplot(224)
    top_left.hist(PROFILE_section_1_times, bins=50)
    top_right.hist(PROFILE_section_2_times, bins=50)
    bot_left.hist(PROFILE_section_3_times, bins=50)
    bot_right.hist(PROFILE_section_4_times, bins=50)
    top_left.set_title("(Void cells only) Section 1 times (sec)")
    top_right.set_title("(Void cells only) Section 2 times (sec)")
    bot_left.set_title("(Void cells only) Section 3 times (sec)")
    bot_right.set_title("(Void cells only) Section 4 times (sec)")
    #plt.show()
    plt.savefig("void_cell_section_breakdown_dist.png")
    plt.close()
    
    
    
    
    
        
    return myvoids_x, myvoids_y, myvoids_z, myvoids_r, n_holes



    