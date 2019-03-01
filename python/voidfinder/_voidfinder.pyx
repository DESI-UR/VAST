






import numpy as np

import time

from sklearn import neighbors

from .voidfinder_functions import not_in_mask

from multiprocessing import Queue, Process, cpu_count

from queue import Empty

from copy import deepcopy



def _main_hole_finder(cell_ID_dict, 
                      ngrid, 
                      dl, 
                      dr,
                      coord_min, 
                      mask,
                      min_dist,
                      max_dist,
                      w_coord,
                      batch_size=1000,
                      verbose=True,
                      num_cpus=1):
    '''
    Description:
    ============
    We kinda know what this does, Kelly can fill this in
    
    
    
    Parameters:
    ===========
    
    cell_ID_dict : python dictionary
        keys are tuples of (i,j,k) locations which correspond to a grid cell.
        if the key (i,j,k) is in the dictionary, then that means there is at least 1 galaxy at the corresponding
        grid cell so we should pass over that grid cell since it isn't empty.
    
    ngrid : numpy.ndarray of shape (3,)
        the number of grid cells in each of the 3 x,y,z dimensions
    
    dl : scalar float
        length of each cell in Mpc/h
        
    dr : scalar float
        distance to shift hole centers during iterative void hole growing in Mpc/h
        
    coord_min : numpy.ndarray of shape (3,) (actually shape (1,3)?)
        minimum coordinates of the survey in x,y,z in Mpc/h
        
    mask : numpy.ndarray of shape (N,M) type bool
        represents the survey footprint in ra/dec space.  Value of True indicates that a location
        is within the survey
    
    min_dist : scalar
        minimum redshift in units of Mpc/h
        
    max_dist : scalar
        maximum redshift in units of Mpc/h
        
    w_coord : numpy.ndarray of shape ()
        x,y,z coordinates of the galaxies used in building the query tree
    
    
    
    Returns:
    ========
    
    Fill in later!
    
    
    
    '''
    
    
    '''
    print(mask)
    print(type(mask)) #numpy ndarray
    print(mask.shape)
    print(mask.dtype)
    print(mask.dtype.itemsize)
    
    exit()
    '''
    
    start_time = time.time()
    
    
    
    ################################################################################
    #
    # Initialize some output containers and counter variables
    #
    ################################################################################
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
    n_empty_cells = ngrid[0]*ngrid[1]*ngrid[2] - len(cell_ID_dict)
    
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
        
        print('KDTree creation time:', time.time() - start_time)
        
        
    ################################################################################
    #
    # Pop all the relevant grid cell IDs onto the job queue
    #
    ################################################################################
    cell_ID_list = []

    for i in range(ngrid[0]):
        
        for j in range(ngrid[1]):
            
            for k in range(ngrid[2]):

                check_bin_ID = (i,j,k)

                if check_bin_ID not in cell_ID_dict:
                    
                    #job_queue.put(check_bin_ID)
                    
                    cell_ID_list.append(check_bin_ID)
                    
                    empty_cell_counter += 1
    
    if verbose:
        print("cell_ID_list finish time: ", time.time() - start_time)
    ################################################################################
    #
    # Set up worker processes
    #
    ################################################################################
    
    #job_queue = Queue()
    
    return_queue = Queue()
    
    if num_cpus is None:
       
        num_cpus = cpu_count() - 1
    
    processes = []
    
    job_queues = []
    
    workers_waiting = []
    
    num_active_processes = 0
    
    #num_cell_per_process = int(np.ceil(len(cell_ID_list)/num_cpus))
    
    for proc_idx in range(num_cpus):
        
        #start_idx = proc_idx*num_cell_per_process
        
        #end_idx = (proc_idx+1)*num_cell_per_process
        
        #worker_cell_IDs = cell_ID_list[start_idx:end_idx]
        
        curr_job_queue = Queue()
        
        job_queues.append(curr_job_queue)
        
        workers_waiting.append(True)
        
        worker_args = (proc_idx,
                       deepcopy(galaxy_tree), 
                       ngrid, 
                       dl, 
                       dr,
                       coord_min, 
                       deepcopy(mask),
                       min_dist,
                       max_dist,
                       deepcopy(w_coord),
                       curr_job_queue,
                       return_queue)
        
        p = Process(target=_main_hole_finder_worker, args=worker_args)
        
        p.start()
        
        processes.append(p)
        
        num_active_processes += 1

    if verbose:
        print("Worker processes started time: ", time.time() - start_time)
    ################################################################################
    #
    # Listen on the return_queue for results
    # and feed workers when they become inactive
    #
    ################################################################################
    num_cells_processed = 0
    
    curr_cell_idx = 0
    
    while num_cells_processed < empty_cell_counter:
        
        try:
            
            message = return_queue.get(False)
            
        except Empty:
            
            pass
            #time.sleep(.01)
            
        else:
            
            if message[0] == 'Done':
                
                num_active_processes -= 1
                
            elif message[0] == "empty_job":
                
                workers_waiting[message[1]] = True
                
            elif message[0] == "data":
                
                #append to the correct lists and stuff
                
                x_val, y_val, z_val, r_val = message[1]
                
                if not np.isnan(x_val):
                
                    myvoids_x.append(x_val)
        
                    myvoids_y.append(y_val)
                    
                    myvoids_z.append(z_val)
                    
                    myvoids_r.append(r_val)
                    
                    n_holes += 1
                
                num_cells_processed += 1
                
                
                if verbose:
            
                    if num_cells_processed % 10000 == 0:
                        
                       print('Processed', num_cells_processed, 'cells of', n_empty_cells)
                
                
        for proc_idx, worker_waiting in enumerate(workers_waiting):
            
            
            if curr_cell_idx > len(cell_ID_list):
                break
            
            
            curr_qsize = job_queues[proc_idx].qsize()
            
            if job_queues[proc_idx].qsize() < 3:
                
                num_put = 3 - curr_qsize
                
                for _ in range(num_put):
                    
                    if curr_cell_idx > len(cell_ID_list):
                        break
                
                    job_queues[proc_idx].put(cell_ID_list[curr_cell_idx:(curr_cell_idx+batch_size)])
                    
                    curr_cell_idx += batch_size
                
            
            
            if worker_waiting:
                
                #put an extra item on the queue for a worker who said they are waiting
                
                if curr_cell_idx > len(cell_ID_list):
                    break
                
                job_queues[proc_idx].put(cell_ID_list[curr_cell_idx:(curr_cell_idx+batch_size)])
                
                curr_cell_idx += batch_size
                
                workers_waiting[proc_idx] = False        
        
                
    if verbose:
        print("Main task finish time: ", time.time() - start_time)
                
    ################################################################################
    #
    # Clean up worker processes
    #
    ################################################################################
    
    for proc_idx in range(num_cpus):
        
        job_queues[proc_idx].put(u"exit")
        
    while num_active_processes > 0:
        
        try:
            
            message = return_queue.get(False)
            
        except Empty:
            
            time.sleep(.1)
            
        else:
            
            if message[0] == 'Done':
                
                num_active_processes -= 1
                
    for p in processes:
        
        p.join(None)
                    
    ################################################################################
    #
    # Some summary print statements
    #
    ################################################################################
                    
    if verbose:
        
        print("Num empty cells: ", empty_cell_counter)
        
    return myvoids_x, myvoids_y, myvoids_z, myvoids_r, n_holes
    
    
    




def _main_hole_finder_worker(process_id,
                             galaxy_tree, 
                             ngrid, 
                             dl, 
                             dr,
                             coord_min, 
                             mask,
                             min_dist,
                             max_dist,
                             w_coord,
                             job_queue,
                             return_queue
                             ):
    
    #galaxy_tree = neighbors.KDTree(w_coord)
    
    exit_process = False
    
    
    worker_lifetime_start = time.time()
    time_main = 0.0
    time_message = 0.0
    time_sleeping = 0.0
    time_empty = 0.0
    
    num_message_checks = 0
    
    num_cells_processed = 0
    
    is_waiting = False
    
    num_empty_job_put = 0
    
    total_loops = 0
    
    time_returning = 0
    
    return_array = np.empty(4, dtype=np.float64)
    
    
    while not exit_process:
        
        total_loops += 1
        
        #print("Main loop")
        
        try:
            
            #print("Checking for message")
            
            num_message_checks += 1
            
            message_time_start = time.time()
            
            message = job_queue.get(False)
            
        except Empty:
            
            
            #print("Except Empty")
            
            if is_waiting:
                
                
                #print("Called long sleep!")
                
                time.sleep(.01)
                
                time_sleeping += time.time() - message_time_start
                
            else:
                
                #print("Empty job message")
                
                put_start = time.time()
                
                return_queue.put(("empty_job", process_id))
                
                time_returning += time.time() - put_start
            
                is_waiting = True
                
                num_empty_job_put += 1
                
            time_empty += time.time() - message_time_start
                
            
        else:
            
            #print("Main processing section")
            
            
            curr_msg_check_time = time.time() - message_time_start
            
            time_message += curr_msg_check_time
            
            is_waiting = False
            
            #print("Message check time: ", curr_msg_check_time)
            
            if isinstance(message, str) and message == 'exit':
                
                #print("Received Exit")
                
                exit_process = True
                
            else:
                
                #profiling
                
                #print("Main processing loop section")
                
                main_proc_start_time = time.time()
                
                cell_ID_list = message
                
                for hole_center_coords in cell_ID_list:
                    
                    num_cells_processed += 1
                    
                    
                    #if num_cells_processed % 10000 == 0:
                    #    print("Processed: ", num_cells_processed, "time: ", time.time() - worker_lifetime_start, "main: ", time_main, "empty: ", time_empty)
                    
                        
                    i, j, k = hole_center_coords
                    
                    
                    main_algorithm(i,
                                   j,
                                   k,
                                   galaxy_tree,
                                   w_coord,
                                   dl, 
                                   dr,
                                   coord_min,
                                   mask,
                                   min_dist,
                                   max_dist,
                                   return_array
                                   )
                    
                    x_val = return_array[0]
                    y_val = return_array[1]
                    z_val = return_array[2]
                    r_val = return_array[3]
                    
                    
                    
                    '''
            
                    hole_center = (np.array([[i, j, k]]) + 0.5)*dl + coord_min  # Purposefully making hole_center have shape (1,3) for KDtree queries
                                    
                    #hole_center = to_vector(hole_center_table)
                    
                    # Check to make sure that the hole center is still within the survey
                    if not_in_mask(hole_center, mask, min_dist, max_dist):
                        
                        
                        put_start = time.time()
                        
                        return_queue.put(("data", (None, None, None, None)))
                        
                        time_returning += time.time() - put_start
                        
                        continue
                    
                    
                    ############################################################
                    #
                    # Find Galaxy 1 (closest to cell center)
                    #
                    # and calculate Unit vector pointing from cell 
                    # center to the closest galaxy
                    #
                    ############################################################
                    modv1, k1g = galaxy_tree.query(hole_center, k=1)
                    
                    modv1 = modv1[0][0]
                    
                    k1g = k1g[0][0]
                
                    v1_unit = (w_coord[k1g] - hole_center)/modv1
                
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
                
                    hole_center_2 = hole_center
                
                    in_mask_2 = True
                
                    while galaxy_search:
                
                        # Shift hole center away from first galaxy
                        hole_center_2 = hole_center_2 - dr*v1_unit
                        
                        # Distance between hole center and nearest galaxy
                        modv1 += dr
                        
                        # Search for nearest neighbors within modv1 of the hole center
                        i_nearest = galaxy_tree.query_radius(hole_center_2, r=modv1)
                
                        i_nearest = i_nearest[0]
                        #dist_nearest = dist_nearest[0]
                
                        # Remove nearest galaxy from list
                        boolean_nearest = i_nearest != k1g
                        
                        i_nearest = i_nearest[boolean_nearest]
                        
                
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
                            
                        elif not_in_mask(hole_center_2, mask, min_dist, max_dist):
                            # Hole is no longer within survey limits
                            galaxy_search = False
                            
                            in_mask_2 = False
                
                    # Check to make sure that the hole center is still within the survey
                    if not in_mask_2:
                        #print('hole not in survey')
                        
                        put_start = time.time()
                        
                        return_queue.put(("data", (None, None, None, None)))
                        
                        time_returning += time.time() - put_start
                        
                        continue
                
                    #print('Found 2nd galaxy')
                
                    ############################################################
                    # Update hole center
                    ############################################################
                    
                    # Calculate new hole center
                    hole_radius = 0.5*np.sum(BA[k2g_x2]**2)/np.dot(BA[k2g_x2], v1_unit.T)  # shape (1,)
                    
                    hole_center = w_coord[k1g] - hole_radius*v1_unit  # shape (1,3)
                   
                    # Check to make sure that the hole center is still within the survey
                    if not_in_mask(hole_center, mask, min_dist, max_dist):
                        #print('hole not in survey')
                        
                        put_start = time.time()
                        
                        return_queue.put(("data", (None, None, None, None)))
                        
                        time_returning += time.time() - put_start
                        
                        continue
                
                    ########################################################################
                    # Find Galaxy 3 (closest to cell center)
                    #
                    # (Same methodology as for finding the second galaxy)
                    ########################################################################
                    
                
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
                        i_nearest = galaxy_tree.query_radius(hole_center_3, r=search_radius)
                
                        i_nearest = i_nearest[0]
                        #dist_nearest = dist_nearest[0]
                
                        # Remove two nearest galaxies from list
                        boolean_nearest = np.logical_and(i_nearest != k1g, i_nearest != k2g)
                        i_nearest = i_nearest[boolean_nearest]
                        #dist_nearest = dist_nearest[boolean_nearest]
                
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
                        elif not_in_mask(hole_center_3, mask, min_dist, max_dist):
                            # Hole is no longer within survey limits
                            galaxy_search = False
                            in_mask_3 = False
                
                    # Check to make sure that the hole center is still within the survey
                    #if not in_mask(hole_center_3, mask, [min_dist, max_dist]):
                    #if not_in_mask(hole_center_3, mask, min_dist, max_dist):
                    if not in_mask_3:
                        #print('hole not in survey')
                        
                        put_start = time.time()
                        
                        return_queue.put(("data", (None, None, None, None)))
                        
                        time_returning += time.time() - put_start
                        
                        continue
                
                    #print('Found 3rd galaxy')
                    
                    ############################################################
                    # Update hole center
                    ############################################################
                    hole_center = hole_center + minx3*v2_unit  # shape (1,3)
                    
                    hole_radius = np.linalg.norm(hole_center - w_coord[k1g])  # shape ()
                
                    # Check to make sure that the hole center is still within the survey
                    if not_in_mask(hole_center, mask, min_dist, max_dist):
                        #print('hole not in survey')
                        
                        put_start = time.time()
                        
                        return_queue.put(("data", (None, None, None, None)))
                        
                        time_returning += time.time() - put_start
                        
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
                            Dcenter = w_coord[i_nearest] - hole_center  # shape (N,3)
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
                            Dcenter = w_coord[i_nearest] - hole_center  # shape (N,3)
                
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
                        
                        put_start = time.time()
                        
                        return_queue.put(("data", (None, None, None, None)))
                        
                        time_returning += time.time() - put_start
                        
                        continue
                
                    ########################################################################
                    # Calculate Radius of the hole
                    ########################################################################
                    hole_radius = np.linalg.norm(hole_center - w_coord[k1g])
                
                    ############################################################
                    # Calculate center coordinates of cell
                    ############################################################
                    hole_center = (np.array([[i, j, k]]) + 0.5)*dl + coord_min  # Purposefully making hole_center have shape (1,3) for KDtree queries
                    
                    # Check to make sure that the hole center is still within the survey
                    if not_in_mask(hole_center, mask, min_dist, max_dist):
                        
                        
                        put_start = time.time()
                        
                        return_queue.put(("data", (None, None, None, None)))
                        
                        time_returning += time.time() - put_start
                        
                        continue
                
                    
                
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
                    '''
                    put_start = time.time()
                    
                    return_queue.put(("data", (x_val, y_val, z_val, r_val)))
                    
                    time_returning += time.time() - put_start
                    
                    
        
                
                time_main += time.time() - main_proc_start_time
                
                
                #print("Processed2: ", num_cells_processed, "time: ", time.time() - worker_lifetime_start, "main: ", time_main, time_empty)
                    
        
        
    
    return_queue.put(("Done", None))
    
    #print("Worker lifetime: ", time.time() - worker_lifetime_start)
    print("Time process: ", time_main, "num: ", num_cells_processed)
    #print("Time sleeping: ", time_sleeping)
    #print("Time returning: ", time_returning)
    #print("Time message: ", time_message)
    #print("Msg chks: ", num_message_checks, "Cells procd: ", num_cells_processed, "Empty puts: ", num_empty_job_put, "total loop: ", total_loops)
    
    return None
    
    
'''
#Attempt at modularizing the big behemoth in _main_hole_finder
def _find_galaxy_2(hole_center, w_coord, galaxy_tree, dr, v1_unit, modv1, k1g, mask, min_dist, max_dist):
    
    
    galaxy_search = True

    hole_center_2 = hole_center

    in_mask_2 = True

    while galaxy_search:

        # Shift hole center away from first galaxy
        hole_center_2 = hole_center_2 - dr*v1_unit
        
        # Distance between hole center and nearest galaxy
        modv1 += dr
        
        # Search for nearest neighbors within modv1 of the hole center
        i_nearest = galaxy_tree.query_radius(hole_center_2, r=modv1)

        i_nearest = i_nearest[0]
        #dist_nearest = dist_nearest[0]

        # Remove nearest galaxy from list
        boolean_nearest = i_nearest != k1g
        
        i_nearest = i_nearest[boolean_nearest]
        

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
            
        elif not_in_mask(hole_center_2, mask, min_dist, max_dist):
            # Hole is no longer within survey limits
            galaxy_search = False
            
            in_mask_2 = False

    return k2g, k2g_x2, minx2, in_mask_2, BA
'''





cimport cython
#import numpy
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

from numpy.math cimport NAN




cpdef void main_algorithm(int i, 
                          int j, 
                          int k,
                          galaxy_tree,
                          DTYPE_F64_t[:,:] w_coord,
                          DTYPE_F64_t dl, 
                          DTYPE_F64_t dr,
                          DTYPE_F64_t[:,:] coord_min, 
                          mask,
                          DTYPE_F64_t min_dist,
                          DTYPE_F64_t max_dist,
                          DTYPE_F64_t[:] return_array
                          ) except *:
    
   
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
    
    cdef DTYPE_B_t galaxy_search
    

    hole_center = (np.array([[i, j, k]], dtype=np.float64) + 0.5)*dl + coord_min  # Purposefully making hole_center have shape (1,3) for KDtree queries
                    
    '''
    print("Hole center")
    print(hole_center)
    print(type(hole_center))
    print(hole_center.dtype)
    '''
    #exit()
    
                    
    # Check to make sure that the hole center is still within the survey
    if not_in_mask(hole_center, mask, min_dist, max_dist):
        
        
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
    #
    #
    ############################################################
    modv1, k1g = galaxy_tree.query(hole_center, k=1)
    
    print(modv1.shape)
    print(k1g.shape)
    
    modv1 = modv1[0][0]
    
    k1g = k1g[0][0]

    v1_unit = (w_coord[k1g] - hole_center)/modv1
    
    print(modv1)
    print(k1g)
    print(v1_unit)

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

    hole_center_2 = hole_center

    in_mask_2 = True

    while galaxy_search:

        # Shift hole center away from first galaxy
        hole_center_2 = hole_center_2 - dr*v1_unit
        
        # Distance between hole center and nearest galaxy
        modv1 += dr
        
        # Search for nearest neighbors within modv1 of the hole center
        i_nearest = galaxy_tree.query_radius(hole_center_2, r=modv1)

        i_nearest = i_nearest[0]
        #dist_nearest = dist_nearest[0]

        # Remove nearest galaxy from list
        boolean_nearest = i_nearest != k1g
        
        i_nearest = i_nearest[boolean_nearest]
        
        
        #print(type(w_coord))
        #print(i_nearest.dtype)
        #print(type(k1g))

        if len(i_nearest) > 0:
            # Found at least one other nearest neighbor!

            # Calculate vector pointing from next nearest galaxies to the nearest galaxy
            
            #print(w_coord.shape)
            #print(i_nearest)
            
            
            temp1 = w_coord[k1g]
            
            #temp2 = w_coord[i_nearest]
            
            temp2 = np.take(w_coord, i_nearest, axis=0)
            
            '''
            try:
                temp2 = w_coord[i_nearest]
            except Exception as err:
                print("AHHHH-2")
                print(w_coord)
                print(type(w_coord))
                print(i_nearest)
                print(type(i_nearest))
                print(i_nearest.dtype)
            
            
            try:
                BA = np.subtract(temp1, temp2)  # shape (N,3)
            except Exception as err:
                print("ERROR")
                print(err)
                print(w_coord)
                print(k1g)
                print(i_nearest)
                raise
            '''
            
            #print(temp1.shape)
            #print(temp2.shape)
            
            BA = np.subtract(temp1, temp2)  # shape (N,3)
            
            #print(BA.shape)
            
            
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
    # Update hole center
    ############################################################
    
    # Calculate new hole center
    hole_radius = 0.5*np.sum(BA[k2g_x2]**2)/np.dot(BA[k2g_x2], v1_unit.T)  # shape (1,)
    
    hole_center = w_coord[k1g] - hole_radius*v1_unit  # shape (1,3)
   
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
    # Find Galaxy 3 (closest to cell center)
    #
    # (Same methodology as for finding the second galaxy)
    ########################################################################
    

    # Find the midpoint between the two nearest galaxies
    midpoint = 0.5*(np.add(w_coord[k1g], w_coord[k2g]))  # shape (3,)
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
        i_nearest = galaxy_tree.query_radius(hole_center_3, r=search_radius)

        i_nearest = i_nearest[0]
        #dist_nearest = dist_nearest[0]

        # Remove two nearest galaxies from list
        boolean_nearest = np.logical_and(i_nearest != k1g, i_nearest != k2g)
        i_nearest = i_nearest[boolean_nearest]
        #dist_nearest = dist_nearest[boolean_nearest]

        if len(i_nearest) > 0:
            # Found at least one other nearest neighbor!

            # Calculate vector pointing from hole center to next nearest galaxies
            
            temp_1 = np.take(w_coord, i_nearest, axis=0)
            
            Ccenter = np.subtract(temp_1, hole_center)  # shape (N,3)
            
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
    # Update hole center
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

    ############################################################
    # Calculate center coordinates of cell
    ############################################################
    hole_center = (np.array([[i, j, k]]) + 0.5)*dl + coord_min  # Purposefully making hole_center have shape (1,3) for KDtree queries
    
    # Check to make sure that the hole center is still within the survey
    if not_in_mask(hole_center, mask, min_dist, max_dist):
        
        
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



