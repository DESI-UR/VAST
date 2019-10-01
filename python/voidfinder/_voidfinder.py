


import multiprocessing


#multiprocessing.set_start_method('spawn')

import os
import sys


import fcntl
import mmap
import struct
import socket
import select
import atexit

import numpy as np

import time

from sklearn import neighbors

from .voidfinder_functions import not_in_mask

from ._voidfinder_cython import main_algorithm



from multiprocessing import Queue, Process, cpu_count, RLock, Value, Array

from ctypes import c_int64, c_double, c_float

from queue import Empty

from copy import deepcopy

import pickle

from astropy.table import Table

from .table_functions import to_array

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt



def _main_hole_finder(cell_ID_dict, 
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
                      num_cpus=1):
    '''
    Description:
    ============

    Grow a sphere in each empty cell until the sphere is bounded by four 
    galaxies on its surface.
    
    
    
    Parameters:
    ===========
    
    cell_ID_dict : python dictionary
        keys are tuples of (i,j,k) locations which correspond to a grid cell.
        if the key (i,j,k) is in the dictionary, then that means there is at 
        least 1 galaxy at the corresponding grid cell so we should pass over 
        that grid cell since it isn't empty.
    
    ngrid : numpy.ndarray of shape (3,)
        the number of grid cells in each of the 3 x,y,z dimensions
    
    dl : scalar float
        length of each cell in Mpc/h
        
    dr : scalar float
        distance to shift hole centers during iterative void hole growing in 
        Mpc/h
        
    coord_min : numpy.ndarray of shape (3,) (actually shape (1,3)?)
        minimum coordinates of the survey in x,y,z in Mpc/h
        
    mask : numpy.ndarray of shape (N,M) type bool
        represents the survey footprint in scaled ra/dec space.  Value of True 
        indicates that a location is within the survey

    mask_resolution : integer
        Scale factor of coordinates needed to index mask
    
    min_dist : scalar
        minimum redshift in units of Mpc/h
        
    max_dist : scalar
        maximum redshift in units of Mpc/h
        
    w_coord : numpy.ndarray of shape ()
        x,y,z coordinates of the galaxies used in building the query tree

    batch_size : scalar float
        Number of empty cells to pass into each process.  Initialized to 1000.

    verbose : int
        value to determine whether or not to display status messages while 
        running.  0 is off, 1 is print after N updates, 2 is full debugging prints.

    num_cpus : scalar float
        Number of CPUs to use in parallel.  Default is 1.  Set to None to use 
        maximum number of CPUs available.
    
    
    
    Returns:
    ========
    
    myvoids_x : numpy.ndarray of shape (N,)
        x-coordinates of the N hole centers found in units of Mpc/h

    myvoids_y : numpy.ndarray of shape (N,)
        y-coordinates of the N hole centers found in units of Mpc/h

    myvoids_z : numpy.ndarray of shape (N,)
        z-coordinates of the N hole centers found in units of Mpc/h

    myvoids_r : numpy.ndarray of shape (N,)
        Radii of the N holes found in units of Mpc/h

    n_holes : scalar float
        Number of holes found
    '''

    
    start_time = time.time()
    
    
    
    
    #empty_cell_counter = 0
        
    ################################################################################
    #
    # Pop all the relevant grid cell IDs onto the job queue
    #
    ################################################################################
    '''
    cell_ID_list = []

    for i in range(ngrid[0]):
        
        for j in range(ngrid[1]):
            
            for k in range(ngrid[2]):

                check_bin_ID = (i,j,k)

                if check_bin_ID not in cell_ID_dict:
                    
                    #job_queue.put(check_bin_ID)
                    
                    cell_ID_list.append(check_bin_ID)
                    
                    #empty_cell_counter += 1
    '''
    
    #cell_ID_list = CellIDGenerator(ngrid[0], ngrid[1], ngrid[2], cell_ID_dict, start_from=0, stop_after=40000000)
    #cell_ID_list = CellIDGenerator(ngrid[0], ngrid[1], ngrid[2], cell_ID_dict, start_from=40000000, stop_after=40000000)
    
    cell_ID_list = CellIDGenerator(ngrid[0], ngrid[1], ngrid[2], cell_ID_dict)
    
    #num_empty_cells = ngrid[0]*ngrid[1]*ngrid[2] - len(cell_ID_dict)
    
    if verbose > 0:
        print("cell_ID_list finish time: ", time.time() - start_time, flush=True)
        print("Len cell_ID_dict (eliminated cells): ", len(cell_ID_dict), flush=True)
    
    ################################################################################
    # Run single or multi-processed
    ################################################################################
    
    if isinstance(num_cpus, int) and num_cpus == 1:
        
        #myvoids_x, myvoids_y, myvoids_z, myvoids_r, n_holes = run_single_process(cell_ID_list, 
        myvoids_x, myvoids_y, myvoids_z, myvoids_r, n_holes = run_single_process_cython(cell_ID_list, 
                                                                                   ngrid, 
                                                                                   dl, 
                                                                                   dr,
                                                                                   coord_min, 
                                                                                   mask,
                                                                                   mask_resolution,
                                                                                   min_dist,
                                                                                   max_dist,
                                                                                   w_coord,
                                                                                   batch_size=batch_size,
                                                                                   verbose=verbose,
                                                                                   print_after=print_after,
                                                                                   num_cpus=num_cpus
                                                                                   )
    else:
        
        myvoids_x, myvoids_y, myvoids_z, myvoids_r, n_holes = run_multi_process(cell_ID_dict, 
                                                                                   ngrid, 
                                                                                   dl, 
                                                                                   dr,
                                                                                   coord_min, 
                                                                                   mask,
                                                                                   mask_resolution,
                                                                                   min_dist,
                                                                                   max_dist,
                                                                                   w_coord,
                                                                                   batch_size=batch_size,
                                                                                   verbose=verbose,
                                                                                   print_after=print_after,
                                                                                   num_cpus=num_cpus
                                                                                   )
    
    
        
    return myvoids_x, myvoids_y, myvoids_z, myvoids_r, n_holes
    
    

class CellIDGenerator(object):
    
    def __init__(self, grid_dim_1, grid_dim_2, grid_dim_3, cell_ID_dict, start_from=None, stop_after=None):
        
        self.num_grid_1 = grid_dim_1
        self.num_grid_2 = grid_dim_2
        self.num_grid_3 = grid_dim_3
        
        #self.i = -1
        #self.j = -1
        #self.k = -1
        
        self.i = Value(c_int64, -1, lock=False) #we will protect all access with a lock in the next method
        self.j = Value(c_int64, -1, lock=False)
        self.k = Value(c_int64, -1, lock=False)
        
        self.out_idx = Value(c_int64, 0, lock=False)
        
        self.cell_ID_dict = cell_ID_dict
        
        self.lock = RLock()
        
        self.num_returned = 0
        
        self.stop_after = stop_after
        
        #Can't calculate ijk using moduluses because it isnt a 1-to-1
        #calculation because some positions get skipped over due to known
        #empties
        if start_from is not None:
            
            self.start_from = start_from
            
            for idx in range(start_from):
            
                next(self)
                
                self.num_returned = 0
                
        else:
            self.start_from = 0
        
    def reset(self):
        
        self.i.value = -1
        self.j.value = -1
        self.k.value = -1
        
        self.out_idx.value = 0
        
    def __iter__(self):
        
        return self
    
    def __next__(self):
        
        self.lock.acquire()
        
        if self.stop_after is not None and self.num_returned >= self.stop_after:
            raise StopIteration
        
        found_valid = False
        
        while not found_valid:
            
            next_cell_ID = self.gen_cell_ID()
        
            if next_cell_ID not in self.cell_ID_dict:
                
                break
            
        self.num_returned += 1
            
        self.lock.release()
            
        return next_cell_ID
    
    def __len__(self):
        
        return self.num_grid_1*self.num_grid_2*self.num_grid_3 - len(self.cell_ID_dict)
        
        
    def gen_cell_ID(self):
        
        inc_j = False
        
        if self.k.value >= (self.num_grid_3 - 1) or self.k.value < 0:
            
            inc_j = True
        
        inc_i = False
        
        if (inc_j and self.j.value >= (self.num_grid_2 - 1)) or self.j.value < 0:
            
            inc_i = True
            
            
        #print(inc_i, self.i, self.j, self.k)
            
        self.k.value = (self.k.value + 1) % self.num_grid_3
        
        if inc_j:
            
            self.j.value = (self.j.value + 1) % self.num_grid_2
            
        if inc_i:
            
            self.i.value += 1
            
            if self.i.value >= self.num_grid_1:
                
                raise StopIteration    
        
        out_cell_ID = (self.i.value, self.j.value, self.k.value)
        
        return out_cell_ID
    
    def gen_cell_ID_batch(self, batch_size, output_array=None):
        
        if output_array is None:
            output_array = np.empty((batch_size, 3), dtype=np.int64)
        
        self.lock.acquire()
        
        last_call = False
        
        for idx in range(batch_size):
            
            try:
                
                cell_ID = next(self)
                
            except StopIteration:
                
                last_call = True
                
                break
            
            output_array[idx,0] = cell_ID[0]
            output_array[idx,1] = cell_ID[1]
            output_array[idx,2] = cell_ID[2]
            
        
        
        out_idx = self.out_idx.value
        
        self.out_idx.value += batch_size
        
        self.lock.release()
        
        if last_call:
            
            return output_array[0:idx], out_idx
        
        else:
        
            return output_array, out_idx
        
        
        
        
        
        


class MultiCellIDGenerator(object):
    
    def __init__(self, 
                 grid_dim_1, 
                 grid_dim_2, 
                 grid_dim_3, 
                 cell_ID_dict,
                 cell_ID_gen_file_handle,
                 cell_ID_gen_memory,
                 cell_ID_gen_mmap
                 ):
        
        self.num_grid_1 = grid_dim_1
        
        self.num_grid_2 = grid_dim_2
        
        self.num_grid_3 = grid_dim_3
        
        self.cell_ID_dict = cell_ID_dict
        
        self.cell_ID_gen_file_handle = cell_ID_gen_file_handle
        
        self.cell_ID_gen_memory = cell_ID_gen_memory
        
        self.cell_ID_gen_mmap = cell_ID_gen_mmap
        
        self.i = np.empty(1, dtype=np.int64)
        self.j = np.empty(1, dtype=np.int64)
        self.k = np.empty(1, dtype=np.int64)
        self.out_idx = np.empty(1, dtype=np.int64)
        
        
    def __iter__(self):
        
        return self
    
    def __next__(self):
        
        found_valid = False
        
        while not found_valid:
            
            next_cell_ID = self.gen_cell_ID()
        
            if next_cell_ID not in self.cell_ID_dict:
                
                break
            
        return next_cell_ID
    
    def __len__(self):
        
        return self.num_grid_1*self.num_grid_2*self.num_grid_3 - len(self.cell_ID_dict)
        
        
    def gen_cell_ID(self):
        
        
        
        #i = self.cell_ID_gen_memory[0]
        #j = self.cell_ID_gen_memory[1]
        #k = self.cell_ID_gen_memory[2]
        
        
        inc_j = False
        
        if self.k[0] >= (self.num_grid_3 - 1) or self.k[0] < 0:
            
            inc_j = True
        
        inc_i = False
        
        if (inc_j and self.j[0] >= (self.num_grid_2 - 1)) or self.j[0] < 0:
            
            inc_i = True
            
            
        #print(inc_i, self.i, self.j, self.k)
            
        self.k[0] = (self.k[0] + 1) % self.num_grid_3
        
        if inc_j:
            
            self.j[0] = (self.j[0] + 1) % self.num_grid_2
            
        if inc_i:
            
            self.i[0] += 1
            
            if self.i[0] >= self.num_grid_1:
                
                raise StopIteration    
        
        out_cell_ID = (self.i[0], self.j[0], self.k[0])
        
        #self.cell_ID_gen_memory[0] = i
        #self.cell_ID_gen_memory[1] = j
        #self.cell_ID_gen_memory[2] = k
        
        return out_cell_ID
    
    def gen_cell_ID_batch(self, batch_size, output_array=None):
        
        if output_array is None:
            
            output_array = np.empty((batch_size, 3), dtype=np.int64)
        
        
        fcntl.lockf(self.cell_ID_gen_file_handle, fcntl.LOCK_EX)
        
        self.cell_ID_gen_mmap.seek(0)
        
        i_bytes = self.cell_ID_gen_mmap.read(8)
        j_bytes = self.cell_ID_gen_mmap.read(8)
        k_bytes = self.cell_ID_gen_mmap.read(8)
        out_idx_bytes = self.cell_ID_gen_mmap.read(8)
        
        
        #print(len(i_bytes), i_bytes)
        
        self.i[0] = struct.unpack("=q", i_bytes)[0]
        self.j[0] = struct.unpack("=q", j_bytes)[0]
        self.k[0] = struct.unpack("=q", k_bytes)[0]
        self.out_idx[0] = struct.unpack("=q", out_idx_bytes)[0]
        
        return_out_idx = self.out_idx[0]
        
        
        
        last_call = False
        
        for idx in range(batch_size):
            
            try:
                
                cell_ID = next(self)
                
            except StopIteration:
                
                last_call = True
                
                break
            
            output_array[idx,0] = cell_ID[0]
            output_array[idx,1] = cell_ID[1]
            output_array[idx,2] = cell_ID[2]
            
        
        
        #out_idx = self.cell_ID_gen_memory[3]
        
        #out_idx += batch_size
        
        #self.cell_ID_gen_memory[3] = out_idx
        
        self.out_idx[0] += batch_size
        
        out_bytes = b""
        out_bytes += struct.pack("=q", self.i[0])
        out_bytes += struct.pack("=q", self.j[0])
        out_bytes += struct.pack("=q", self.k[0])
        out_bytes += struct.pack("=q", self.out_idx[0])
        
        self.cell_ID_gen_mmap.seek(0)
        
        self.cell_ID_gen_mmap.write(out_bytes)
        
        self.cell_ID_gen_mmap.flush()
    
        fcntl.lockf(self.cell_ID_gen_file_handle, fcntl.LOCK_UN)
        
        if last_call:
            
            return output_array[0:idx], return_out_idx
        
        else:
        
            return output_array, return_out_idx
        
       



    
def run_single_process(cell_ID_list, 
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



    

    
def run_single_process_cython(cell_ID_gen, 
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
    
    
    ################################################################################
    #
    # Profiling
    #
    ################################################################################
    
    PROFILE_loop_times = []
    
    ################################################################################
    #
    # Initialize some output containers and counter variables
    #
    ################################################################################
    
    print("Running single-process mode")
    '''
    myvoids_x = []
    
    myvoids_y = []
    
    myvoids_z = []
    
    myvoids_r = []
    '''
    n_holes = 0

    #empty_cell_counter = 0
    
    n_empty_cells = len(cell_ID_gen)
    
    #ALLOCATE A BIGASS ARRAY
    RETURN_ARRAY = np.empty((n_empty_cells, 4), dtype=np.float64)
    RETURN_ARRAY.fill(np.NAN)
    
    
    PROFILE_ARRAY = np.empty((85000000,3), dtype=np.float64)
    #RETURN_ARRAY = np.zeros((2000000000, 4), dtype=np.float64)
    #RETURN_ARRAY.fill(3.5)
    
    print("RETURN ARRAY SHAPE: ", RETURN_ARRAY.shape)
    print(RETURN_ARRAY.nbytes)
    
    out_start_idx = 0
    
    ################################################################################
    # Convert the mask to an array of uint8 values for running in the cython code
    ################################################################################
    
    mask = mask.astype(np.uint8)
    
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
    
    num_processed = 0
    
    num_in_batch = 0
    
    cell_ID_list = []
    
    return_array = np.empty((batch_size, 4), dtype=np.float64)
    
    i_j_k_array = np.empty((batch_size, 3), dtype=np.int64)
    
    ################################################################################
    # PROFILE ARRAY elements are:
    # 0 - total cell time
    # 1 - cell exit stage
    # 2 - kdtree_time
    ################################################################################
    PROFILE_process_start_time = time.time()
    PROFILE_array = np.empty((batch_size, 3), dtype=np.float32)
    PROFILE_samples = []
    PROFILE_start_time = time.time()
    PROFILE_sample_time = 5.0
    
    
    exit_condition = False
    
    while not exit_condition:
        
        ######################################################################
        # Print progress and profiling statistics
        ######################################################################
        if num_processed >= 80000000:
        #if num_processed >= 8000000:
            
            break
        
        if time.time() - PROFILE_start_time > PROFILE_sample_time:
            
            PROFILE_samples.append(num_processed)
            
            PROFILE_start_time = time.time()
        
            if verbose > 0:
            
                print("Processing cell "+str(num_processed)+" of "+str(n_empty_cells), time.time() - PROFILE_process_start_time)
            
            if len(PROFILE_samples) > 3:
                
                cells_per_sec = (PROFILE_samples[-1] - PROFILE_samples[-2])/PROFILE_sample_time
                
                print(cells_per_sec, "cells per sec")
                #print("DR: ", dr)
                
        ######################################################################
        # Generate the next batch and run the main algorithm
        ######################################################################
        
        i_j_k_array, out_start_idx = cell_ID_gen.gen_cell_ID_batch(batch_size, i_j_k_array)
        
        num_cells_to_process = i_j_k_array.shape[0]
        
        if num_cells_to_process > 0:

            if return_array.shape[0] != num_cells_to_process:

                return_array = np.empty((num_cells_to_process, 4), dtype=np.float64)
                
                PROFILE_array = np.empty((num_cells_to_process, 3), dtype=np.float32)
        
            main_algorithm(i_j_k_array,
                           galaxy_tree,
                           w_coord,
                           dl, 
                           dr,
                           coord_min,
                           mask,
                           mask_resolution,
                           min_dist,
                           max_dist,
                           return_array,
                           0,  #verbose level
                           PROFILE_array
                           )
        
            #print("Exited main algorithm returned to python")
            #print(return_array.shape, out_start_idx)
            
            RETURN_ARRAY[out_start_idx:(out_start_idx+return_array.shape[0]),:] = return_array
            
            PROFILE_ARRAY[out_start_idx:(out_start_idx+return_array.shape[0]),:] = PROFILE_array
            
            n_holes += np.sum(np.logical_not(np.isnan(return_array[:,0])))
            
            num_processed += return_array.shape[0]
            
        else:
            
            exit_condition = True
        
    outfile = open("/home/oneills2/VoidFinder/doc/profiling/single_thread_profile.pickle", 'wb')
    pickle.dump(PROFILE_samples, outfile)
    outfile.close()    
    
    
    if verbose > 0:
        
        
        PROFILE_ARRAY_SUBSET = PROFILE_ARRAY[0:num_processed]
        '''
        fig, axes = plt.subplots(3,3, figsize=(10,10))
        
        fig.suptitle("All Cell Processing Times [s]")
        
        axes_list = []
        
        axes_list.append(axes[0,0])
        axes_list.append(axes[0,1])
        axes_list.append(axes[0,2])
        axes_list.append(axes[1,0])
        axes_list.append(axes[1,1])
        axes_list.append(axes[1,2])
        axes_list.append(axes[2,0])
        '''
        for idx in range(7):
            
            #curr_axes = axes_list[idx]
            
            
            curr_idx = PROFILE_ARRAY_SUBSET[:,1] == idx
            
            curr_data = PROFILE_ARRAY_SUBSET[curr_idx, 0]
            
            if idx == 0:
                outfile = open("/home/oneills2/VoidFinder/doc/profiling/Cell_Processing_Times_SingleThreadCython.pickle", 'wb')
                pickle.dump(curr_data, outfile)
                outfile.close()
            
            plot_cell_processing_times(curr_data, idx, "Single")
            
            curr_data = PROFILE_ARRAY_SUBSET[curr_idx, 2]
            
            plot_cell_kdtree_times(curr_data, idx, 'Single')
        
        
        
        
        
    if verbose > 2:
        print('Plotting single cell processing times distribution')
        plt.figure(figsize=(14,10))
        plt.hist(PROFILE_loop_times, bins=50)
        plt.title("All Single Cell processing times Cython (sec)")
        plt.xlabel('Time [s]')
        plt.ylabel('Count')
        #plt.show()
        plt.savefig("Cell_time_dist_Cython.png")
        plt.close()
    
        
    return myvoids_x, myvoids_y, myvoids_z, myvoids_r, n_holes



    
    
    
    
    


def run_multi_process(cell_ID_dict, 
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
    
    
    
    max_cpus = cpu_count()
    
    if (num_cpus is None) or (num_cpus > max_cpus):
          
        num_cpus = max_cpus
    
    start_time = time.time()
    
    CONFIG_PATH = "/tmp/voidfinder_config.pickle"
    
    SOCKET_PATH = "/tmp/voidfinder.sock"
    
    RESULT_BUFFER_PATH = "/tmp/voidfinder_result_buffer.dat"
    
    CELL_ID_BUFFER_PATH = "/tmp/voidfinder_cell_ID_gen.dat"
    
    PROFILE_BUFFER_PATH = "/tmp/voidfinder_profile_buffer.dat"
    
    ################################################################################
    #
    # Initialize some output containers and counter variables
    #
    ################################################################################
    n_holes = 0

    n_empty_cells = ngrid[0]*ngrid[1]*ngrid[2] - len(cell_ID_dict)
    
    result_buffer = open(RESULT_BUFFER_PATH, 'w+b') #USE WB HERE BUT USE 'r+b' IN CHILD WORKERS
    #https://docs.python.org/3.7/library/functions.html#open
    
    result_buffer_length = n_empty_cells*4*8 #float64 so 8 bytes per element
    
    result_buffer.write(b"0"*result_buffer_length)
    
    #result_mmap_buffer = mmap.mmap(result_buffer.fileno(), result_buffer_length)
    
    #RETURN_ARRAY = np.frombuffer(result_mmap_buffer, dtype=c_double)
    
    #RETURN_ARRAY.shape = (n_empty_cells, 4)
    
    #RETURN_ARRAY.fill(np.NAN)
    
    ################################################################################
    # Memory for PROFILING
    ################################################################################
    PROFILE_buffer = open(PROFILE_BUFFER_PATH, 'w+b')
    
    PROFILE_buffer_length = 85000000*3*4 #float32 so 4 bytes per element
    
    PROFILE_buffer.write(b"0"*PROFILE_buffer_length)
    
    #PROFILE_mmap_buffer = mmap.mmap(PROFILE_buffer.fileno(), PROFILE_buffer_length)
    
    #PROFILE_ARRAY = np.frombuffer(PROFILE_mmap_buffer, dtype=c_float)
    
    #PROFILE_ARRAY.shape = (85000000, 3)
    
    #PROFILE_ARRAY.fill(np.NAN)
    
    ################################################################################
    # Build Cell ID generator
    ################################################################################
    cell_ID_buffer = open(CELL_ID_BUFFER_PATH, 'w+b')
    
    cell_ID_buffer_length = 4*8 #need 4 8-byte integers: i, j, k, out_idx
    
    cell_ID_buffer.write(b"\x00"*cell_ID_buffer_length)
    
    cell_ID_buffer.flush()
    
    #data = cell_ID_buffer.read()
    #print("Cell ID Buffer length: ", len(data), cell_ID_buffer_length)
    
    #cell_ID_mmap_buffer = mmap.mmap(cell_ID_buffer.fileno(), cell_ID_buffer_length)
    
    #cell_ID_mem_array = np.frombuffer(cell_ID_mmap_buffer, dtype=np.int64)
    
    #cell_ID_mem_array.shape = (4,)
    
    #cell_ID_mem_array.fill(0)
    
    
    ################################################################################
    # mask needs to be 1 byte per bool to match the cython dtype
    ################################################################################
    
    print("Grid: ", ngrid)
    
    
    config_object = (SOCKET_PATH,
                     RESULT_BUFFER_PATH,
                     CELL_ID_BUFFER_PATH,
                     PROFILE_BUFFER_PATH,
                     cell_ID_dict, 
                     ngrid, 
                       dl, 
                       dr,
                       coord_min, 
                       mask.astype(np.uint8),
                       mask_resolution,
                       min_dist,
                       max_dist,
                       w_coord,
                       batch_size,
                       verbose,
                       print_after,
                       num_cpus
                     )
    
    outfile = open(CONFIG_PATH, 'wb')
    
    pickle.dump(config_object, outfile)
    
    outfile.close()
    
    
    #mask = mask.astype(np.uint8)
    
    ################################################################################
    #
    #   BUILD NEAREST-NEIGHBOR TREE
    #   galaxy_tree : sklearn.neighbors/scipy KDTree or similar implementing sklearn interface
    #   nearest neighbor finder for the galaxies in x,y,z space
    #
    ################################################################################
    '''
    if verbose:
        
        kdtree_start_time = time.time()

    galaxy_tree = neighbors.KDTree(w_coord)
    
    if verbose:
        
        print('KDTree creation time:', time.time() - kdtree_start_time)
    
    '''
    ################################################################################
    #
    # Set up worker processes
    #
    ################################################################################
    '''
    command_queue = Queue()
    
    return_queue = Queue()

    processes = []
    
    workers_waiting = []
    
    num_active_processes = 0
    
    for proc_idx in range(num_cpus):
        
        worker_args = (proc_idx,
                       command_queue,
                       return_queue,
                       
                       cell_ID_dict,
                       galaxy_tree, 
                       ngrid, 
                       dl, 
                       dr,
                       coord_min, 
                       mask,
                       mask_resolution,
                       min_dist,
                       max_dist,
                       w_coord,
                       
                       batch_size,
                       RESULT_BUFFER_PATH,
                       CELL_ID_BUFFER_PATH,
                       PROFILE_BUFFER_PATH,
                       )
        
        p = Process(target=_main_hole_finder_worker, args=worker_args)
        
        p.start()
        
        processes.append(p)
        
        num_active_processes += 1
    '''
    ################################################################################
    # Register some functions to be called when the python interpreter exits
    # to clean up any leftover file memory on disk or socket files, etc
    ################################################################################
    def cleanup_config():
        
        os.remove(CONFIG_PATH)
        
    def cleanup_socket():
        
        os.remove(SOCKET_PATH)
        
    def cleanup_result():
        
        os.remove(RESULT_BUFFER_PATH)
        
    def cleanup_cellID():
        
        os.remove(CELL_ID_BUFFER_PATH)
        
    def cleanup_profile():
        
        os.remove(PROFILE_BUFFER_PATH)
        
        
    atexit.register(cleanup_config)
    atexit.register(cleanup_socket)
    atexit.register(cleanup_result)
    atexit.register(cleanup_cellID)
    atexit.register(cleanup_profile)
    
    ################################################################################
    # Start the worker processes
    ################################################################################
        
    listener_socket = socket.socket(socket.AF_UNIX)
    
    listener_socket.bind(SOCKET_PATH)
    
    listener_socket.listen(num_cpus)
        
    processes = []
    
    for proc_idx in range(num_cpus):
        
        p = Process(target=_main_hole_finder_startup, args=(proc_idx, CONFIG_PATH))
        
        p.start()
        
        processes.append(p)
        
    #Not sure if we need to join the processes or not, I think maybe we don't
    #for p in processes:
        
    #    p.join()
    
    #Connect child processes
    
    num_active_processes = 0
    
    worker_sockets = []
    
    message_buffers = []
    
    socket_index = {}
    
    for idx in range(num_cpus):
        
        worker_sock, worker_addr = listener_socket.accept()
        
        worker_sockets.append(worker_sock)
        
        num_active_processes += 1
        
        message_buffers.append(b"")
        
        #print("Creating socket: ", worker_sock.fileno())
        
        socket_index[worker_sock.fileno()] = idx
        
    if verbose:
        print("Worker processes started time: ", time.time() - start_time)
    
    ################################################################################
    #
    # Listen on the return_queue for results
    # and feed workers when they become inactive
    #
    ################################################################################
    num_cells_processed = 0
    
    
    ################################################################################
    # PROFILING VARIABLES
    ################################################################################
    PROFILE_process_start_time = time.time()
    PROFILE_samples = []
    PROFILE_start_time = time.time()
    PROFILE_sample_time = 30.0
    
    
    ################################################################################
    # LOOP TO LISTEN FOR RESULTS WHILE WORKERS WORKING
    ################################################################################
    empty1 = []
    empty2 = []
    
    select_timeout = 2.0
    
    sent_exit_commands = False
    
    while num_active_processes > 0:
        
        
        #Debugging stop early condition
        
        if num_cells_processed >= 80000000:
        #if num_cells_processed >= n_empty_cells:
        
            print("Breaking debug loop", num_cells_processed, num_active_processes)
            
            for idx in range(num_cpus):
                
                worker_sockets[idx].send(b"exit")
                
            sent_exit_commands = True
            
            break
        
        
        
        if time.time() - PROFILE_start_time > PROFILE_sample_time:
            PROFILE_samples.append(num_cells_processed)
            PROFILE_start_time = time.time()
            
            if verbose > 0:
                
                print('Processed', num_cells_processed, 'cells of', n_empty_cells, time.time() - PROFILE_process_start_time)
                
                if len(PROFILE_samples) > 3:
                    cells_per_sec = (PROFILE_samples[-1] - PROFILE_samples[-2])/PROFILE_sample_time
                    print(cells_per_sec, "cells per sec")
            
            
        read_socks, empty3, empty4 = select.select(worker_sockets, empty1, empty2, select_timeout)
        
        if read_socks:
            
            for worker_sock in read_socks:
                
                curr_read = worker_sock.recv(1024)
                
                #print("Reading sock: ", worker_sock.fileno(), curr_read)
                
                sock_idx = socket_index[worker_sock.fileno()]
                
                
                curr_message_buffer = message_buffers[sock_idx]
                
                curr_message_buffer += curr_read
                
                #message_buffer += curr_read
                '''
                if b"\n" not in message_buffers[sock_idx]:
                    
                    messages = []
                    
                else:
                
                    messages = message_buffers[sock_idx].split(b"\n")
                
                    message_buffers[sock_idx] = messages[-1]
                '''
                #print("MESSAGES: ", messages)
                
                messages = []
                
                if len(curr_message_buffer) > 0:
                
                    messages_remaining_in_buffer = True
                    
                else:
                    
                    messages_remaining_in_buffer = False
                    
                #print("Process Buffer? ", messages_remaining_in_buffer)
                
                while messages_remaining_in_buffer:
                
                    #print("Processing buffer: ", sock_idx, len(curr_message_buffer))
                    
                    #print(type(curr_message_buffer))
                    
                    #https://stackoverflow.com/questions/28249597/why-do-i-get-an-int-when-i-index-bytes
                    #implicitly converts the 0th byte to an integer
                    msg_fields = curr_message_buffer[0]
                    
                    #print("Head char: ", head_char)
                    
                    #print(type(head_char))
                
                    #msg_fields = struct.unpack('b', head_char)[0]
                    
                    #print("Msg fields: ", msg_fields)
                    
                    msg_len = 1 + 8*msg_fields
                    
                    #print("Message len: ", msg_len)
                    
                    if len(curr_message_buffer) >= msg_len:
                    
                        curr_msg = curr_message_buffer[1:msg_len]
                        
                        #print("Curr msg: ", curr_msg)
                        
                        messages.append(curr_msg)
                        
                        curr_message_buffer = curr_message_buffer[msg_len:]
                        
                        #print(curr_message_buffer[msg_len:])
                        #print(curr_message_buffer)
                        
                        if len(curr_message_buffer) > 0:
                            
                            messages_remaining_in_buffer = True
                            
                        else:
                            
                            messages_remaining_in_buffer = False
                            
                    #else:
                    #    messages_remaining_in_buffer = False
                            
                        
                message_buffers[sock_idx] = curr_message_buffer
                    
                
                for message in messages:
                    
                    if message == b"":
                        continue
                    
                    message_type = struct.unpack("=q", message[0:8])[0]
                    
                    if message_type == 0:
                        
                        try:
                            num_result = struct.unpack("=q", message[8:16])[0]
                            num_hole = struct.unpack("=q", message[16:24])[0]
                        except struct.error as e:
                            print("Error on message: ", message)
                            raise e
                        
                        num_cells_processed += num_result
                        n_holes += num_hole
                        
                    elif message_type == 1:
                        
                        num_active_processes -= 1
                    
    if not sent_exit_commands:
        
        for idx in range(num_cpus):
                
            worker_sockets[idx].send(b"exit")
            
        '''
        try:
            
            message = return_queue.get(timeout=2.0)
            
        except Empty:
            
            pass
            
        else:
            
            if message[0] == 'Done':
                
                num_active_processes -= 1
                
            elif message[0] == "data":
                
                num_cells_processed += message[1]
                
                n_holes += message[2]
        '''
                
                
                
    ################################################################################
    # PROFILING - SAVE OFF RESULTS
    ################################################################################
    
    outfile = open("/home/oneills2/VoidFinder/doc/profiling/multi_thread_profile.pickle", 'wb')
    
    pickle.dump(PROFILE_samples, outfile)
    
    outfile.close()
    
    if verbose > 0:
        
        PROFILE_ARRAY = np.memmap(PROFILE_buffer, dtype=np.float32, shape=(85000000,3))
        
        PROFILE_ARRAY_SUBSET = PROFILE_ARRAY[0:num_cells_processed]
        
        for idx in range(7):
            
            #curr_axes = axes_list[idx]
            
            curr_idx = PROFILE_ARRAY_SUBSET[:,1] == idx
            
            curr_data = PROFILE_ARRAY_SUBSET[curr_idx, 0]
            
            if idx == 0:
                outfile = open("/home/oneills2/VoidFinder/doc/profiling/Cell_Processing_Times_MultiThreadCython.pickle", 'wb')
                pickle.dump(curr_data, outfile)
                outfile.close()
            
            plot_cell_processing_times(curr_data, idx, "Multi")
            
            curr_data = PROFILE_ARRAY_SUBSET[curr_idx, 2]
            
            plot_cell_kdtree_times(curr_data, idx, 'Multi')
        
    ################################################################################
    #
    # Clean up worker processes
    #
    ################################################################################
    
    if verbose:
        
        print("Main task finish time: ", time.time() - start_time)
    
    for p in processes:
        
        p.join(None) #block till join
        
    ################################################################################
    # Close the unneeded file handles to the shared memory, and return the file
    # handle to the result memory buffer
    ################################################################################
    cell_ID_buffer.close()
    
    PROFILE_buffer.close()
        
    return result_buffer, n_holes
                    
    
    
    
    
    
def _main_hole_finder_startup(worker_idx, config_path):
    """
    Helper function called from run_multi_process() to help
    create worker processes for voidfinder parallelization.
    Basically creates a new python process via os.execv()
    and gives that new process the location of a 
    configuration pickle file which it can read in to 
    configure itself.
    """
    
    curr_dir = os.path.dirname(os.path.abspath(__file__))
    
    worker_script_name = os.path.join(curr_dir, "_voidfinder_worker_startup.py")
    
    #Shouldn't strictly be necessary to add this next parameter cause theoretically
    #the voidfinder package should be installed in the python environment, but for now
    #lets ensure that the voidfinder directory is in the path for the worker process
    voidfinder_dir = os.path.abspath(os.path.join(curr_dir, ".."))
    
    working_dir = os.getcwd()
    
    args = ["VoidFinderWorker", worker_script_name, voidfinder_dir, working_dir, str(worker_idx), config_path]
    
    print(sys.executable, args)
    
    os.execv(sys.executable, args)
    
    
    


'''
def _main_hole_finder_worker(process_id,
                             command_queue,
                             return_queue,
                             
                             cell_ID_dict,
                             galaxy_tree, 
                             ngrid, 
                             dl, 
                             dr,
                             coord_min, 
                             mask,
                             mask_resolution,
                             min_dist,
                             max_dist,
                             w_coord,
                             
                             batch_size,
                             RESULT_BUFFER_PATH,
                             CELL_ID_BUFFER_PATH,
                             PROFILE_BUFFER_PATH,
                             ):
'''

def _main_hole_finder_worker(worker_idx, config_path):
    
    #galaxy_tree = neighbors.KDTree(w_coord)
    
    '''
    w_coord_table = Table.read('SDSS_dr7_' + 'wall_gal_file.txt', format='ascii.commented_header')
    w_coord = to_array(w_coord_table)
        
        
    temp_infile = open("filter_galaxies_output.pickle", 'rb')
    coord_min_table, mask, ngrid = pickle.load(temp_infile)
    temp_infile.close()
    del coord_min_table
    
    
    mask = mask.astype(np.uint8)
    
    
    #print(type(w_coord))
    
    w_coord_2 = np.copy(w_coord)
    
    galaxy_tree = neighbors.KDTree(w_coord_2)
    '''
    #print("Process id: ", process_id, id(mask), id(w_coord), id(galaxy_tree), w_coord.__array_interface__['data'][0])
    
    #print("MAIN HOLE FINDER WORKER: ", worker_idx, config_path)
    
    ################################################################################
    #
    ################################################################################
    infile = open(config_path, 'rb')
    
    config = pickle.load(infile)
    
    infile.close()
    
    
    SOCKET_PATH = config[0]
    RESULT_BUFFER_PATH = config[1]
    CELL_ID_BUFFER_PATH = config[2]
    PROFILE_BUFFER_PATH = config[3]
    cell_ID_dict = config[4]
    ngrid = config[5]
    dl = config[6]
    dr = config[7]
    coord_min = config[8]
    mask = config[9]
    mask_resolution = config[10]
    min_dist = config[11]
    max_dist = config[12]
    w_coord = config[13]
    batch_size = config[14]
    verbose = config[15]
    print_after = config[16]
    num_cpus = config[17]



    worker_socket = socket.socket(socket.AF_UNIX)
    
    worker_socket.connect(SOCKET_PATH)
    
    worker_socket.setblocking(False)
    
    

    if verbose:
        
        kdtree_start_time = time.time()

    galaxy_tree = neighbors.KDTree(w_coord)
    
    if verbose:
        
        print('KDTree creation time:', time.time() - kdtree_start_time)
    
    ################################################################################
    #
    ################################################################################
    
    
    n_empty_cells = ngrid[0]*ngrid[1]*ngrid[2] - len(cell_ID_dict)
    
    result_buffer = open(RESULT_BUFFER_PATH, 'r+b')
    
    result_buffer_length = n_empty_cells*4*8 #float64 so 8 bytes per element
    
    #result_mmap_buffer = mmap.mmap(result_buffer.fileno(), result_buffer_length)
    
    #RETURN_ARRAY = np.frombuffer(result_mmap_buffer, dtype=c_double)
    
    #RETURN_ARRAY.shape = (n_empty_cells, 4)
    
    RETURN_ARRAY = np.memmap(result_buffer, dtype=np.float64, shape=(n_empty_cells, 4))
    
    ################################################################################
    # Memory for PROFILING
    ################################################################################
    PROFILE_buffer = open(PROFILE_BUFFER_PATH, 'r+b')
    
    PROFILE_buffer_length = 85000000*3*4 #float32 so 4 bytes per element
    
    #PROFILE_mmap_buffer = mmap.mmap(PROFILE_buffer.fileno(), PROFILE_buffer_length)
    
    #PROFILE_ARRAY = np.frombuffer(PROFILE_mmap_buffer, dtype=c_float)
    
    #PROFILE_ARRAY.shape = (85000000, 3)
    
    PROFILE_ARRAY = np.memmap(PROFILE_buffer, dtype=np.float32, shape=(85000000,3))
    
    ################################################################################
    # Build Cell ID generator
    ################################################################################
    cell_ID_buffer = open(CELL_ID_BUFFER_PATH, 'r+b')
    
    cell_ID_buffer_length = 4*8 #need 4 8-byte integers: i, j, k, out_idx
    
    cell_ID_mmap_buffer = mmap.mmap(cell_ID_buffer.fileno(), cell_ID_buffer_length)
    
    cell_ID_mem_array = np.frombuffer(cell_ID_mmap_buffer, dtype=np.int64)
    
    cell_ID_mem_array.shape = (4,)
    
    cell_ID_gen = MultiCellIDGenerator(ngrid[0],
                                       ngrid[1],
                                       ngrid[2],
                                       cell_ID_dict,
                                       cell_ID_buffer,
                                       cell_ID_mem_array,
                                       cell_ID_mmap_buffer)
    
    
    ################################################################################
    #
    # Profiling parameters
    #
    ################################################################################
    
    worker_lifetime_start = time.time()
    time_main = 0.0
    time_message = 0.0
    time_sleeping = 0.0
    time_empty = 0.0
    
    num_message_checks = 0
    
    num_cells_processed = 0
    
    num_empty_job_put = 0
    
    total_loops = 0
    
    time_returning = 0
    
    
    ################################################################################
    #
    # exit_process - flag for reading an exit command off the queue
    #
    # return_array - some memory for cython code to pass return values back to
    #
    ################################################################################
    
    
    received_exit_command = False
    
    exit_process = False
    
    return_array = np.empty((batch_size, 4), dtype=np.float64)
    
    PROFILE_array = np.empty((batch_size, 3), dtype=np.float32)
    
    i_j_k_array = np.empty((batch_size, 3), dtype=np.int64)
    
    worker_sockets = [worker_socket]
    empty1 = []
    empty2 = []
    
    while not exit_process:
        
        total_loops += 1
        
        read_socks, empty3, empty4 = select.select(worker_sockets, empty1, empty2, 0)
        
        if read_socks:
            
            message = worker_socket.recv(1024)
            
            if len(message) == 4 and message == b'exit':
                
                exit_process = True
                
                received_exit_command = True
                
                continue
        
        
        ################################################################################
        # Locked access to cell ID generation
        ################################################################################
        i_j_k_array, out_start_idx = cell_ID_gen.gen_cell_ID_batch(batch_size, i_j_k_array)
        
        #print("Worker: ", worker_idx, i_j_k_array[0,:], out_start_idx)
        
        ################################################################################
        #
        ################################################################################
        num_cells_to_process = i_j_k_array.shape[0]
        
        if num_cells_to_process > 0:

            if return_array.shape[0] != num_cells_to_process:

                return_array = np.empty((num_cells_to_process, 4), dtype=np.float64)
                
                PROFILE_array = np.empty((num_cells_to_process, 3), dtype=np.float32)
                
            main_algorithm(i_j_k_array,
                           galaxy_tree,
                           w_coord,
                           dl, 
                           dr,
                           coord_min,
                           mask,
                           mask_resolution,
                           min_dist,
                           max_dist,
                           return_array,
                           0,  #verbose level
                           PROFILE_array
                           )
            
            num_cells_processed += return_array.shape[0]
            
            RETURN_ARRAY[out_start_idx:(out_start_idx+return_array.shape[0])] = return_array
            
            PROFILE_ARRAY[out_start_idx:(out_start_idx+return_array.shape[0]),:] = PROFILE_array
            
            n_hole = np.sum(np.logical_not(np.isnan(return_array[:,0])), axis=None, dtype=np.int64)
            
            if not isinstance(n_hole, np.int64):
                print("N_hole not integer: ", n_hole, type(n_hole))
            
            #return_queue.put(("data", return_array.shape[0], n_hole))
            out_msg = b""
            #out_msg += struct.pack("=q", 4)
            out_msg += struct.pack("b", 3)
            out_msg += struct.pack("=q", 0)
            #out_msg += b","
            out_msg += struct.pack("=q", return_array.shape[0])
            #out_msg += b","
            out_msg += struct.pack("=q", n_hole)
            #out_msg += b"\n"
            
            #print(worker_idx, out_msg)
            
            worker_socket.send(out_msg)
            
        else:
            
            out_msg = b""
            out_msg += struct.pack("b", 1)
            out_msg += struct.pack("=q", 1)
            #out_msg += b"\n"
            
            worker_socket.send(out_msg)
            
            exit_process = True
                
    
    
    while not received_exit_command:
        
        read_socks, empty3, empty4 = select.select(worker_sockets, empty1, empty2, 10.0)
        
        if read_socks:
            
            message = worker_socket.recv(1024)
            
            if len(message) == 4 and message == b'exit':
                
                received_exit_command = True
                
                continue
        
    worker_socket.close()
    
    del RETURN_ARRAY #flushes np.memmap 
    del PROFILE_ARRAY #flushes np.memmap
    #return_queue.put(("done", None))
    
    print("WORKER EXITING GRACEFULLY", worker_idx)
    
    return None











def plot_cell_processing_times(curr_data, idx, single_or_multi):
    
    
    if curr_data.shape[0] > 0:
    
        plt.hist(curr_data, bins=50)
        
        curr_title = ""
        #curr_title += "Cell Processing Times [s]\n"
        curr_title += "Cell Exit Stage: " + str(idx) + "\n"
        curr_title += "Total vals: " + str(curr_data.shape[0])
        
        plt.title(curr_title)
        
        
        x_min = curr_data.min()
        x_max = curr_data.max()
        
        x_range = x_max - x_min
        
        x_ticks = np.linspace(x_min - .05*x_range, x_max + .05*x_range, 5)
        x_ticks = x_ticks[1:-1]
        x_tick_labels = ["{:.2E}".format(x_tick) for x_tick in x_ticks]
        
        
        
        plt.xticks(x_ticks, x_tick_labels)
        
        plt.ylabel("Cell Count")
        plt.xlabel("Cell Processing Time [s]")
        
        plt.savefig("Cell_Processing_Times_"+single_or_multi+"ThreadCython_"+str(idx)+".png")
        plt.close()

def plot_cell_kdtree_times(curr_data, idx, single_or_multi):
    
    if curr_data.shape[0] > 0:
    
        plt.hist(curr_data, bins=50)
        
        curr_title = ""
        #curr_title += "Cell Processing Times [s]\n"
        curr_title += "Cell Exit Stage: " + str(idx) + "\n"
        curr_title += "Total vals: " + str(curr_data.shape[0])
        
        plt.title(curr_title)
        
        
        x_min = curr_data.min()
        x_max = curr_data.max()
        
        x_range = x_max - x_min
        
        x_ticks = np.linspace(x_min - .05*x_range, x_max + .05*x_range, 5)
        x_ticks = x_ticks[1:-1]
        x_tick_labels = ["{:.2E}".format(x_tick) for x_tick in x_ticks]
        
        
        
        plt.xticks(x_ticks, x_tick_labels)
        
        plt.ylabel("Cell Count")
        plt.xlabel("KDTree Processing Time [s]")
        
        plt.savefig("Cell_KDTree_Times_"+single_or_multi+"ThreadCython_"+str(idx)+".png")
        plt.close()




class LockedWrapper():
        
    def __init__(self, galaxy_tree):
        
        self.tree = galaxy_tree
        
        self.lock = RLock()
        
    def query_radius(self, *args, **kwargs):
        
        self.lock.acquire()
        
        results = self.tree.query_radius(*args, **kwargs)
        
        self.lock.release()
        
        return results
    
    def query(self, *args, **kwargs):
        
        self.lock.acquire()
        
        results = self.tree.query(*args, **kwargs)
        
        self.lock.release()
        
        return results

if __name__ == "__main__":
    """
    For running in multi-process mode, the run_multi_process function above now uses
    the helper function _main_hole_finder_startup() and os.execv(), which in turn
    invokes this file _voidfinder.py as a script, and will enter this
    if __name__ == "__main__" block, where the real worker function is called.
    
    The worker then will load a pickled config file based on the 
    `config_path` argument which the worker can then use to
    configure itself and connect via socket to the main process.
    """
    
    run_args = sys.argv
    
    worker_idx = sys.argv[1]
    
    config_path = sys.argv[2]
    
    print("WORKER STARTED WITH ARGS: ", sys.argv)
    
    _main_hole_finder_worker(worker_idx, config_path)
    
    
    
    
    
    
    
    
    
    
    
    
    