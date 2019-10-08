


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
import tempfile
from psutil import cpu_count


import cProfile

import numpy as np

import time

#from sklearn import neighbors

from .voidfinder_functions import not_in_mask

from ._voidfinder_cython import main_algorithm, fill_ijk
from ._voidfinder_cython_find_next import GalaxyMap, Cell_ID_Memory



from multiprocessing import Queue, Process, RLock, Value, Array

from ctypes import c_int64, c_double, c_float

from queue import Empty

from copy import deepcopy

import pickle

from astropy.table import Table

from .table_functions import to_array

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt



def _main_hole_finder(void_grid_shape, 
                      void_grid_edge_length, 
                      hole_center_iter_dist,
                      search_grid_edge_length,
                      coord_min, 
                      mask,
                      mask_resolution,
                      min_dist,
                      max_dist,
                      galaxy_coords,
                      batch_size=1000,
                      verbose=0,
                      print_after=5.0,
                      num_cpus=1):
    '''
    Description
    ===========

    See help(voidfinder.find_voids)

    This function is basically a glorified switch between single-threaded mode and
    multi-processed mode of running VoidFinder.

    
    Parameters
    ==========
    
    void_grid_shape : array or tuple of length 3
        the number of grid cells in each of the 3 x,y,z dimensions
    
    void_grid_edge_length : scalar float
        length of each cell in Mpc/h
        
    hole_center_iter_dist : scalar float
        distance to shift hole centers during iterative void hole growing in 
        Mpc/h
        
        
    search_grid_edge_length : float or None
        edge length in Mpc/h for the secondary grid for finding nearest neighbor
        galaxies.  If None, will default to 3*void_grid_edge_length (which results
        in a cell volume of 3^3 = 27 times larger cube volume).  This parameter
        yields a tradeoff between number of galaxies in a cell, and number of
        cells to search when growing a sphere.  Too large and many redundant galaxies
        may be searched, too small and too many cells will need to be searched.
        (xyz space)
        
    coord_min : numpy.ndarray of shape (1,3) 
        minimum coordinates of the survey in x,y,z in Mpc/h
        Note that this coordinate is used for transforming values into the i,j,k search
        grid space and also into the p,q,r galaxy map grid space
        
    mask : numpy.ndarray of shape (N,M) type bool
        represents the survey footprint in scaled ra/dec space.  Value of True 
        indicates that a location is within the survey

    mask_resolution : integer
        Scale factor of coordinates needed to index mask
    
    min_dist : float
        minimum redshift in units of Mpc/h
        
    max_dist : float
        maximum redshift in units of Mpc/h
        
    galaxy_coords : numpy.ndarray of shape (num_galaxies, 3)
        coordinates of the galaxies in the survey, units of Mpc/h
        (xyz space)

    batch_size : scalar float
        Number of empty cells to pass into each process.  Initialized to 1000.

    verbose : int
        value to determine whether or not to display status messages while 
        running.  0 is off, 1 is print after N updates, 2 is full debugging prints.

    num_cpus : int or None
        number of cpus to use while running the main algorithm.  None will result
        in using number of physical cores on the machine.  Some speedup benefit
        may be obtained from using additional logical cores via Intel Hyperthreading
        but with diminishing returns.  This can safely be set above the number of 
        physical cores without issue if desired.
    
    
    
    Returns
    =======
    
    x_y_z_r_array : numpy.ndarray of shape (N,4)
        x,y,z coordinates of the N hole centers found in units of Mpc/h (cols 0,1,2)
        and Radii of the N holes found in units of Mpc/h

    n_voids : scalar float
        Number of voids found - note this number is prior to the void combining
        stage so will not represent the final output of VoidFinder
    '''

    ################################################################################
    # Run single or multi-processed
    ################################################################################
    
    if isinstance(num_cpus, int) and num_cpus == 1:
        
        
        #cProfile.runctx("run_single_process_cython(ngrid, dl, dr, coord_min, mask, mask_resolution, min_dist, max_dist, w_coord, batch_size=batch_size, verbose=verbose, print_after=print_after, num_cpus=num_cpus)", globals(), locals(), 'prof_single.prof')
        #x_y_z_r_array = None
        #n_holes = None
        
        
        x_y_z_r_array, n_voids = run_single_process_cython(void_grid_shape, 
                                                           void_grid_edge_length, 
                                                           hole_center_iter_dist,
                                                           search_grid_edge_length,
                                                           coord_min, 
                                                           mask,
                                                           mask_resolution,
                                                           min_dist,
                                                           max_dist,
                                                           galaxy_coords,
                                                           batch_size=batch_size,
                                                           verbose=verbose,
                                                           print_after=print_after,
                                                           num_cpus=num_cpus
                                                           )
        
        
        return x_y_z_r_array, n_voids
        
        
    else:
        
        x_y_z_r_array, n_voids = run_multi_process(void_grid_shape, 
                                                   void_grid_edge_length, 
                                                   hole_center_iter_dist,
                                                   search_grid_edge_length,
                                                   coord_min, 
                                                   mask,
                                                   mask_resolution,
                                                   min_dist,
                                                   max_dist,
                                                   galaxy_coords,
                                                   batch_size=batch_size,
                                                   verbose=verbose,
                                                   print_after=print_after,
                                                   num_cpus=num_cpus
                                                   )
    
    
        
    return x_y_z_r_array, n_voids


    
    

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
        
        #self.ijko = np.empty(4, dtype=np.int64)
        
        
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
        
        #ijko_bytes = self.cell_ID_gen_mmap.read(32)
        
        
        #print(len(i_bytes), i_bytes)
        
        self.i[0] = struct.unpack("=q", i_bytes)[0]
        self.j[0] = struct.unpack("=q", j_bytes)[0]
        self.k[0] = struct.unpack("=q", k_bytes)[0]
        self.out_idx[0] = struct.unpack("=q", out_idx_bytes)[0]
        
        #self.ijko[:] = np.frombuffer(ijko_bytes, dtype=np.int64)
        
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
        
       
class MultiCellIDGenerator_2(object):
    
    def __init__(self, 
                 grid_dim_1, 
                 grid_dim_2, 
                 grid_dim_3, 
                 cell_ID_dict,
                 ):
        
        
        self.data = np.empty(4, dtype=np.int64)
        
        self.data[0] = grid_dim_1
        self.data[1] = grid_dim_2
        self.data[2] = grid_dim_3 #also modulus for finding j
        self.data[3] = grid_dim_2*grid_dim_3 #modulus for finding i
        #456 for ijk
        
        
        self.cell_ID_dict = cell_ID_dict
        
    def gen_cell_ID_batch(self, start_idx, batch_size, output_array=None):
        
        if output_array is None:
            
            output_array = np.empty((batch_size, 3), dtype=np.int64)
            
        #i = integer division of index by number of j*ks
        i, remainder = divmod(start_idx, self.data[3]) 
        
        #j = integer division of remainder after i by number of ks
        #k is the final remainder
        j, k = divmod(remainder, self.data[2]) 
        
        num_out = fill_ijk(output_array, 
                           i, 
                           j, 
                           k, 
                           batch_size,
                           self.data[0],
                           self.data[1],
                           self.data[2],
                           self.cell_ID_dict
                           )
        
        return num_out
        
        
        
        

    
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



    

    
def run_single_process_cython(void_grid_shape, 
                              void_grid_edge_length, 
                              hole_center_iter_dist,
                              search_grid_edge_length,
                              coord_min, 
                              mask,
                              mask_resolution,
                              min_dist,
                              max_dist,
                              galaxy_coords,
                              batch_size=1000,
                              verbose=0,
                              print_after=5.0,
                              num_cpus=None,
                              DEBUG_DIR="/home/moose/VoidFinder/doc/debug_dir"
                              ):
    if verbose > 0:
        start_time = time.time()
    
    if verbose > 1:
        print("Running single-process mode", flush=True)
    
    ################################################################################
    #
    #   BUILD NEAREST-NEIGHBOR TREE
    #   galaxy_tree : sklearn.neighbors/scipy KDTree or similar implementing sklearn interface
    #   nearest neighbor finder for the galaxies in x,y,z space
    #
    ################################################################################
    
    if verbose > 1:
        
        kdtree_start_time = time.time()
        
        
    from sklearn import neighbors
    galaxy_kdtree = neighbors.KDTree(galaxy_coords)

    galaxy_tree = GalaxyMap(galaxy_coords, coord_min, search_grid_edge_length)
    
    if verbose > 1:
        
        print('Galaxy Map creation time:', time.time() - kdtree_start_time, flush=True)
    
    cell_ID_mem = Cell_ID_Memory(len(galaxy_tree.galaxy_map))
    
    ################################################################################
    # Create the Cell ID generator
    ################################################################################
    
    mesh_indices = ((galaxy_coords - coord_min)/void_grid_edge_length).astype(np.int64)
        
    cell_ID_dict = {}
        
    for idx in range(mesh_indices.shape[0]):

        bin_ID = tuple(mesh_indices[idx])

        cell_ID_dict[bin_ID] = 1
    
    
    ################################################################################
    # Create the Cell ID generator
    ################################################################################
    
    
    
    start_idx = 0
    
    out_start_idx = 0
    
    cell_ID_gen = MultiCellIDGenerator_2(void_grid_shape[0], 
                                         void_grid_shape[1], 
                                         void_grid_shape[2], 
                                         cell_ID_dict)
    
    if verbose > 1:
        
        print("Len cell_ID_dict (eliminated cells): ", len(cell_ID_dict), flush=True)
    
    ################################################################################
    # Convert the mask to an array of uint8 values for running in the cython code
    ################################################################################
    
    mask = mask.astype(np.uint8)
    
    ################################################################################
    # Main loop
    ################################################################################
    
    n_empty_cells = void_grid_shape[0]*void_grid_shape[1]*void_grid_shape[2] \
                    - len(cell_ID_dict)
    
    RETURN_ARRAY = np.empty((n_empty_cells, 4), dtype=np.float64)
    
    RETURN_ARRAY.fill(np.NAN)
    
    return_array = np.empty((batch_size, 4), dtype=np.float64)
    
    i_j_k_array = np.empty((batch_size, 3), dtype=np.int64)
    
    ################################################################################
    #
    # PROFILING VARIABLES
    #
    # PROFILE ARRAY elements are:
    # 0 - total cell time
    # 1 - cell exit stage
    # 2 - kdtree_time
    ################################################################################
    
    '''
    PROFILE_COUNT = 80000000
    
    PROFILE_ARRAY = np.empty((85000000,3), dtype=np.float64)
    
    PROFILE_array = np.empty((batch_size, 3), dtype=np.float32)
    
    PROFILE_process_start_time = time.time()
    
    PROFILE_sample_times = []
    
    PROFILE_samples = []
    
    PROFILE_start_time = time.time()
    
    PROFILE_sample_time = 5.0
    '''
    
    ################################################################################
    # Set up print timer if verbose is 1 or greater
    ################################################################################
    if verbose > 0:
        
        print_start_time = time.time()
        
    
    ################################################################################
    # Mainloop
    ################################################################################
    
    num_cells_processed = 0
    
    exit_condition = False
    
    while not exit_condition:
        
        '''
        if num_cells_processed >= PROFILE_COUNT:
            
            exit_condition = True
            
            break
        
        if (time.time() - PROFILE_start_time) > PROFILE_sample_time:
            
            curr_sample_time = time.time() - PROFILE_process_start_time
            
            curr_sample_interval = time.time() - PROFILE_start_time
            
            PROFILE_sample_times.append(curr_sample_time)
            
            PROFILE_samples.append(num_cells_processed)
            
            PROFILE_start_time = time.time()
        
            if verbose > 0:
            
                print("Processing cell "+str(num_cells_processed)+" of "+str(n_empty_cells), str(round(curr_sample_time, 2)))
            
            if len(PROFILE_samples) > 3:
                
                cells_per_sec = (PROFILE_samples[-1] - PROFILE_samples[-2])/curr_sample_interval
                
                print(str(round(cells_per_sec, 2)), "cells per sec")
        '''
        
        if verbose > 0:
            
            curr_time = time.time()
            
            if (curr_time - print_start_time) > print_after:
                
                print("Processed cell "+str(num_cells_processed)+" of "+str(n_empty_cells), str(round(curr_time - start_time, 2)))
            
                print_start_time = curr_time
        
        ######################################################################
        # Generate the next batch and run the main algorithm
        ######################################################################
        
        num_write = cell_ID_gen.gen_cell_ID_batch(start_idx, batch_size, i_j_k_array)
        
        start_idx += batch_size
        
        num_cells_to_process = num_write
        
        if num_cells_to_process > 0:

            if return_array.shape[0] != num_cells_to_process:

                return_array = np.empty((num_cells_to_process, 4), dtype=np.float64)
                
                #PROFILE_array = np.empty((num_cells_to_process, 3), dtype=np.float32)
        
            main_algorithm(i_j_k_array[0:num_write],
                           galaxy_tree,
                           #galaxy_kdtree,
                           galaxy_coords,
                           void_grid_edge_length, 
                           hole_center_iter_dist,
                           coord_min,
                           mask,
                           mask_resolution,
                           min_dist,
                           max_dist,
                           return_array,
                           cell_ID_mem,
                           0,  
                           #PROFILE_array
                           )
        
            RETURN_ARRAY[out_start_idx:(out_start_idx+num_write),:] = return_array[0:num_write]
            
            #PROFILE_ARRAY[out_start_idx:(out_start_idx+num_write),:] = PROFILE_array[0:num_write]
            
            num_cells_processed += num_write
            
            out_start_idx += num_write
        
        elif num_cells_to_process == 0 and num_cells_processed == n_empty_cells:
        
            exit_condition = True
            
            
    if verbose > 1:
        
        print("Main task finish time: ", time.time() - start_time)
        
    ######################################################################
    # PROFILING CODE
    ######################################################################
    '''
    outfile = open(os.path.join(DEBUG_DIR, "single_thread_profile.pickle"), 'wb')
    pickle.dump((PROFILE_sample_times, PROFILE_samples), outfile)
    outfile.close()    
    
    
    if verbose > 0:
        
        
        PROFILE_ARRAY_SUBSET = PROFILE_ARRAY[0:PROFILE_COUNT]
        
        for idx in range(7):
            
            curr_idx = PROFILE_ARRAY_SUBSET[:,1] == idx
            
            curr_data = PROFILE_ARRAY_SUBSET[curr_idx, 0]
            
            if idx == 6:
                outfile = open(os.path.join(DEBUG_DIR,"Cell_Processing_Times_SingleThreadCython.pickle"), 'wb')
                pickle.dump(curr_data, outfile)
                outfile.close()
                print("Avg Cell time: ", np.mean(curr_data), len(curr_data))
            
            plot_cell_processing_times(curr_data, idx, "Single", DEBUG_DIR)
            
            curr_data = PROFILE_ARRAY_SUBSET[curr_idx, 2]
            
            if idx == 6:
                print("Avg Cell KDTree time: ", np.mean(curr_data))
            
            plot_cell_kdtree_times(curr_data, idx, 'Single', DEBUG_DIR)
        
    
        
    n_holes = np.sum(np.logical_not(np.isnan(RETURN_ARRAY[0:PROFILE_COUNT,0])), axis=None, dtype=np.int64)
    
    print("N holes: ", n_holes)
    '''
    ######################################################################
    # END PROFILING CODE
    ######################################################################
    
    valid_idx = np.logical_not(np.isnan(RETURN_ARRAY[:,0]))
    
    n_holes = np.sum(valid_idx, axis=None, dtype=np.int64)
    
    return RETURN_ARRAY[valid_idx,:], n_holes



    
    
    
    
    


def run_multi_process(ngrid, 
                      dl, 
                      dr,
                      search_grid_edge_length,
                      coord_min, 
                      mask,
                      mask_resolution,
                      min_dist,
                      max_dist,
                      w_coord,
                      batch_size=1000,
                      verbose=0,
                      print_after=10000,
                      num_cpus=None,
                      CONFIG_PATH="/tmp/voidfinder_config.pickle",
                      SOCKET_PATH="/tmp/voidfinder.sock",
                      RESULT_BUFFER_PATH="/tmp/voidfinder_result_buffer.dat",
                      CELL_ID_BUFFER_PATH="/tmp/voidfinder_cell_ID_gen.dat",
                      PROFILE_BUFFER_PATH="/tmp/voidfinder_profile_buffer.dat",
                      RESOURCE_DIR="/dev/shm",
                      DEBUG_DIR="/home/moose/VoidFinder/doc/debug_dir"
                      ):
    """
    Work-horse method for running VoidFinder with the Cython code in parallel
    multi-process form.  Currently a little bitch because there is some kind of
    blocking going on which actually makes it slower than single-thread.
    
    This method contains the logic for:
    
    1). Sanity check the num_cpus to use
    2). Open a few file handles and allocate memory for workers to memmap to
    3). write a config object to a temporary disk location for the workers
    4). Register some cleanup helpers with the python interpreters for 
            making sure the disk space gets reclaimed when we're done
    5). Start the workers
    6). Make sure workers connect to the comm socket
    7). Collect progress results from the workers
    8). DEBUG - Make some plots of processing timings
    
    
    TODO: Remove the cell_ID_dict functionality.  It has something like 500,000 entries
          out of 226 million possible cell locations, and if we remove that check we won't
          have to build an i_j_k_array object every call, we'll literally just have to
          get the next out_start_idx and each worker can increment itself while only
          updating the shared memory super duper quickly and not having to wait for
          the whole cell_ID_gen.gen_cell_ID_batch() to return.  There is a memory
          tradeoff for this, thats 500,000*4*8 additional bytes we need to
          allocate which will be useless
          
          
    TODO: Check that /dev/shm actually exists, if not fall back to /tmp
    """
    
    if verbose > 0:
        start_time = time.time()
    
    ################################################################################
    # Start by converting the num_cpus argument into the real value we will use
    # by maxing sure its reasonable, or if it was none use the max val available
    #
    # Maybe should use psutil.cpu_count(logical=False) instead of the
    # multiprocessing version?
    #
    ################################################################################
    
    if (num_cpus is None):
          
        num_cpus = cpu_count(logical=False)
    
    ################################################################################
    # An output counter for total number of holes found, and calculate the
    # total number of cells we're going to have to check based on the grid
    # dimensions and the total number of previous cells in the cell_ID_dict which
    # we already discovered we do NOT have to check.
    ################################################################################
    
    mesh_indices = ((w_coord - coord_min)/dl).astype(np.int64)
        
    cell_ID_dict = {}
        
    for idx in range(mesh_indices.shape[0]):

        bin_ID = tuple(mesh_indices[idx])

        cell_ID_dict[bin_ID] = 1
    
    
    
    

    n_empty_cells = ngrid[0]*ngrid[1]*ngrid[2] - len(cell_ID_dict)
    

    ################################################################################
    # Setup a file handle for output memory, we're going to memmap in the
    # worker processes to store results and then we'll use numpy.frombuffer
    # to convert it back into an array to pass back up the chain.
    ################################################################################
    
    result_fd, RESULT_BUFFER_PATH = tempfile.mkstemp(prefix="voidfinder", dir=RESOURCE_DIR, text=False)
    
    #result_fd = os.open(RESULT_BUFFER_PATH, os.O_TRUNC | os.O_CREAT | os.O_RDWR | os.O_CLOEXEC)
    
    #os.unlink(RESULT_BUFFER_PATH)
    
    result_buffer_length = n_empty_cells*4*8
    
    os.ftruncate(result_fd, result_buffer_length)
    
    result_buffer = open(result_fd, 'w+b')
    
    #result_buffer.write(b"0"*result_buffer_length)
    
    ################################################################################
    # Memory for PROFILING
    ################################################################################
    '''
    PROFILE_fd, PROFILE_BUFFER_PATH = tempfile.mkstemp(prefix="voidfinder", dir=RESOURCE_DIR, text=False)
    
    #PROFILE_fd = os.open(PROFILE_BUFFER_PATH, os.O_TRUNC | os.O_CREAT | os.O_RDWR | os.O_CLOEXEC)
    
    PROFILE_buffer_length = 85000000*3*4 #float32 so 4 bytes per element
    
    os.ftruncate(PROFILE_fd, PROFILE_buffer_length)
    
    PROFILE_buffer = open(PROFILE_fd, 'w+b')
    
    #PROFILE_buffer_length = 85000000*3*4 #float32 so 4 bytes per element
    
    #PROFILE_buffer.write(b"0"*PROFILE_buffer_length)
    '''
    ################################################################################
    # Build Cell ID generator memory
    ################################################################################
    #cell_ID_fd, CELL_ID_BUFFER_PATH = tempfile.mkstemp(prefix="voidfinder", dir=RESOURCE_DIR, text=False)
    
    #Don't actually need this memory anymore if we use the ijk_start and write_start vals instead
    #os.close(cell_ID_fd)
    '''
    #cell_ID_fd = os.open(CELL_ID_BUFFER_PATH, os.O_TRUNC | os.O_CREAT | os.O_RDWR | os.O_CLOEXEC)
    
    cell_ID_buffer_length = 4*8 #need 4 8-byte integers: i, j, k, out_idx
    
    os.ftruncate(cell_ID_fd, cell_ID_buffer_length)
    
    cell_ID_buffer = open(cell_ID_fd, 'w+b')
    
    #cell_ID_buffer_length = 4*8 #need 4 8-byte integers: i, j, k, out_idx
    
    cell_ID_buffer.write(b"\x00"*cell_ID_buffer_length)
    
    cell_ID_buffer.flush()
    '''
    ################################################################################
    # Shared memory values to help generating Cell IDs
    ################################################################################
    ijk_start = Value(c_int64, 0, lock=True)
    
    write_start = Value(c_int64, 0, lock=True)
    
    
    
    
    ################################################################################
    # Dump configuration to a file for worker processes to get it
    ################################################################################
    if verbose > 0:
        print("Grid: ", ngrid, flush=True)
    
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
                     num_cpus,
                     search_grid_edge_length,
                     DEBUG_DIR
                     )
    
    outfile = open(CONFIG_PATH, 'wb')
    
    pickle.dump(config_object, outfile)
    
    outfile.close()
    
    
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
        
    #def cleanup_cellID():
        
    #    os.remove(CELL_ID_BUFFER_PATH)
        
    #def cleanup_profile():
        
    #    os.remove(PROFILE_BUFFER_PATH)
    
    def possibly_send_exit_commands():
        
        if not sent_exit_commands:
        
            for idx in range(num_cpus):
                
                worker_sockets[idx].send(b"exit")
        
    
    
    atexit.register(cleanup_config)
    
    atexit.register(cleanup_socket)
    
    atexit.register(cleanup_result)
    
    atexit.register(possibly_send_exit_commands)
    
    #atexit.register(cleanup_cellID)
    
    #atexit.register(cleanup_profile)
    
    ################################################################################
    # Start the worker processes
    ################################################################################
        
    listener_socket = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM | socket.SOCK_CLOEXEC)
    
    listener_socket.bind(SOCKET_PATH)
    
    listener_socket.listen(num_cpus)
    
    startup_context = multiprocessing.get_context("fork")
        
    processes = []
    
    for proc_idx in range(num_cpus):
        
        #p = startup_context.Process(target=_main_hole_finder_startup, args=(proc_idx, CONFIG_PATH))
        p = startup_context.Process(target=_main_hole_finder_worker, 
                                    args=(proc_idx, 
                                          ijk_start, 
                                          write_start, 
                                          CONFIG_PATH))
        #p = startup_context.Process(target=_main_hole_finder_profile, args=(proc_idx, CONFIG_PATH))
        
        p.start()
        
        processes.append(p)
    
    ################################################################################
    # Make sure each worker process connects to the main socket, so we block on
    # the accept() call below until we get a connection, and make sure we get 
    # exactly num_cpus connections
    ################################################################################
    if verbose > 0:
        print("Waiting on workers to connect", flush=True)
    
    num_active_processes = 0
    
    worker_sockets = []
    
    message_buffers = []
    
    socket_index = {}
    
    for idx in range(num_cpus):
        
        worker_sock, worker_addr = listener_socket.accept()
        
        worker_sockets.append(worker_sock)
        
        num_active_processes += 1
        
        message_buffers.append(b"")
        
        socket_index[worker_sock.fileno()] = idx
        
    if verbose > 0:
        
        print("Worker processes started time: ", time.time() - start_time)
    
    ################################################################################
    # PROFILING VARIABLES
    ################################################################################
    '''
    PROFILE_process_start_time = time.time()
    
    PROFILE_sample_times = []
    
    PROFILE_samples = []
    
    PROFILE_start_time = time.time()
    
    PROFILE_sample_time = 5.0
    '''
    
    
    ################################################################################
    # LOOP TO LISTEN FOR RESULTS WHILE WORKERS WORKING
    ################################################################################
    if verbose > 0:
        print_after_time = time.time()
    
    num_cells_processed = 0
    
    empty1 = []
    
    empty2 = []
    
    select_timeout = 2.0
    
    sent_exit_commands = False
    
    while num_active_processes > 0:
        
        ################################################################################
        # DEBUGGING CODE
        ################################################################################
        '''
        if num_cells_processed >= 80000000:
        
            print("Breaking debug loop", num_cells_processed, num_active_processes)
            
            for idx in range(num_cpus):
                
                worker_sockets[idx].send(b"exit")
                
            sent_exit_commands = True
            
            break
        
        
        if time.time() - PROFILE_start_time > PROFILE_sample_time:
            
            curr_sample_time = time.time() - PROFILE_process_start_time
            
            curr_sample_interval = time.time() - PROFILE_start_time
            
            PROFILE_sample_times.append(curr_sample_time)
            
            PROFILE_samples.append(num_cells_processed)
            
            PROFILE_start_time = time.time()
            
            if verbose > 0:
                
                print('Processed', num_cells_processed, 'cells of', n_empty_cells, str(round(curr_sample_time,2)))
                
                if len(PROFILE_samples) > 3:
                    
                    cells_per_sec = (PROFILE_samples[-1] - PROFILE_samples[-2])/curr_sample_interval
                    
                    print(str(round(cells_per_sec,2)), "cells per sec")
        '''
        
        if verbose > 0:
            
            curr_time = time.time()
            
            if (curr_time - print_after_time) > print_after:
            
                print('Processed', num_cells_processed, 'cells of', n_empty_cells, str(round(curr_time-start_time,2)))
                
                print_after_time = curr_time
            
        ################################################################################
        # END DEBUGGING CODE
        ################################################################################
            
            
        read_socks, empty3, empty4 = select.select(worker_sockets, empty1, empty2, select_timeout)
        
        if read_socks:
            
            for worker_sock in read_socks:
                
                sock_idx = socket_index[worker_sock.fileno()]
                
                curr_read = worker_sock.recv(1024)
                
                curr_message_buffer = message_buffers[sock_idx]
                
                curr_message_buffer += curr_read
                
                messages, remaining_buffer = process_message_buffer(curr_message_buffer)
                
                message_buffers[sock_idx] = remaining_buffer
                    
                for message in messages:
                    
                    if message == b"":
                        continue
                    
                    message_type = struct.unpack("=q", message[0:8])[0]
                    
                    if message_type == 0:
                        
                        num_result = struct.unpack("=q", message[8:16])[0]
                        
                        #num_hole = struct.unpack("=q", message[16:24])[0]
                        
                        num_cells_processed += num_result
                        
                        #n_holes += num_hole
                        
                    elif message_type == 1:
                        
                        num_active_processes -= 1


    ################################################################################
    # Clean up worker processes
    ################################################################################
    
    if verbose > 0:
        
        print("Main task finish time: ", time.time() - start_time)
    
    
    if not sent_exit_commands:
        
        for idx in range(num_cpus):
                
            worker_sockets[idx].send(b"exit")
    
    
    for p in processes:
        
        p.join(None) #block till join
    
    ################################################################################
    # PROFILING - SAVE OFF RESULTS
    ################################################################################
    '''
    outfile = open(os.path.join(DEBUG_DIR, "multi_thread_profile.pickle"), 'wb')
    
    pickle.dump((PROFILE_sample_times, PROFILE_samples), outfile)
    
    outfile.close()
    
    if verbose > 0:
        
        PROFILE_ARRAY = np.memmap(PROFILE_buffer, dtype=np.float32, shape=(85000000,3))
        
        PROFILE_ARRAY_SUBSET = PROFILE_ARRAY[0:80000000]
        
        for idx in range(7):
            
            #curr_axes = axes_list[idx]
            
            curr_idx = PROFILE_ARRAY_SUBSET[:,1] == idx
            
            curr_data = PROFILE_ARRAY_SUBSET[curr_idx, 0]
            
            if idx == 6:
                
                print("Count of profile stage 6: ", curr_data.shape[0])
                print("Avg Cell time: ", np.mean(curr_data))
            
            if idx == 6:
                outfile = open(os.path.join(DEBUG_DIR, "Cell_Processing_Times_MultiThreadCython.pickle"), 'wb')
                pickle.dump(curr_data, outfile)
                outfile.close()
            
            plot_cell_processing_times(curr_data, idx, "Multi", DEBUG_DIR)
            
            curr_data = PROFILE_ARRAY_SUBSET[curr_idx, 2]
            
            if idx == 6:
                print("Avg Cell KDTree time: ", np.mean(curr_data))
            
            plot_cell_kdtree_times(curr_data, idx, 'Multi', DEBUG_DIR)
    '''
    ################################################################################
    # Close the unneeded file handles to the shared memory, and return the file
    # handle to the result memory buffer
    ################################################################################
    #cell_ID_buffer.close()
    
    #PROFILE_buffer.close()
    
    result_buffer.seek(0)
    
    result_array = np.frombuffer(result_buffer.read(), dtype=np.float64)
    
    result_array.shape = (n_empty_cells, 4)
    
    result_buffer.close()
    
    #n_holes = np.sum(np.logical_not(np.isnan(result_array[0:80000000,0])), axis=None, dtype=np.int64)
    
    valid_idx = np.logical_not(np.isnan(result_array[:,0]))
    
    n_holes = np.sum(valid_idx, axis=None, dtype=np.int64)
    
    #print(result_array.shape, valid_idx.shape, n_holes)
        
    return result_array[valid_idx,:], n_holes
                    
    
def process_message_buffer(curr_message_buffer):
    """
    Helper function to process the communication between worker processes and 
    main voidfinder processes.  Since communication over a socket is only guaranteed
    to be in order, we have to process an arbitrary number of bytes depending on
    the message format.  The message format is as such:  the first byte gives the number
    of 8-byte fields in the message.  So a first byte of 3 means the message on the head
    of the buffer should be 1 + 3*8 = 25 bytes long.
    
    curr_message_buffer : bytes
        the current string of bytes to process
    """
    
    messages = []
    
    
    if len(curr_message_buffer) > 0:
                
        messages_remaining_in_buffer = True
        
    else:
        
        messages_remaining_in_buffer = False
        
        
    while messages_remaining_in_buffer:
    
        #https://stackoverflow.com/questions/28249597/why-do-i-get-an-int-when-i-index-bytes
        #implicitly converts the 0th byte to an integer
        msg_fields = curr_message_buffer[0]
        
        msg_len = 1 + 8*msg_fields
        
        if len(curr_message_buffer) >= msg_len:
        
            curr_msg = curr_message_buffer[1:msg_len]
            
            messages.append(curr_msg)
            
            curr_message_buffer = curr_message_buffer[msg_len:]
            
            if len(curr_message_buffer) > 0:
                
                messages_remaining_in_buffer = True
                
            else:
                
                messages_remaining_in_buffer = False
                
        else:
            messages_remaining_in_buffer = False
    
    return messages, curr_message_buffer
    
    
    
def _main_hole_finder_profile(worker_idx, ijk_start, write_start, config_path):
    
    cProfile.runctx("_main_hole_finder_worker(worker_idx, ijk_start, write_start, config_path)", globals(), locals(), 'prof%d.prof' %worker_idx)
    
    
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
    
    env = {"PYTHONHOME" : os.path.abspath(os.path.join(os.path.dirname(sys.executable), "..")),
           "USER" : os.environ["USER"]}
    
    print(sys.executable, args)
    
    #os.spawnve(os.P_NOWAIT, sys.executable, args, os.environ)
    #os.execve(sys.executable, args, env)
    os.execve(sys.executable, args, env)
    
    
    


def _main_hole_finder_worker(worker_idx, ijk_start, write_start, config_path):
    
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
    search_grid_edge_length = config[18]
    DEBUG_DIR = config[19]



    worker_socket = socket.socket(socket.AF_UNIX)
    
    worker_socket.connect(SOCKET_PATH)
    
    worker_socket.setblocking(False)
    
    ################################################################################
    #
    #   BUILD NEAREST-NEIGHBOR TREE
    #   galaxy_tree : sklearn.neighbors/scipy KDTree or similar implementing sklearn interface
    #   nearest neighbor finder for the galaxies in x,y,z space
    #
    ################################################################################

    if verbose:
        
        kdtree_start_time = time.time()
        
    #from sklearn import neighbors
    #galaxy_tree = neighbors.KDTree(w_coord)
    
    #from scipy.spatial import KDTree
    #galaxy_tree = KDTree(w_coord)
    
    galaxy_tree = GalaxyMap(w_coord, coord_min, search_grid_edge_length)
    
    
    if verbose:
        
        print('KDTree creation time:', time.time() - kdtree_start_time)
        
        
    cell_ID_mem = Cell_ID_Memory(len(galaxy_tree.galaxy_map))
    
    
    ################################################################################
    #
    ################################################################################
    
    
    n_empty_cells = ngrid[0]*ngrid[1]*ngrid[2] - len(cell_ID_dict)
    
    result_buffer = open(RESULT_BUFFER_PATH, 'r+b')
    
    result_buffer_length = n_empty_cells*4*8 #float64 so 8 bytes per element
    
    result_mmap_buffer = mmap.mmap(result_buffer.fileno(), result_buffer_length)
    
    
    
    
    
    ################################################################################
    # Build Cell ID generator
    ################################################################################
    '''
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
    '''
    
    
    
    cell_ID_gen = MultiCellIDGenerator_2(ngrid[0],
                                         ngrid[1],
                                         ngrid[2],
                                         cell_ID_dict)
    
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
    # Memory for PROFILING
    ################################################################################
    '''
    PROFILE_buffer = open(PROFILE_BUFFER_PATH, 'r+b')
    
    PROFILE_buffer_length = 85000000*3*4 #float32 so 4 bytes per element
    
    PROFILE_mmap_buffer = mmap.mmap(PROFILE_buffer.fileno(), PROFILE_buffer_length)
    
    PROFILE_array = np.empty((batch_size, 3), dtype=np.float32)
    
    PROFILE_gen_times = []
    
    PROFILE_main_times = []
    '''
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
    
    i_j_k_array = np.empty((batch_size, 3), dtype=np.int64)
    
    worker_sockets = [worker_socket]
    
    empty1 = []
    
    empty2 = []
    
    message_buffer = b""
    
    while not exit_process:
        
        total_loops += 1
        
        read_socks, empty3, empty4 = select.select(worker_sockets, empty1, empty2, 0)
        
        if read_socks:
            
            message_buffer += worker_socket.recv(1024)
            
            if len(message_buffer) >= 4 and message_buffer[0:4] == b'exit':
                
                exit_process = True
                
                received_exit_command = True
                
                continue
        
        ################################################################################
        # Locked access to cell ID generation
        ################################################################################
        
        
        
        ijk_start.acquire()
        
        start_idx = ijk_start.value
        
        ijk_start.value += batch_size
        
        ijk_start.release()
        
        
        
        
        #PROFILE_gen_start = time.time_ns()
        
        num_write = cell_ID_gen.gen_cell_ID_batch(start_idx, batch_size, i_j_k_array)
        
        #PROFILE_gen_end = time.time_ns()
        
        #PROFILE_gen_times.append((PROFILE_gen_end, PROFILE_gen_start))
        
        #print("Worker: ", worker_idx, i_j_k_array[0,:], out_start_idx)
        
        ################################################################################
        #
        ################################################################################
        num_cells_to_process = num_write
        
        if num_cells_to_process > 0:
            
            #PROFILE_main_start = time.time_ns()

            if return_array.shape[0] != num_cells_to_process:

                return_array = np.empty((num_cells_to_process, 4), dtype=np.float64)
                
                #PROFILE_array = np.empty((num_cells_to_process, 3), dtype=np.float32)
                
            main_algorithm(i_j_k_array[0:num_write],
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
                           cell_ID_mem,
                           0,  #verbose level
                           #PROFILE_array
                           )
            
            num_cells_processed += num_write
            
            #RETURN_ARRAY[out_start_idx:(out_start_idx+return_array.shape[0])] = return_array
            
            
            
            write_start.acquire()
            
            out_start_idx = write_start.value
            
            write_start.value += num_write
            
            write_start.release()
            
            
            
            seek_location = 32*out_start_idx
            result_mmap_buffer.seek(seek_location)
            result_mmap_buffer.write(return_array[0:num_write].tobytes())
            
            
            
            #PROFILE_ARRAY[out_start_idx:(out_start_idx+return_array.shape[0]),:] = PROFILE_array
            #seek_location = 12*out_start_idx
            #PROFILE_mmap_buffer.seek(seek_location)
            #PROFILE_mmap_buffer.write(PROFILE_array[0:num_write].tobytes())
            
            
            
            #n_hole = np.sum(np.logical_not(np.isnan(return_array[:,0])), axis=None, dtype=np.int64)
            
            #if not isinstance(n_hole, np.int64):
            #    print("N_hole not integer: ", n_hole, type(n_hole))
            
            #return_queue.put(("data", return_array.shape[0], n_hole))
            out_msg = b""
            #out_msg += struct.pack("=q", 4)
            out_msg += struct.pack("b", 2)
            out_msg += struct.pack("=q", 0)
            #out_msg += b","
            out_msg += struct.pack("=q", num_write)
            #out_msg += b","
            #out_msg += struct.pack("=q", n_hole)
            #out_msg += b"\n"
            
            #print(worker_idx, out_msg)
            
            worker_socket.send(out_msg)
            
            #PROFILE_main_end = time.time_ns()
            
            #PROFILE_main_times.append((PROFILE_main_end, PROFILE_main_start, return_array.shape[0]))
            
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
    
    #outfile = open(os.path.join(DEBUG_DIR, "multi_gen_times_"+str(worker_idx)+".pickle"), 'wb')
    #pickle.dump((PROFILE_gen_times, PROFILE_main_times), outfile)
    #outfile.close()
    
    #del RETURN_ARRAY #flushes np.memmap 
    #del PROFILE_ARRAY #flushes np.memmap
    #return_queue.put(("done", None))
    
    print("WORKER EXITING GRACEFULLY", worker_idx)
    
    return None






def plot_cell_processing_times(curr_data, idx, single_or_multi, out_dir):
    
    
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
        
        plt.savefig(os.path.join(out_dir, "Cell_Processing_Times_"+single_or_multi+"ThreadCython_"+str(idx)+".png"))
        plt.close()

def plot_cell_kdtree_times(curr_data, idx, single_or_multi, out_dir):
    
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
        
        plt.savefig(os.path.join(out_dir, "Cell_KDTree_Times_"+single_or_multi+"ThreadCython_"+str(idx)+".png"))
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
    
    
    
    
    
    
    
    
    
    
    
    
    