



import os
import stat
import sys
import fcntl
import mmap
import struct
import socket
import select
import atexit
import signal
import tempfile
import multiprocessing
from psutil import cpu_count


import cProfile

import numpy as np

import time

from .voidfinder_functions import not_in_mask

from ._voidfinder_cython import main_algorithm, \
                                fill_ijk


from ._voidfinder_cython_find_next import GalaxyMap, \
                                          Cell_ID_Memory, \
                                          GalaxyMapCustomDict



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
    """
    This class generates an array of 'batch_size' (i,j,k) cell IDs from the
    VoidFinder search grid, for the main VoidFinder algorithm to run on.
    
    To generate the cell IDs, this class stores the shape of the search grid, and 
    a dictionary of the grid cells which do not need to be searched.
    """
    
    def __init__(self, 
                 grid_dim_i, 
                 grid_dim_j, 
                 grid_dim_k, 
                 cell_ID_dict,
                 ):
        
        self.data = np.empty(4, dtype=np.int64)
        
        ################################################################################
        # Store off the shape of the 3D search grid, but also store a 4th value so 
        # we don't keep having to recalculate the modulus number used for finding the
        # i value.
        ################################################################################
        self.data[0] = grid_dim_i
        self.data[1] = grid_dim_j
        self.data[2] = grid_dim_k #also modulus for finding j
        self.data[3] = grid_dim_j*grid_dim_k #modulus for finding i
        
        self.cell_ID_dict = cell_ID_dict
        
    def gen_cell_ID_batch(self, start_idx, batch_size, output_array):
        """
        Description
        ===========
        
        Given the user-provided 'output_array', fill in at most 'batch_size' cell IDs into
        'output_array'. 
        
        Note that this class may return less than 'batch_size' values if the end of
        the grid is reached.  Also, the returned cell_IDs will be filtered by the
        provided cell_ID_dict, so while they will be returned in-order, they are not
        guaranteed to be perfectly sequential.  The actual number of cell IDs written into
        the 'output_array' will be returned as an integer from this method.
        
        This class uses the natural grid ordering (ex: 0->(0,0,0), 1->(0,0,1), 2->(0,0,2), ...)
        to map from a 'start_idx' to a starting grid cell,  start_idx->(i,j,k). 
        
        
        Parameters
        ==========
        
        start_idx : int
            starting cell index in the natural grid ordering 0->(0,0,0), 1->(0,0,1), 2->(0,0,2)
            
        batch_size : int
            how many (i,j,k) cell IDs to write out
            
        output_array : numpy.ndarray of shape (N,3), of dtype numpy.int64
            The memory to write the resultant cell IDs into
            N must be greater than or equal to batch_size
        
        
        Returns
        =======
        
        num_out : int
            number of cell IDs actually written into the output_array
            (may be less than or equal to batch_size)
        
        """
        
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
        
        print("Running single-process mode", flush=True)
    
    ################################################################################
    #
    #   BUILD NEAREST-NEIGHBOR TREE
    #   galaxy_tree : sklearn.neighbors/scipy KDTree or similar implementing sklearn interface
    #   nearest neighbor finder for the galaxies in x,y,z space
    #
    ################################################################################
    '''
    if verbose > 1:
        
        kdtree_start_time = time.time()
        
        
    from sklearn import neighbors
    
    galaxy_kdtree = neighbors.KDTree(galaxy_coords)

    galaxy_tree = GalaxyMap(galaxy_coords, coord_min, search_grid_edge_length)
    
    if verbose > 1:
        
        print('Galaxy Map creation time:', time.time() - kdtree_start_time, flush=True)
    '''
        
    ################################################################################
    # An output counter for total number of holes found, and calculate the
    # total number of cells we're going to have to check based on the grid
    # dimensions and the total number of previous cells in the cell_ID_dict which
    # we already discovered we do NOT have to check.
    #
    #
    # Create the GalaxyMap index and GalaxyMap data array and memmap them for the 
    # workers
    ################################################################################
    mesh_indices = ((galaxy_coords - coord_min)/search_grid_edge_length).astype(np.int64)
        
    #cell_ID_dict = {}
        
    galaxy_map = {}

    for idx in range(mesh_indices.shape[0]):

        bin_ID = tuple(mesh_indices[idx])

        #cell_ID_dict[bin_ID] = 1
        
        if bin_ID not in galaxy_map:
            
            galaxy_map[bin_ID] = []
        
        galaxy_map[bin_ID].append(idx)
        
    del mesh_indices
    
    
    num_in_galaxy_map = len(galaxy_map)
        
        
    ################################################################################
    # Convert the galaxy map from a map of grid_cell_ID -> belonging galaxy indices
    # to a map from grid_cell_ID -> (offset, num) into 
    ################################################################################
    
    offset = 0
    
    galaxy_map_list = []
    
    for key in galaxy_map:
        
        indices = np.array(galaxy_map[key], dtype=np.int64)
        
        num_elements = indices.shape[0]
        
        galaxy_map_list.append(indices)
        
        galaxy_map[key] = (offset, num_elements)
        
        offset += num_elements

    galaxy_map_array = np.concatenate(galaxy_map_list)
    
    del galaxy_map_list
        
    ################################################################################
    # Convert the galaxy_map dictionary into a custom dictionary that we can
    # use to memmap down to our workers
    ################################################################################
    
    num_galaxy_map_elements = len(galaxy_map)
    
    next_prime = find_next_prime(2*num_galaxy_map_elements)
    
    lookup_memory = np.zeros(next_prime, dtype=[("filled_flag", np.uint8, 1),
                                                   ("i", np.uint16, 1),
                                                   ("j", np.uint16, 1),
                                                   ("k", np.uint16, 1),
                                                   ("offset", np.int64, 1),
                                                   ("num_elements", np.int64, 1)])
    
    new_galaxy_map = GalaxyMapCustomDict(void_grid_shape,
                                         lookup_memory)
    
    for curr_ijk in galaxy_map:
        
        offset, num_elements = galaxy_map[curr_ijk]
        
        new_galaxy_map.setitem(*curr_ijk, offset, num_elements)
        
    del galaxy_map
    
    galaxy_map = new_galaxy_map
    
    
    if verbose > 0:
        print("Rebuilt galaxy map (size", num_in_galaxy_map, "total slots ", next_prime,")")
        print("Num collisions in rebuild: ", new_galaxy_map.num_collisions, flush=True)
        
        
    cell_ID_mem = Cell_ID_Memory(10000)
    
    ################################################################################
    # 
    ################################################################################
    '''
    mesh_indices = ((galaxy_coords - coord_min)/void_grid_edge_length).astype(np.int64)
        
    cell_ID_dict = {}
        
    for idx in range(mesh_indices.shape[0]):

        bin_ID = tuple(mesh_indices[idx])

        cell_ID_dict[bin_ID] = 1
    
    '''
    
    galaxy_tree = GalaxyMap(galaxy_coords, 
                            coord_min, 
                            search_grid_edge_length,
                            galaxy_map,
                            galaxy_map_array)
    
    
    
    ################################################################################
    # Create the Cell ID generator
    ################################################################################
    
    
    
    start_idx = 0
    
    out_start_idx = 0
    
    cell_ID_gen = CellIDGenerator(void_grid_shape[0], 
                                  void_grid_shape[1], 
                                  void_grid_shape[2], 
                                  galaxy_map)
    
    if verbose > 1:
        
        print("Len cell_ID_dict (eliminated cells): ", num_in_galaxy_map, flush=True)
    
    ################################################################################
    # Convert the mask to an array of uint8 values for running in the cython code
    ################################################################################
    
    mask = mask.astype(np.uint8)
    
    ################################################################################
    # Main loop
    ################################################################################
    
    n_empty_cells = void_grid_shape[0]*void_grid_shape[1]*void_grid_shape[2] \
                    - num_in_galaxy_map
    
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
            
            
    if verbose > 0:
        
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
                      #RESULT_BUFFER_PATH="/tmp/voidfinder_result_buffer.dat",
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
    
    
    TODO: Replace the cell_ID_dict with the galaxy_map, they both have the same exact
          cell ID keys in them
          
          
    TODO: Add a check that /dev/shm actually exists, if not fall back to /tmp?
    """
    
    if verbose > 0:
        start_time = time.time()
        
    if not os.path.isdir(RESOURCE_DIR):
        
        print("WARNING: RESOURCE DIR ", RESOURCE_DIR, "does not exist.  Falling back to /tmp but could be slow", flush=True)
        
        RESOURCE_DIR = "/tmp"
        
        
    
    ################################################################################
    # Start by converting the num_cpus argument into the real value we will use
    # by making sure its reasonable, or if it was none use the max val available
    #
    # Maybe should use psutil.cpu_count(logical=False) instead of the
    # multiprocessing version?
    #
    ################################################################################
    
    if (num_cpus is None):
          
        num_cpus = cpu_count(logical=False)
        
    if verbose > 0:
        
        print("Running multi-process mode,", str(num_cpus), "cpus", flush=True)
        
        
    ################################################################################
    # Memmap the w_coord array to our worker processes 
    #
    # This is achieved with file descriptors (an integer referring to
    # the file descriptors as given by the kernel).  The tempfile.mkstemp() call 
    # below returns an integer file descriptor which will correspond to this
    # particular memory mapping.  When we create child processes using fork(), 
    # those child processes inherit a copy of the file descriptor with the same
    # integer value.  We pass this value to the child processes via the config
    # object below, and they can use that file descriptor to memmap the same
    # memory location.  We could have also used the path returned by the
    # tempfile.mkstemp() call, but in that case we would have to leave the link
    # to the memory mapping it created on the filesystem.  By using the file
    # descriptor instead of the file path, we can immediately os.unlink() the path,
    # so if VoidFinder crashes, the kernel reference count for that memory mapping
    # will drop to 0 and it will be able to free that memory automatically.  If we
    # left the link (the path) on the filesystem and VF crashed, the RAM that it 
    # refers to isn't freed until the filesystem link is manually removed.
    #
    # Note that this scheme probably doesn't work for fork() + exec() child process
    # creation, because the child isn't guaranteed that the same file descriptor
    # values point to the same entries in the kernel's open file description table.
    # 'spawn' and 
    ################################################################################
    
    w_coord_fd, WCOORD_BUFFER_PATH = tempfile.mkstemp(prefix="voidfinder", dir=RESOURCE_DIR, text=False)
    
    if verbose > 0:
        
        print("Mem-mapping galaxy coordinates", flush=True)
        
        print("WCOORD MEMMAP PATH: ", WCOORD_BUFFER_PATH, w_coord_fd, flush=True)
    
    num_galaxies = w_coord.shape[0]
    
    w_coord_buffer_length = num_galaxies*3*8 # 3 for xyz and 8 for float64
    
    os.ftruncate(w_coord_fd, w_coord_buffer_length)
    
    w_coord_buffer = mmap.mmap(w_coord_fd, w_coord_buffer_length)
    
    w_coord_buffer.write(w_coord.astype(np.float64).tobytes())
    
    del w_coord
    
    w_coord = np.frombuffer(w_coord_buffer, dtype=np.float64)
    
    w_coord.shape = (num_galaxies, 3)
    
    #os.close(w_coord_fd)
    
    os.unlink(WCOORD_BUFFER_PATH)
        
    
    ################################################################################
    # An output counter for total number of holes found, and calculate the
    # total number of cells we're going to have to check based on the grid
    # dimensions and the total number of previous cells in the cell_ID_dict which
    # we already discovered we do NOT have to check.
    #
    #
    # Create the GalaxyMap index and GalaxyMap data array and memmap them for the 
    # workers
    ################################################################################
    
    if verbose > 0:
        
        galaxy_map_start_time = time.time()
        
        print("Building galaxy map", flush=True)
    
    mesh_indices = ((w_coord - coord_min)/search_grid_edge_length).astype(np.int64)
        
    galaxy_map = {}

    for idx in range(mesh_indices.shape[0]):

        bin_ID = tuple(mesh_indices[idx])
        
        if bin_ID not in galaxy_map:
            
            galaxy_map[bin_ID] = []
        
        galaxy_map[bin_ID].append(idx)
        
    del mesh_indices
    
    num_in_galaxy_map = len(galaxy_map)
    
    ################################################################################
    # Convert the galaxy map from a map of grid_cell_ID -> belonging galaxy indices
    # to a map from grid_cell_ID -> (offset, num) into 
    ################################################################################
    
    offset = 0
    
    galaxy_map_list = []
    
    for key in galaxy_map:
        
        indices = np.array(galaxy_map[key], dtype=np.int64)
        
        num_elements = indices.shape[0]
        
        galaxy_map_list.append(indices)
        
        galaxy_map[key] = (offset, num_elements)
        
        offset += num_elements

    galaxy_map_array = np.concatenate(galaxy_map_list)
    
    del galaxy_map_list
    
    
    ################################################################################
    # Convert the galaxy_map dictionary into a custom dictionary that we can
    # use to memmap down to our workers
    ################################################################################
    
    num_galaxy_map_elements = len(galaxy_map)
    
    next_prime = find_next_prime(2*num_galaxy_map_elements)
    
    lookup_memory = np.zeros(next_prime, dtype=[("filled_flag", np.uint8, 1),
                                                   ("i", np.uint16, 1),
                                                   ("j", np.uint16, 1),
                                                   ("k", np.uint16, 1),
                                                   ("offset", np.int64, 1),
                                                   ("num_elements", np.int64, 1)])
    
    new_galaxy_map = GalaxyMapCustomDict(ngrid, lookup_memory)
    
    for curr_ijk in galaxy_map:
        
        offset, num_elements = galaxy_map[curr_ijk]
        
        new_galaxy_map.setitem(*curr_ijk, offset, num_elements)
        
    del galaxy_map
    
    if verbose > 0:
        
        print("Galaxy Map build time: ", time.time() - galaxy_map_start_time, flush=True)
        
        print("Size: ", num_in_galaxy_map, "Total slots: ", next_prime, flush=True)
        
        print("Num collisions in rebuild: ", new_galaxy_map.num_collisions, flush=True)
    
    
    ################################################################################
    # memmap the lookup memory for the galaxy map
    # maybe rename it to the galaxy map hash table
    ################################################################################
    
    lookup_fd, LOOKUPMEM_BUFFER_PATH = tempfile.mkstemp(prefix="voidfinder", dir=RESOURCE_DIR, text=False)
    
    if verbose > 0:
        
        print("Galaxy map lookup memmap: ", LOOKUPMEM_BUFFER_PATH, lookup_fd, flush=True)
    
    lookup_buffer_length = next_prime*23 #23 bytes per element
    
    os.ftruncate(lookup_fd, lookup_buffer_length)
    
    lookup_buffer = mmap.mmap(lookup_fd, lookup_buffer_length)
    
    lookup_buffer.write(lookup_memory.tobytes())
    
    del lookup_memory
    
    os.unlink(LOOKUPMEM_BUFFER_PATH)
    
    
    ################################################################################
    # Memmap the galaxy map array to our worker processes 
    ################################################################################
    
    gma_fd, GMA_BUFFER_PATH = tempfile.mkstemp(prefix="voidfinder", dir=RESOURCE_DIR, text=False)
    
    if verbose > 0:
        
        print("Galaxy map array memmap: ", GMA_BUFFER_PATH, gma_fd, flush=True)
    
    num_gma_indices = galaxy_map_array.shape[0]
    
    gma_buffer_length = num_gma_indices*8 # 8 for int64
    
    os.ftruncate(gma_fd, gma_buffer_length)
    
    gma_buffer = mmap.mmap(gma_fd, gma_buffer_length)
    
    gma_buffer.write(galaxy_map_array.tobytes())
    
    del galaxy_map_array
    
    os.unlink(GMA_BUFFER_PATH)
    
    galaxy_map_array = np.frombuffer(gma_buffer, dtype=np.int64)
    
    galaxy_map_array.shape = (num_gma_indices,)
    
    ################################################################################
    # Calculate the number of cells we need to search
    ################################################################################

    n_empty_cells = ngrid[0]*ngrid[1]*ngrid[2] - num_in_galaxy_map
    

    ################################################################################
    # Setup a file handle for output memory, we're going to memmap in the
    # worker processes to store results and then we'll use numpy.frombuffer
    # to convert it back into an array to pass back up the chain.
    ################################################################################
    
    result_fd, RESULT_BUFFER_PATH = tempfile.mkstemp(prefix="voidfinder", dir=RESOURCE_DIR, text=False)
    
    if verbose > 0:
        
        print("Result array memmap: ", RESULT_BUFFER_PATH, result_fd, flush=True)
    
    #result_fd = os.open(RESULT_BUFFER_PATH, os.O_TRUNC | os.O_CREAT | os.O_RDWR | os.O_CLOEXEC)
    
    #os.unlink(RESULT_BUFFER_PATH)
    
    result_buffer_length = n_empty_cells*4*8
    
    os.ftruncate(result_fd, result_buffer_length)
    
    result_buffer = mmap.mmap(result_fd, 0)
    
    #result_buffer.write(b"0"*result_buffer_length)
    
    os.unlink(RESULT_BUFFER_PATH)
    
    
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
    # Shared memory values to help generating Cell IDs
    ################################################################################
    ijk_start = Value(c_int64, 0, lock=True)
    
    write_start = Value(c_int64, 0, lock=True)
    
    
    
    
    ################################################################################
    # Dump configuration to a file for worker processes to get it
    ################################################################################
    if verbose > 0:
        
        print("Grid: ", ngrid, flush=True)
        
        #print("Dumping config pickle to disk for workers at: ", CONFIG_PATH)
    
    config_object = {"SOCKET_PATH" : SOCKET_PATH,
                     "RESULT_BUFFER_PATH" : RESULT_BUFFER_PATH,
                     "result_fd" : result_fd,
                     "WCOORD_BUFFER_PATH" : WCOORD_BUFFER_PATH,
                     "w_coord_fd" : w_coord_fd,
                     "num_galaxies" : num_galaxies,
                     "GMA_BUFFER_PATH" : GMA_BUFFER_PATH,
                     "gma_fd" : gma_fd,
                     "num_gma_indices" : num_gma_indices,
                     "LOOKUPMEM_BUFFER_PATH" : LOOKUPMEM_BUFFER_PATH,
                     "lookup_fd" : lookup_fd,
                     "next_prime" : next_prime,
                     "CELL_ID_BUFFER_PATH" : CELL_ID_BUFFER_PATH,
                     "PROFILE_BUFFER_PATH" : PROFILE_BUFFER_PATH,
                     #"cell_ID_dict" : cell_ID_dict,
                     #"galaxy_map" : galaxy_map,
                     "num_in_galaxy_map" : num_in_galaxy_map,
                     "ngrid" : ngrid, 
                     "dl" : dl, 
                     "dr" : dr,
                     "coord_min" : coord_min, 
                     "mask" : mask.astype(np.uint8),
                     "mask_resolution" : mask_resolution,
                     "min_dist" : min_dist,
                     "max_dist" : max_dist,
                     #w_coord,
                     "batch_size" : batch_size,
                     "verbose" : verbose,
                     "print_after" : print_after,
                     "num_cpus" : num_cpus,
                     "search_grid_edge_length" : search_grid_edge_length,
                     "DEBUG_DIR" : DEBUG_DIR
                     }
    
    '''
    outfile = open(CONFIG_PATH, 'wb')
    
    pickle.dump(config_object, outfile)
    
    outfile.close()
    '''
    
    ################################################################################
    # Register some functions to be called when the python interpreter exits
    # to clean up any leftover file memory on disk or socket files, etc
    ################################################################################
    
    def cleanup_config():
        
        if os.path.isfile(CONFIG_PATH):
        
            os.remove(CONFIG_PATH)
        
    def cleanup_socket():
        
        if os.path.exists(SOCKET_PATH):
            
            mode = os.stat(SOCKET_PATH).st_mode
        
            is_socket = stat.S_ISSOCK(mode)
            
            if is_socket:
        
                os.remove(SOCKET_PATH)
                
                print("CLEANING UP MAIN SOCKET")
        
    def cleanup_result():
        
        if os.path.isfile(RESULT_BUFFER_PATH):
        
            os.remove(RESULT_BUFFER_PATH)
        
    def cleanup_wcoord():
        
        if os.path.isfile(WCOORD_BUFFER_PATH):
        
            os.remove(WCOORD_BUFFER_PATH)
        
    def cleanup_gma():
        
        if os.path.isfile(GMA_BUFFER_PATH):
        
            os.remove(GMA_BUFFER_PATH)
        
    def cleanup_lookupmem():
        
        if os.path.isfile(LOOKUPMEM_BUFFER_PATH):
        
            os.remove(LOOKUPMEM_BUFFER_PATH)
        
    
    atexit.register(cleanup_config)
    
    atexit.register(cleanup_socket)
    
    atexit.register(cleanup_result)
    
    atexit.register(cleanup_wcoord)
    
    atexit.register(cleanup_gma)
    
    atexit.register(cleanup_lookupmem)
    
    
    ################################################################################
    # Start the worker processes
    #
    # For whatever reason, OSX doesn't define the socket.SOCK_CLOEXEC constants
    # so check for that before opening the listener socket
    ################################################################################
    
    if hasattr(socket, "SOCK_CLOEXEC"):
        
        listener_socket = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM | socket.SOCK_CLOEXEC)
        
    else:
        
        listener_socket = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    
    listener_socket.bind(SOCKET_PATH)
    
    listener_socket.listen(num_cpus)
    
    
    #print("TEST SPOT")
    
    #time.sleep(30.0)
    
    
    startup_context = multiprocessing.get_context("fork")
        
    processes = []
    
    for proc_idx in range(num_cpus):
        
        #p = startup_context.Process(target=_main_hole_finder_startup, args=(proc_idx, CONFIG_PATH))
        p = startup_context.Process(target=_main_hole_finder_worker, 
                                    args=(proc_idx, 
                                          ijk_start, 
                                          write_start, 
                                          config_object))
        #p = startup_context.Process(target=_main_hole_finder_profile, args=(proc_idx, CONFIG_PATH))
        
        p.start()
        
        processes.append(p)
    
    worker_start_time = time.time()
    
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
    
    all_successful_connections = True
    
    listener_socket.settimeout(10.0)
    
    for idx in range(num_cpus):
        
        #print("Waiting for connection: ", idx)
        
        try:
            
            worker_sock, worker_addr = listener_socket.accept()
            
        except:
            
            all_successful_connections = False
            
            break
        
        worker_sockets.append(worker_sock)
        
        num_active_processes += 1
        
        message_buffers.append(b"")
        
        socket_index[worker_sock.fileno()] = idx
        
    if verbose > 0:
        
        if all_successful_connections:
            
            print("Worker processes time to connect: ", time.time() - worker_start_time, flush=True)
    
    
    
    listener_socket.shutdown(socket.SHUT_RDWR)
    
    listener_socket.close()
    
    os.unlink(SOCKET_PATH)
    
    
    def cleanup_worker_sockets():
        
        #print("CLEANING UP WORKER SOCKETS")
        
        for worker_sock in worker_sockets:
            
            worker_sock.close()
            
    atexit.register(cleanup_worker_sockets)
    
    
    if not all_successful_connections:
        
        for worker_sock in worker_sockets:
                
            worker_sock.send(b"exit")
        
        print("FAILED TO CONNECT ALL WORKERS SUCCESSFULLY, EXITING")
            
        exit()
        
        
    
    #cleanup_config()
    
    #os.unlink(CONFIG_PATH)
    
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
            
                print('Processed', num_cells_processed, 'cells of', n_empty_cells, "empty cells", str(round(curr_time-start_time,2)), flush=True)
                
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
        
        print("Main task finish time: ", time.time() - start_time, flush=True)
    
    
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
    
    result_array = np.frombuffer(result_buffer, dtype=np.float64)
    
    result_array.shape = (n_empty_cells, 4)
    
    valid_idx = np.logical_not(np.isnan(result_array[:,0]))
    
    n_holes = np.sum(valid_idx, axis=None, dtype=np.int64)
    
    valid_result_array = result_array[valid_idx,:]
    
    ################################################################################
    # Since we just indexed into the result array with a boolean index (valid_idx)
    # this will force a copy of the data into the new valid_result_array.  Since
    # its a copy and not a view anymore, we can safely close the mmap
    # result_buffer.close() to the original memory, but if for some reason we change
    # this to not copy the memory, we can't call .close() on the memmap or we lose
    # access to the underlying data buffer, and VoidFinder crashes with no
    # traceback for some reason.
    ################################################################################
    result_buffer.close()
        
    return valid_result_array, n_holes
                    
    
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
    """
    Helper used in profiling the worker processes.
    """
    
    cProfile.runctx("_main_hole_finder_worker(worker_idx, ijk_start, write_start, config_path)", globals(), locals(), 'prof%d.prof' %worker_idx)
    

def _main_hole_finder_worker(worker_idx, ijk_start, write_start, config):
    
    
    ################################################################################
    # Unpack the configuration from the master process.
    ################################################################################
    '''
    infile = open(config_path, 'rb')
    
    config = pickle.load(infile)
    
    infile.close()
    '''
    
    SOCKET_PATH = config["SOCKET_PATH"]
    RESULT_BUFFER_PATH = config["RESULT_BUFFER_PATH"]
    result_fd = config["result_fd"]
    WCOORD_BUFFER_PATH = config["WCOORD_BUFFER_PATH"]
    w_coord_fd = config["w_coord_fd"]
    num_galaxies = config["num_galaxies"]
    GMA_BUFFER_PATH = config["GMA_BUFFER_PATH"]
    gma_fd = config["gma_fd"]
    num_gma_indices = config["num_gma_indices"]
    LOOKUPMEM_BUFFER_PATH = config["LOOKUPMEM_BUFFER_PATH"]
    lookup_fd = config["lookup_fd"]
    next_prime = config["next_prime"]
    CELL_ID_BUFFER_PATH = config["CELL_ID_BUFFER_PATH"]
    PROFILE_BUFFER_PATH = config["PROFILE_BUFFER_PATH"]
    #cell_ID_dict = config["cell_ID_dict"]
    #galaxy_map = config["galaxy_map"]
    num_in_galaxy_map = config["num_in_galaxy_map"]
    ngrid = config["ngrid"]
    dl = config["dl"]
    dr = config["dr"]
    coord_min = config["coord_min"]
    mask = config["mask"]
    mask_resolution = config["mask_resolution"]
    min_dist = config["min_dist"]
    max_dist = config["max_dist"]
    #w_coord = config[13]
    batch_size = config["batch_size"]
    verbose = config["verbose"]
    print_after = config["print_after"]
    num_cpus = config["num_cpus"]
    search_grid_edge_length = config["search_grid_edge_length"]
    DEBUG_DIR = config["DEBUG_DIR"]


    #print("WORKER EXITING FOR DEBUG")
    #exit()

    worker_socket = socket.socket(socket.AF_UNIX)
    
    worker_socket.settimeout(10.0)
    
    connect_start = time.time()
    
    try:
        
        worker_socket.connect(SOCKET_PATH)
        
    except Exception as E:
        
        print("WORKER", worker_idx, "UNABLE TO CONNECT, EXITING", flush=True)
        
        raise E
    
    
    #print("Worker", worker_idx, "connect time: ", time.time() - connect_start, flush=True)
    #worker_socket.setblocking(True)
    
    ################################################################################
    # Load up w_coord from shared memory
    ################################################################################
    
    #wcoord_buffer = open(WCOORD_BUFFER_PATH, 'r+b')
    
    wcoord_buffer_length = num_galaxies*3*8 # 3 since xyz and 8 since float64
    
    #wcoord_mmap_buffer = mmap.mmap(wcoord_buffer.fileno(), wcoord_buffer_length)
    wcoord_mmap_buffer = mmap.mmap(w_coord_fd, wcoord_buffer_length)
    
    w_coord = np.frombuffer(wcoord_mmap_buffer, dtype=np.float64)
    
    w_coord.shape = (num_galaxies, 3)
    
    
    
    ################################################################################
    # Load up galaxy_map_array from shared memory
    ################################################################################
    
    #gma_buffer = open(GMA_BUFFER_PATH, 'r+b')
    
    gma_buffer_length = num_gma_indices*8 # 3 since xyz and 8 since float64
    
    #gma_mmap_buffer = mmap.mmap(gma_buffer.fileno(), gma_buffer_length)
    gma_mmap_buffer = mmap.mmap(gma_fd, gma_buffer_length)
    
    galaxy_map_array = np.frombuffer(gma_mmap_buffer, dtype=np.int64)
    
    galaxy_map_array.shape = (num_gma_indices,)
    
    
    
    
    ################################################################################
    #
    # Primary data structure for the lookup of galaxies in cells.  Used to be
    # a scipy KDTree, then switched to sklearn KDTree for better performance, then
    # re-wrote to use a map from elements of the search grid to the galaxies it
    # is closest to, and lastly, re-wrote a custom dict class, I was using the 
    # built-in python dict, but I needed the underlying memory to be exposed so 
    # I could memmap it, so I wrote a new class which I can do that with.
    #
    ################################################################################


    #lookup_buffer = open(LOOKUPMEM_BUFFER_PATH, 'r+b')
    
    lookup_buffer_length = next_prime*23 # 23 bytes per element
    
    #lookup_mmap_buffer = mmap.mmap(lookup_buffer.fileno(), lookup_buffer_length)
    lookup_mmap_buffer = mmap.mmap(lookup_fd, lookup_buffer_length)
    
    lookup_dtype = [("filled_flag", np.uint8, 1),
                    ("i", np.uint16, 1),
                    ("j", np.uint16, 1),
                    ("k", np.uint16, 1),
                    ("offset", np.int64, 1),
                    ("num_elements", np.int64, 1)]

    input_numpy_dtype = np.dtype(lookup_dtype, align=False)
    
    lookup_memory = np.frombuffer(lookup_mmap_buffer, dtype=input_numpy_dtype)
    
    lookup_memory.shape = (next_prime,)

    galaxy_map = GalaxyMapCustomDict(ngrid,
                                     lookup_memory)

    #if verbose:
        
    #    kdtree_start_time = time.time()
        
    #from sklearn import neighbors
    #galaxy_tree = neighbors.KDTree(w_coord)
    
    #from scipy.spatial import KDTree
    #galaxy_tree = KDTree(w_coord)
    
    galaxy_tree = GalaxyMap(w_coord, 
                            coord_min, 
                            search_grid_edge_length,
                            galaxy_map,
                            galaxy_map_array)
    
    
    #if verbose:
        
    #    print('KDTree creation time:', time.time() - kdtree_start_time)
        
        
    #cell_ID_mem = Cell_ID_Memory(len(galaxy_tree.galaxy_map))
    cell_ID_mem = Cell_ID_Memory(10000)
    
    
    ################################################################################
    #
    ################################################################################
    
    
    n_empty_cells = ngrid[0]*ngrid[1]*ngrid[2] - num_in_galaxy_map
    
    #result_buffer = open(RESULT_BUFFER_PATH, 'r+b')
    
    result_buffer_length = n_empty_cells*4*8 #float64 so 8 bytes per element
    
    #result_mmap_buffer = mmap.mmap(result_buffer.fileno(), result_buffer_length)
    result_mmap_buffer = mmap.mmap(result_fd, result_buffer_length)
    
    
    
    
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
    
    
    
    cell_ID_gen = CellIDGenerator(ngrid[0],
                                  ngrid[1],
                                  ngrid[2],
                                  galaxy_map)
    
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
    
    do_work = True
    
    have_result_to_write = False
    
    no_cells_left_to_process = False
    
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
        
        if do_work:
        
            
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
                
                have_result_to_write = True
            
            else:
                no_cells_left_to_process = True
            
            
            #PROFILE_ARRAY[out_start_idx:(out_start_idx+return_array.shape[0]),:] = PROFILE_array
            #seek_location = 12*out_start_idx
            #PROFILE_mmap_buffer.seek(seek_location)
            #PROFILE_mmap_buffer.write(PROFILE_array[0:num_write].tobytes())
            
        if have_result_to_write:   
            
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
            
            
            try:
                worker_socket.send(out_msg)
            except:
                do_work = False
            else:
                do_work = True
                have_result_to_write = False
            
            #PROFILE_main_end = time.time_ns()
            
            #PROFILE_main_times.append((PROFILE_main_end, PROFILE_main_start, return_array.shape[0]))
            
        if no_cells_left_to_process:
            
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
    
    print("WORKER EXITING GRACEFULLY "+str(worker_idx)+"\n", flush=True)
    
    return None




def find_next_prime(threshold_value):
    """
    Given an input integer threshold_value, find the next prime number
    greater than threshold_value.  This is used as a helper in creating
    the memory backing array for the galaxy map, because taking an index
    modulus a prime number is a nice way to hash an integer.
    
    Uses Bertrams(?) theorem that for every n > 1 there is a prime number
    p such that n < p < 2n
    """
    
    
    for check_val in range(threshold_value+1, 2*threshold_value):
        
        if check_val%2 == 0:
            continue
        
        sqrt_check = int(np.sqrt(check_val))+1
        
        at_least_one_divisor = False
        
        for j in range(3, sqrt_check):
            
            if check_val % j == 0:
                
                at_least_one_divisor = True
                
                break
            
        if not at_least_one_divisor:
            
            return check_val





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




if __name__ == "__main__":
    
    print("This module is not intended to be run as a script")
    
    
    
    
    
    
    
    
    
    
    
    