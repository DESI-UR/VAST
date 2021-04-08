



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
import h5py
from psutil import cpu_count


import cProfile

import numpy as np

import time

from .voidfinder_functions import not_in_mask

from ._voidfinder_cython import main_algorithm, \
                                fill_ijk, \
                                fill_ijk_zig_zag


from ._voidfinder_cython_find_next import GalaxyMap, \
                                          Cell_ID_Memory, \
                                          GalaxyMapCustomDict, \
                                          HoleGridCustomDict, \
                                          NeighborMemory, \
                                          MaskChecker



from multiprocessing import Queue, Process, RLock, Value, Array

from ctypes import c_int64, c_double, c_float

from queue import Empty

from copy import deepcopy

import pickle

from astropy.table import Table

from .table_functions import to_array

import matplotlib
#matplotlib.use('TkAgg')
import matplotlib.pyplot as plt






def _hole_finder(hole_grid_shape, 
                 hole_grid_edge_length, 
                 hole_center_iter_dist,
                 galaxy_map_grid_edge_length,
                 coord_min, 
                 galaxy_coords,
                 survey_name,
                 mask_mode=0,
                 mask=None,
                 mask_resolution=None,
                 min_dist=None,
                 max_dist=None,
                 xyz_limits=None,
                 #hole_radial_mask_check_dist,
                 save_after=None,
                 use_start_checkpoint=False,
                 batch_size=1000,
                 verbose=0,
                 print_after=5.0,
                 num_cpus=1):
    
    '''
    Description
    ===========

    See help(voidfinder.find_voids)

    This function is basically a glorified switch between single-threaded mode 
    and multi-processed mode of running VoidFinder.

    
    Parameters
    ==========
    
    hole_grid_shape : array or tuple of length 3
        the number of grid cells in each of the 3 x,y,z dimensions
    
    hole_grid_edge_length : scalar float
        length of each cell in Mpc/h
        
    hole_center_iter_dist : scalar float
        distance to shift hole centers during iterative void hole growing in 
        Mpc/h
        
        
    galaxy_map_grid_edge_length : float or None
        edge length in Mpc/h for the secondary grid for finding nearest neighbor 
        galaxies.  If None, will default to 3*void_grid_edge_length (which 
        results in a cell volume of 3^3 = 27 times larger cube volume).  This 
        parameter yields a tradeoff between number of galaxies in a cell, and 
        number of cells to search when growing a sphere.  Too large and many 
        redundant galaxies may be searched, too small and too many cells will 
        need to be searched.
        (xyz space)
        
    coord_min : numpy.ndarray of shape (1,3) 
        minimum coordinates of the survey in x,y,z in Mpc/h

        Note that this coordinate is used for transforming values into the i,j,k 
        search grid space and also into the p,q,r galaxy map grid space
        
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
        
    survey_name : str
        identifier for the survey running, may be prepended or appended to 
        output filenames including the checkpoint filename
        
    DEPRECATED hole_radial_mask_check_dist : float in (0.0,1.0)
        radial distance to check whether or not a hole overlaps with outside the 
        mask too much
        
    save_after : int or None
        save a VoidFinderCheckpoint.h5 file after *approximately* every 
        save_after cells have been processed.  This will over-write this 
        checkpoint file every save_after cells, NOT append to it.  Also, saving 
        the checkpoint file forces the worker processes to pause and synchronize 
        with the master process to ensure the correct values get written, so 
        choose a good balance between saving too often and not often enough if 
        using this parameter.  Note that it is an approximate value because it 
        depends on the number of worker processes and the provided batch_size 
        value, if you batch size is 10,000 and your save_after is 1,000,000 you 
        might actually get a checkpoint at say 1,030,000.  If None, disables 
        saving the checkpoint file
        
    
    use_start_checkpoint : bool
        Whether to attempt looking for a  VoidFinderCheckpoint.h5 file which can 
        be used to restart the VF run
        If False, VoidFinder will start fresh from 0    

    batch_size : scalar float
        Number of empty cells to pass into each process.  Initialized to 1000.

    verbose : int
        value to determine whether or not to display status messages while 
        running.  0 is off, 1 is print after N updates, 2 is full debugging 
        prints.

    num_cpus : int or None
        number of cpus to use while running the main algorithm.  None will 
        result in using number of physical cores on the machine.  Some speedup 
        benefit may be obtained from using additional logical cores via Intel 
        Hyperthreading but with diminishing returns based on some basic
        spot testing
    
    
    
    Returns
    =======
    
    x_y_z_r_array : numpy.ndarray of shape (N,4)
        x,y,z coordinates of the N hole centers found in units of Mpc/h (cols 
        0,1,2) and Radii of the N holes found in units of Mpc/h

    n_holes : scalar float
        Number of potential holes found - note this number is prior to the void 
        combining stage so will not represent the final output of VoidFinder
    '''


    '''
    #ijk reordertest
    
    batch_size = 126
    
    start_idx = 0

    test_cell_ID_dict = {(4,4,0) : 1,
                         (0,2,4) : 1}
    
    i_j_k_array = np.empty((batch_size, 3), dtype=np.int64)

    test_gen = CellIDGenerator(5,5,5,test_cell_ID_dict)
    
    num_write = test_gen.gen_cell_ID_batch(start_idx, batch_size, i_j_k_array)

    print("num_write: ", num_write)
    
    for idx, row in enumerate(i_j_k_array):
        
        print(idx, row)
    
    
    print("Start idx 10")
    
    num_write = test_gen.gen_cell_ID_batch(10, 10, i_j_k_array)
    
    print(num_write)
    print(i_j_k_array[0:10])
    
    print("Start idx 125")
    
    num_write = test_gen.gen_cell_ID_batch(126, 10, i_j_k_array)
    
    print(num_write)
    
    exit()
    '''

    ############################################################################
    # Run single or multi-processed
    ############################################################################
    
    if isinstance(num_cpus, int) and num_cpus == 1:
        
        #cProfile.runctx("run_single_process_cython(ngrid, dl, dr, coord_min, mask, mask_resolution, min_dist, max_dist, w_coord, batch_size=batch_size, verbose=verbose, print_after=print_after, num_cpus=num_cpus)", globals(), locals(), 'prof_single.prof')
        #x_y_z_r_array = None
        #n_holes = None
        
        
        x_y_z_r_array, n_holes = _hole_finder_single_process(hole_grid_shape, 
                                                             hole_grid_edge_length, 
                                                             hole_center_iter_dist,
                                                             galaxy_map_grid_edge_length,
                                                             coord_min, 
                                                             mask,
                                                             mask_resolution,
                                                             min_dist,
                                                             max_dist,
                                                             galaxy_coords,
                                                             survey_name,
                                                             #hole_radial_mask_check_dist,
                                                             save_after=save_after,
                                                             use_start_checkpoint=use_start_checkpoint,
                                                             batch_size=batch_size,
                                                             verbose=verbose,
                                                             print_after=print_after,
                                                             num_cpus=num_cpus
                                                             )
        
    else:
        
        x_y_z_r_array, n_holes = _hole_finder_multi_process(hole_grid_shape, 
                                                            hole_grid_edge_length, 
                                                            hole_center_iter_dist,
                                                            galaxy_map_grid_edge_length,
                                                            coord_min, 
                                                            galaxy_coords,
                                                            survey_name,
                                                            mask_mode=mask_mode,
                                                            mask=mask,
                                                            mask_resolution=mask_resolution,
                                                            min_dist=min_dist,
                                                            max_dist=max_dist,
                                                            xyz_limits=xyz_limits,
                                                            #hole_radial_mask_check_dist,
                                                            save_after=save_after,
                                                            use_start_checkpoint=use_start_checkpoint,
                                                            batch_size=batch_size,
                                                            verbose=verbose,
                                                            print_after=print_after,
                                                            num_cpus=num_cpus
                                                            )

    return x_y_z_r_array, n_holes


    
    
        
        
        
        
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
        
        
        
        '''
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
        '''
        
        
        
        '''
        num_out = fill_ijk(output_array, 
                           start_idx,
                           batch_size,
                           self.data[0],
                           self.data[1],
                           self.data[2],
                           self.cell_ID_dict
                           )
                           
        '''
        
        
        
        
        num_out = fill_ijk_zig_zag(output_array, 
                                   start_idx,
                                   batch_size,
                                   self.data[0],
                                   self.data[1],
                                   self.data[2],
                                   self.cell_ID_dict
                                   )
        
        return num_out
        
        
        

    
def _hole_finder_single_process(void_grid_shape, 
                                void_grid_edge_length, 
                                hole_center_iter_dist,
                                galaxy_map_grid_edge_length,
                                coord_min, 
                                mask,
                                mask_resolution,
                                min_dist,
                                max_dist,
                                galaxy_coords,
                                survey_name,
                                #hole_radial_mask_check_dist,
                                save_after=None,
                                use_start_checkpoint=False,
                                batch_size=1000,
                                verbose=0,
                                print_after=5.0,
                                num_cpus=None,
                                DEBUG_DIR="/home/moose/VoidFinder/doc/debug_dir"
                                ):
    """
    Run VoidFinder using the cython code, except just in single-process mode.
    """
    
    start_time = time.time()
        
    if verbose > 0:

        print("Running single-process mode", flush=True)
        
        print("Grid: ", void_grid_shape, flush=True)
        
        
        
        
        
        
    ############################################################################
    # First build a helper for the i,j,k generator, using the hole grid edge 
    # length.  We basically need a flag that says "there is a galaxy in this ijk 
    # cell" so VoidFinder can skip that i,j,k value when growing holes
    ############################################################################
    mesh_indices = ((galaxy_coords - coord_min)/void_grid_edge_length).astype(np.int64)
    
    hole_cell_ID_dict = {}
    
    for row in mesh_indices:
        
        hole_cell_ID_dict[tuple(row)] = 1
    
    
    num_nonempty_hole_cells = len(hole_cell_ID_dict)
    
    
    
    hole_next_prime = find_next_prime(2*num_nonempty_hole_cells)
    
    hole_lookup_memory = np.zeros(hole_next_prime, dtype=[("filled_flag", np.uint8, ()), #() indicates scalar, or length 1 shape
                                                          ("i", np.int16, ()),
                                                          ("j", np.int16, ()),
                                                          ("k", np.int16, ())])
    
    
    new_hole_cell_ID_dict = HoleGridCustomDict(void_grid_shape, 
                                               hole_lookup_memory)
    
    for curr_ijk in hole_cell_ID_dict:
        
        new_hole_cell_ID_dict.setitem(*curr_ijk)
        
    del hole_cell_ID_dict
    
    del mesh_indices
    
    if verbose > 0:
        
        print("Num nonempty hole cells: ", num_nonempty_hole_cells, flush=True)
        
        print("Total slots in hole_cell_ID_dict: ", hole_next_prime, flush=True)
        
        print("Num collisions hole_cell_ID_dict: ", 
              new_hole_cell_ID_dict.num_collisions, 
              flush=True)
        
        
        
        
        
        
        
        
        
    ############################################################################
    # Create the GalaxyMap index and GalaxyMap data array 
    ############################################################################
    mesh_indices = ((galaxy_coords - coord_min)/galaxy_map_grid_edge_length).astype(np.int64)
        
    galaxy_map = {}

    for idx in range(mesh_indices.shape[0]):

        bin_ID = tuple(mesh_indices[idx])
        
        if bin_ID not in galaxy_map:
            
            galaxy_map[bin_ID] = []
        
        galaxy_map[bin_ID].append(idx)
        
    del mesh_indices
    
    num_in_galaxy_map = len(galaxy_map)
        
        
    ############################################################################
    # Convert the galaxy map from a map of grid_cell_ID -> belonging galaxy 
    # indices to a map from grid_cell_ID -> (offset, num) into 
    ############################################################################
    
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
        
    ############################################################################
    # Convert the galaxy_map dictionary into a custom dictionary 
    ############################################################################
    
    num_galaxy_map_elements = len(galaxy_map)
    
    next_prime = find_next_prime(2*num_galaxy_map_elements)
    
    lookup_memory = np.zeros(next_prime, dtype=[("filled_flag", np.uint8, ()),
                                                   ("i", np.int16, ()),
                                                   ("j", np.int16, ()),
                                                   ("k", np.int16, ()),
                                                   ("offset", np.int64, ()),
                                                   ("num_elements", np.int64, ())])
    
    new_galaxy_map = GalaxyMapCustomDict(void_grid_shape,
                                         lookup_memory)
    
    for curr_ijk in galaxy_map:
        
        offset, num_elements = galaxy_map[curr_ijk]
        
        new_galaxy_map.setitem(*curr_ijk, offset, num_elements)
        
    del galaxy_map
    
    galaxy_map = new_galaxy_map
    
    
    if verbose > 0:
        print("Rebuilt galaxy map (size", num_in_galaxy_map, 
              "total slots", next_prime,")", 
              flush=True)
        print("Num collisions in rebuild:", new_galaxy_map.num_collisions, 
              flush=True)
        
        
    cell_ID_mem = Cell_ID_Memory(2)
    
    neighbor_mem = NeighborMemory(50)
    
    ############################################################################
    # Right now this object is a glorified data holder
    ############################################################################
    
    galaxy_tree = GalaxyMap(galaxy_coords, 
                            coord_min, 
                            galaxy_map_grid_edge_length,
                            galaxy_map,
                            galaxy_map_array)
    
    
    
    ############################################################################
    # Create the Cell ID generator
    ############################################################################
    
    start_idx = 0
    
    out_start_idx = 0
    
    cell_ID_gen = CellIDGenerator(void_grid_shape[0], 
                                  void_grid_shape[1], 
                                  void_grid_shape[2], 
                                  new_hole_cell_ID_dict)
    
    if verbose > 1:
        
        print("Len galaxy map (eliminated cells):", num_in_galaxy_map, flush=True)
    
    ############################################################################
    # Convert the mask to an array of uint8 values for running in the cython 
    # code
    ############################################################################
    
    mask = mask.astype(np.uint8)
    
    ############################################################################
    # Allocate memory for output/results
    ############################################################################
    
    n_empty_cells = void_grid_shape[0]*void_grid_shape[1]*void_grid_shape[2] \
                    - num_nonempty_hole_cells
    
    RETURN_ARRAY = np.empty((n_empty_cells, 4), dtype=np.float64)
    
    RETURN_ARRAY.fill(np.NAN)
    
    return_array = np.empty((batch_size, 4), dtype=np.float64)
    
    ############################################################################
    # memory for a batch of cells to work on
    ############################################################################
    i_j_k_array = np.empty((batch_size, 3), dtype=np.int64)
    
    ############################################################################
    #
    # PROFILING VARIABLES
    #
    # PROFILE ARRAY elements are:
    # 0 - total cell time
    # 1 - cell exit stage
    # 2 - kdtree_time
    ############################################################################
    
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
    
    ############################################################################
    # Set up print timer
    ############################################################################
        
    print_start_time = time.time()
    
    main_task_start_time = time.time()
        
    
    ############################################################################
    # Mainloop
    ############################################################################
    
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
            
        curr_time = time.time()
        
        if (curr_time - print_start_time) > print_after:
            
            print("Processed cell", num_cells_processed, 
                  "of", n_empty_cells, 
                  round(curr_time - main_task_start_time, 2))
        
            print_start_time = curr_time
        
        ########################################################################
        # Generate the next batch and run the main algorithm
        ########################################################################
        
        num_write = cell_ID_gen.gen_cell_ID_batch(start_idx, 
                                                  batch_size, 
                                                  i_j_k_array)
        
        start_idx += batch_size
        
        num_cells_to_process = num_write
        
        if num_cells_to_process > 0:

            if return_array.shape[0] != num_cells_to_process:

                return_array = np.empty((num_cells_to_process, 4), 
                                        dtype=np.float64)
                
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
                           #hole_radial_mask_check_dist,
                           return_array,
                           cell_ID_mem,
                           neighbor_mem,
                           0,  
                           #PROFILE_array
                           )
        
            RETURN_ARRAY[out_start_idx:(out_start_idx+num_write),:] = return_array[0:num_write]
            
            #PROFILE_ARRAY[out_start_idx:(out_start_idx+num_write),:] = PROFILE_array[0:num_write]
            
            num_cells_processed += num_write
            
            out_start_idx += num_write
        
        elif num_cells_to_process == 0 and num_cells_processed == n_empty_cells:
        
            exit_condition = True
            
            
    print("Total time growing holes:", time.time() - main_task_start_time)
        
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



    
    
    
    
    


def _hole_finder_multi_process(ngrid, 
                               dl, 
                               dr,
                               search_grid_edge_length,
                               coord_min, 
                               w_coord,
                               survey_name,
                               mask_mode=0,
                               mask=None,
                               mask_resolution=None,
                               min_dist=None,
                               max_dist=None,
                               xyz_limits=None,
                               #hole_radial_mask_check_dist,
                               batch_size=1000,
                               verbose=0,
                               print_after=10000,
                               
                               save_after=None,
                               use_start_checkpoint=False,
                               
                               num_cpus=None,
                               CONFIG_PATH="/tmp/voidfinder_config.pickle",
                               SOCKET_PATH="/tmp/voidfinder.sock",
                               #RESULT_BUFFER_PATH="/tmp/voidfinder_result_buffer.dat",
                               #CELL_ID_BUFFER_PATH="/tmp/voidfinder_cell_ID_gen.dat",
                               #PROFILE_BUFFER_PATH="/tmp/voidfinder_profile_buffer.dat",
                               RESOURCE_DIR="/dev/shm",
                               DEBUG_DIR="/home/moose/VoidFinder/doc/debug_dir"
                               ):
    """
    Description
    ===========
    
    Work-horse method for running VoidFinder with the Cython code in parallel
    multi-process form.  
    
    This method contains the logic for:
    
    1). Sanity check the num_cpus to use
    2). Open file handles and allocate memory for workers to memmap to
    3). Build a few data structures for the workers to share
    4). Register some cleanup helpers with the python interpreters for 
        making sure the disk space gets reclaimed when we are done
    5). Start the workers
    6). Make sure workers connect to the comm socket
    7). Checkpoint the progress if those parameters are enabled
    8). Collect progress results from the workers
    
    This function is designed to be run on Linux on an SMP (Symmetric 
    Multi-Processing) architecture.  It takes advantage of 2 Linux-specific 
    properties: the /dev/shm filesystem and the fork() method of spawning 
    processes. /dev/shm is used as the preferred location for creating memory 
    maps to share information between the worker processes since on Linux is is 
    a RAMdisk, and the implementation of fork() on Linux is used to share file 
    descriptor values between the master and worker processes, whereas on 
    mac/OSX fork() is wonky and Windows does not offer fork() at all.  This has 
    run successfully on mac/OSX, in which case the /tmp directory is used for 
    the memory maps and such, but the fork() on OSX as far as this author 
    understands is not 100% reliable, as the engineers at Apple seem to have 
    certain cases which enforce a fork()-then-exec() paradigm, and others which 
    do not.  Use at your own risk.  However, the single-threaded version of 
    VoidFinder should have no trouble running on Linux, Windows, or OSx.
    
    Parameters
    ==========
    
    FILL IN LATER
    
    Returns
    =======
    
    valid_result_array : numpy.ndarray shape (?,4)
        x,y,z and radius values for all holes which were found
    
    n_holes : int
        number of valid holes found
    
    
    """
    
    
    #print("_hole_finder_mult mask_mode: ", mask_mode, xyz_limits)
    
    if mask_mode == 0:
        if mask is None or \
           mask_resolution is None or \
           min_dist is None or \
           max_dist is None:
            raise ValueError("Mask mode is 0 (ra-dec-z) but a required mask parameter is None")
    
    if mask_mode == 1 and xyz_limits is None:
        raise ValueError("Mask mode is 1 (xyz) but required mask parameter xyz_limits is None")
       
    
    if mask_mode == 0:
        
        mask = mask.astype(np.uint8)
    
    
    ############################################################################
    # If /dev/shm is not available, use /tmp as the shared resource filesystem
    # location instead.  Since on Linux /dev/shm is guaranteed to be a mounted
    # RAMdisk, I do not know if /tmp will be as fast or not, probably depends on
    # kernel settings.
    #---------------------------------------------------------------------------
    if not os.path.isdir(RESOURCE_DIR):
        
        print("WARNING: RESOURCE DIR", RESOURCE_DIR, 
              "does not exist.  Falling back to /tmp but could be slow", 
              flush=True)
        
        RESOURCE_DIR = "/tmp"
    ############################################################################
        
    

    ############################################################################
    # Start by converting the num_cpus argument into the real value we will use
    # by making sure its reasonable, or if it was none use the max val available
    #
    # Maybe should use psutil.cpu_count(logical=False) instead of the
    # multiprocessing version?
    #---------------------------------------------------------------------------
    if (num_cpus is None):
          
        num_cpus = cpu_count(logical=False)
        
    if verbose > 0:
        
        print("Running multi-process mode,", str(num_cpus), "cpus", flush=True)
        
        print("Grid: ", ngrid, flush=True)
    ############################################################################

        
        
    ############################################################################
    # Set up so that VoidFinder will periodically save a checkpoint file so that 
    # it can be restarted from the middle of a run, this saves a checkpoint file
    # after every 'save_after' cells have been processed
    #---------------------------------------------------------------------------
    ENABLE_SAVE_MODE = False
    
    if save_after is not None:
        
        print("ENABLED SAVE MODE", flush=True)
        
        #save every 'save_after' cells have been processed
    
        ENABLE_SAVE_MODE = True
        
        save_after_counter = save_after
        
        sent_syncs = False
        
        num_acknowledges = 0
    ############################################################################

    
    
    ############################################################################
    # Set up so that VoidFinder can be restarted from an on-disk checkpoint
    # file, if the use_start_checkpoint parameter has been enabled
    #---------------------------------------------------------------------------
    START_FROM_CHECKPOINT = False
    
    if use_start_checkpoint == True:
        
        if os.path.isfile(survey_name+"VoidFinderCheckpoint.h5"):
            
            print("Starting from checkpoint: ", 
                  survey_name+"VoidFinderCheckpoint.h5", 
                  flush=True)
        
            start_checkpoint_infile = h5py.File(survey_name+"VoidFinderCheckpoint.h5", 'r')
        
            START_FROM_CHECKPOINT = True
            
        else:
            raise ValueError("Since use_start_checkpoint was True, expected to find"+survey_name+"VoidFinderCheckpoint.h5 file to use as starting checkpoint file, file was not found")
    ############################################################################


    
    ############################################################################
    # First build a helper for the i,j,k generator, using the hole grid edge 
    # length.  We basically need a flag that says "there is a galaxy in this ijk 
    # cell" so VoidFinder can skip that i,j,k value when growing holes
    #---------------------------------------------------------------------------
    mesh_indices = ((w_coord - coord_min)/dl).astype(np.int64)
    
    hole_cell_ID_dict = {}
    
    for row in mesh_indices:
        
        hole_cell_ID_dict[tuple(row)] = 1
    
    num_nonempty_hole_cells = len(hole_cell_ID_dict)
    ############################################################################

    
    
    ############################################################################
    # Now convert this ijk helper into the cython class so we can share its
    # memory array among the processes to not duplicate memory, also the custom
    # hash function it uses is faster than the built-in python one since we are
    # taking advantage of the sequential nature of grid cells
    #---------------------------------------------------------------------------
    hole_next_prime = find_next_prime(2*num_nonempty_hole_cells)
    
    hole_lookup_memory = np.zeros(hole_next_prime, dtype=[("filled_flag", np.uint8, ()), #() indicates scalar, or length 1 shape
                                                          ("i", np.int16, ()),
                                                          ("j", np.int16, ()),
                                                          ("k", np.int16, ())])
    
    new_hole_cell_ID_dict = HoleGridCustomDict(ngrid, 
                                               hole_lookup_memory)
    
    for curr_ijk in hole_cell_ID_dict:
        
        new_hole_cell_ID_dict.setitem(*curr_ijk)
        
    del hole_cell_ID_dict
    
    del mesh_indices
    
    if verbose > 0:
        
        print("Number of nonempty hole cells: ", num_nonempty_hole_cells, 
              flush=True)
        
        print("Total slots in hole_cell_ID_dict: ", hole_next_prime, flush=True)
        
        print("Num collisions hole_cell_ID_dict: ", 
              new_hole_cell_ID_dict.num_collisions, 
              flush=True)
    ############################################################################

    
    
    ############################################################################
    # Next create the GalaxyMap p-q-r-space index, which is constructed 
    # identically to the hole_grid i-j-k-space, except that we use a larger cell 
    # edge length so we get more galaxies per cell
    #---------------------------------------------------------------------------
    if verbose > 0:
        
        galaxy_map_start_time = time.time()
        
        print("Building galaxy map", flush=True)
    
    mesh_indices = ((w_coord - coord_min)/search_grid_edge_length).astype(np.int64)
        
    galaxy_map = {}

    for idx in range(mesh_indices.shape[0]):

        bin_ID_pqr = tuple(mesh_indices[idx])
        
        if bin_ID_pqr not in galaxy_map:
            
            galaxy_map[bin_ID_pqr] = []
        
        galaxy_map[bin_ID_pqr].append(idx)
        
    del mesh_indices
    
    num_in_galaxy_map = len(galaxy_map)
    ############################################################################

    
    
    ############################################################################
    # When we use the GalaxyMap, we are going to need to get all the galaxies in 
    # the survey that belong to a desired p-q-r cell.  So, the GalaxyMap itself 
    # is going to store key-value pairs where the key is the p-q-r cell ID, and 
    # the value is an (offset, num_elements) pair.  The offset represents the 
    # index into the galaxy_map_array, which in turn holds the indices of the 
    # galaxies belonging to the p-q-r cell that we are interested in.  The 
    # num_elements tells us how many elements at the offset in the 
    # galaxy_map_array belong to this p-q-r cell.  The values in 
    # galaxy_map_array tell us the rows in the main galaxy_coords array where 
    # the galaxies in our p-q-r cell of interest are.
    #---------------------------------------------------------------------------
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
    
    num_galaxy_map_elements = len(galaxy_map)
    ############################################################################

    
    
    ############################################################################
    # Now convert the galaxy_map python dict created above into a custom 
    # dictionary type which exposes the backing hash-table array, so we can 
    # mem-map that array and share it among our worker processes.
    #---------------------------------------------------------------------------
    next_prime = find_next_prime(2*num_galaxy_map_elements)
    
    lookup_memory = np.zeros(next_prime, dtype=[("filled_flag", np.uint8, ()), #() indicates scalar, or length 1 shape
                                                ("p", np.int16, ()),
                                                ("q", np.int16, ()),
                                                ("r", np.int16, ()),
                                                ("offset", np.int64, ()),
                                                ("num_elements", np.int64, ())])
    
    new_galaxy_map = GalaxyMapCustomDict(ngrid, 
                                         lookup_memory)
    
    for curr_pqr in galaxy_map:
        
        offset, num_elements = galaxy_map[curr_pqr]
        
        new_galaxy_map.setitem(*curr_pqr, offset, num_elements)
        
    del galaxy_map
    
    if verbose > 0:
        
        print("Galaxy Map build time:", time.time() - galaxy_map_start_time, 
              flush=True)
        
        print("Num items in Galaxy Map:", num_in_galaxy_map, flush=True)
        
        print("Total slots in galaxy map hash table:", next_prime, flush=True)
        
        print("Num collisions in rebuild:", new_galaxy_map.num_collisions, 
              flush=True)
    ############################################################################

    
    
    ############################################################################
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
    # descriptor instead of the file path, we can immediately os.unlink() the 
    # path, so if VoidFinder crashes, the kernel reference count for that memory 
    # mapping will drop to 0 and it will be able to free that memory 
    # automatically.  If we left the link (the path) on the filesystem and VF 
    # crashed, the RAM that it refers to is not freed until the filesystem link 
    # is manually removed.
    #
    # I am assuming that this scheme does not work for fork() + exec() child 
    # process creation, but I really should open my textbook and double check to 
    # be sure.  If I remember correctly a lot of file descriptors "close on 
    # exec()" so maybe it would work, maybe it would not - either way in 
    # practice this means that I do not know if the python multiprocessing 
    # "spawn" or "forkserver" methods would work correctly on this code, and if 
    # we needed to use them it might require re-engineering this code.
    #---------------------------------------------------------------------------
    w_coord_fd, WCOORD_BUFFER_PATH = tempfile.mkstemp(prefix="voidfinder", 
                                                      dir=RESOURCE_DIR, 
                                                      text=False)
    
    if verbose > 0:
        
        print("Mem-mapping galaxy coordinates", flush=True)
        
        print("WCOORD MEMMAP PATH:", WCOORD_BUFFER_PATH, w_coord_fd, flush=True)
    
    num_galaxies = w_coord.shape[0]
    
    w_coord_buffer_length = num_galaxies*3*8 # 3 for xyz and 8 for float64
    
    os.ftruncate(w_coord_fd, w_coord_buffer_length)
    
    w_coord_buffer = mmap.mmap(w_coord_fd, w_coord_buffer_length)
    
    w_coord_buffer.write(w_coord.astype(np.float64).tobytes())
    
    del w_coord
    
    w_coord = np.frombuffer(w_coord_buffer, dtype=np.float64)
    
    w_coord.shape = (num_galaxies, 3)
    
    os.unlink(WCOORD_BUFFER_PATH)
    ############################################################################

    
    
    ############################################################################
    # memmap the lookup memory for the galaxy map
    # maybe rename it to the galaxy map hash table
    #---------------------------------------------------------------------------
    lookup_fd, LOOKUPMEM_BUFFER_PATH = tempfile.mkstemp(prefix="voidfinder", 
                                                        dir=RESOURCE_DIR, 
                                                        text=False)
    
    if verbose > 0:
        
        print("Galaxy map lookup memmap:", LOOKUPMEM_BUFFER_PATH, lookup_fd, 
              flush=True)
    
    lookup_buffer_length = next_prime*23 #23 bytes per element
    
    os.ftruncate(lookup_fd, lookup_buffer_length)
    
    lookup_buffer = mmap.mmap(lookup_fd, lookup_buffer_length)
    
    lookup_buffer.write(lookup_memory.tobytes())
    
    del lookup_memory
    
    os.unlink(LOOKUPMEM_BUFFER_PATH)
    ############################################################################

    
    
    ############################################################################
    # memmap the lookup memory for the hole_cell_ID_dict
    # maybe rename it to the hole_cell_ID hash table
    #---------------------------------------------------------------------------
    hole_lookup_fd, HOLE_LOOKUPMEM_BUFFER_PATH = tempfile.mkstemp(prefix="voidfinder", 
                                                                  dir=RESOURCE_DIR, 
                                                                  text=False)
    
    if verbose > 0:
        
        print("Hole cell lookup memmap:", 
              HOLE_LOOKUPMEM_BUFFER_PATH, 
              hole_lookup_fd, 
              flush=True)
    
    hole_lookup_buffer_length = hole_next_prime*7 #7 bytes per element
    
    os.ftruncate(hole_lookup_fd, hole_lookup_buffer_length)
    
    hole_lookup_buffer = mmap.mmap(hole_lookup_fd, hole_lookup_buffer_length)
    
    hole_lookup_buffer.write(hole_lookup_memory.tobytes())
    
    del hole_lookup_memory
    
    os.unlink(HOLE_LOOKUPMEM_BUFFER_PATH)
    ############################################################################

    
    
    ############################################################################
    # Memmap the galaxy map array to our worker processes 
    #---------------------------------------------------------------------------
    gma_fd, GMA_BUFFER_PATH = tempfile.mkstemp(prefix="voidfinder", 
                                               dir=RESOURCE_DIR, 
                                               text=False)
    
    if verbose > 0:
        
        print("Galaxy map array memmap:", GMA_BUFFER_PATH, gma_fd, flush=True)
    
    num_gma_indices = galaxy_map_array.shape[0]
    
    gma_buffer_length = num_gma_indices*8 # 8 for int64
    
    os.ftruncate(gma_fd, gma_buffer_length)
    
    gma_buffer = mmap.mmap(gma_fd, gma_buffer_length)
    
    gma_buffer.write(galaxy_map_array.tobytes())
    
    del galaxy_map_array
    
    os.unlink(GMA_BUFFER_PATH)
    
    galaxy_map_array = np.frombuffer(gma_buffer, dtype=np.int64)
    
    galaxy_map_array.shape = (num_gma_indices,)
    ############################################################################

    
    
    ############################################################################
    # Calculate the number of cells we need to search
    #---------------------------------------------------------------------------
    n_empty_cells = ngrid[0]*ngrid[1]*ngrid[2] - num_nonempty_hole_cells
    ############################################################################
    
    

    ############################################################################
    # Setup a mem-map for output memory, we are going to memmap in the worker 
    # processes to store results and then we will use numpy.frombuffer to 
    # convert it back into an array to pass back up the chain.
    #---------------------------------------------------------------------------
    result_fd, RESULT_BUFFER_PATH = tempfile.mkstemp(prefix="voidfinder", 
                                                     dir=RESOURCE_DIR, 
                                                     text=False)
    
    if verbose > 0:
        
        print("Result array memmap:", RESULT_BUFFER_PATH, result_fd, flush=True)
    
    result_buffer_length = n_empty_cells*4*8
    
    if verbose > 0:
        
        print("RESULT BUFFER LENGTH (bytes):", result_buffer_length, flush=True)
    
    os.ftruncate(result_fd, result_buffer_length)
    
    result_buffer = mmap.mmap(result_fd, 0)
    
    #result_buffer.write(b"0"*result_buffer_length)
    
    os.unlink(RESULT_BUFFER_PATH)
    ############################################################################

    
    
    ############################################################################
    # If we are starting from a voidfinder checkpoint file, we need to write
    # the already calculated results back into the result buffer before we begin
    #---------------------------------------------------------------------------
    if START_FROM_CHECKPOINT:
        
        starting_result_data = start_checkpoint_infile["result_array"][()]
        
        result_buffer.seek(0)
        
        result_buffer.write(starting_result_data.tobytes())
        
        del starting_result_data
    ############################################################################
    

    
    ############################################################################
    # Memory for PROFILING
    #---------------------------------------------------------------------------
    '''
    PROFILE_fd, PROFILE_BUFFER_PATH = tempfile.mkstemp(prefix="voidfinder", dir=RESOURCE_DIR, text=False)
    
    #PROFILE_fd = os.open(PROFILE_BUFFER_PATH, os.O_TRUNC | os.O_CREAT | os.O_RDWR | os.O_CLOEXEC)
    
    PROFILE_buffer_length = 85000000*3*4 #float32 so 4 bytes per element
    
    os.ftruncate(PROFILE_fd, PROFILE_buffer_length)
    
    PROFILE_buffer = open(PROFILE_fd, 'w+b')
    
    #PROFILE_buffer_length = 85000000*3*4 #float32 so 4 bytes per element
    
    #PROFILE_buffer.write(b"0"*PROFILE_buffer_length)
    '''
    ############################################################################


    
    ############################################################################
    # We need 2 lock-protected values in order to synchronize our worker 
    # processes correctly - the ijk index to start a batch of calculations at, 
    # and the index into the result memory to write your block of results at.
    #
    # ijk_start is a simple integer which represents a sequential index into a 
    # hole_grid location, for example:
    #     0 -> (0,0,0), 1 -> (0,0,1), 2 -> (0,0,2)
    # The CellIDGenerator will use this value to generate a batch of cell IDs, 
    # and by synchronizing access to this value we can ensure 2 workers never 
    # work on the same cell ID
    #
    # write_start is also an integer which represents the next row index into 
    # the result buffer at which new results can be written.  Workers acquire 
    # access to this value after processing a batch of results, and update it 
    # with however many rows they have to write, then release it, effectively 
    # guaranteeing their own access to a block of rows in the result buffer.
    #
    # If we are starting from a VoidFinder checkpoint file, we have to update 
    # these 2 values to wherever we left off.  We can also close the checkpoint 
    # infile at this point.
    #---------------------------------------------------------------------------
    if START_FROM_CHECKPOINT:
        
        next_cell_idx = start_checkpoint_infile.attrs["next_cell_idx"]
        
        num_written_rows = start_checkpoint_infile.attrs["num_written_rows"]
        
        if verbose > 0:
            
            print("Closing:", start_checkpoint_infile, flush=True)
        
        start_checkpoint_infile.close()
        
        ijk_start = Value(c_int64, next_cell_idx, lock=True)
        
        write_start = Value(c_int64, num_written_rows, lock=True)
        
        num_cells_processed = num_written_rows
        
        print("Starting from cell index:", next_cell_idx, flush=True)
        
    else:
    
        ijk_start = Value(c_int64, 0, lock=True)
        
        write_start = Value(c_int64, 0, lock=True)
        
        num_cells_processed = 0
    ############################################################################

    
    
    ############################################################################
    # In previous versions of voidfinder, we were dumping this configuration 
    # object to disk, and re-reading it in from the worker processes.  Since we 
    # are using fork(), there is really no need to do that, but all the data 
    # which the worker processes need are packed up into this config_object 
    # below to make the process of starting the worker processes a lot simpler.
    #---------------------------------------------------------------------------
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
                     "HOLE_LOOKUPMEM_BUFFER_PATH" : HOLE_LOOKUPMEM_BUFFER_PATH,
                     "hole_lookup_fd" : hole_lookup_fd,
                     "hole_next_prime" : hole_next_prime,
                     "num_nonempty_hole_cells" : num_nonempty_hole_cells,
                     #"hole_radial_mask_check_dist" : hole_radial_mask_check_dist,
                     #"CELL_ID_BUFFER_PATH" : CELL_ID_BUFFER_PATH,
                     #"PROFILE_BUFFER_PATH" : PROFILE_BUFFER_PATH,
                     #"cell_ID_dict" : cell_ID_dict,
                     #"galaxy_map" : galaxy_map,
                     "num_in_galaxy_map" : num_in_galaxy_map,
                     "ngrid" : ngrid, 
                     "dl" : dl, 
                     "dr" : dr,
                     "coord_min" : coord_min, 
                     "mask_mode" : mask_mode,
                     "xyz_limits" : xyz_limits,
                     "mask" : mask,
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
    ############################################################################

    
    
    ############################################################################
    # Register some functions to be called when the python interpreter exits to 
    # clean up any leftover file memory on disk or socket files, etc
    #
    # Note - needs an additional check, but I believe most of these atexit 
    # functions are now obsolete, since we are using os.unlink() on all the 
    # filesystem paths correctly above.  Also worth noting that these functions 
    # do not get called on a SIGKILL, which kinda defeats their purpose anyway.
    #---------------------------------------------------------------------------
    def cleanup_config():
        
        if os.path.isfile(CONFIG_PATH):
        
            os.remove(CONFIG_PATH)
        
    def cleanup_socket():
        
        if os.path.exists(SOCKET_PATH):
            
            mode = os.stat(SOCKET_PATH).st_mode
        
            is_socket = stat.S_ISSOCK(mode)
            
            if is_socket:
        
                os.remove(SOCKET_PATH)
        
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
    ############################################################################

    
    
    ############################################################################
    # Start the worker processes
    #
    # For whatever reason, OSX does not define the socket.SOCK_CLOEXEC constants
    # so check for that attribute on the socket module before opening the 
    # listener socket.  Not super critical, but the child processes do not need 
    # a file descriptor for the listener socket so I was trying to be clean and 
    # have it "close on exec"
    #---------------------------------------------------------------------------
    if hasattr(socket, "SOCK_CLOEXEC"):
        
        listener_socket = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM | socket.SOCK_CLOEXEC)
        
    else:
        
        listener_socket = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    
    listener_socket.bind(SOCKET_PATH)
    
    listener_socket.listen(num_cpus)
    
    startup_context = multiprocessing.get_context("fork")
        
    processes = []
    
    for proc_idx in range(num_cpus):
        
        '''
        p = startup_context.Process(target=_main_hole_finder_startup, 
                                     args=(proc_idx, CONFIG_PATH))
        '''
        p = startup_context.Process(target=_hole_finder_worker, 
                                    args=(proc_idx, 
                                          ijk_start, 
                                          write_start, 
                                          config_object))
        
        '''
        p = startup_context.Process(target=_hole_finder_worker_profile, 
                                    args=(proc_idx, 
                                          ijk_start, 
                                          write_start, 
                                          config_object))
        
        '''
        p.start()
        
        processes.append(p)
    
    worker_start_time = time.time()
    ############################################################################


    
    ############################################################################
    # Make sure each worker process connects to the main socket, so we block on
    # the accept() call below until we get a connection, and make sure we get 
    # exactly num_cpus connections.
    #
    # To avoid waiting for hours and hours without getting a successful socket 
    # connection, we set the timeout to the reasonably high value of 10.0 
    # seconds (remember, 0.1 seconds is on the order of 100 million cycles for a 
    # 1GHz processor), and if we do not get a connection within that time frame, 
    # we are going to intentionally raise a RunTimeError.
    #
    # If successful, we save off references to our new worker sockets by their
    # file descriptor integer value so we can refer to them by that value using
    # select() later, then shut down and close our listener/server socket since
    # we are done with it.
    #---------------------------------------------------------------------------
    if verbose > 0:
        
        print("Attempting to connect workers", flush=True)
    
    num_active_processes = 0
    
    worker_sockets = []
    
    message_buffers = []
    
    socket_index = {}
    
    all_successful_connections = True
    
    listener_socket.settimeout(10.0)
    
    for idx in range(num_cpus):
        
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
            
            print("Worker processes time to connect:", 
                  time.time() - worker_start_time, 
                  flush=True)
    
    # This try-except clause was added for weird behavior on mac/OSX
    try:
        listener_socket.shutdown(socket.SHUT_RDWR)
    except:
        pass
    
    listener_socket.close()
    
    os.unlink(SOCKET_PATH)
    
    
    def cleanup_worker_sockets():
        
        #print("CLEANING UP WORKER SOCKETS", flush=True)
        
        for worker_sock in worker_sockets:
            
            worker_sock.close()
            
    atexit.register(cleanup_worker_sockets)
    
    
    if not all_successful_connections:
        
        for worker_sock in worker_sockets:
                
            worker_sock.send(b"exit")
        
        print("FAILED TO CONNECT ALL WORKERS SUCCESSFULLY, EXITING", flush=True)
            
        raise RunTimeError("Worker sockets failed to connect properly")
    ############################################################################

        
    
    ############################################################################
    # PROFILING VARIABLES
    #---------------------------------------------------------------------------
    '''
    PROFILE_process_start_time = time.time()
    
    PROFILE_sample_times = []
    
    PROFILE_samples = []
    
    PROFILE_start_time = time.time()
    
    PROFILE_sample_time = 5.0
    '''
    ############################################################################

    
    
    ############################################################################
    # LOOP TO LISTEN FOR RESULTS WHILE WORKERS WORKING
    # 
    # This loop has 3 primary jobs 
    # 1) Accumulate results from reading the worker sockets
    # 2) Periodically print the status/results from the workers
    # 3) Save checkpoint files after every 'safe_after' results
    #---------------------------------------------------------------------------
    print_after_time = time.time()
    
    main_task_start_time = time.time()
    
    empty1 = []
    
    empty2 = []
    
    select_timeout = 2.0
    
    sent_exit_commands = False
    
    while num_active_processes > 0:
        
        #-----------------------------------------------------------------------
        # DEBUGGING CODE
        #-----------------------------------------------------------------------
        '''
        if num_cells_processed >= 80000000:
        
            print("Breaking debug loop", 
                  num_cells_processed, 
                  num_active_processes, 
                  flush=True)
            
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
                
                print('Processed', num_cells_processed, 
                      'cells of', n_empty_cells, 
                      str(round(curr_sample_time,2)), 
                      flush=True)
                
                if len(PROFILE_samples) > 3:
                    
                    cells_per_sec = (PROFILE_samples[-1] - PROFILE_samples[-2])/curr_sample_interval
                    
                    print(str(round(cells_per_sec,2)), "cells per sec", flush=True)
        '''
        #-----------------------------------------------------------------------
        

        #-----------------------------------------------------------------------
        # Print status updates if verbose is on
        #-----------------------------------------------------------------------
        curr_time = time.time()
        
        if (curr_time - print_after_time) > print_after:
        
            print('Processed', num_cells_processed, 
                  'cells of', n_empty_cells, "empty cells", 
                  str(round(curr_time - main_task_start_time, 2)), 
                  flush=True)
            
            print_after_time = curr_time
        #-----------------------------------------------------------------------
        
            
        #-----------------------------------------------------------------------
        # Accumulate status updates from the worker sockets
        #-----------------------------------------------------------------------
        read_socks, empty3, empty4 = select.select(worker_sockets, 
                                                   empty1, 
                                                   empty2, 
                                                   select_timeout)
        
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
                        
                        if ENABLE_SAVE_MODE:
                            save_after_counter -= num_result
                        
                        #n_holes += num_hole
                        
                    elif message_type == 1:
                        
                        num_active_processes -= 1
                        
                    elif message_type == 2:
                        
                        num_acknowledges += 1
        #-----------------------------------------------------------------------

                        
        #-----------------------------------------------------------------------
        # Save checkpoint if that has been enabled
        #-----------------------------------------------------------------------
        if ENABLE_SAVE_MODE and save_after_counter <= 0:
            
            if not sent_syncs:
            
                for worker_sock in worker_sockets:
                    
                    worker_sock.send(b"sync")
                    
                sent_syncs = True
                
                
            if num_acknowledges == num_cpus:
                
                #---------------------------------------------------------------
                # We do not need synchronized/locked access to ijk_start and 
                # write_start since we sync-paused all the workers and we have 
                # received an acknowledgement from all workers
                #---------------------------------------------------------------
                next_cell_idx = ijk_start.value
    
                num_written_rows = write_start.value

                print("Saving checkpoint:", 
                      num_cells_processed, 
                      next_cell_idx, 
                      num_written_rows, 
                      flush=True)
                
                temp_result_array = np.frombuffer(result_buffer, 
                                                  dtype=np.float64)
    
                temp_result_array.shape = (n_empty_cells, 4)
                
                temp_result = temp_result_array[0:num_written_rows,:]
                
                if os.path.isfile(survey_name+"VoidFinderCheckpoint.h5"):
                    
                    os.remove(survey_name+"VoidFinderCheckpoint.h5")
                    
                outfile = h5py.File(survey_name+"VoidFinderCheckpoint.h5", 'w')
                
                outfile.attrs["num_written_rows"] = num_written_rows
                
                outfile.attrs["next_cell_idx"] = next_cell_idx
                
                outfile.create_dataset("result_array", data=temp_result)
                
                outfile.close()
                
                print("Saved checkpoint", survey_name+"VoidFinderCheckpoint.h5", 
                      flush=True)
                
                del next_cell_idx
                del num_written_rows
                del temp_result_array
                del temp_result
                #---------------------------------------------------------------

                
                #---------------------------------------------------------------
                # Now the checkpoint is complete, tell the workers to resume 
                # working and resent the sync, ack, and save_after_counter 
                # variables to track until the next checkpoint
                #---------------------------------------------------------------
                for worker_sock in worker_sockets:
                    
                    worker_sock.send(b"resume")
                    
                sent_syncs = False
                
                num_acknowledges = 0
                
                save_after_counter = save_after
                #---------------------------------------------------------------
        #-----------------------------------------------------------------------
    ############################################################################

                

    ############################################################################
    # We are done the main work!  Clean up worker processes.  Block until we 
    # have joined everybody so that we know everything completed correctly.
    #---------------------------------------------------------------------------
    if verbose > 0:
        
        print("Main task finish time:", time.time() - main_task_start_time, 
              flush=True)
    
    if not sent_exit_commands:
        
        for idx in range(num_cpus):
            
            worker_sockets[idx].send(b"exit")
    
    for p in processes:
        
        p.join(None) #block till join
    ############################################################################


    
    ############################################################################
    # PROFILING - SAVE OFF RESULTS
    #---------------------------------------------------------------------------
    '''
    outfile = open(os.path.join(DEBUG_DIR, "multi_thread_profile.pickle"), 'wb')
    
    pickle.dump((PROFILE_sample_times, PROFILE_samples), outfile)
    
    outfile.close()
    
    if verbose > 0:
        
        PROFILE_ARRAY = np.memmap(PROFILE_buffer, 
                                  dtype=np.float32, 
                                  shape=(85000000,3))
        
        PROFILE_ARRAY_SUBSET = PROFILE_ARRAY[0:80000000]
        
        for idx in range(7):
            
            #curr_axes = axes_list[idx]
            
            curr_idx = PROFILE_ARRAY_SUBSET[:,1] == idx
            
            curr_data = PROFILE_ARRAY_SUBSET[curr_idx, 0]
            
            if idx == 6:
                
                print("Count of profile stage 6: ", curr_data.shape[0])
                print("Avg Cell time: ", np.mean(curr_data))
            
            if idx == 6:
                outfile = open(os.path.join(DEBUG_DIR, 
                                            "Cell_Processing_Times_MultiThreadCython.pickle"), 
                                            'wb')
                pickle.dump(curr_data, outfile)
                outfile.close()
            
            plot_cell_processing_times(curr_data, idx, "Multi", DEBUG_DIR)
            
            curr_data = PROFILE_ARRAY_SUBSET[curr_idx, 2]
            
            if idx == 6:
                print("Avg Cell KDTree time: ", np.mean(curr_data))
            
            plot_cell_kdtree_times(curr_data, idx, 'Multi', DEBUG_DIR)
    '''
    ############################################################################



    ############################################################################
    # Extract the results from the shared memory map into a numpy array (using
    # frombuffer() so that we do not copy the memory), then filter out the 
    # invalid rows.
    #
    # A row will have a NAN in the 0th column if it is invalid - aka was outside
    # the survey mask, or such.
    #---------------------------------------------------------------------------
    result_buffer.seek(0)
    
    result_array = np.frombuffer(result_buffer, dtype=np.float64)
    
    result_array.shape = (n_empty_cells, 4)
    
    valid_idx = np.logical_not(np.isnan(result_array[:,0]))
    
    n_holes = np.sum(valid_idx, axis=None, dtype=np.int64)
    
    valid_result_array = result_array[valid_idx,:]
    ############################################################################

    
    
    ############################################################################
    # Since we just indexed into the result array with a boolean index 
    # (valid_idx) this will force a copy of the data into the new 
    # valid_result_array.  Since it is a copy and not a view anymore, we can 
    # safely close the mmap result_buffer.close() to the original memory, but if 
    # for some reason we change this to not copy the memory, we cannot call 
    # .close() on the memmap or we lose access to the underlying data buffer, 
    # and VoidFinder crashes with no traceback for some reason.
    #
    # Also close all our other mem-maps.
    #---------------------------------------------------------------------------
    result_buffer.close()
    
    gma_buffer.close()
    
    lookup_buffer.close()
    
    hole_lookup_buffer.close()
    
    w_coord_buffer.close()
    ############################################################################
        
    return valid_result_array, n_holes





     
    
def process_message_buffer(curr_message_buffer):
    """
    Description
    ===========
    
    Helper function to process the communication between the master 
    _hole_finder_multi_process and _hole_finder_worker processes.
    
    Since communication over a socket is only guaranteed to be in order, we have 
    to process an arbitrary number of bytes depending on the message format.  
    The message format is as such:  the first byte gives the number of 8-byte 
    fields in the message.  So a first byte of 3 means the message on the head 
    of the buffer should be 1 + 3*8 = 25 bytes long.
    
    Right now there are 3 types of message: 
    
    type 0 - "status" messages from the workers, where field 0's value should be 
    2 for 2 fields, field 1 is the number of results the worker has written, and 
    field 2 (currently unused) was the number of new holes the worker has found
    
    type 1 - "worker finished" message
    
    type 2 - "sync acknowledge" message
    
    
    Parameters
    ==========
    
    curr_message_buffer : bytes
        the current string of bytes to process
        
        
    Returns
    =======
    
    messages : list
        list of parsed messages
        
    curr_message_buffer : bytes
        any remaining data in the current message buffer which was not yet able
        to be parsed, likely due to a socket read ending not perfectly on a 
        message border
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
    




    
    
def _hole_finder_worker_profile(worker_idx, ijk_start, write_start, config):
    """
    Helper used in profiling the worker processes.
    """
    
    cProfile.runctx("_hole_finder_worker(worker_idx, ijk_start, write_start, config)", 
                    globals(), 
                    locals(), 
                    'prof%d.prof' %worker_idx)
    





def _hole_finder_worker(worker_idx, ijk_start, write_start, config):
    """
    Description
    ===========
    
    Worker process for the _hole_finder_multi_process function above.
    
    
    Parameters
    ==========
    
    worker_idx : int
       ID number for this worker
       
    ijk_start : multiprocessing.Value, integer
        locked value for synchronizing access to cell ID generation
        
    write_start : multiprocessing.Value, integer
        locked value for synchronizing access to writing to the output buffer
        
    config : dict
        configuration values for running this worker
        
        
    Output
    ======
    
    Writes out x,y,z and radius values to result buffer directly via
    shared memory.
    
    Sends update messages to master process over socket for number
    of cells which have been processed.
    """
    
    
    ############################################################################
    # Unpack the configuration from the master process.
    #---------------------------------------------------------------------------
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
    HOLE_LOOKUPMEM_BUFFER_PATH = config["HOLE_LOOKUPMEM_BUFFER_PATH"]
    hole_lookup_fd = config["hole_lookup_fd"]
    hole_next_prime = config["hole_next_prime"]
    num_nonempty_hole_cells = config["num_nonempty_hole_cells"]
    #hole_radial_mask_check_dist = config["hole_radial_mask_check_dist"]
    num_in_galaxy_map = config["num_in_galaxy_map"]
    ngrid = config["ngrid"]
    dl = config["dl"]
    dr = config["dr"]
    coord_min = config["coord_min"]
    mask_mode = config["mask_mode"]
    mask = config["mask"]
    mask_resolution = config["mask_resolution"]
    min_dist = config["min_dist"]
    max_dist = config["max_dist"]
    xyz_limits = config["xyz_limits"]
    batch_size = config["batch_size"]
    verbose = config["verbose"]
    print_after = config["print_after"]
    num_cpus = config["num_cpus"]
    search_grid_edge_length = config["search_grid_edge_length"]
    DEBUG_DIR = config["DEBUG_DIR"]
    ############################################################################



    ############################################################################
    # Open a UNIX-domain socket for communication to the master process.  We set
    # the timeout to be 10.0 seconds, so this worker will try notifying the 
    # master that it has results for up to 10.0 seconds, then it will loop again 
    # and check for input from the master, and if necessary wait and try to push 
    # results for 10 seconds again.  Right now the workers only exit after a 
    # b'exit' message has been received from the master.
    #---------------------------------------------------------------------------
    worker_socket = socket.socket(socket.AF_UNIX)
    
    worker_socket.settimeout(10.0)
    
    connect_start = time.time()
    
    try:
        
        worker_socket.connect(SOCKET_PATH)
        
    except Exception as E:
        
        print("WORKER", worker_idx, "UNABLE TO CONNECT, EXITING", flush=True)
        
        raise E
    ############################################################################

    
    
    ############################################################################
    # Load up w_coord from shared memory
    #
    # Note that since we are using the fork() method of spawning workers, that 
    # means that this worker has access to the same file descriptors that the 
    # master process had, for example file descriptor 5 means the same thing in 
    # the master process and in this worker process.  I do not know if this 
    # holds true for fork() + exec() style child process creation, so I do not 
    # know if this method will work if we use the python multiprocessing "spawn" 
    # or "forkserver" methods.  
    #
    # Note that the python mmap() call creates a duplicate file descriptor, 
    # which we have no access to (in both the master and in each child).  
    # Hopefully this never becomes a problem (so far it has not been an issue, 
    # but every system does have a limit on how many file descriptors can be 
    # open for a process and on the whole system).
    #---------------------------------------------------------------------------
    wcoord_buffer_length = num_galaxies*3*8 # 3 since xyz and 8 since float64
    
    wcoord_mmap_buffer = mmap.mmap(w_coord_fd, wcoord_buffer_length)
    
    w_coord = np.frombuffer(wcoord_mmap_buffer, dtype=np.float64)
    
    w_coord.shape = (num_galaxies, 3)
    ############################################################################

    
    
    ############################################################################
    # Load up galaxy_map_array from shared memory
    #---------------------------------------------------------------------------
    gma_buffer_length = num_gma_indices*8 # 3 since xyz and 8 since float64
    
    gma_mmap_buffer = mmap.mmap(gma_fd, gma_buffer_length)
    
    galaxy_map_array = np.frombuffer(gma_mmap_buffer, dtype=np.int64)
    
    galaxy_map_array.shape = (num_gma_indices,)
    ############################################################################

    
    
    ############################################################################
    # Primary data structure for the lookup of galaxies in cells.  Used to be
    # a scipy KDTree, then switched to sklearn KDTree for better performance, 
    # then re-wrote to use a map from elements of the search grid to the 
    # galaxies it is closest to, and lastly, re-wrote a custom dict class. I was 
    # using the built-in python dict, but I needed the underlying memory to be 
    # exposed so I could memmap it, so I wrote a new class which I can do that 
    # with.  Also with the new dict class, which works just like the python 
    # class, backed by a hash-table array, we can take advantage of the 
    # uniformness of our key values which are cell IDs, and their sequential 
    # nature, so our custom dict class actually sped up voidfinder by about a 
    # factor of 2.
    #
    # The galaxy_map keys are p-q-r triplet cell ID values, and the value 
    # associated with each key is the offset into the galaxy_map_array, and the 
    # number of elements at that offset, which correspond to the galaxies which 
    # belong to that p-q-r cell.  The num_elements galaxy_map_array values are 
    # the indices into w_coord for the galaxies at the p-q-r cell
    #---------------------------------------------------------------------------
    lookup_buffer_length = next_prime*23 # 23 bytes per element
    
    lookup_mmap_buffer = mmap.mmap(lookup_fd, lookup_buffer_length)
    
    lookup_dtype = [("filled_flag", np.uint8, ()), #() indicating scalar length 1
                    ("i", np.int16, ()),
                    ("j", np.int16, ()),
                    ("k", np.int16, ()),
                    ("offset", np.int64, ()),
                    ("num_elements", np.int64, ())]

    input_numpy_dtype = np.dtype(lookup_dtype, align=False)
    
    lookup_memory = np.frombuffer(lookup_mmap_buffer, dtype=input_numpy_dtype)
    
    lookup_memory.shape = (next_prime,)

    galaxy_map = GalaxyMapCustomDict(ngrid,
                                     lookup_memory)

    galaxy_tree = GalaxyMap(w_coord, 
                            coord_min, 
                            search_grid_edge_length,
                            galaxy_map,
                            galaxy_map_array)
    ############################################################################

    
    
    ############################################################################
    # Create an object which controls the allocation of memory for storing cell 
    # IDs when searching the grid for galaxy neighbors.  The parameter 10 in 
    # Cell_ID_Memory(10) refers to "level 10" meaning enough memory to store a 
    # 21x21x21 grid (2*10+1 = 21 = grid edge size).  If we knew the largest 
    # query to be performed for this run we could set this value appropriately 
    # here, but we typically do not, so Cell_ID_Memory is capable of 
    # self-resizing to accomodate larger queries.
    #
    # Also now an object similar to Cell_ID_Memory, for storing the indices of 
    # nearest neighbor galaxies
    #---------------------------------------------------------------------------
    cell_ID_mem = Cell_ID_Memory(10)
    
    neighbor_mem = NeighborMemory(50)
    ############################################################################


    
    ############################################################################
    # Memmap in the memory for the results
    #---------------------------------------------------------------------------
    n_empty_cells = ngrid[0]*ngrid[1]*ngrid[2] - num_nonempty_hole_cells
    
    result_buffer_length = n_empty_cells*4*8 #float64 so 8 bytes per element
    
    result_mmap_buffer = mmap.mmap(result_fd, result_buffer_length)
    ############################################################################

    
    
    ############################################################################
    # Load/mmap the dictionary corresponding to the hole grid.
    #---------------------------------------------------------------------------
    hole_lookup_buffer_length = hole_next_prime*7 # 7 bytes per element
    
    hole_lookup_mmap_buffer = mmap.mmap(hole_lookup_fd, 
                                        hole_lookup_buffer_length)
    
    hole_lookup_dtype = [("filled_flag", np.uint8, ()), #() indicating scalar length 1
                         ("i", np.int16, ()),
                         ("j", np.int16, ()),
                         ("k", np.int16, ())]
                    

    hole_input_numpy_dtype = np.dtype(hole_lookup_dtype, align=False)
    
    hole_lookup_memory = np.frombuffer(hole_lookup_mmap_buffer, 
                                       dtype=hole_input_numpy_dtype)
    
    hole_lookup_memory.shape = (hole_next_prime,)
    
    hole_cell_ID_dict = HoleGridCustomDict(ngrid,
                                           hole_lookup_memory)
    ############################################################################


    
    ############################################################################
    # Build Cell ID generator
    #---------------------------------------------------------------------------
    cell_ID_gen = CellIDGenerator(ngrid[0],
                                  ngrid[1],
                                  ngrid[2],
                                  hole_cell_ID_dict)
    ############################################################################




    ############################################################################
    # Build class to help process mask checks
    #---------------------------------------------------------------------------
    #mask_mode = 0
    
    if mask_mode == 0:
        mask_checker = MaskChecker(mask_mode,
                                   survey_mask_ra_dec=mask,
                                   n=mask_resolution,
                                   rmin=min_dist,
                                   rmax=max_dist,
                                   )
        
    elif mask_mode == 1:
        mask_checker = MaskChecker(mask_mode,
                                   xyz_limits=xyz_limits)
    ############################################################################


    
    ############################################################################
    # Profiling parameters
    #---------------------------------------------------------------------------
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
    ############################################################################


    
    ############################################################################
    # Memory for PROFILING
    #---------------------------------------------------------------------------
    '''
    PROFILE_buffer = open(PROFILE_BUFFER_PATH, 'r+b')
    
    PROFILE_buffer_length = 85000000*3*4 #float32 so 4 bytes per element
    
    PROFILE_mmap_buffer = mmap.mmap(PROFILE_buffer.fileno(), PROFILE_buffer_length)
    
    PROFILE_array = np.empty((batch_size, 3), dtype=np.float32)
    
    PROFILE_gen_times = []
    
    PROFILE_main_times = []
    '''
    ############################################################################



    ############################################################################
    # Main Loop for the worker process begins here.
    #
    #    exit_process - flag for reading an exit command off the queue
    #
    #    document the additional below variables here please
    #
    # If this worker process has reached the end of the Cell ID generator, we 
    # want to tell the master process we are done working, and wait for an exit 
    # command, so increase the select_timeout from 0 (instant) to 2.0 seconds to 
    # allow the operating system to wake us up during that 2.0 second interval 
    # and avoid using unnecessary CPU.
    #---------------------------------------------------------------------------
    received_exit_command = False
    
    exit_process = False
    
    return_array = np.empty((batch_size, 4), dtype=np.float64)
    
    i_j_k_array = np.empty((batch_size, 3), dtype=np.int64)
    
    worker_sockets = [worker_socket]
    
    empty1 = []
    
    empty2 = []
    
    message_buffer = b""
    
    do_work = True
    
    sync = False
    
    sent_sync_ack = False
    
    have_result_to_write = False
    
    no_cells_left_to_process = False
    
    sent_deactivation = False
    
    select_timeout = 0
    
    while not exit_process:
        
        total_loops += 1
        
        #-----------------------------------------------------------------------
        # As the first part of the main loop, use the select() method to check 
        # for any messages from the master process.  It may send us an "exit" 
        # command, to tell us to terminate, a "sync" command, to tell us to stop 
        # processing momentarily while it writes out a save checkpoint, or a 
        # "resume" command to tell us that we may continue processing after a 
        # "sync".
        #-----------------------------------------------------------------------
        #print("Worker "+str(worker_idx)+" "+str(message_buffer), flush=True)
        
        read_socks, empty3, empty4 = select.select(worker_sockets, 
                                                   empty1, 
                                                   empty2, 
                                                   select_timeout)
        
        if read_socks:
            
            message_buffer += worker_socket.recv(1024)
            
        if len(message_buffer) > 0:
            
            if len(message_buffer) >= 4 and message_buffer[0:4] == b'exit':
                
                exit_process = True
                
                received_exit_command = True
                
                continue
            
            elif len(message_buffer) >= 4 and message_buffer[0:4] == b"sync":
                
                sync = True
                
                message_buffer = message_buffer[4:]
                
            elif len(message_buffer) >= 6 and message_buffer[0:6] == b"resume":
                
                sync = False
                
                sent_sync_ack = False
                
                message_buffer = message_buffer[6:]
        #-----------------------------------------------------------------------
        
        
        #-----------------------------------------------------------------------
        # Here we do the main work of VoidFinder.  We synchronize the work with 
        # the other worker processes using 2 lock-protected values, 'ijk_start' 
        # and 'write_start'.  ijk_start gives us the starting cell_ID index to 
        # generate the next batch of cell ID's at, and write_start gives us the 
        # index to write our batch of results at.  Note that we will process AT 
        # MOST 'batch_size' indexes per loop, because we use the Galaxy Map to 
        # filter out cell IDs which do not need to be checked (since they have 
        # galaxies in them they are non-empty and we will not find a hole 
        # there).  Since we may process LESS than batch_size locations, when we 
        # update 'write_start' we update it with the actual number of cells 
        # which we have worked in our current batch. 
        #
        # Note if we are in 'sync' mode, we do not want to do any work since the 
        # master process is making a checkpoint file.
        #-----------------------------------------------------------------------
        if do_work and not sync:
        
            #-------------------------------------------------------------------
            # Get the next index of the starting cell ID to process for our 
            # current batch
            #-------------------------------------------------------------------
            ijk_start.acquire()
            
            start_idx = ijk_start.value
            
            ijk_start.value += batch_size
            
            ijk_start.release()
            #-------------------------------------------------------------------

            
            #-------------------------------------------------------------------
            # Fill the current batch of cell IDs into our i_j_k_array variable
            #-------------------------------------------------------------------
            num_write = cell_ID_gen.gen_cell_ID_batch(start_idx, 
                                                      batch_size, 
                                                      i_j_k_array)
            
            num_cells_to_process = num_write
            
            if num_cells_to_process > 0:
                
                if return_array.shape[0] != num_cells_to_process:
    
                    return_array = np.empty((num_cells_to_process, 4), 
                                            dtype=np.float64)
                    
                main_algorithm(i_j_k_array[0:num_write],
                               galaxy_tree,
                               w_coord,
                               dl, 
                               dr,
                               coord_min,
                               mask_checker,
                               #mask,
                               #mask_resolution,
                               #min_dist,
                               #max_dist,
                               #hole_radial_mask_check_dist,
                               return_array,
                               cell_ID_mem,
                               neighbor_mem,
                               0,  #verbose level
                               #PROFILE_array
                               )
                
                num_cells_processed += num_write
                
                #---------------------------------------------------------------
                # We have some results, now we synchronize with the other 
                # processes to figure out where to write those results into the 
                # result memmap
                #---------------------------------------------------------------
                write_start.acquire()
                
                out_start_idx = write_start.value
                
                write_start.value += num_write
                
                write_start.release()
                #---------------------------------------------------------------
                
                
                #---------------------------------------------------------------
                # Write out the current results into our memmap
                #---------------------------------------------------------------
                '''
                print("Worker", worker_idx, 
                      "writing row:", out_start_idx, 
                      "num write:", num_write, 
                      flush=True)
                '''
                seek_location = 32*out_start_idx
                
                result_mmap_buffer.seek(seek_location)
                
                write_data = return_array[0:num_write].tobytes()
                
                result_mmap_buffer.write(write_data)
                
                have_result_to_write = True
                #---------------------------------------------------------------
                
            else:
                #---------------------------------------------------------------
                # If the cell_ID_generator ever returns '0', that means that we 
                # have reached the end of the whole search grid, so this worker 
                # can notify the master that it is done working.
                #---------------------------------------------------------------
                no_cells_left_to_process = True
                #---------------------------------------------------------------
        #-----------------------------------------------------------------------

            
        #-----------------------------------------------------------------------
        # Update the master process that we have processed some number of cells, 
        # using our socket connection.  Note the actual results get written 
        # directly to the shared memmap, but the socket just updates the master 
        # with the number of new results (an integer).
        #-----------------------------------------------------------------------  
        if have_result_to_write:   
            '''
            n_hole = np.sum(np.logical_not(np.isnan(return_array[:,0])), 
                                           axis=None, 
                                           dtype=np.int64)
            '''
            out_msg = b""
            out_msg += struct.pack("b", 2) #1 byte - number of 8 byte fields
            out_msg += struct.pack("=q", 0) #8 byte field - message type 0
            out_msg += struct.pack("=q", num_write) #8 byte field - payload for num-write
            
            try:
                worker_socket.send(out_msg)
            except:
                do_work = False
            else:
                do_work = True
                have_result_to_write = False
        #-----------------------------------------------------------------------

            
        #-----------------------------------------------------------------------
        # If we are done working (cell ID generator reached the end/returned 0), 
        # notify the master process that this worker is going into a "wait for 
        # exit" state where we just sleep and check the input socket for the 
        # b'exit' message
        #-----------------------------------------------------------------------
        if no_cells_left_to_process:
            
            if not sent_deactivation:
            
                out_msg = b""
                out_msg += struct.pack("b", 1) #1 byte - number of 8 byte fields
                out_msg += struct.pack("=q", 1) #8 byte field - message type 1 (no payload)
                
                worker_socket.send(out_msg)
                
                sent_deactivation = True
                
                select_timeout = 2.0
        #-----------------------------------------------------------------------

            
        #-----------------------------------------------------------------------
        # If the master process wants to save a checkpoint, it needs the workers 
        # to sync up.  It sends a b'sync' message, and then it waits for all the 
        # workers to acknowledge that they have received the 'sync', so here we 
        # send that acknowledgement.  After we have received the sync, we just 
        # want to sleep and check the socket for a b'resume' message.
        #-----------------------------------------------------------------------
        if sync:
            
            if not sent_sync_ack:
                
                acknowledge_sync = b""
                acknowledge_sync += struct.pack("b", 1) #1 byte - number of 8 byte fields
                acknowledge_sync += struct.pack("=q", 2) #8 byte field - message type 2
                
                try:
                    worker_socket.send(acknowledge_sync)
                except:
                    pass
                else:
                    sent_sync_ack = True
            else:
            
                time.sleep(1.0)
        #-----------------------------------------------------------------------
    ############################################################################


                
    ############################################################################
    # We are all done!  Close the socket and any other resources, and finally 
    # return.
    #---------------------------------------------------------------------------
    worker_socket.close()
    
    print("WORKER EXITING GRACEFULLY", worker_idx, flush=True)
    ############################################################################

    
    return None




def find_next_prime(threshold_value):
    """
    Description
    ===========
    
    Given an input integer threshold_value, find the next prime number
    greater than threshold_value.  This is used as a helper in creating
    the memory backing array for the galaxy map, because taking an index
    modulus a prime number is a nice way to hash an integer.
    
    Uses Bertrams(?) theorem that for every n > 1 there is a prime number
    p such that n < p < 2n
    
    
    Parameters
    ==========
    
    threshold_value : int
        find the next prime number after this value
        
    
    Returns
    =======
    
    check_val : int
        next prime number after threshold_value
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
    
    
    
    
    
    
    
    
    
    
    
    