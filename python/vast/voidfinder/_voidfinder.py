



import os
import stat
import mmap
import struct
import socket
import select
import atexit
import tempfile
import multiprocessing
import h5py
#import pickle
import time
from psutil import cpu_count

import cProfile

import numpy as np

from ._voidfinder_cython import fill_ijk_zig_zag, grow_spheres
#from ._voidfinder_cython import fill_ijk
                                

from ._voidfinder_cython_find_next import SpatialMap, \
                                          Cell_ID_Memory, \
                                          GalaxyMapCustomDict, \
                                          HoleGridCustomDict, \
                                          NeighborMemory, \
                                          MaskChecker, \
                                          SphereGrower

from multiprocessing import Process, Value

from ctypes import c_int64

#import matplotlib
#matplotlib.use('TkAgg')
#import matplotlib.pyplot as plt



#from .viz import VoidRender

    
    
        
        
        
        
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
        
        
        

    

def SaveCheckpointFile(checkpoint_filepath,
                       num_written_rows,
                       next_cell_idx,
                       temp_result):


    if os.path.isfile(checkpoint_filepath):
                    
        os.remove(checkpoint_filepath)
        
    outfile = h5py.File(checkpoint_filepath, 'w')
    
    outfile.attrs["num_written_rows"] = num_written_rows
    
    outfile.attrs["next_cell_idx"] = next_cell_idx
    
    outfile.create_dataset("result_array", data=temp_result)
    
    outfile.close()




def get_common_divisors(values):
    
    
    min_value = int(np.ceil(min(values)/2))
    
    divisors = []
    
    for idx in range(1, min_value):
        
        if all([value % idx == 0 for value in values]):
        
            divisors.append(idx)
            
    return divisors
            
        







def _hole_finder(galaxy_coords,
                 hole_grid_edge_length, 
                 hole_center_iter_dist,
                 galaxy_map_grid_edge_length,
                 survey_name,
                 mask_mode=0,
                 mask=None,
                 mask_resolution=None,
                 min_dist=None,
                 max_dist=None,
                 xyz_limits=None,
                 check_only_empty_cells=True,
                 save_after=None,
                 use_start_checkpoint=False,
                 batch_size=1000,
                 verbose=0,
                 print_after=10000,
                 num_cpus=None,
                 
                 CONFIG_PATH="/tmp/voidfinder_config.pickle",
                 SOCKET_PATH="/tmp/voidfinder.sock",
                 RESOURCE_DIR="/dev/shm",
                 DEBUG_DIR="/home/moose/VoidFinder/doc/debug_dir"
                 ):
    """
    See help(voidfinder.find_voids)
    
    Work-horse method for running VoidFinder with the Cython code in parallel
    multi-process form. Also the single threaded version.
    
    This method contains the logic for:
    
    1) Sanity check the num_cpus to use
    2) Open file handles and allocate memory for workers to memmap to
    3) Build a few data structures for the workers to share
    4) Register some cleanup helpers with the python interpreters for making 
       sure the disk space gets reclaimed when we are done
    5) Start the workers
    6) Make sure workers connect to the comm socket
    7) Checkpoint the progress if those parameters are enabled
    8) Collect & print progress results from the workers
    
    This function is designed to be run on Linux on an SMP (Symmetric 
    Multi-Processing) architecture.  It takes advantage of 2 Linux-specific 
    properties: the /dev/shm filesystem and the fork() method of spawning 
    processes. /dev/shm is used as the preferred location for creating memory 
    maps to share information between the worker processes since on Linux it is 
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
    
    galaxy_coords : length-2 list of numpy.ndarrays of shape (num_galaxies, 3)
        Cartesian coordinates of the galaxies in the survey, units of Mpc/h.  
        The first element in the list is the wall galaxies, and the second is 
        the field galaxies.
    
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
        
    survey_name : str
        identifier for the survey running, may be prepended or appended to 
        output filenames including the checkpoint filename
        
    mask_mode : int, one of [0,1,2]
        Determines which mode VoidFinder is running in with regards to the Mask
        checking.  0 == 'ra-dec-redshift', 1 == 'xyz' and 2 == 'periodic'
        
    mask : numpy.ndarray of shape (N,M) type bool
        represents the survey footprint in scaled ra/dec space.  Value of True 
        indicates that a location is within the survey

    mask_resolution : integer
        Scale factor of coordinates needed to index mask
    
    min_dist : float
        minimum redshift in units of Mpc/h
        
    max_dist : float
        maximum redshift in units of Mpc/h
        
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
    """
    
    
    
    
    ############################################################################
    # Do some sanity checking on the mask modes and various inputs since we
    # have a lot of None type optional inputs
    #---------------------------------------------------------------------------
    if mask_mode == 0:
        if mask is None or \
            mask_resolution is None or \
            min_dist is None or \
            max_dist is None:
            raise ValueError("Mask mode is 0 (ra-dec-z) but a required mask parameter is None")
        
        # The cython requires some very specific types and shapes
        mask = mask.astype(np.uint8)
    
    if mask_mode == 1 and xyz_limits is None:
        raise ValueError("Mask mode is 1 (xyz) but required mask parameter xyz_limits is None")
    
    if mask_mode == 2 and xyz_limits is None:
        raise ValueError("Mask mode is 2 (periodic) but required mask parameter xyz_limits is None")
    ############################################################################
    
    
    
    
    ############################################################################
    # Next, depending on the mask mode, calculate the transform origin and grid
    # cell parameters for our Hole grid and our Galaxy Map grids.
    #
    # For 'ra-dec-z' - just use the min and max of the provided coordinates of 
    #     the survey for the transform
    #
    # For 'xyz' mode - use the required/provided xyz_limits
    #
    # For 'periodic' mode - use the required/provided xyz_limits, but we also 
    #     have to ensure the GalaxyMap cells align with the xyz_limits 
    #     boundaries, so that VoidFinder doesn't accidentally introduce dead 
    #     space into cells because of misalignment.  
    #
    # The important values calculated below are:
    #    'coords_min' - origin for transforming between real and index spaces
    #    'hole_grid_shape' - grid cells for the hole growing grid
    #    'galaxy_map_grid_shape' - grid cells for the galaxy finding grid
    #---------------------------------------------------------------------------
    if mask_mode == 0: #ra-dec-redshift
        
        if galaxy_map_grid_edge_length is None:
        
            galaxy_map_grid_edge_length = 3.0*hole_grid_edge_length
        
        coords_max = np.max(np.concatenate(galaxy_coords), axis=0)
    
        coords_min = np.min(np.concatenate(galaxy_coords), axis=0)
        
        box = coords_max - coords_min
    
        ngrid = box/hole_grid_edge_length
        
        hole_grid_shape = tuple(np.ceil(ngrid).astype(int))
        
        ngrid_galaxymap = box/galaxy_map_grid_edge_length
        
        galaxy_map_grid_shape = tuple(np.ceil(ngrid_galaxymap).astype(int))
        
        
    elif mask_mode == 1: #xyz
        
        if galaxy_map_grid_edge_length is None:
        
            galaxy_map_grid_edge_length = 3.0*hole_grid_edge_length
        
        box = xyz_limits[1,:] - xyz_limits[0,:]
        
        ngrid = box/hole_grid_edge_length
        
        hole_grid_shape = tuple(np.ceil(ngrid).astype(int))
        
        coords_min = xyz_limits[0,:]
        
        ngrid_galaxymap = box/galaxy_map_grid_edge_length
        
        galaxy_map_grid_shape = tuple(np.ceil(ngrid_galaxymap).astype(int))
        
        
    elif mask_mode == 2: #periodic
        
        box = xyz_limits[1,:] - xyz_limits[0,:]
        
        ngrid = box/hole_grid_edge_length
        
        hole_grid_shape = tuple(np.ceil(ngrid).astype(int))
        
        coords_min = xyz_limits[0,:]
        
        if galaxy_map_grid_edge_length is None:
            
            desired_length = 3.0*hole_grid_edge_length
            
            #Find the common integer divisors of the length dimensions of the survey limits
            common_divisors = get_common_divisors(box)
            
            if len(common_divisors) == 0 or \
               (len(common_divisors) == 1 and common_divisors[0] == 1):
                
                error_str = """Could not automatically determine meaningful galaxy_map_grid_edge_length 
                from the provided xyz_limits.  In mask_mode==periodic, the survey limits 
                provided by the xyz_limits variable must be divisible by a common integer 
                in all dimensions"""
                
                raise ValueError(error_str)
            
            common_divisors = np.array(common_divisors)
            
            argmin = np.abs(common_divisors - desired_length).argmin()
            
            galaxy_map_grid_edge_length = float(common_divisors[argmin])
            
            ngrid_galaxymap = box/galaxy_map_grid_edge_length
        
            galaxy_map_grid_shape = tuple(np.ceil(ngrid_galaxymap).astype(int))
            
        else:
            
            ngrid_galaxymap = box/galaxy_map_grid_edge_length
            
            rounded = np.rint(ngrid_galaxymap)
            
            print(rounded)
            
            close_to_round = np.isclose(ngrid_galaxymap, rounded)
            
            print(close_to_round)
            
            if np.all(close_to_round):
                #Vals are good, just proceed with given
                galaxy_map_grid_shape = tuple(np.rint(ngrid_galaxymap).astype(int))
            else:
                #Attempt to adjust galaxy_map_grid_edge_length
                error_str = """The provided combination of xyz_limits and galaxy_map_grid_edge length
                will not work.  In mask_mode==periodic, the edge length must be an integer
                divisor of all dimensions of the survey as provided by the xyz_limits input."""
    
                raise ValueError(error_str)
        
        
    coords_min = coords_min.reshape(1,3).astype(np.float64)
    
    if verbose > 0:
        
        print("Hole-growing Grid:", hole_grid_shape, flush=True)
        
        print("Galaxy-searching Grid:", galaxy_map_grid_shape, flush=True)
        
        print("Galaxy-searching edge length:", galaxy_map_grid_edge_length, 
              flush=True)
    ############################################################################
    
    
    
    
    ############################################################################
    # If /dev/shm is not available, use /tmp as the shared resource filesystem
    # location instead.  Since on Linux /dev/shm is guaranteed to be a mounted
    # RAMdisk, I do not know if /tmp will be as fast or not, probably depends on
    # kernel settings.
    # Also future updates to Linux on some distros might be using /run/shm
    # instead of /dev/shm
    #---------------------------------------------------------------------------
    if not os.path.isdir(RESOURCE_DIR):
        
        print("WARNING: RESOURCE DIR", RESOURCE_DIR, 
              "does not exist.  Falling back to /tmp", 
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
    if num_cpus is None:
          
        num_cpus = cpu_count(logical=False)
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
    # cell" so VoidFinder can skip that i,j,k value when growing holes.
    #
    # The HoleGridCustomDict class now creates and manages its own internal
    # memmaped file so it can be passed directly across fork() and has a resize
    # method so we can just use it directly without the helper.
    #---------------------------------------------------------------------------
    mesh_indices = ((galaxy_coords[0] - coords_min)/hole_grid_edge_length).astype(np.int64)
    
    #test_hole_cell_ID_dict = {}
    
    hole_cell_ID_dict = HoleGridCustomDict(hole_grid_shape,
                                           RESOURCE_DIR)
    
    if check_only_empty_cells:
        
        #print("Num rows: ", len(mesh_indices))
    
        for row in mesh_indices:
            
            #test_hole_cell_ID_dict[tuple(row)] = 1
            
            hole_cell_ID_dict.setitem(*tuple(row))
            
    
    num_nonempty_hole_cells = len(hole_cell_ID_dict)
    
    
    del mesh_indices
    
    if verbose > 0:
        
        print("Number of filtered out hole-growing cells:", 
              num_nonempty_hole_cells, 
              flush=True)
        
        print("Total slots in hole_cell_ID_dict:", 
              hole_cell_ID_dict.mem_length, 
              flush=True)
        
        print("Num collisions hole_cell_ID_dict:", 
              hole_cell_ID_dict.num_collisions, 
              flush=True)
    
    #print("Test: ", len(test_hole_cell_ID_dict), num_nonempty_hole_cells)
    ############################################################################


    
    ############################################################################
    # Next create the GalaxyMap p-q-r-space index, which is constructed 
    # identically to the hole_grid i-j-k-space, except that we use a larger cell 
    # edge length so we get more galaxies per cell, and we actually store 
    # information about that cell, so we have to do a first pass through
    # all the galaxies to sort their indices into their cells first before we 
    # can actually construct the hash map based GalaxyMapCustomDict
    #---------------------------------------------------------------------------
    if verbose > 0:
        
        galaxy_map_start_time = time.time()
        
        print("Building galaxy map", flush=True)
    
    mesh_indices = ((galaxy_coords[0] - coords_min)/galaxy_map_grid_edge_length).astype(np.int64)
        
    pre_galaxy_map = {}

    for idx in range(mesh_indices.shape[0]):

        bin_ID_pqr = tuple(mesh_indices[idx])
        
        if bin_ID_pqr not in pre_galaxy_map:
            
            pre_galaxy_map[bin_ID_pqr] = []
        
        pre_galaxy_map[bin_ID_pqr].append(idx)
        
    del mesh_indices
    
    num_in_galaxy_map = len(pre_galaxy_map)
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
    galaxy_search_cell_dict = GalaxyMapCustomDict(galaxy_map_grid_shape,
                                                  RESOURCE_DIR)
    
    offset = 0
    
    galaxy_map_list = []
    
    for key in pre_galaxy_map:
        
        indices = np.array(pre_galaxy_map[key], dtype=np.int64)
        
        num_elements = indices.shape[0]
        
        galaxy_map_list.append(indices)
        
        galaxy_search_cell_dict.setitem(*key, offset, num_elements)
        
        offset += num_elements

    galaxy_map_array = np.concatenate(galaxy_map_list)
    
    del galaxy_map_list
    
    num_galaxy_map_elements = len(galaxy_search_cell_dict)
    
    
    
    galaxy_map = SpatialMap(RESOURCE_DIR,
                            mask_mode,
                            galaxy_coords[0],
                            hole_grid_edge_length,
                            coords_min[0,:], 
                            galaxy_map_grid_edge_length,
                            galaxy_search_cell_dict,
                            galaxy_map_array)
    
    
    
    if verbose > 0:
        
        print("Total slots in galaxy map:", 
              galaxy_search_cell_dict.mem_length, 
              flush=True)
        
        print("Num gma indices:", galaxy_map.num_gma_indices, flush=True)
    
    
    if mask_mode == 2:
        #Brute force all the cells including a shell around the survey when in
        #periodic mode
        for i in range(-1, galaxy_map_grid_shape[0]+1):
            for j in range(-1, galaxy_map_grid_shape[1]+1):
                for k in range(-1, galaxy_map_grid_shape[2]+1):
                    
                    #Yes using contains not setitem here
                    galaxy_map.contains(i,j,k)
                
                
    
    
    
    
    
    
    '''
    viz = VoidRender(
                 
                 galaxy_xyz=galaxy_coords[0],
                 wall_galaxy_xyz=galaxy_map.wall_galaxy_coords,
                 galaxy_display_radius=10,
                 SPHERE_TRIANGULARIZATION_DEPTH=3,
                 canvas_size=(1600,1200))

    viz.run()
    '''
    
    
    
    
    
    if verbose > 0:
        
        print("Galaxy Map build time:", time.time() - galaxy_map_start_time, 
              flush=True)
        
        print("Num items in Galaxy Map:", num_in_galaxy_map, flush=True)
        
        print("Total slots in galaxy map hash table:", 
              galaxy_search_cell_dict.mem_length, 
              flush=True)
        
        print("Num collisions in rebuild:", 
              galaxy_search_cell_dict.num_collisions, 
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
    '''
    w_coord_fd, WCOORD_BUFFER_PATH = tempfile.mkstemp(prefix="voidfinder", 
                                                      dir=RESOURCE_DIR, 
                                                      text=False)
    
    if verbose > 0:
        
        print("Mem-mapping galaxy coordinates", flush=True)
        
        print("WCOORD MEMMAP PATH:", WCOORD_BUFFER_PATH, w_coord_fd, flush=True)
    
    num_galaxies = galaxy_coords[0].shape[0]
    
    w_coord_buffer_length = num_galaxies*3*8 # 3 for xyz and 8 for float64
    
    os.ftruncate(w_coord_fd, w_coord_buffer_length)
    
    w_coord_buffer = mmap.mmap(w_coord_fd, w_coord_buffer_length)
    
    w_coord_buffer.write(galaxy_coords[0].astype(np.float64).tobytes())
    
    del galaxy_coords
    
    galaxy_coords = np.frombuffer(w_coord_buffer, dtype=np.float64)
    
    galaxy_coords.shape = (num_galaxies, 3)
    
    os.unlink(WCOORD_BUFFER_PATH)
    '''
    ############################################################################


    
    

    
    
    ############################################################################
    # Memmap the galaxy map array to our worker processes 
    #---------------------------------------------------------------------------
    '''
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
    '''
    ############################################################################

    
    
    ############################################################################
    # Calculate the number of cells we need to search
    #---------------------------------------------------------------------------
    n_empty_cells = hole_grid_shape[0]*hole_grid_shape[1]*hole_grid_shape[2] - num_nonempty_hole_cells
    
    if n_empty_cells == 0:
        raise ValueError("Found 0 cells to grow holes in.  Either set 'check_only_empty_cells' to False or check your grid and grid edge length parameters")
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
                     #"WCOORD_BUFFER_PATH" : WCOORD_BUFFER_PATH,
                     #"w_coord_fd" : w_coord_fd,
                     #"num_galaxies" : num_galaxies,
                     #"GMA_BUFFER_PATH" : GMA_BUFFER_PATH,
                     #"gma_fd" : gma_fd,
                     #"num_gma_indices" : num_gma_indices,
                     #"LOOKUPMEM_BUFFER_PATH" : LOOKUPMEM_BUFFER_PATH,
                     #"lookup_fd" : lookup_fd,
                     #"next_prime" : next_prime,
                     "galaxy_map" : galaxy_map,
                     #"HOLE_LOOKUPMEM_BUFFER_PATH" : HOLE_LOOKUPMEM_BUFFER_PATH,
                     #"hole_lookup_fd" : hole_lookup_fd,
                     #"hole_next_prime" : hole_next_prime,
                     "hole_cell_ID_dict" : hole_cell_ID_dict,
                     "num_nonempty_hole_cells" : num_nonempty_hole_cells,
                     #"hole_radial_mask_check_dist" : hole_radial_mask_check_dist,
                     #"CELL_ID_BUFFER_PATH" : CELL_ID_BUFFER_PATH,
                     #"PROFILE_BUFFER_PATH" : PROFILE_BUFFER_PATH,
                     #"cell_ID_dict" : cell_ID_dict,
                     #"galaxy_map" : galaxy_map,
                     "num_in_galaxy_map" : num_in_galaxy_map,
                     "ngrid" : hole_grid_shape, 
                     "dl" : hole_grid_edge_length, 
                     "dr" : hole_center_iter_dist,
                     "coords_min" : coords_min, 
                     "mask_mode" : mask_mode,
                     "xyz_limits" : xyz_limits,
                     "mask" : mask,
                     "mask_resolution" : mask_resolution,
                     "min_dist" : min_dist,
                     "max_dist" : max_dist,
                     "ENABLE_SAVE_MODE" : ENABLE_SAVE_MODE,
                     "save_after" : save_after,
                     "survey_name" : survey_name,
                     #w_coord,
                     "batch_size" : batch_size,
                     "verbose" : verbose,
                     "print_after" : print_after,
                     "num_cpus" : num_cpus,
                     "search_grid_edge_length" : galaxy_map_grid_edge_length,
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
        
    #def cleanup_wcoord():
    #    
    #    if os.path.isfile(WCOORD_BUFFER_PATH):
    #    
    #        os.remove(WCOORD_BUFFER_PATH)
        
    #def cleanup_gma():
    #    
    #    if os.path.isfile(GMA_BUFFER_PATH):
    #    
    #        os.remove(GMA_BUFFER_PATH)
        
    #def cleanup_lookupmem():
    #    
    #    if os.path.isfile(LOOKUPMEM_BUFFER_PATH):
    #    
    #        os.remove(LOOKUPMEM_BUFFER_PATH)
        
    
    atexit.register(cleanup_config)
    
    atexit.register(cleanup_socket)
    
    atexit.register(cleanup_result)
    
    #atexit.register(cleanup_wcoord)
    
    #atexit.register(cleanup_gma)
    
    #atexit.register(cleanup_lookupmem)
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
    
    if num_cpus > 1:
        
        if verbose > 0:
            print("Running multi-process mode,", str(num_cpus), "cpus", flush=True)
        
        
    
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
            
    else:
        
        
        if verbose > 0:
            print("Running single-process mode,", str(num_cpus), "cpus", flush=True)
        
        main_task_start_time = time.time()
        
        #Single process mode
        _hole_finder_worker(0, 
                            ijk_start, 
                            write_start, 
                            config_object)
        
        if verbose > 0:
            print("Main task finish time:", time.time() - main_task_start_time, 
                      flush=True)
        
    
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
    
    num_active_processes = 0
    
    if num_cpus > 1:
        
        if verbose > 0:
            
            print("Attempting to connect workers", flush=True)
        
        
        
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
                  'cells of', n_empty_cells, " cells", 
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
                
                
                SaveCheckpointFile(survey_name+"VoidFinderCheckpoint.h5",
                                   num_written_rows,
                                   next_cell_idx,
                                   temp_result)
                
                
                
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
        
        if num_cpus > 1:
            print("Main task finish time:", time.time() - main_task_start_time, 
                  flush=True)
    
    
    if num_cpus > 1:
        
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
    #result_buffer.close()
    # Commented this out due to:
    # BufferError: cannot close exported pointers exist.
    # https://stackoverflow.com/questions/53339931/properly-discarding-ctypes-pointers-to-mmap-memory-in-python
    # https://github.com/ercius/openNCEM/issues/39
    
    # Since the worker function closes this stuff too, we only have to close
    # our parent copies if we're running multi-processed
    
    if num_cpus > 1:
        
        hole_cell_ID_dict.close()
        
        galaxy_map.close()
    
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
    #WCOORD_BUFFER_PATH = config["WCOORD_BUFFER_PATH"]
    #w_coord_fd = config["w_coord_fd"]
    #num_galaxies = config["num_galaxies"]
    #GMA_BUFFER_PATH = config["GMA_BUFFER_PATH"]
    #gma_fd = config["gma_fd"]
    #num_gma_indices = config["num_gma_indices"]
    #LOOKUPMEM_BUFFER_PATH = config["LOOKUPMEM_BUFFER_PATH"]
    #lookup_fd = config["lookup_fd"]
    #next_prime = config["next_prime"]
    galaxy_map = config["galaxy_map"]
    #HOLE_LOOKUPMEM_BUFFER_PATH = config["HOLE_LOOKUPMEM_BUFFER_PATH"]
    #hole_lookup_fd = config["hole_lookup_fd"]
    #hole_next_prime = config["hole_next_prime"]
    hole_cell_ID_dict = config["hole_cell_ID_dict"]
    num_nonempty_hole_cells = config["num_nonempty_hole_cells"]
    #hole_radial_mask_check_dist = config["hole_radial_mask_check_dist"]
    num_in_galaxy_map = config["num_in_galaxy_map"]
    ngrid = config["ngrid"]
    dl = config["dl"]
    dr = config["dr"]
    coords_min = config["coords_min"]
    mask_mode = config["mask_mode"]
    mask = config["mask"]
    mask_resolution = config["mask_resolution"]
    min_dist = config["min_dist"]
    max_dist = config["max_dist"]
    ENABLE_SAVE_MODE = config["ENABLE_SAVE_MODE"]
    save_after = config["save_after"]
    survey_name = config["survey_name"]
    xyz_limits = config["xyz_limits"]
    batch_size = config["batch_size"]
    verbose = config["verbose"]
    print_after = config["print_after"]
    num_cpus = config["num_cpus"]
    search_grid_edge_length = config["search_grid_edge_length"]
    DEBUG_DIR = config["DEBUG_DIR"]
    ############################################################################




    if ENABLE_SAVE_MODE:
        save_after_counter = save_after
        
    if num_cpus == 1:
        print_after_time = time.time()
        
        main_task_start_time = time.time()


    ############################################################################
    # Open a UNIX-domain socket for communication to the master process.  We set
    # the timeout to be 10.0 seconds, so this worker will try notifying the 
    # master that it has results for up to 10.0 seconds, then it will loop again 
    # and check for input from the master, and if necessary wait and try to push 
    # results for 10 seconds again.  Right now the workers only exit after a 
    # b'exit' message has been received from the master.
    #---------------------------------------------------------------------------
    
    
    if num_cpus > 1:
        
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
    '''
    wcoord_buffer_length = num_galaxies*3*8 # 3 since xyz and 8 since float64
    
    wcoord_mmap_buffer = mmap.mmap(w_coord_fd, wcoord_buffer_length)
    
    w_coord = np.frombuffer(wcoord_mmap_buffer, dtype=np.float64)
    
    w_coord.shape = (num_galaxies, 3)
    '''
    ############################################################################

    
    
    ############################################################################
    # Load up galaxy_map_array from shared memory
    #---------------------------------------------------------------------------
    '''
    gma_buffer_length = num_gma_indices*8 # 3 since xyz and 8 since float64
    
    gma_mmap_buffer = mmap.mmap(gma_fd, gma_buffer_length)
    
    galaxy_map_array = np.frombuffer(gma_mmap_buffer, dtype=np.int64)
    
    galaxy_map_array.shape = (num_gma_indices,)
    '''
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
    '''
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
    '''
        
    '''
    galaxy_tree = GalaxyMap(w_coord, 
                            coords_min, 
                            search_grid_edge_length,
                            galaxy_map,
                            galaxy_map_array)
    '''
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
    '''
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
    '''
    ############################################################################


    
    ############################################################################
    # Build Cell ID generator
    #---------------------------------------------------------------------------
    cell_ID_gen = CellIDGenerator(ngrid[0],
                                  ngrid[1],
                                  ngrid[2],
                                  hole_cell_ID_dict)
    ############################################################################


    #We need an instance of this class basically for memory efficiency, the call
    # to grow_spheres uses this guy and its arrays to loop over sphere growing
    sphere_grower = SphereGrower()

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
        
    elif mask_mode in [1,2]:
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
    
    if num_cpus > 1:
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
    
    local_cells_processed = 0
    
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
        
        if num_cpus > 1:
        
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
            
            if num_write > 0:
                
                if return_array.shape[0] != num_write:
    
                    return_array = np.empty((num_write, 4), 
                                             dtype=np.float64)
                
                    
                grow_spheres(i_j_k_array[0:num_write],
                             num_write,
                             return_array,
                             galaxy_map,
                             sphere_grower,
                             mask_checker)
                
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
                
                #If we're in single threaded mode, exit now
                if num_cpus == 1:
                    exit_process = True
                #---------------------------------------------------------------
        #-----------------------------------------------------------------------

            
        #-----------------------------------------------------------------------
        # Update the master process that we have processed some number of cells, 
        # using our socket connection.  Note the actual results get written 
        # directly to the shared memmap, but the socket just updates the master 
        # with the number of new results (an integer).
        #-----------------------------------------------------------------------
        if num_cpus == 1 and have_result_to_write:
            
            local_cells_processed += num_write
            
            have_result_to_write = False
            
            if ENABLE_SAVE_MODE:
                
                save_after_counter -= num_write
                
                if save_after_counter <= 0:
                    
                    temp_result_array = np.frombuffer(result_mmap_buffer, 
                                                  dtype=np.float64)
        
                    temp_result_array.shape = (n_empty_cells, 4)
                    
                    temp_result = temp_result_array[0:local_cells_processed,:]
                    
                    next_cell_idx = ijk_start.value
                    
                    SaveCheckpointFile(survey_name+"VoidFinderCheckpoint.h5",
                                       local_cells_processed,
                                       next_cell_idx,
                                       temp_result)
                    
                    
                    print("Saving checkpoint file at: ", local_cells_processed)
                    
                    
                    #Save checkpoint
                    pass
            
                    save_after_counter = save_after
        
        
        
            #-----------------------------------------------------------------------
            # Print status updates if verbose is on
            #-----------------------------------------------------------------------
            curr_time = time.time()
            
            if (curr_time - print_after_time) > print_after:
            
                print('Processed', num_cells_processed, 
                      'cells of', n_empty_cells, " cells", 
                      str(round(curr_time - main_task_start_time, 2)), 
                      flush=True)
                
                print_after_time = curr_time
        
        
        if num_cpus > 1:
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
    if num_cpus > 1:
        worker_socket.close()
        
    hole_cell_ID_dict.close()
    
    galaxy_map.close()
    
    if verbose > 1:
        print("WORKER EXITING GRACEFULLY", worker_idx, flush=True)
    ############################################################################

    
    return None




def find_next_prime_DEPRECATED(threshold_value):
    """
    replaced by cythonized version from _voidfinder_cython_find_next just keeping
    this function here until new one is tested
    
    
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
    
    
    
    
    
    
    
    
    
    
    
    