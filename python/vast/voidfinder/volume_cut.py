#imports 

import numpy as np
import os
import mmap
import struct
import socket
import select
import tempfile
import multiprocessing
from psutil import cpu_count
from astropy.table import Table


from multiprocessing import Process, Value
from ctypes import c_int64


from .voidfinder_functions import in_mask#, not_in_mask
from .hole_combine import spherical_cap_volume
from ._voidfinder_cython_find_next import not_in_mask as nim_cython
from ._vol_cut_cython import _check_holes_mask_overlap#, _check_holes_mask_overlap_2
from ._voidfinder import process_message_buffer

import time


# function to find which spheres stick out of the mask
def max_range_check(spheres_table, direction, sign, survey_mask, mask_resolution, r_limits):
    '''
    Given the list of potential hole locations and their radii in spheres_table,
    and an axes x,y,z and direction +/-, add the radii of each hole to the hole
    location and check if that location is within the mask.
    
    Returns a boolean array of length N where True indicates the location is valid.
    '''

    #print("Max Range Check", direction, sign, "hole_table ID: ", id(spheres_table))
    #print(spheres_table['x'][0])



    if sign == '+':
       spheres_table[direction] += spheres_table['radius']
    else:
       spheres_table[direction] -= spheres_table['radius']
       
       
       
    #print(spheres_table['x'][0])
    #print(spheres_table)

    boolean = in_mask(spheres_table, survey_mask, mask_resolution, r_limits)

    return boolean





def check_coordinates(coord, direction, sign, survey_mask, mask_resolution, r_limits):

    dr = 0
    check_coord = coord
    #mask_check = True
    mask_check2 = False
    #mask_check3 = False
    
    #print(id(check_coord), id(coord))
    
    np_check_coord = np.empty((1,3), dtype=np.float64)
    np_check_coord[0,0] = coord['x']
    np_check_coord[0,1] = coord['y']
    np_check_coord[0,2] = coord['z']
    
    if direction == 'x':
        np_dir = 0
    elif direction == 'y':
        np_dir = 1
    elif direction == 'z':
        np_dir = 2
    
    #out_log = open("VF_DEBUG_volume_cut.txt", 'a')

    #while dr < coord['radius'] and mask_check:
    while dr < coord['radius'] and not mask_check2:

        dr += 1

        if sign == '+':
        #    check_coord[direction] = coord[direction] + dr
            np_check_coord[0,np_dir] = np_check_coord[0,np_dir] + dr
        else:
        #    check_coord[direction] = coord[direction] - dr
            np_check_coord[0,np_dir] = np_check_coord[0,np_dir] - dr

        #mask_check = in_mask(check_coord, survey_mask, mask_resolution, r_limits)
        
        mask_check2 = nim_cython(np_check_coord, survey_mask, mask_resolution, r_limits[0], r_limits[1])
        
        #mask_check3 = not_in_mask(np_check_coord, survey_mask, mask_resolution, r_limits[0], r_limits[1])
        
        #if mask_check == mask_check3: # or \
        #   mask_check != mask_check3 or \
        #if mask_check2 != mask_check3:
            #out_log.write(str(check_coord)+"\n")
            #out_log.write(str(np_check_coord)+","+str(mask_check)+","+str(mask_check2)+","+str(mask_check3)+"\n")
            
    #out_log.close()
        

    height_i = check_coord['radius'] - dr
    cap_volume_i = spherical_cap_volume(check_coord['radius'], height_i)
    sphere_volume = np.pi*(4/3)*(check_coord['radius']**3)
    
    return cap_volume_i, sphere_volume





def volume_cut(hole_table, survey_mask, mask_resolution, r_limits):
    
    #print("Vol cut hole_table ID: ", id(hole_table))
    #print(hole_table['x'][0])
    
    
    
    # xpos, xneg, etc are True when the hole center + hole_radius in that direction
    # is within the mask
    xpos = max_range_check(Table(hole_table), 'x', '+', survey_mask, mask_resolution, r_limits)
    xneg = max_range_check(Table(hole_table), 'x', '-', survey_mask, mask_resolution, r_limits)

    ypos = max_range_check(Table(hole_table), 'y', '+', survey_mask, mask_resolution, r_limits)
    yneg = max_range_check(Table(hole_table), 'y', '-', survey_mask, mask_resolution, r_limits)

    zpos = max_range_check(Table(hole_table), 'z', '+', survey_mask, mask_resolution, r_limits)
    zneg = max_range_check(Table(hole_table), 'z', '-', survey_mask, mask_resolution, r_limits)


    comb_bool = np.logical_and.reduce((xpos, xneg, ypos, yneg, zpos, zneg))
    
    
    
    #print("Comb bool: ", np.sum(comb_bool))
    
    

    false_indices = np.where(comb_bool == False)

    out_spheres_indices = []

    for i in false_indices[0]:

        not_removed = True

        coord = hole_table[i]

        # Check x-direction 

        if not xpos[i]:

            cap_volume, sphere_volume = check_coordinates(Table(coord), 'x', '+', survey_mask, mask_resolution, r_limits)

            if cap_volume > 0.1*sphere_volume:
                out_spheres_indices.append(i)
                not_removed = False

        elif xneg[i] == False and not_removed:

            cap_volume, sphere_volume = check_coordinates(Table(coord), 'x', '-', survey_mask, mask_resolution, r_limits)

            if cap_volume > 0.1*sphere_volume:
                out_spheres_indices.append(i)
                not_removed = False

        # Check y-direction

        if ypos[i] == False and not_removed:

            cap_volume, sphere_volume = check_coordinates(Table(coord), 'y', '+', survey_mask, mask_resolution, r_limits)

            if cap_volume > 0.1*sphere_volume:
                out_spheres_indices.append(i)
                not_removed = False


        elif yneg[i] == False and not_removed:

            cap_volume, sphere_volume = check_coordinates(Table(coord), 'y', '-', survey_mask, mask_resolution, r_limits)

            if cap_volume > 0.1*sphere_volume:
                out_spheres_indices.append(i)
                not_removed = False


        # Check z-direction

        if zpos[i] == False and not_removed:
            cap_volume, sphere_volume = check_coordinates(Table(coord), 'z', '+', survey_mask, mask_resolution, r_limits)

            if cap_volume > 0.1*sphere_volume:
                out_spheres_indices.append(i)
                not_removed = False

        elif zneg[i] == False and not_removed:

            cap_volume, sphere_volume = check_coordinates(Table(coord), 'z', '-', survey_mask, mask_resolution, r_limits)

            if cap_volume > 0.1*sphere_volume:
                out_spheres_indices.append(i)
                not_removed = False
    
    out_spheres_indices = np.unique(out_spheres_indices)


    if len(out_spheres_indices) > 0:
    
        hole_table.remove_rows(out_spheres_indices)

    return hole_table





def check_hole_bounds(x_y_z_r_array, 
                      mask_checker,
                      cut_pct=0.1,
                      pts_per_unit_volume=3,
                      num_surf_pts=20,
                      num_cpus=1,
                      verbose=0):
    """
    Remove holes from the output of _hole_finder() whose volume falls outside of 
    the mask by X % or more.  
    
    This is accomplished by a 2-phase approach.  First, N points are distributed 
    on the surface of each sphere, and those N points are checked against the 
    mask.  If any of those N points fall outside the mask, the percentage of the
    volume of the sphere which falls outside the mask is calculated by using a
    monte-carlo-esque method whereby the hole in question is filled with points
    corresponding to some minimum density, and each of those points is checked.
    The percentage of volume outside the mask is then approximated as the 
    percentage of those points which fall outside the mask.
    

    Parameters
    ==========
    
    x_y_z_r_array : numpy.ndarray of shape (N,4)
        x,y,z locations of the holes, and radius, in that order
        
    mask_checker : 
        
    r_limits : 2-tuple (min_r, max_r)
        min and max radius limits of the survey
        
    cut_pct : float in [0,1)
        if this fraction of a hole volume overlaps with the mask, discard that 
        hole
        
    num_surf_pts : int
        distribute this many points on the surface of each sphere and check them 
        against the mask before doing the monte-carlo volume calculation.
        
    num_cpus : int
        number of processes to use
        
        
    Returns
    =======
    
    valid_index : numpy.ndarray shape (N,)
        boolean array of length corresponding to input x_y_z_r_array
        True if hole is within bounds, False is hole falls outside
        the mask too far based on the cut_pct criteria
        
    monte_index : numpy.ndarray of shape (N,)
        boolean array - True if the current point underwent
        the additional monte-carlo analysis, and False if all the points
        on the shell were inside the mask and therefore no volume
        analysis was necessary
    """

   

    if num_cpus == 1:
        
        valid_index, monte_index = oob_cut_single(x_y_z_r_array, 
                                                  mask_checker,
                                                  cut_pct,
                                                  pts_per_unit_volume,
                                                  num_surf_pts)
        
    else:
        
        valid_index, monte_index = oob_cut_multi(x_y_z_r_array, 
                                                 mask_checker,
                                                 cut_pct,
                                                 pts_per_unit_volume,
                                                 num_surf_pts,
                                                 num_cpus,
                                                 verbose=verbose)
        
    return valid_index, monte_index
    




def build_unit_sphere_points(num_surf_pts):
    '''
    Distribute N points on a unit sphere.

    Reference algorithm "Golden Spiral" method: 
    https://stackoverflow.com/questions/9600801/evenly-distributing-n-points-on-a-sphere


    PARAMETERS
    ==========

    num_surf_pts : int
        Number of points to distribute on the surface of the sphere


    RETURNS
    =======

    unit_sphere_pts : ndarray of shape (num_surf_pts, 3)
        Cartesian coordinates of the surface points
    '''

    indices = np.arange(0, num_surf_pts, dtype=float) + 0.5
    
    phi = np.arccos(1 - 2*indices/num_surf_pts)
    
    theta = np.pi * (1 + 5**0.5) * indices
    
    x = np.cos(theta) * np.sin(phi)
    y = np.sin(theta) * np.sin(phi)
    z = np.cos(phi)
    
    unit_sphere_pts = np.empty((num_surf_pts, 3), dtype=np.float64)
    unit_sphere_pts[:,0] = x
    unit_sphere_pts[:,1] = y
    unit_sphere_pts[:,2] = z

    return unit_sphere_pts





def generate_mesh(radius, pts_per_unit_volume):
    '''
    Generate a mesh of constant density such that a sphere will fit in this 
    mesh.

    Cut the extraneous points, and sort all of the points in order of smallest 
    radius to largest radius, so that when we iterate later for the smaller 
    holes we can stop early at the largest necessary radius - the cythonized 
    code critically depends on this sort.


    PARAMETERS
    ==========

    radius : float
        Radius of sphere around which to define the mesh grid.

    pts_per_unit_volume : int
        Number of points per unit volume in the mesh grid.


    RETURNS
    =======

    mesh_points : ndarray of shape (N,3)
        Cartesian coordinates of all points in mesh

    mesh_points_radii : ndarray of shape (N,)
        Distance from center to each point in mesh_points
    '''

    gen_radius = radius*1.05 # add a bit of margin for the mesh
    
    step = 1.0/np.power(pts_per_unit_volume, 0.33)
    
    mesh_pts = np.arange(-1.0*gen_radius, gen_radius, step)
    
    n_pts = mesh_pts.shape[0]
    
    mesh_x, mesh_y, mesh_z = np.meshgrid(mesh_pts, mesh_pts, mesh_pts)
    
    mesh_points = np.concatenate((mesh_x.ravel().reshape(n_pts**3, 1),
                                  mesh_y.ravel().reshape(n_pts**3, 1),
                                  mesh_z.ravel().reshape(n_pts**3, 1)), axis=1)
    
    mesh_point_radii = np.linalg.norm(mesh_points, axis=1)
    
    keep_idx = mesh_point_radii < radius
    
    mesh_points = mesh_points[keep_idx]
    
    mesh_point_radii = mesh_point_radii[keep_idx]
    
    sort_order = mesh_point_radii.argsort()
    
    mesh_points = mesh_points[sort_order]
    
    mesh_points_radii = mesh_point_radii[sort_order]

    return mesh_points, mesh_points_radii




        
def oob_cut_single(x_y_z_r_array, 
                   mask_checker,
                   cut_pct,
                   pts_per_unit_volume,
                   num_surf_pts):
    """
    Out-Of-Bounds cut single threaded version.
    """

    valid_index = np.ones(x_y_z_r_array.shape[0], dtype=np.uint8)
    
    monte_index = np.zeros(x_y_z_r_array.shape[0], dtype=np.uint8)
    
    ############################################################################
    # Distrubute N points on a unit sphere
    #---------------------------------------------------------------------------
    unit_sphere_pts = build_unit_sphere_points(num_surf_pts)
    ############################################################################

    
    
    ############################################################################
    # Find the largest radius hole in the results, and generate a mesh of
    # constant density such that the largest hole will fit in this mesh
    #---------------------------------------------------------------------------
    mesh_points, mesh_points_radii = generate_mesh(x_y_z_r_array[:,3].max(), 
                                                   pts_per_unit_volume)
    ############################################################################


    
    ############################################################################
    # Iterate through our holes
    #---------------------------------------------------------------------------
    _check_holes_mask_overlap(x_y_z_r_array,
    #_check_holes_mask_overlap_2(x_y_z_r_array,
                              mask_checker,
                              unit_sphere_pts,
                              mesh_points,
                              mesh_points_radii,
                              cut_pct,
                              valid_index,
                              monte_index)
    ############################################################################
    
    
    '''
    for idx, curr_hole in enumerate(x_y_z_r_array):
        
        #if idx%100 == 0:
        #    print(idx)
        
        curr_hole_position = curr_hole[0:3]
        
        curr_hole_radius = curr_hole[3]
        
        ########################################################################
        # First, check the shell points to see if we need to do the monte carlo
        # volume
        ########################################################################
        curr_sphere_pts = curr_hole_radius*unit_sphere_pts + curr_hole_position
        
        require_monte_carlo = False
        
        for curr_sphere_edge_pt in curr_sphere_pts:
            
            not_in_mask = nim_cython(curr_sphere_edge_pt.reshape(1,3), mask, mask_resolution, r_limits[0], r_limits[1])

            if not_in_mask:
                
                require_monte_carlo = True
                
                break

        ########################################################################
        # Do the monte carlo if any of the shell points failed
        ########################################################################
        if require_monte_carlo:
            
            #print("REQ MONT")
            monte_index[idx] = True
            
            total_checked_pts = 0
            
            total_outside_mask = 0
            
            for jdx, (mesh_pt, mesh_pt_radius) in enumerate(zip(mesh_points, mesh_points_radii)):
                
                if mesh_pt_radius > curr_hole_radius:
                    
                    break
                
                check_pt = curr_hole_position + mesh_pt
                
                not_in_mask = nim_cython(check_pt.reshape(1,3), mask, mask_resolution, r_limits[0], r_limits[1])

                if not_in_mask:
                    
                    total_outside_mask += 1
                    
                total_checked_pts += 1
                
            vol_pct_outside = float(total_outside_mask)/float(total_checked_pts)
            
            if vol_pct_outside > cut_pct:
                
                valid_index[idx] = False
            
        else:
            #do nothing, the hole is valid
            pass
    '''

    return valid_index.astype(np.bool_), monte_index.astype(np.bool_)




def oob_cut_multi(x_y_z_r_array, 
                  mask_checker,
                  cut_pct,
                  pts_per_unit_volume,
                  num_surf_pts,
                  num_cpus,
                  batch_size=1000,
                  verbose=0,
                  print_after=5.0,
                  SOCKET_PATH="/tmp/voidfinder2.sock",
                  RESOURCE_DIR="/dev/shm"):
    """
    Out-Of-Bounds cut multi processed version.
    """

    num_holes = x_y_z_r_array.shape[0]

    valid_index = np.ones(num_holes, dtype=np.uint8)
    
    monte_index = np.zeros(num_holes, dtype=np.uint8)
    
    ############################################################################
    # Distrubute N points on a unit sphere
    #---------------------------------------------------------------------------
    unit_sphere_pts = build_unit_sphere_points(num_surf_pts)
    ############################################################################
    
    
    ############################################################################
    # Find the largest radius hole in the results, and generate a mesh of
    # constant density such that the largest hole will fit in this mesh
    #---------------------------------------------------------------------------
    mesh_points, mesh_points_radii = generate_mesh(x_y_z_r_array[:,3].max(), 
                                                   pts_per_unit_volume)

    mesh_points = mesh_points.astype(np.float64)
    mesh_points_radii = mesh_points_radii.astype(np.float64)
    
    num_mesh_points = mesh_points.shape[0]
    ############################################################################
    

    ############################################################################
    # If /dev/shm is not available, use /tmp as the shared resource filesystem
    # location instead.  Since on Linux /dev/shm is guaranteed to be a mounted
    # RAMdisk, I don't know if /tmp will be as fast or not, probably depends on
    # kernel settings.
    #---------------------------------------------------------------------------
    if not os.path.isdir(RESOURCE_DIR):
        
        print("WARNING: RESOURCE DIR ", RESOURCE_DIR, "does not exist.  Falling back to /tmp but could be slow", flush=True)
        
        RESOURCE_DIR = "/tmp"
    ############################################################################
        
    
    ############################################################################
    # Start by converting the num_cpus argument into the real value we will use
    # by making sure it's reasonable, or if it was None use the max val 
    # available.
    #---------------------------------------------------------------------------
    if num_cpus is None:
          
        num_cpus = cpu_count(logical=False)
        
    if verbose > 0:
        
        print("Running hole cut in multi-process mode,", str(num_cpus), "cpus", 
              flush=True)
    ############################################################################
    
    
    ############################################################################
    # Set up memmap for x_y_z_r_array
    #---------------------------------------------------------------------------
    xyzr_fd, XYZR_BUFFER_PATH = tempfile.mkstemp(prefix="voidfinder", 
                                                 dir=RESOURCE_DIR, 
                                                 text=False)
    
    if verbose > 1:
        
        print("XYZR MEMMAP PATH:", XYZR_BUFFER_PATH, xyzr_fd, flush=True)
    
    xyzr_buffer_length = num_holes*4*8 # n by 4 by 8 per float64
    
    os.ftruncate(xyzr_fd, xyzr_buffer_length)
    
    xyzr_buffer = mmap.mmap(xyzr_fd, xyzr_buffer_length)
    
    xyzr_buffer.write(x_y_z_r_array.tobytes())
    
    del x_y_z_r_array
    
    x_y_z_r_array = np.frombuffer(xyzr_buffer, dtype=np.float64)
    
    x_y_z_r_array.shape = (num_holes,4)
    
    os.unlink(XYZR_BUFFER_PATH)
    ############################################################################
    
    
    ############################################################################
    # Set up memmap for valid_idx
    #---------------------------------------------------------------------------
    valid_idx_fd, VALID_IDX_BUFFER_PATH = tempfile.mkstemp(prefix="voidfinder", 
                                                           dir=RESOURCE_DIR, 
                                                           text=False)
    
    if verbose > 1:
        
        print("VALID_IDX MEMMAP PATH:", VALID_IDX_BUFFER_PATH, valid_idx_fd, 
              flush=True)
    
    valid_idx_buffer_length = num_holes*1 # 1 per uint8
    
    os.ftruncate(valid_idx_fd, valid_idx_buffer_length)
    
    valid_idx_buffer = mmap.mmap(valid_idx_fd, valid_idx_buffer_length)
    
    valid_idx_buffer.write(valid_index.tobytes())
    
    del valid_index
    
    valid_index = np.frombuffer(valid_idx_buffer, dtype=np.uint8)
    
    valid_index.shape = (num_holes,)
    
    os.unlink(VALID_IDX_BUFFER_PATH)
    ############################################################################

    
    ############################################################################
    # Set up memmap for monte_idx
    #---------------------------------------------------------------------------
    monte_idx_fd, MONTE_IDX_BUFFER_PATH = tempfile.mkstemp(prefix="voidfinder", 
                                                           dir=RESOURCE_DIR, 
                                                           text=False)
    
    if verbose > 1:
        
        print("MONTE_IDX MEMMAP PATH:", MONTE_IDX_BUFFER_PATH, monte_idx_fd, 
              flush=True)
    
    monte_idx_buffer_length = num_holes*1 # 1 per uint8
    
    os.ftruncate(monte_idx_fd, monte_idx_buffer_length)
    
    monte_idx_buffer = mmap.mmap(monte_idx_fd, monte_idx_buffer_length)
    
    monte_idx_buffer.write(monte_index.tobytes())
    
    del monte_index
    
    monte_index = np.frombuffer(monte_idx_buffer, dtype=np.uint8)
    
    monte_index.shape = (num_holes,)
    
    os.unlink(MONTE_IDX_BUFFER_PATH)
    ############################################################################

    
    ############################################################################
    # Set up memmap for unit_sphere_pts
    #---------------------------------------------------------------------------
    unit_sphere_fd, UNIT_SHELL_BUFFER_PATH = tempfile.mkstemp(prefix="voidfinder", 
                                                              dir=RESOURCE_DIR, 
                                                              text=False)
    
    if verbose > 1:
        
        print("UNIT SHELL MEMMAP PATH:", UNIT_SHELL_BUFFER_PATH, unit_sphere_fd, 
              flush=True)
    
    unit_sphere_buffer_length = num_surf_pts*3*8 # n by 3 by 8 per float64
    
    os.ftruncate(unit_sphere_fd, unit_sphere_buffer_length)
    
    unit_sphere_buffer = mmap.mmap(unit_sphere_fd, unit_sphere_buffer_length)
    
    unit_sphere_buffer.write(unit_sphere_pts.tobytes())
    
    del unit_sphere_pts
    
    unit_sphere_pts = np.frombuffer(unit_sphere_buffer, dtype=np.float64)
    
    unit_sphere_pts.shape = (num_surf_pts, 3)
    
    os.unlink(UNIT_SHELL_BUFFER_PATH)
    ############################################################################

    
    ############################################################################
    # Set up memmap for mesh_pts
    #---------------------------------------------------------------------------
    mesh_pts_fd, MESH_PTS_BUFFER_PATH = tempfile.mkstemp(prefix="voidfinder", 
                                                         dir=RESOURCE_DIR, 
                                                         text=False)
    
    if verbose > 1:
        
        print("MESH PTS MEMMAP PATH:", MESH_PTS_BUFFER_PATH, mesh_pts_fd, 
              flush=True)
    
    mesh_pts_buffer_length = num_mesh_points*3*8 # n by 3 by 8 per float64
    
    os.ftruncate(mesh_pts_fd, mesh_pts_buffer_length)
    
    mesh_pts_buffer = mmap.mmap(mesh_pts_fd, mesh_pts_buffer_length)
    
    mesh_pts_buffer.write(mesh_points.tobytes())
    
    del mesh_points
    
    mesh_points = np.frombuffer(mesh_pts_buffer, dtype=np.float64)
    
    mesh_points.shape = (num_mesh_points, 3)
    
    os.unlink(MESH_PTS_BUFFER_PATH)
    ############################################################################

    
    ############################################################################
    # Set up memmap for mesh_points_radii
    #---------------------------------------------------------------------------
    mesh_radii_fd, MESH_RADII_BUFFER_PATH = tempfile.mkstemp(prefix="voidfinder", 
                                                             dir=RESOURCE_DIR, 
                                                             text=False)
    
    if verbose > 1:
        
        print("MESH RADII MEMMAP PATH:", MESH_RADII_BUFFER_PATH, mesh_radii_fd, 
              flush=True)
    
    mesh_radii_buffer_length = num_mesh_points*8 # n by 3 by 8 per float64
    
    os.ftruncate(mesh_radii_fd, mesh_radii_buffer_length)
    
    mesh_radii_buffer = mmap.mmap(mesh_radii_fd, mesh_radii_buffer_length)
    
    if verbose > 1:
        
        print(mesh_radii_buffer_length, len(mesh_points_radii.tobytes()), 
              flush=True)
    
    mesh_radii_buffer.write(mesh_points_radii.tobytes())
    
    del mesh_points_radii
    
    mesh_points_radii = np.frombuffer(mesh_radii_buffer, dtype=np.float64)
    
    mesh_points_radii.shape = (num_mesh_points,)
    
    os.unlink(MESH_RADII_BUFFER_PATH)
    ############################################################################
    
    
    ############################################################################
    #
    ############################################################################
    index_start = Value(c_int64, 0, lock=True)
    
    num_cells_processed = 0
    
    ############################################################################
    #
    ############################################################################
    
    config_object = {"SOCKET_PATH" : SOCKET_PATH,
                     "batch_size" : batch_size,
                     
                     "mask_checker" : mask_checker,
                     #"mask" : mask,
                     #"mask_resolution" : mask_resolution,
                     #"min_dist" : r_limits[0],
                     #"max_dist" : r_limits[1],
                     "cut_pct" : cut_pct,
                     
                     "XYZR_BUFFER_PATH" : XYZR_BUFFER_PATH,
                     "xyzr_fd" : xyzr_fd,
                     "num_holes" : num_holes,
                     "VALID_IDX_BUFFER_PATH" : VALID_IDX_BUFFER_PATH,
                     "valid_idx_fd" : valid_idx_fd,
                     "MONTE_IDX_BUFFER_PATH" : MONTE_IDX_BUFFER_PATH,
                     "monte_idx_fd" : monte_idx_fd,
                     "UNIT_SHELL_BUFFER_PATH" : UNIT_SHELL_BUFFER_PATH,
                     "unit_sphere_fd" : unit_sphere_fd,
                     "num_surf_pts" : num_surf_pts,
                     "MESH_PTS_BUFFER_PATH" : MESH_PTS_BUFFER_PATH,
                     "mesh_pts_fd" : mesh_pts_fd,
                     "num_mesh_points" : num_mesh_points,
                     "MESH_RADII_BUFFER_PATH" : MESH_RADII_BUFFER_PATH,
                     "mesh_radii_fd" : mesh_radii_fd,
                     }
    
    ############################################################################
    # Start the worker processes
    #
    # For whatever reason, OSX doesn't define the socket.SOCK_CLOEXEC constants
    # so check for that attribute on the socket module before opening the 
    # listener socket.  Not super critical, but the child processes don't need a 
    # file descriptor for the listener socket so I was trying to be clean and 
    # have it "close on exec"
    ############################################################################
    
    if hasattr(socket, "SOCK_CLOEXEC"):
        
        listener_socket = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM | socket.SOCK_CLOEXEC)
        
    else:
        
        listener_socket = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    
    listener_socket.bind(SOCKET_PATH)
    
    listener_socket.listen(num_cpus)
    
    startup_context = multiprocessing.get_context("fork")
        
    processes = []
    
    for proc_idx in range(num_cpus):
        
        #p = startup_context.Process(target=_main_hole_finder_startup, args=(proc_idx, CONFIG_PATH))
        
        p = startup_context.Process(target=_oob_cut_worker, 
                                    args=(proc_idx, 
                                          index_start, 
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
    # Make sure each worker process connects to the main socket, so we block on
    # the accept() call below until we get a connection, and make sure we get 
    # exactly num_cpus connections.
    #
    # To avoid waiting for hours and hours without getting a successful socket 
    # connection, we set the timeout to the reasonably high value of 10.0 seconds
    # (remember, 0.1 seconds is on the order of 100 million cycles for a 1GHz
    # processor), and if we don't get a connection within that time frame we're
    # going to intentionally raise a RunTimeError
    #
    # If successful, we save off references to our new worker sockets by their
    # file descriptor integer value so we can refer to them by that value using
    # select() later, then shut down and close our listener/server socket since
    # we're done with it.
    ############################################################################
    if verbose > 0:
        
        print("Attempting to connect workers for volume cut/out of bounds check", 
              flush=True)
    
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
        
    
    if not all_successful_connections:
        
        for worker_sock in worker_sockets:
                
            worker_sock.send(b"exit")
        
        print("FAILED TO CONNECT ALL WORKERS SUCCESSFULLY, EXITING")
            
        raise RunTimeError("Worker sockets failed to connect properly")
        
        
        
    
    ############################################################################
    # LOOP TO LISTEN FOR RESULTS WHILE WORKERS WORKING
    # This loop has 3 primary jobs 
    # 1). accumulate results from reading the worker sockets
    # 2). periodically print the status/results from the workers
    # 3). Save checkpoint files after every 'safe_after' results
    ############################################################################
    if verbose > 0:
        
        print_after_time = time.time()
        
        main_task_start_time = time.time()
    
    empty1 = []
    
    empty2 = []
    
    select_timeout = 2.0
    
    sent_exit_commands = False
    
    while num_active_processes > 0:
        
        ########################################################################
        # Print status updates if verbose is on
        ########################################################################
        if verbose > 0:
            
            curr_time = time.time()
            
            if (curr_time - print_after_time) > print_after:
            
                print('Processed', num_cells_processed, 
                      'holes of', num_holes, 
                      "at", str(round(curr_time-main_task_start_time,2)), 
                      flush=True)
                
                print_after_time = curr_time
            
        
            
        ########################################################################
        # Accumulate status updates from the worker sockets
        ########################################################################
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
                        
                        #if ENABLE_SAVE_MODE:
                        #    save_after_counter -= num_result
                        
                        #n_holes += num_hole
                        
                    elif message_type == 1:
                        
                        num_active_processes -= 1
                        
                    elif message_type == 2:
                        
                        num_acknowledges += 1
                        
        

    ############################################################################
    # We're done the main work! Clean up worker processes.  Block until we've
    # joined everybody so that we know everything completed correctly.
    ############################################################################
    if verbose > 0:
        
        print("Vol cut finish time: ", time.time() - main_task_start_time, 
              flush=True)
    
    if not sent_exit_commands:
        
        for idx in range(num_cpus):
            
            worker_sockets[idx].send(b"exit")
    
    for p in processes:
        
        p.join(None) #block till join
    
    ############################################################################
    # DONE
    ############################################################################
        
    return valid_index.astype(np.bool), monte_index.astype(np.bool)






def _oob_cut_worker(worker_idx, index_start, config):
    
    
    SOCKET_PATH = config["SOCKET_PATH"]
    batch_size = config["batch_size"]
    
    
    mask_checker = config["mask_checker"]
    #mask = config["mask"]
    #mask_resolution = config["mask_resolution"]
    #min_dist = config["min_dist"]
    #max_dist = config["max_dist"]
    cut_pct = config["cut_pct"]
    
    
    XYZR_BUFFER_PATH = config["XYZR_BUFFER_PATH"]
    xyzr_fd = config["xyzr_fd"]
    num_holes = config["num_holes"]
    VALID_IDX_BUFFER_PATH = config["VALID_IDX_BUFFER_PATH"]
    valid_idx_fd = config["valid_idx_fd"]
    MONTE_IDX_BUFFER_PATH = config["MONTE_IDX_BUFFER_PATH"]
    monte_idx_fd = config["monte_idx_fd"]
    UNIT_SHELL_BUFFER_PATH = config["UNIT_SHELL_BUFFER_PATH"]
    unit_sphere_fd = config["unit_sphere_fd"]
    num_surf_pts = config["num_surf_pts"]
    MESH_PTS_BUFFER_PATH = config["MESH_PTS_BUFFER_PATH"]
    mesh_pts_fd = config["mesh_pts_fd"]
    num_mesh_points = config["num_mesh_points"]
    MESH_RADII_BUFFER_PATH = config["MESH_RADII_BUFFER_PATH"]
    mesh_radii_fd = config["mesh_radii_fd"]
    
    
    
    
    ############################################################################
    # Open a UNIX-domain socket for communication to the master process.  We set
    # the timeout to be 10.0 seconds, so this worker will try notifying the 
    # master that it has results for up to 10.0 seconds, then it will loop again 
    # and check for input from the master, and if necessary wait and try to push 
    # results for 10 seconds again.  Right now the workers only exit after a 
    # b'exit' message has been received from the master.
    ############################################################################
    worker_socket = socket.socket(socket.AF_UNIX)
    
    worker_socket.settimeout(10.0)
    
    connect_start = time.time()
    
    try:
        
        worker_socket.connect(SOCKET_PATH)
        
    except Exception as E:
        
        print("WORKER", worker_idx, "UNABLE TO CONNECT, EXITING", flush=True)
        
        raise E
    
    ############################################################################
    #
    ############################################################################
    xyzr_buffer_length = num_holes*4*8 # 4 for xyzr and 8 for float64
    
    xyzr_mmap_buffer = mmap.mmap(xyzr_fd, xyzr_buffer_length)
    
    x_y_z_r_array = np.frombuffer(xyzr_mmap_buffer, dtype=np.float64)
    
    x_y_z_r_array.shape = (num_holes, 4)
    
    ############################################################################
    #
    ############################################################################
    valid_idx_buffer_length = num_holes*1 # uint8
    
    valid_idx_mmap_buffer = mmap.mmap(valid_idx_fd, valid_idx_buffer_length)
    
    valid_index = np.frombuffer(valid_idx_mmap_buffer, dtype=np.uint8)
    
    valid_index.shape = (num_holes,)
    
    ############################################################################
    #
    ############################################################################
    monte_idx_buffer_length = num_holes*1 # uint8
    
    monte_idx_mmap_buffer = mmap.mmap(monte_idx_fd, monte_idx_buffer_length)
    
    monte_index = np.frombuffer(monte_idx_mmap_buffer, dtype=np.uint8)
    
    monte_index.shape = (num_holes,)
    
    ############################################################################
    #
    ############################################################################
    unit_shell_buffer_length = num_surf_pts*3*8 # n by 3 by float64
    
    unit_shell_mmap_buffer = mmap.mmap(unit_sphere_fd, unit_shell_buffer_length)
    
    unit_sphere_pts = np.frombuffer(unit_shell_mmap_buffer, dtype=np.float64)
    
    unit_sphere_pts.shape = (num_surf_pts,3)
    
    ############################################################################
    #
    ############################################################################
    mesh_pts_buffer_length = num_mesh_points*3*8 # n by 3 by float64
    
    mesh_pts_mmap_buffer = mmap.mmap(mesh_pts_fd, mesh_pts_buffer_length)
    
    mesh_points = np.frombuffer(mesh_pts_mmap_buffer, dtype=np.float64)
    
    mesh_points.shape = (num_mesh_points,3)
    
    
    ############################################################################
    #
    ############################################################################
    mesh_radii_buffer_length = num_mesh_points*8 # n by float64
    
    mesh_radii_mmap_buffer = mmap.mmap(mesh_radii_fd, mesh_radii_buffer_length)
    
    mesh_points_radii = np.frombuffer(mesh_radii_mmap_buffer, dtype=np.float64)
    
    mesh_points_radii.shape = (num_mesh_points,)
    
    ############################################################################
    # Main Loop for the worker process begins here.
    #
    #    exit_process - flag for reading an exit command off the queue
    #
    #    document the additional below variables here please
    #
    # If this worker process has reached the end of the Cell ID generator, we 
    # want to tell the master process we're done working, and wait for an exit 
    # command, so increase the select_timeout from 0 (instant) to 2.0 seconds to 
    # allow the operating system to wake us up during that 2.0 second interval 
    # and avoid using unnecessary CPU
    ############################################################################
    
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
        
        #total_loops += 1
        
        ########################################################################
        # As the first part of the main loop, use the select() method to check 
        # for any messages from the master process.  It may send us an "exit" 
        # command, to tell us to terminate, a "sync" command, to tell us to stop 
        # processing momentarily while it writes out a save checkpoint, or a 
        # "resume" command to tell us that we may continue processing after a 
        # "sync"
        ########################################################################
        
        #print("Worker "+str(worker_idx)+" "+str(message_buffer), flush=True)
        
        read_socks, empty3, empty4 = select.select(worker_sockets, empty1, empty2, select_timeout)
        
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
        
        
        ########################################################################
        # Here we do the main work of VoidFinder.  We synchronize the work with 
        # the other worker processes using 2 lock-protected values, 'ijk_start' 
        # and 'write_start'.  ijk_start gives us the starting cell_ID index to 
        # generate the next batch of cell ID's at, and write_start gives us the 
        # index to write our batch of results at.  Note that we will process AT 
        # MOST 'batch_size' indexes per loop, because we use the Galaxy Map to 
        # filter out cell IDs which do not need to be checked (since they have 
        # galaxies in them they are non-empty and we won't find a hole there).  
        # Since we may process LESS than batch_size locations, when we update 
        # 'write_start' we update it with the actual number of cells which we 
        # have worked in our current batch. 
        #
        # Note if we're in 'sync' mode, we don't want to do any work since the 
        # master process is making a checkpoint file.
        ########################################################################
        if do_work and not sync:
        
            ####################################################################
            # Get the next index of the starting cell ID to process for our 
            # current batch
            ####################################################################
            index_start.acquire()
            
            start_idx = index_start.value
            
            index_start.value += batch_size
            
            index_start.release()
    
            
            ####################################################################
            # Setup the work
            ####################################################################
            
            if start_idx + batch_size <= num_holes:
                num_cells_to_process = batch_size
                
            elif start_idx + batch_size > num_holes:
                
                num_cells_to_process = max(0, num_holes - start_idx)
            
            if num_cells_to_process > 0:
                
                end_idx = start_idx + num_cells_to_process
                
                _check_holes_mask_overlap(x_y_z_r_array[start_idx:end_idx],
                #_check_holes_mask_overlap_2(x_y_z_r_array[start_idx:end_idx],
                                          mask_checker,
                                          unit_sphere_pts,
                                          mesh_points,
                                          mesh_points_radii,
                                          cut_pct,
                                          valid_index[start_idx:end_idx],
                                          monte_index[start_idx:end_idx])
                
                
                have_result_to_write = True
            
            ####################################################################
            # If the cell_ID_generator ever returns '0', that means we've 
            # reached the end of the whole search grid, so this worker can 
            # notify the master that it is done working
            ####################################################################
            else:
                
                no_cells_left_to_process = True
            
        ########################################################################
        # Update the master process that we have processed some number of cells, 
        # using our socket connection.  Note the actual results get written 
        # directly to the shared memmap, but the socket just updates the master 
        # with the number of new results (an integer)
        ########################################################################  
        if have_result_to_write:   
            
            #n_hole = np.sum(np.logical_not(np.isnan(return_array[:,0])), axis=None, dtype=np.int64)
            
            out_msg = b""
            out_msg += struct.pack("b", 2) #1 byte - number of 8 byte fields
            out_msg += struct.pack("=q", 0) #8 byte field - message type 0
            out_msg += struct.pack("=q", num_cells_to_process) #8 byte field - payload for num-write
            
            try:
                worker_socket.send(out_msg)
            except:
                do_work = False
            else:
                do_work = True
                have_result_to_write = False
            
        ########################################################################
        # If we're done working (cell ID generator reached the end/returned 0), 
        # notify the master process that this worker is going into a "wait for 
        # exit" state where we just sleep and check the input socket for the 
        # b'exit' message
        #########################################################################
        if no_cells_left_to_process:
            
            if not sent_deactivation:
            
                out_msg = b""
                out_msg += struct.pack("b", 1) #1 byte - number of 8 byte fields
                out_msg += struct.pack("=q", 1) #8 byte field - message type 1 (no payload)
                
                worker_socket.send(out_msg)
                
                sent_deactivation = True
                
                select_timeout = 2.0
            
        ########################################################################
        # If the master process wants to save a checkpoint, it needs the workers 
        # to sync up.  It sends a b'sync' message, and then it waits for all the 
        # workers to acknowledge that they have received the 'sync', so here we 
        # send that acknowledgement.  After we've received the sync, we just 
        # want to sleep and check the socket for a b'resume' message.
        ########################################################################
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
                
    ############################################################################
    # We're all done!  Close the socket and any other resources, and finally 
    # return.
    ############################################################################
    worker_socket.close()
    
    #print("WORKER EXITING GRACEFULLY "+str(worker_idx), flush=True)
    
    return None
    
    
    
    
    
    
    
