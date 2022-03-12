#cython: language_level=3

from __future__ import print_function

cimport cython

import numpy as np

cimport numpy as np

np.import_array()  # required in order to use C-API


from .typedefs cimport DTYPE_CP128_t, \
                      DTYPE_CP64_t, \
                      DTYPE_F64_t, \
                      DTYPE_F32_t, \
                      DTYPE_B_t, \
                      ITYPE_t, \
                      DTYPE_INT32_t, \
                      DTYPE_INT64_t, \
                      DTYPE_INT8_t, \
                      CELL_ID_t

from numpy.math cimport NAN, INFINITY

from libc.math cimport fabs, sqrt, asin, atan, ceil#, exp, pow, cos, sin, asin

from ._voidfinder import find_next_prime

from ._voidfinder_cython_find_next cimport GalaxyMapCustomDict, \
                                           Cell_ID_Memory, \
                                           _gen_cube, \
                                           OffsetNumPair



cdef DTYPE_F64_t pi = np.pi

import time



@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cpdef np.ndarray remove_duplicates_2(DTYPE_F64_t[:,:] x_y_z_r_array,
                                     DTYPE_F64_t tolerance):
    """
    Description
    ===========
    
    This function is not intuitive - it serves the unique purpose of removing
    only a small subset of duplicate hole candidates.  It is assumes the
    x_y_z_r_array is sorted such that the largest hole is at index 0, and we
    compare pairwise adjacent holes to remove duplicates only from that 
    selection.
    
    Note that we do not check for holes which are completely contained within
    other holes, because the main voidfinder will continue growing any sphere
    until it is bounded by galaxies - so the only way for a hole to be
    completely contained within another hole is to have the exact same bounding
    galaxies - so the centers will have a separation distance of less than
    the provided tolerance.
    
    Parameters
    ==========
    
    x-y_z_r_array : numpy.ndarray of shape (N,4)
    
    tolerance : float
        absolute tolerance (ex: Mpc/h) beyond which holes are 
        considered unique
    
    """
    
    cdef ITYPE_t num_holes = x_y_z_r_array.shape[0]
    
    cdef ITYPE_t idx, last_idx
    
    cdef DTYPE_F64_t[3] diff
    cdef DTYPE_F64_t curr_radius, last_radius, separation
    
    out_index = np.zeros(num_holes, dtype=np.uint8)
    
    cdef DTYPE_B_t[:] out_index_memview = out_index
    
    out_index[0] = True
    
    last_idx = 0


    for idx in range(1, num_holes):
        
        curr_radius = x_y_z_r_array[idx, 3]
        
        last_radius = x_y_z_r_array[last_idx, 3]
        
        diff[0] = x_y_z_r_array[idx, 0] - x_y_z_r_array[last_idx, 0]
        diff[1] = x_y_z_r_array[idx, 1] - x_y_z_r_array[last_idx, 1]
        diff[2] = x_y_z_r_array[idx, 2] - x_y_z_r_array[last_idx, 2]
        
        separation = sqrt(diff[0]*diff[0] + diff[1]*diff[1] + diff[2]*diff[2])
        
        if (separation > tolerance) or (last_radius - curr_radius > tolerance):
            
            out_index_memview[idx] = 1
            
            last_idx = idx
            
    return out_index.astype(np.bool)








@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef DTYPE_F64_t cap_height(DTYPE_F64_t R, 
                            DTYPE_F64_t r, 
                            DTYPE_F64_t d):
    '''Calculate the height of a spherical cap.

    Parameters
    __________

    R : radius of sphere

    r : radius of other sphere

    d : distance between sphere centers


    Output
    ______

    h : height of cap
    '''
    '''for elem in d:
        if np.isnan(elem) or elem == 0:
            print(d)
            print(':(')'''
            
    cdef DTYPE_F64_t h
    
    h = (r - R + d)*(r + R - d)/(2*d)
    
    return h



@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef DTYPE_F64_t spherical_cap_volume(DTYPE_F64_t radius, 
                                      DTYPE_F64_t height):
    '''Calculate the volume of a spherical cap'''

    cdef DTYPE_F64_t volume
    
    volume = pi*(height**2)*(3*radius - height)/3.

    return volume



@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cpdef np.ndarray find_maximals_2(DTYPE_F64_t[:,:] x_y_z_r_array,
                                 DTYPE_F64_t frac,
                                 DTYPE_F64_t min_radius):
                                               
    """
    
    DEPRECATED BY find_maximals_3 REMOVE AFTER TESTING
    
    Description
    ===========
    
    Build an array of the indices which correspond to maximal
    holes in the input x_y_z_r_array.  Since x_y_z_r_array is sorted by radius,
    we can assume the hole at index 0 is the first maximal hole.
    
    Parameters
    =========
    
    x_y_z_r_array : ndarray of shape (N,4)
        must be sorted such that index 0 corresponds to the largest hole radius.
        
    frac : float in [0, 1)
        any 2 potential maximals which overlap
        by more than this percentage means the smaller hole will not be
        considered a maximal
    
    min_radius : float
    
    
    Returns
    =======
    
    out_maximals_index : ndarray of shape (N,)
    """
                                               
    ################################################################################
    # Create the output array and add a memview for fast access to it.  Since
    # x_y_z_r_array is sorted by size we set the first element to be a maximal
    ################################################################################
    out_maximals_index = np.empty(x_y_z_r_array.shape[0], dtype=np.int64)
    
    out_maximals_index[0] = 0
                                             
    cdef DTYPE_INT64_t[:] out_index_memview = out_maximals_index
    
    ################################################################################
    # Set up some memory, again we have at least 1 maximal to start with
    ################################################################################
    cdef DTYPE_F64_t[3] diffs
    cdef DTYPE_F64_t separation, curr_radius, curr_maximal_radius
    cdef DTYPE_F64_t curr_cap_height, maximal_cap_height, overlap_volume, curr_sphere_volume_thresh
    cdef ITYPE_t N_voids = 1
    cdef ITYPE_t idx, jdx, maximal_idx, num_holes
    cdef DTYPE_B_t is_maximal
    
    
    ################################################################################
    # Iterate through all our holes, starting from index 1 since we assume
    # index 0 is already a maximal due to sorting.  Check to ensure each
    # hole radius is larger than the minimum size required to be a maximal, and if
    # it is a candidate, compare it against every existing maximal for overlap, and
    # if it doesn't overlap with any other existing maximals, then it too gets
    # to be a maximal and is added to the output index. 
    ################################################################################
    num_holes = x_y_z_r_array.shape[0]
    
    for idx in range(1, num_holes):
        
        if x_y_z_r_array[idx,3] <= min_radius:
            
            continue
        
        is_maximal = True
        
        curr_radius = x_y_z_r_array[idx,3]
        
        curr_sphere_volume_thresh = frac*(4.0/3.0)*np.pi*curr_radius*curr_radius*curr_radius
        
        ################################################################################
        # If the current sphere is large enough to be a maximal candidate, compare 
        # it against all the previously marked maximals.  When we compare, we basically
        # have 3 cases - the centers are very far apart (no overlap)
        ################################################################################
        for jdx in range(N_voids):
            
            maximal_idx = out_maximals_index[jdx]
            
            diffs[0] = x_y_z_r_array[idx,0] - x_y_z_r_array[maximal_idx,0]
            diffs[1] = x_y_z_r_array[idx,1] - x_y_z_r_array[maximal_idx,1]
            diffs[2] = x_y_z_r_array[idx,2] - x_y_z_r_array[maximal_idx,2]
            
            separation = sqrt(diffs[0]*diffs[0] + diffs[1]*diffs[1] + diffs[2]*diffs[2])
            
            curr_maximal_radius = x_y_z_r_array[maximal_idx,3]
            
            if curr_maximal_radius - curr_radius >= separation:
                #hole is fully contained in another hole, it is not a maximal
                is_maximal = False
                break
            
            elif curr_maximal_radius + curr_radius >= separation:
                
                curr_cap_height = cap_height(curr_radius, curr_maximal_radius, separation)
                
                maximal_cap_height = cap_height(curr_maximal_radius, curr_radius, separation)
            
                overlap_volume = spherical_cap_volume(curr_radius, curr_cap_height) + spherical_cap_volume(curr_maximal_radius, maximal_cap_height)
            
                if overlap_volume > curr_sphere_volume_thresh:
                    
                    is_maximal = False
                    
                    break
                
                else:
                    # Didn't overlap too much with this maximal, but still
                    # need to check all the others so keep going
                    pass
            
            else:
                # Didn't overlap with this maximal at all, but still need to check
                # all the others
                pass
            
        if is_maximal:
            
            #out_xyzr_memview[N_voids,:] = x_y_z_r_array[idx,:]
            out_maximals_index[N_voids] = idx
            N_voids += 1
            
            
    return out_maximals_index[0:N_voids]
        










@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def find_maximals_3(DTYPE_F64_t[:,:] x_y_z_r_array,
                                 DTYPE_F64_t frac,
                                 DTYPE_F64_t min_radius):
                                               
    """
    Description
    ===========
    
    Build an array of the indices which correspond to maximal
    holes in the input x_y_z_r_array.  Since x_y_z_r_array is sorted by radius,
    we can assume the hole at index 0 is the first maximal hole.
    
    
    Use a 2-pass scheme.  First, just eliminate holes which are too small.
    
    
    Parameters
    =========
    
    x_y_z_r_array : ndarray of shape (N,4)
        must be sorted such that index 0 corresponds to the largest hole radius.
        
    frac : float in [0, 1)
        any 2 potential maximals which overlap
        by more than this percentage means the smaller hole will not be
        considered a maximal
    
    min_radius : float
    
    
    Returns
    =======
    
    out_maximals_index : ndarray of shape (N,)
    """
                                               
    ################################################################################
    # Create the output array and add a memview for fast access to it, and an array
    # for the candidate maximals who are big enough holes
    ################################################################################
    out_maximals_index = np.empty(x_y_z_r_array.shape[0], dtype=np.int64)
                                             
    cdef DTYPE_INT64_t[:] out_index_memview = out_maximals_index
    
    candidate_maximals_index = np.empty(x_y_z_r_array.shape[0], dtype=np.int64)
    
    cdef DTYPE_INT64_t[:] candidate_index_memview = candidate_maximals_index
    
    
    
    
    ################################################################################
    # Set up some memory
    ################################################################################
    cdef DTYPE_F64_t[3] diffs
    cdef DTYPE_F64_t separation, curr_radius, curr_maximal_radius
    cdef DTYPE_F64_t largest_radius, twice_largest_radius
    cdef DTYPE_F64_t curr_cap_height, maximal_cap_height, overlap_volume, curr_sphere_volume_thresh
    cdef ITYPE_t N_voids = 0
    cdef ITYPE_t idx, jdx, kdx, maximal_idx, num_holes, num_candidates, cell_ID_idx
    cdef DTYPE_B_t is_maximal
    
    
    cdef DTYPE_INT64_t n_grid_x, n_grid_y, n_grid_z
    cdef CELL_ID_t grid_i, grid_j, grid_k
    
    
    cdef DTYPE_F64_t min_x = x_y_z_r_array[0,0]
    cdef DTYPE_F64_t min_y = x_y_z_r_array[0,1]
    cdef DTYPE_F64_t min_z = x_y_z_r_array[0,2]
    
    cdef DTYPE_F64_t max_x = x_y_z_r_array[0,0]
    cdef DTYPE_F64_t max_y = x_y_z_r_array[0,1]
    cdef DTYPE_F64_t max_z = x_y_z_r_array[0,2]
    
    
    
    
    
    cdef OffsetNumPair curr_offset_num_pair
    
    cdef DTYPE_INT64_t num_cell_IDs
    
    cdef CELL_ID_t id1, id2, id3
    
    cdef DTYPE_INT64_t offset, num_elements
    
    
    
    ################################################################################
    # Take a first pass through all the holes and filter out all the holes which
    # are too small.  Also grab the largest radius for help building an index 
    # structure to help optimize maximal comparisons, and find the coordinates
    # of the minimum corner of the candidates to help with that structure
    ################################################################################
    num_holes = x_y_z_r_array.shape[0]
    
    num_candidates = 0
    
    largest_radius = x_y_z_r_array[0,3]
    
    for idx in range(num_holes):
        
        if x_y_z_r_array[idx,3] <= min_radius:
            
            continue
        
        else:
            
            candidate_index_memview[num_candidates] = idx
            
            num_candidates += 1
            
            if min_x > x_y_z_r_array[idx,0]:
                
                min_x = x_y_z_r_array[idx,0]
                
            if min_y > x_y_z_r_array[idx,1]:
                
                min_y = x_y_z_r_array[idx,1]
                
            if min_z > x_y_z_r_array[idx,2]:
                
                min_z = x_y_z_r_array[idx,2]
                
                
                
            if max_x < x_y_z_r_array[idx,0]:
                
                max_x = x_y_z_r_array[idx,0]
                
            if max_y < x_y_z_r_array[idx,1]:
                
                max_y = x_y_z_r_array[idx,1]
                
            if max_z < x_y_z_r_array[idx,2]:
                
                max_z = x_y_z_r_array[idx,2]
            
            
    '''
    print("num holes: ", num_holes)
    print("Num candidates: ", num_candidates)
    print("largest radius: ", largest_radius)
    print("Mins: ", min_x, min_y, min_z)
    print("Maxs: ", max_x, max_y, max_z)
    '''
            
    ################################################################################
    # Set up a cubical grid based on the galaxy positions with edge length of
    # twice the largest sphere radius to help us find maximals we need to do
    # overlap comparisons on
    ################################################################################
    min_x = min_x - largest_radius
    min_y = min_y - largest_radius
    min_z = min_z - largest_radius
    
    max_x += largest_radius
    max_y += largest_radius
    max_z += largest_radius
    
    twice_largest_radius = 2.0*largest_radius
    
    n_grid_x = <DTYPE_INT64_t>ceil((max_x - min_x)/twice_largest_radius)
    n_grid_y = <DTYPE_INT64_t>ceil((max_y - min_y)/twice_largest_radius)
    n_grid_z = <DTYPE_INT64_t>ceil((max_z - min_z)/twice_largest_radius)
    
    '''
    print("Mins: ", min_x, min_y, min_z)
    print("Maxs: ", max_x, max_y, max_z)
    print("Twice radius: ", twice_largest_radius)
    print("Grid: ", n_grid_x, n_grid_y, n_grid_z)
    '''
    
    ################################################################################
    # Now convert all the candidate galaxies into grid cell locations, using the
    # same galaxy_map technique as in the main part of voidfinder, and count how
    # many go into each bin, so we can allocate an array for dynamically updating
    # how many maximals we have when querying the grid
    ################################################################################
    
    galaxy_map = {}
    
    for jdx in range(num_candidates):
        
        idx = candidate_index_memview[jdx]
        
        grid_i = <CELL_ID_t>((x_y_z_r_array[idx,0] - min_x)/twice_largest_radius)
        grid_j = <CELL_ID_t>((x_y_z_r_array[idx,1] - min_y)/twice_largest_radius)
        grid_k = <CELL_ID_t>((x_y_z_r_array[idx,2] - min_z)/twice_largest_radius)
        
        bin_ID = (grid_i, grid_j, grid_k)
        
        if bin_ID not in galaxy_map:
            
            galaxy_map[bin_ID] = 0
        
        galaxy_map[bin_ID] += 1
        
        
    #print("Num non-empty grid cells: ", len(galaxy_map))
        
    next_prime = find_next_prime(2*num_candidates)
    
    lookup_memory = np.zeros(next_prime, dtype=[("filled_flag", np.uint8, ()), #() indicates scalar, or length 1 shape
                                                ("p", np.int16, ()),
                                                ("q", np.int16, ()),
                                                ("r", np.int16, ()),
                                                ("offset", np.int64, ()),
                                                ("num_elements", np.int64, ())])
    
    new_galaxy_map = GalaxyMapCustomDict((n_grid_x, n_grid_y, n_grid_z), 
                                         lookup_memory)
    
    
    offset = 0
    
    for curr_pqr in galaxy_map:
        
        new_galaxy_map.setitem(curr_pqr[0],
                               curr_pqr[1],
                               curr_pqr[2],
                               offset, 
                               0)
        
        num_elements = galaxy_map[curr_pqr]
        
        offset += num_elements
        
    
    
    candidate_hole_map_array = np.zeros(num_candidates, dtype=np.int64)
    
    cdef DTYPE_INT64_t[:] candidate_hole_map_array_memview = candidate_hole_map_array
    
    cell_ID_mem = Cell_ID_Memory(10)
    
    reference_cell_ijk = np.zeros((1,3), dtype=np.int16)
    
    cdef CELL_ID_t[:,:] reference_cell_ijk_memview = reference_cell_ijk
    
    ################################################################################
    # Iterate through all our holes, 
    ################################################################################
    
    
    for jdx in range(num_candidates):
        
        idx = candidate_index_memview[jdx]
        
        
        ################################################################################
        # Calculate some properties of the current candidate we're looking at
        ################################################################################
        is_maximal = True
        
        curr_radius = x_y_z_r_array[idx,3]
        
        curr_sphere_volume_thresh = frac*(4.0/3.0)*pi*curr_radius*curr_radius*curr_radius
        
        
        
        reference_cell_ijk_memview[0,0] = <CELL_ID_t>((x_y_z_r_array[idx,0] - min_x)/twice_largest_radius)
        reference_cell_ijk_memview[0,1] = <CELL_ID_t>((x_y_z_r_array[idx,1] - min_y)/twice_largest_radius)
        reference_cell_ijk_memview[0,2] = <CELL_ID_t>((x_y_z_r_array[idx,2] - min_z)/twice_largest_radius)
        
        
        ################################################################################
        # Generate all the grid cell IDs that could potentially hold conflicting
        # maximals
        ################################################################################
        num_cell_IDs = _gen_cube(reference_cell_ijk_memview, 
                                 1,
                                 cell_ID_mem,
                                 new_galaxy_map)
    
        for cell_ID_idx in range(<ITYPE_t>num_cell_IDs):
            
            id1 = cell_ID_mem.data[3*cell_ID_idx]
            id2 = cell_ID_mem.data[3*cell_ID_idx+1]
            id3 = cell_ID_mem.data[3*cell_ID_idx+2]
            
            if not new_galaxy_map.contains(id1, id2, id3):
                
                continue
            
            
            
            curr_offset_num_pair = new_galaxy_map.getitem(id1, id2, id3)
                
            offset = curr_offset_num_pair.offset
            
            num_elements = curr_offset_num_pair.num_elements
            
            
            for kdx in range(num_elements):
                
                maximal_idx = <ITYPE_t>candidate_hole_map_array_memview[offset+kdx]
                
        
        
                diffs[0] = x_y_z_r_array[idx,0] - x_y_z_r_array[maximal_idx,0]
                diffs[1] = x_y_z_r_array[idx,1] - x_y_z_r_array[maximal_idx,1]
                diffs[2] = x_y_z_r_array[idx,2] - x_y_z_r_array[maximal_idx,2]
                
                separation = sqrt(diffs[0]*diffs[0] + diffs[1]*diffs[1] + diffs[2]*diffs[2])
                
                curr_maximal_radius = x_y_z_r_array[maximal_idx,3]
                
                if curr_maximal_radius - curr_radius >= separation:
                    #hole is fully contained in another hole, it is not a maximal
                    is_maximal = False
                    
                    break
                
                elif curr_maximal_radius + curr_radius >= separation:
                    
                    curr_cap_height = cap_height(curr_radius, curr_maximal_radius, separation)
                    
                    maximal_cap_height = cap_height(curr_maximal_radius, curr_radius, separation)
                
                    overlap_volume = spherical_cap_volume(curr_radius, curr_cap_height) + spherical_cap_volume(curr_maximal_radius, maximal_cap_height)
                
                    if overlap_volume > curr_sphere_volume_thresh:
                        
                        is_maximal = False
                        
                        break
                    
                    else:
                        # Didn't overlap too much with this maximal, but still
                        # need to check all the others so keep going
                        pass
                
                else:
                    # Didn't overlap with this maximal at all, but still need to check
                    # all the others
                    pass
        
        
        
            if not is_maximal:
                break
        
        
        ################################################################################
        # The jth candidate was a maximal!  Add its idx to the output index, update
        # the number of maximals (N_voids), and also insert this maximal idx into the
        # array holding the lists of maximals which correspond to a given cell
        ################################################################################
        if is_maximal:
            
            out_index_memview[N_voids] = idx
            
            N_voids += 1
            
            
            '''
            #Shouldnt need to do this check since our current hole should always
            #have a valid grid cell
            if not new_galaxy_map.contains(reference_cell_ijk[0,0], 
                                           reference_cell_ijk[0,1],
                                           reference_cell_ijk[0,2]):
                
                print(min_x, min_y, min_z)
                
                print(x_y_z_r_array[idx,0], x_y_z_r_array[idx,1], x_y_z_r_array[idx,2])
                
                raise ValueError("GalaxymapCustomDict doesnt contain valid key"+str(reference_cell_ijk))
            '''
                
            
            curr_offset_num_pair = new_galaxy_map.getitem(reference_cell_ijk_memview[0,0],
                                                          reference_cell_ijk_memview[0,1],
                                                          reference_cell_ijk_memview[0,2])
            
            offset = curr_offset_num_pair.offset
            
            num_elements = curr_offset_num_pair.num_elements
            
            
            candidate_hole_map_array_memview[offset+num_elements] = idx
            
            new_galaxy_map.setitem(reference_cell_ijk_memview[0,0],
                                   reference_cell_ijk_memview[0,1],
                                   reference_cell_ijk_memview[0,2], 
                                   offset, 
                                   num_elements+1)
            
            
            
    maximals_info = {"maximals_map" : new_galaxy_map,
                   "maximals_cell_array" : candidate_hole_map_array,
                   "min_x" : min_x,
                   "min_y" : min_y,
                   "min_z" : min_z,
                   "twice_largest_radius" : twice_largest_radius,
                  }
    
    
    return out_maximals_index[0:N_voids], maximals_info
        
        
        
        
        
        







@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cpdef np.ndarray find_holes_2(DTYPE_F64_t[:,:] x_y_z_r_array,
                                            DTYPE_INT64_t[:] maximals_index,
                                            DTYPE_F64_t frac,
                                            ):
    """
    
    Rename this to 'build_void_groups'
    
    We have found the maximals, so intuitively one might think the 'holes' would
    just be all the remaining rows in the table.  However, some may be discarded 
    still because they are completely contained within another hole.  Also,
    the maximals themselves are also holes, so they get included in the 
    holes index.
    """

    cdef ITYPE_t idx, jdx, maximal_idx, num_holes, num_maximals, num_out_holes, num_matches, last_maximal_idx
    
    cdef DTYPE_F64_t[3] diffs
    cdef DTYPE_F64_t separation, curr_radius, curr_maximal_radius, curr_sphere_volume_thresh
    cdef DTYPE_F64_t curr_cap_height, maximal_cap_height, overlap_volume
    
    cdef DTYPE_B_t is_hole
    
    num_holes = x_y_z_r_array.shape[0]
    
    num_maximals = maximals_index.shape[0]

    holes_index = np.zeros(num_holes, dtype=np.int64)
    
    flag_column = np.zeros(num_holes, dtype=np.int64)
    
    cdef DTYPE_INT64_t[:] holes_index_memview = holes_index
    
    cdef DTYPE_INT64_t[:] flag_column_memview = flag_column
    
    ################################################################################
    # Set up some mappings for convenience
    ################################################################################
    cdef DTYPE_B_t[:] is_maximal_col = np.zeros(num_holes, dtype=np.uint8)
    
    cdef DTYPE_INT64_t[:] maximal_IDs = np.zeros(num_holes, dtype=np.int64)

    for idx in range(num_maximals):

        is_maximal_col[maximals_index[idx]] = 1
        
        maximal_IDs[maximals_index[idx]] = idx
    
    
    ################################################################################
    # Iterate through all the holes
    ################################################################################
    num_out_holes = 0
    
    for idx in range(num_holes):
        
        if is_maximal_col[idx]:
            
            holes_index_memview[num_out_holes] = idx

            flag_column_memview[num_out_holes] = maximal_IDs[idx]
            
            num_out_holes += 1
            
            continue


        curr_radius = x_y_z_r_array[idx,3]
        
        curr_sphere_volume_thresh = frac*(4.0/3.0)*np.pi*curr_radius*curr_radius*curr_radius

        is_hole = False
        
        num_matches = 0
        
        for jdx in range(num_maximals):
            
            maximal_idx = maximals_index[jdx]

            diffs[0] = x_y_z_r_array[idx,0] - x_y_z_r_array[maximal_idx,0]
            diffs[1] = x_y_z_r_array[idx,1] - x_y_z_r_array[maximal_idx,1]
            diffs[2] = x_y_z_r_array[idx,2] - x_y_z_r_array[maximal_idx,2]
            
            curr_maximal_radius = x_y_z_r_array[maximal_idx,3]


            separation = sqrt(diffs[0]*diffs[0] + diffs[1]*diffs[1] + diffs[2]*diffs[2])
            
            
            if (curr_maximal_radius - curr_radius) >= separation:
                #current sphere is completely contained within another,
                #throw it away
                break
            
            elif (curr_maximal_radius + curr_radius) >= separation:

                curr_cap_height = cap_height(curr_radius, curr_maximal_radius, separation)
                
                maximal_cap_height = cap_height(curr_maximal_radius, curr_radius, separation)
            
                overlap_volume = spherical_cap_volume(curr_radius, curr_cap_height) + spherical_cap_volume(curr_maximal_radius, maximal_cap_height)
            

                if overlap_volume > curr_sphere_volume_thresh:
                    
                    num_matches += 1
                    
                    last_maximal_idx = maximal_idx
                    
                    
                    
        ################################################################################
        # To match reference implementation, we need to only attach holes who
        # match up with exactly 1 maximal.  Future schemes may include assigning a hole
        # who matches more than 1 maximal based on the larger volume overlap
        ################################################################################
        if num_matches == 1:
            
            holes_index_memview[num_out_holes] = idx

            flag_column_memview[num_out_holes] = maximal_IDs[last_maximal_idx]
            
            num_out_holes += 1
    
    ################################################################################
    # Cython complained about returning a type (np.ndarray, np.ndarray). instead
    # return an (N,2) array since we're lucky that both return items have the
    # same dtype.
    ################################################################################
    out = np.zeros((num_out_holes,2), dtype=np.int64)
    
    out[:,0] = holes_index[0:num_out_holes]
    
    out[:,1] = flag_column[0:num_out_holes]
    
    return out






@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cpdef np.ndarray join_holes_to_maximals(DTYPE_F64_t[:,:] x_y_z_r_array,
                                        DTYPE_INT64_t[:] maximals_index,
                                        DTYPE_F64_t frac,
                                        dict maximals_info,
                                        ):
    """
    
    
    
    We have found the maximals, so intuitively one might think the 'holes' would
    just be all the remaining rows in the table.  However, some may be discarded 
    still because they are completely contained within another hole.  Also,
    the maximals themselves are also holes, so they get included in the 
    holes index.
    """

    cdef ITYPE_t idx, jdx, kdx, maximal_idx, num_holes, num_maximals, num_out_holes, num_matches, last_maximal_idx
    
    cdef DTYPE_F64_t[3] diffs
    cdef DTYPE_F64_t separation, curr_radius, curr_maximal_radius, curr_sphere_volume_thresh
    cdef DTYPE_F64_t curr_cap_height, maximal_cap_height, overlap_volume
    
    
    num_holes = x_y_z_r_array.shape[0]
    
    num_maximals = maximals_index.shape[0]

    holes_index = np.zeros(num_holes, dtype=np.int64)
    
    flag_column = np.zeros(num_holes, dtype=np.int64)
    
    cdef DTYPE_INT64_t[:] holes_index_memview = holes_index
    
    cdef DTYPE_INT64_t[:] flag_column_memview = flag_column
    
    ################################################################################
    # Set up some mappings for convenience
    ################################################################################
    cdef DTYPE_B_t[:] is_maximal_col = np.zeros(num_holes, dtype=np.uint8)
    
    cdef DTYPE_INT64_t[:] maximal_IDs = np.zeros(num_holes, dtype=np.int64)

    for idx in range(num_maximals):

        is_maximal_col[maximals_index[idx]] = 1
        
        maximal_IDs[maximals_index[idx]] = idx
    
    
    
    
    
    
    ################################################################################
    # Boop Boop a shoop
    ################################################################################
    
    
    cdef DTYPE_F64_t min_x = maximals_info["min_x"]
    cdef DTYPE_F64_t min_y = maximals_info["min_y"]
    cdef DTYPE_F64_t min_z = maximals_info["min_z"]
    
    cdef DTYPE_F64_t twice_largest_radius = maximals_info["twice_largest_radius"]
    
    cdef GalaxyMapCustomDict maximals_map = maximals_info["maximals_map"]
    
    cdef DTYPE_INT64_t[:] maximals_map_array = maximals_info["maximals_cell_array"]
    '''
    maximals_info = {"maximals_map" : new_galaxy_map,
                   "maximals_cell_array" : candidate_hole_map_array,
                   "min_x" : min_x,
                   "min_y" : min_y,
                   "min_z" : min_z,
                   "twice_largest_radius" : twice_largest_radius,
                  }
    '''
    
    cdef Cell_ID_Memory cell_ID_mem = Cell_ID_Memory(10)
    
    cdef ITYPE_t cell_ID_idx
    
    cdef OffsetNumPair curr_offset_num_pair
    
    cdef DTYPE_INT64_t num_cell_IDs
    
    cdef CELL_ID_t id1, id2, id3
    
    cdef DTYPE_INT64_t offset, num_elements
    
    
    reference_cell_ijk = np.zeros((1,3), dtype=np.int16)
    
    cdef CELL_ID_t[:,:] reference_cell_ijk_memview = reference_cell_ijk
    
    ################################################################################
    # Iterate through all the holes
    ################################################################################
    #start_time = time.time()
    
    num_out_holes = 0
    
    #print("Join holes to maximals")
    #print("Total num holes: ", num_holes)
    #print("Total num maximals: ", maximals_index.shape[0])
    
    for idx in range(num_holes):
        
        
        
        #if idx%10000 == 0:
        #    print("Working: ", idx, "at time: ", time.time() - start_time, flush=True)
        
        
        if is_maximal_col[idx]:
            
            holes_index_memview[num_out_holes] = idx

            flag_column_memview[num_out_holes] = maximal_IDs[idx]
            
            num_out_holes += 1
            
            continue


        curr_radius = x_y_z_r_array[idx,3]
        
        curr_sphere_volume_thresh = frac*(4.0/3.0)*pi*curr_radius*curr_radius*curr_radius
        
        num_matches = 0
        
        
        
        
        
        
        
        
        
        reference_cell_ijk_memview[0,0] = <CELL_ID_t>((x_y_z_r_array[idx,0] - min_x)/twice_largest_radius)
        reference_cell_ijk_memview[0,1] = <CELL_ID_t>((x_y_z_r_array[idx,1] - min_y)/twice_largest_radius)
        reference_cell_ijk_memview[0,2] = <CELL_ID_t>((x_y_z_r_array[idx,2] - min_z)/twice_largest_radius)
        
        
        ################################################################################
        # Generate all the grid cell IDs that could potentially hold conflicting
        # maximals
        ################################################################################
        num_cell_IDs = _gen_cube(reference_cell_ijk_memview, 
                                 1,
                                 cell_ID_mem,
                                 maximals_map)
    
        for cell_ID_idx in range(<ITYPE_t>num_cell_IDs):
            
            id1 = cell_ID_mem.data[3*cell_ID_idx]
            id2 = cell_ID_mem.data[3*cell_ID_idx+1]
            id3 = cell_ID_mem.data[3*cell_ID_idx+2]
            
            if not maximals_map.contains(id1, id2, id3):
                
                continue
            
            
            
            curr_offset_num_pair = maximals_map.getitem(id1, id2, id3)
                
            offset = curr_offset_num_pair.offset
            
            num_elements = curr_offset_num_pair.num_elements
            
            
            for kdx in range(num_elements):
                
                maximal_idx = <ITYPE_t>maximals_map_array[offset+kdx]
                
                diffs[0] = x_y_z_r_array[idx,0] - x_y_z_r_array[maximal_idx,0]
                diffs[1] = x_y_z_r_array[idx,1] - x_y_z_r_array[maximal_idx,1]
                diffs[2] = x_y_z_r_array[idx,2] - x_y_z_r_array[maximal_idx,2]
                
                curr_maximal_radius = x_y_z_r_array[maximal_idx,3]
    
    
                separation = sqrt(diffs[0]*diffs[0] + diffs[1]*diffs[1] + diffs[2]*diffs[2])
                
                
                if (curr_maximal_radius - curr_radius) >= separation:
                    #current sphere is completely contained within another,
                    #throw it away
                    break
                
                elif (curr_maximal_radius + curr_radius) >= separation:
    
                    curr_cap_height = cap_height(curr_radius, curr_maximal_radius, separation)
                    
                    maximal_cap_height = cap_height(curr_maximal_radius, curr_radius, separation)
                
                    overlap_volume = spherical_cap_volume(curr_radius, curr_cap_height) + spherical_cap_volume(curr_maximal_radius, maximal_cap_height)
                
    
                    if overlap_volume > curr_sphere_volume_thresh:
                        
                        num_matches += 1
                        
                        last_maximal_idx = maximal_idx
            
        ################################################################################
        # To match reference implementation, we need to only attach holes who
        # match up with exactly 1 maximal.  Future schemes may include assigning a hole
        # who matches more than 1 maximal based on the larger volume overlap
        ################################################################################
        if num_matches == 1:
            
            holes_index_memview[num_out_holes] = idx

            flag_column_memview[num_out_holes] = maximal_IDs[last_maximal_idx]
            
            num_out_holes += 1
    
    ################################################################################
    # Cython complained about returning a type (np.ndarray, np.ndarray). instead
    # return an (N,2) array since we're lucky that both return items have the
    # same dtype.
    ################################################################################
    out = np.zeros((num_out_holes,2), dtype=np.int64)
    
    out[:,0] = holes_index[0:num_out_holes]
    
    out[:,1] = flag_column[0:num_out_holes]
    
    return out








