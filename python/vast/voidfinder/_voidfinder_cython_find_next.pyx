#cython: language_level=3



from __future__ import print_function

cimport cython

from cpython.mem cimport PyMem_Malloc, PyMem_Realloc, PyMem_Free

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
                      DTYPE_UINT16_t, \
                      CELL_ID_t

from numpy.math cimport NAN, INFINITY

from libc.math cimport fabs, sqrt, asin, atan, ceil#, exp, pow, cos, sin, asin

#from libc.stdlib cimport malloc, free

import time
import os
import tempfile
from multiprocessing import RLock, Value
from ctypes import c_int64
import mmap





cdef DTYPE_F64_t RtoD = 180./np.pi
cdef DTYPE_F64_t DtoR = np.pi/180.
cdef DTYPE_F64_t dec_offset = -90





@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
@cython.profile(False)
cdef FindNextReturnVal find_next_galaxy(DTYPE_F64_t[:,:] hole_center_memview, 
                                        DTYPE_F64_t[:,:] temp_hole_center_memview,
                                        DTYPE_F64_t search_radius, 
                                        DTYPE_F64_t dr, 
                                        DTYPE_F64_t direction_mod,
                                        DTYPE_F64_t[:] unit_vector_memview, 
                                        GalaxyMap galaxy_tree, 
                                        DTYPE_INT64_t[:] nearest_gal_index_list, 
                                        ITYPE_t num_neighbors,
                                        MaskChecker mask_checker,
                                        DTYPE_F64_t[:] Bcenter_memview,
                                        Cell_ID_Memory cell_ID_mem,
                                        NeighborMemory neighbor_mem,
                                        ):          

    '''
    Description:
    ============
    
    
    Function to locate the next nearest galaxy during hole center propagation 
    along direction defined by unit_vector_memview.

    The algorithm needs to find 4 bounding galaxies per cell.  The 
    first galaxy is found as the minimum of regular euclidean distance to the hole center.  The
    second galaxy is determined using a ratio of distance with respect to projected distance
    along the hole center search path.  To find galaxy 2, the hole center search path and 
    distance ratios are calculated with respect to neighbor 1.  

    Implementation Detail:
    For galaxies 3 and 4, the 
    distance ratios are calculated with respect to the previous galaxies, so this code uses
    an if-block to detect whether we're calculating for galaxy 2/B, 3/C, or 4/D, and runs slightly
    different distance ratio calculations.  The if block looks for the number of elements
    in the input nearest_gal_index_list, when its 1 it assumes we're finding for galaxy 2
    and when its not 1 it assumes we have the first 2 galaxies and we're looking for galaxy
    3 or 4.



    Parameters:
    ===========

    hole_center_memview : memview of shape (1,3)
        x,y,z coordinate of current center of hole in units of Mpc/h
        
    NEEDS DEFINITION temp_hole_center_memview :

    search_radius : float
        Radius of hole in units of Mpc/h

    dr : float
        Incrememt value for hole propagation
        
    direction_mod : -1.0 or 1.0
        basically a switch to go in the opposite direction of vector propagation for
        finding galaxies 4a and 4b

    unit_vector_memview : memview of shape (3)
        Unit vector indicating direction hole center will shift

    galaxy_tree : custom thingy
        Data structure to query for nearest-neighbor results

    nearest_gal_index_list : memview of shape (N)
        List of row indices in w_coord for existing bounding galaxies
        
    num_neighbors : int
        number of valid neighbor indices in the nearest_gal_index_list object    

    w_coord : memview of shape (N_galaxies, 3)
        x,y,z coordinates of all galaxies in sample in units of Mpc/h

    mask : memview of shape (ra_dim, dec_dim)
        uint8 array of whether location is within survey footprint

    mask_resolution : integer
        Scale factor of coordinates used to index mask

    min_dist : float
        minimum distance (redshift) in survey in units of Mpc/h

    max_dist : float
        maximum distance (redshift) in survey in units of Mpc/h
        
    NEEDS DEFINITION Bcenter_memview : 
        some memory for ???
                           
    


    Returns:
    ========


    retval : FindNextReturnVal
    
        struct with members:
    
        nearest_neighbor_index : index
            Index value to w_coord array of next nearest neighbor

        min_x_ratio : float
            ???

        in_mask : boolean
            Flag indicating whether or not the temporary hole center is within the 
            survey footprint.
    '''



    # We are going to shift the center of the hole by dr along 
    # the direction of the vector pointing from the nearest 
    # galaxy to the center of the empty cell.  From there, we 
    # will search within a radius of length the distance between 
    # the center of the hole and the first galaxy from the 
    # center of the hole to find the next nearest neighbors.  
    # From there, we will minimize top/bottom to find which one 
    # is the next nearest galaxy that bounds the hole.

    ############################################################################
    # Struct type for the output of this function which includes members for
    # the neighbor index, x_ratio value, and in_mask 
    ############################################################################
    cdef FindNextReturnVal retval
    
    retval.in_mask = False
    
    
    ############################################################################
    # helper index and calculation variables, re-useable
    ############################################################################
    cdef ITYPE_t idx
    
    cdef ITYPE_t jdx
    
    cdef ITYPE_t temp_idx
    
    cdef DTYPE_F64_t temp_f64_accum
    
    cdef DTYPE_F64_t temp_f64_val

    ############################################################################
    # Used in filtering exiting neighbors out of results
    ############################################################################
    cdef ITYPE_t num_results
    
    cdef ITYPE_t num_nearest
    
    
    ############################################################################
    # Used in finding valid x ratio values
    ############################################################################
    cdef DTYPE_B_t any_valid = 0
            
    cdef ITYPE_t valid_min_idx
    
    cdef DTYPE_F64_t valid_min_val
    

    ############################################################################
    #
    # Main galaxy finding loop.  Contains internal logic to use slightly different
    # processes depending on whether we're finding galaxy 2/B, 3/C, or 4/D.
    #
    # The general gist is that we take a hole center, move it in the direction
    # of the unit vector by steps of dr, then look at all the galaxies in the
    # sphere centered at the new hole center of larger radius until we find one
    # that bounds the sphere
    #
    ############################################################################
    
    cdef DTYPE_B_t galaxy_search = True
    
    cdef DTYPE_B_t final_level = False
    
    cdef DTYPE_INT32_t curr_level = -1
    
    #cdef CELL_ID_t[3] last_pqr

    #last_pqr[0] = -1
    #last_pqr[1] = -1
    #last_pqr[2] = -1
    
    #cdef CELL_ID_t[3] curr_pqr
    
    while galaxy_search:

        dr *= 1.1
        
        #curr_level += 1
        
        ############################################################################
        # shift hole center
        ############################################################################
        for idx in range(3):

            temp_hole_center_memview[0, idx] = temp_hole_center_memview[0, idx] + direction_mod*dr*unit_vector_memview[idx]
            #temp_hole_center_memview[0, idx] = temp_hole_center_memview[0, idx] + direction_mod*galaxy_tree.dl*unit_vector_memview[idx]
            
            #curr_pqr[idx] = <CELL_ID_t>((temp_hole_center_memview[0,0] - galaxy_tree.coord_min[0,0])/galaxy_tree.dl)
        
        
        #Note if we keep this code, potential bug whereby the dr *= 1.1
        #makes us jump more than 1 level at a time
        #if curr_pqr[0] != last_pqr[0] or \
        #   curr_pqr[1] != last_pqr[1] or \
        #   curr_pqr[2] != last_pqr[2]:
        #    curr_level += 1
        #    
        #else:
        #    continue
        
        
        ############################################################################
        # calculate new search radius
        #
        # If we have the 1st neighbor, just looking for 2nd, easy we just increment
        # search_radius by dr.    
        #
        # For finding 3/C and 4/D neighbors, we're still going to use the distance 
        # between the hole center and the 1/A neighbor galaxy for the search_radius
        #
        # For finding galaxies 3/C and 4/D, the current scheme wants the sphere
        # boundary to still pass through galaxy 1/A, and this results in a case
        # where the previous sphere is not a complete subset of the next sphere to
        # be grown.  Since we're using 1/A as the reference point, moving the
        # hole by an amount of dr will not result in a search radius difference
        # of dr, since the unit vector we're moving along is not aligned with
        # the vector from the moving hole center to galaxy 1/A
        ############################################################################
        
        if num_neighbors == 1:
            
            search_radius += dr
        
        elif num_neighbors > 1:
            
            temp_f64_accum = 0.0
            
            for idx in range(3):
                
                temp_f64_val = galaxy_tree.wall_galaxy_coords[nearest_gal_index_list[0],idx] - temp_hole_center_memview[0,idx]
                
                temp_f64_accum += temp_f64_val*temp_f64_val
                
            search_radius = sqrt(temp_f64_accum)
        
        
        
        ############################################################################
        # use GalaxyMap to find the galaxies within our target sphere
        #
        # _query_shell_radius() fills in index values into the 
        # neighbor_mem.i_nearest array corresponding to the neighbors it finds for
        # this search
        ############################################################################
        
        neighbor_mem.next_neigh_idx = 0
        
        _query_shell_radius(galaxy_tree.reference_point_ijk,
                            galaxy_tree.coord_min,
                            galaxy_tree.dl, 
                            galaxy_tree,
                            cell_ID_mem,
                            neighbor_mem,
                            temp_hole_center_memview, 
                            search_radius)
        
        #When we're looping through, keep track of the last ijk so we can track whether or not
        #we have already queried this "level" with respect to the hole_center on next loop and
        #short circuit if the movement of length dr didn't move us into a new ijk cell                        
        #for idx in range(3):
        
        #    last_pqr[idx] = <CELL_ID_t>((temp_hole_center_memview[0,0] - galaxy_tree.coord_min[0,0])/galaxy_tree.dl)
        
        ############################################################################
        # The resulting galaxies may include galaxies we already found in previous
        # steps, so build a boolean index representing whether a resultant galaxy
        # is valid or not, and track how many valid result galaxies we actually have
        # for the next step.
        ############################################################################
        num_results = <ITYPE_t>(neighbor_mem.next_neigh_idx)


        for idx in range(num_results):
        
            neighbor_mem.boolean_nearest[idx] = 1
            
            
        num_nearest = num_results

        for idx in range(num_results):

            for jdx in range(num_neighbors):
                
                if neighbor_mem.i_nearest[idx] == nearest_gal_index_list[jdx]:
                    
                    neighbor_mem.boolean_nearest[idx] = 0
                    
                    num_nearest -= 1
                    
                    break
                
                
        ############################################################################
        # If we have any valid result galaxies, use the special x ratio distance
        # metric on them.  Note that metric is dependent on whether we are
        # searching for galaxy 2/B, 3/C or 4/D, so this code uses the
        # num_neighbors input parameter to tell which one we are looking for.
        # num_neighbors == 1 means we have 1 bounding galaxy so we're looking
        # for galaxy 2/B, if num_neighbors > 1 it means we have at least 2 galaxies
        # so we're looking for 3/C or 4/D which use the same process.
        ############################################################################
        if num_nearest > 0:
            
            ############################################################################
            # copy the valid galaxy indicies into the i_nearest_reduced memory
            ############################################################################
            jdx = 0
            
            for idx in range(num_results):
                
                if neighbor_mem.boolean_nearest[idx]:
                    
                    neighbor_mem.i_nearest_reduced[jdx] = neighbor_mem.i_nearest[idx]
                    
                    jdx += 1
            
            
            ############################################################################
            # Calculate vectors pointing from hole center and galaxy 1/A to next 
            # nearest candidate galaxy
            ############################################################################
            for idx in range(num_nearest):

                temp_idx = neighbor_mem.i_nearest_reduced[idx]

                for jdx in range(3):
                    
                    if num_neighbors == 1:
                        
                        neighbor_mem.candidate_minus_A[3*idx+jdx] = galaxy_tree.wall_galaxy_coords[nearest_gal_index_list[0], jdx] - galaxy_tree.wall_galaxy_coords[temp_idx, jdx]
                        
                        
                    else:

                        neighbor_mem.candidate_minus_A[3*idx+jdx] = galaxy_tree.wall_galaxy_coords[temp_idx, jdx] - galaxy_tree.wall_galaxy_coords[nearest_gal_index_list[0], jdx]
                        
                    neighbor_mem.candidate_minus_center[3*idx+jdx] = galaxy_tree.wall_galaxy_coords[temp_idx, jdx] - hole_center_memview[0, jdx]

            ############################################################################
            # Calculate bottom of ratio to be minimized
            # 2*dot(candidate_minus_A, unit_vector)
            ############################################################################
            for idx in range(num_nearest):
                
                temp_f64_accum = 0.0
                
                for jdx in range(3):
                    
                    temp_f64_accum += neighbor_mem.candidate_minus_A[3*idx+jdx]*unit_vector_memview[jdx]
                    
                neighbor_mem.bot_ratio[idx] = 2*temp_f64_accum
            
            
            ############################################################################
            # Calculate top of ratio to be minimized
            ############################################################################
            if num_neighbors == 1:

                for idx in range(num_nearest):
                
                    temp_f64_accum = 0.0
                    
                    for jdx in range(3):
                        
                        temp_f64_accum += neighbor_mem.candidate_minus_A[3*idx+jdx]*neighbor_mem.candidate_minus_A[3*idx+jdx]
                        
                    neighbor_mem.top_ratio[idx] = temp_f64_accum

            else:

                for idx in range(3):

                    Bcenter_memview[idx] = galaxy_tree.wall_galaxy_coords[nearest_gal_index_list[1], idx] - hole_center_memview[0, idx]


                temp_f64_accum = 0.0
                
                for idx in range(3):
                    
                    temp_f64_accum += Bcenter_memview[idx]*Bcenter_memview[idx]
                    
                temp_f64_val = temp_f64_accum

                
                for idx in range(num_nearest):
                    
                    temp_f64_accum = 0.0
                    
                    for jdx in range(3):
                        
                        temp_f64_accum += neighbor_mem.candidate_minus_center[3*idx+jdx]*neighbor_mem.candidate_minus_center[3*idx+jdx]
                        
                    neighbor_mem.top_ratio[idx] = temp_f64_accum - temp_f64_val



            ############################################################################
            # Calculate the minimization ratios
            ############################################################################
            for idx in range(num_nearest):

                neighbor_mem.x_ratio[idx] = neighbor_mem.top_ratio[idx]/neighbor_mem.bot_ratio[idx]

            ############################################################################
            # Locate positive values of x_ratio
            ############################################################################
            any_valid = 0
            
            valid_min_idx = 0
            
            valid_min_val = INFINITY
            
            for idx in range(num_nearest):
                
                temp_f64_val = neighbor_mem.x_ratio[idx]
                
                if temp_f64_val > 0.0:
                    
                    any_valid = 1
                    
                    if temp_f64_val < valid_min_val:
                        
                        valid_min_idx = idx
                        
                        valid_min_val = temp_f64_val
                        

            ############################################################################
            # If we found any positive values, we have a result, we need to check
            # one final level of the grid, so set final_level == True and do one
            # more loop, then if that shell returns any values, keep the one with
            # the smaller min_x_ratio value
            ############################################################################
            if any_valid and not final_level:
                
                retval.nearest_neighbor_index = neighbor_mem.i_nearest_reduced[valid_min_idx]
                
                retval.min_x_ratio = neighbor_mem.x_ratio[valid_min_idx]
                
                retval.in_mask = True
                
                final_level = True
                
                return retval
                
            elif any_valid and final_level:
                
                if neighbor_mem.x_ratio[valid_min_idx] < retval.min_x_ratio:
                    
                    retval.nearest_neighbor_index = neighbor_mem.i_nearest_reduced[valid_min_idx]
                
                    retval.min_x_ratio = neighbor_mem.x_ratio[valid_min_idx]
                    
                    retval.in_mask = True
                    
                #galaxy_search = False
                
                return retval
                

        #elif not_in_mask(temp_hole_center_memview, mask, mask_resolution, min_dist, max_dist):
        elif mask_checker.not_in_mask(temp_hole_center_memview):
            
            retval.in_mask = False
            
            return retval
            

    return retval












cdef class MaskChecker:
    
    def __init__(self, 
                 mode,
                 survey_mask_ra_dec=None,
                 n=None,
                 rmin=None,
                 rmax=None,
                 xyz_limits=None):
        
        self.mode = mode
        
        if survey_mask_ra_dec is not None:
            self.survey_mask_ra_dec = survey_mask_ra_dec
            
        if n is not None:
            self.n = n
            
        if rmin is not None:
            self.rmin = rmin
            
        if rmax is not None:
            self.rmax = rmax
            
        if xyz_limits is not None:
            self.xyz_limits = xyz_limits
        
        
    cdef DTYPE_B_t not_in_mask(self, DTYPE_F64_t[:,:] coordinates):
        
        if self.mode == 0:
            
            return not_in_mask(coordinates,
                               self.survey_mask_ra_dec,
                               self.n,
                               self.rmin,
                               self.rmax)
            
        elif self.mode == 1:
            
            return not_in_mask_xyz(coordinates, self.xyz_limits)
        
        
        elif self.mode == 2: #periodic, so everything is always in the mask
            
            return False





@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
@cython.profile(False)
cdef DTYPE_B_t not_in_mask_xyz(DTYPE_F64_t[:,:] coordinates, 
                               DTYPE_F64_t[:,:] xyz_limits):
    """
    Parameters
    ==========
    
    coordinates : array shape (1,3)
        format [x,y,z]
        
    xyz_limits : array shape (2,3)
        format [x_min, y_min, z_min]
               [x_max, y_max, z_max]
    """
    
    cdef DTYPE_F64_t coord_x
    cdef DTYPE_F64_t coord_y
    cdef DTYPE_F64_t coord_z
    
    coord_x = coordinates[0,0]
    coord_y = coordinates[0,1]
    coord_z = coordinates[0,2]
    
    if coord_x < xyz_limits[0,0]:
        return True
    
    if coord_x > xyz_limits[1,0]:
        return True
    
    if coord_y < xyz_limits[0,1]:
        return True
    
    if coord_y > xyz_limits[1,1]:
        return True
    
    if coord_z < xyz_limits[0,2]:
        return True
    
    if coord_z > xyz_limits[1,2]:
        return True
    
    return False
    
    
    
    
    



@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
@cython.profile(False)
cdef DTYPE_B_t not_in_mask(DTYPE_F64_t[:,:] coordinates, 
                           DTYPE_B_t[:,:] survey_mask_ra_dec, 
                           DTYPE_INT32_t n,
                           DTYPE_F64_t rmin, 
                           DTYPE_F64_t rmax):
    '''
    Description
    ===========
    
    Determine whether a given set of coordinates falls within the survey.  Since 
    survey_mask_ra_dec is only used to index the ra (Right Ascension) and dec (Declination)
    of the survey, rmin and rmax are provided as proxies to the minimum and maximum
    Redshift (commonly denoted by z).  In the following code, the radius of the xyz position
    is calculated and compared to rmin and rmax to check against redshift, then the
    ra and dec of the xyz position are calculated.  The ra and dec of the target
    position are then scaled by n, to account for upsampling or downsampling in the
    survey mask, and the integer part of those ra and dec values form the index
    into the survey mask.

    Parameters
    ==========

    coordinates : numpy.ndarray of shape (1,3), 
        Coordinates of a point to check in x,y,z cartesian coordinates
        x, y, and z are measured in Mpc/h

    survey_mask_ra_dec : numpy.ndarray of shape (num_ra, num_dec) 
        the element at [i,j] represents whether or not the scaled ra corresponding to
        i and the scaled dec corresponding to j fall within the survey mask.  RA and dec
        are both measured in degrees and scaled by n.

    n : integer
        Scale factor of coordinates used to index survey_mask_ra_dec

    rmin, rmax : scalar
        min and max values of the survey in x,y,z units of Mpc/h
        Note these values form a proxy for redshift

    Returns
    =======

    return_mask_value : bool
        True if coordinates fall outside the survey mask "not in mask"
        and False if the coordinates lie inside the mask
    '''

    ######################################################################
    # Initialize some cdef memory for our variables
    ######################################################################
    cdef DTYPE_F64_t r_sq
    cdef DTYPE_F64_t r
    
    
    cdef DTYPE_F64_t ra
    cdef DTYPE_F64_t dec
    
    cdef DTYPE_F64_t n_float
    
    
    cdef ITYPE_t idx1
    cdef ITYPE_t idx2
    
    cdef DTYPE_F64_t coord_x
    cdef DTYPE_F64_t coord_y
    cdef DTYPE_F64_t coord_z
    
    cdef DTYPE_B_t return_mask_value

    ######################################################################
    # Unpack our target coordinate into the cdef variables and check
    # in x,y,z space if we've exceeded rmin or rmax
    ######################################################################
    coord_x = coordinates[0,0]
    coord_y = coordinates[0,1]
    coord_z = coordinates[0,2]
    
    r_sq = coord_x*coord_x + coord_y*coord_y + coord_z*coord_z

    if r_sq < rmin*rmin or r_sq > rmax*rmax:
        
        return True

    ######################################################################
    # We'll need to scale by n later, so create a float version of it
    ######################################################################
    n_float = <DTYPE_F64_t>n

    ######################################################################
    # Now calculate the ra and dec of the current point, and convert
    # them into 
    #
    # Double check with Kelly - what is the allowed range of ra and dec 
    # here?
    #
    ######################################################################
    r = sqrt(r_sq)
    
    ra = atan(coord_y/coord_x)*RtoD
    
    dec = asin(coord_z/r)*RtoD
    
    if coord_x < 0.0 and coord_y != 0.0:
        
        ra += 180.0
        
    if ra < 0:
        
        ra += 360.0
        
    ######################################################################
    # Index into the mask by taking the integer part of the scaled
    # ra and dec
    ######################################################################
    idx1 = <ITYPE_t>(n_float*ra)
    
    idx2 = <ITYPE_t>(n_float*dec) - <ITYPE_t>(n_float*dec_offset)
    
    return_mask_value = survey_mask_ra_dec[idx1, idx2]
    
    ######################################################################
    # Since the mask tells us if we're "in" the mask, and we want to
    # return whether we're "not in" the mask, flip the value we got,
    # then return.
    ######################################################################
    
    if return_mask_value == 1:
        
        return_mask_value = 0
        
    elif return_mask_value == 0:
        
        return_mask_value = 1

    return return_mask_value







"""
Note in voidfinder.volume_cut we were using the cpdef wrapper
around the cythonized not_in_mask function so I added this stupid
little workaround so we can keep not_in_mask cdef'd to work with
the typedef stuff 
"""

cpdef DTYPE_B_t not_in_mask2(DTYPE_F64_t[:,:] coordinates, 
                               DTYPE_B_t[:,:] survey_mask_ra_dec, 
                               DTYPE_INT32_t n,
                               DTYPE_F64_t rmin, 
                               DTYPE_F64_t rmax):
                               
    return not_in_mask(coordinates, survey_mask_ra_dec, n, rmin, rmax)




cdef class HoleGridCustomDict:
    """
    Description
    ===========
    
    This class is a dictionary-like object whose sole purpose is to provide a 
    contains(i,j,k) method which tells us whether or not a cell in the VoidFinder
    hole grid is empty or non-empty.  If hole_dict.contains(i,j,k) == True, that
    means there is at least 1 galaxy in that (i,j,k) cell, so it is non-empty
    and VoidFinder will skip growing a hole at that location.
    
    Parameters
    ==========
    
    grid_dimensions : length 3 tuple
        the number of grid cells in each of the 3 i-j-k dimensions.  Used
        in calculating hash values for the grid cell info.
        
    lookup_memory : numpy.ndarray of shape (N,)
        Each element of this array is of type _voidfinder_cython_find_next.pxd 
        HOLE_LOOKUPMEM_t and contains a flag for if the array element is filled,
        and i-j-k values for key comparisons.  No actual value is stored in this
        array besides this flag and the i-j-k key value.
        
        The value of (N,) is chosen to be the first prime number larger than 2 times
        the number of elements to be inserted into this dictionary, to help reduce
        hash-table collisions.
        
    """

    def __init__(self, grid_dimensions, resource_dir, starting_cells=8):
        
        cdef HOLE_LOOKUPMEM_t curr_element
        
        
        ################################################################################
        # First, set up some parameters corresponding to the VoidFinder hole-growing
        # grid, which we use in the hash function to calculate hash table indices
        ################################################################################
        self.i_dim = grid_dimensions[0]
        
        self.j_dim = grid_dimensions[1]
        
        self.k_dim = grid_dimensions[2]
        
        self.jk_mod = self.j_dim*self.k_dim
        
        
        ################################################################################
        # Next, create a memory-mapped file we can use as the shared memory
        # to share among our processes.  Since we're using fork(), the file descriptor
        # which we create here will point to the same file in the child processes
        # meaning we can just pass this whole class across the fork()
        ################################################################################
        next_prime = find_next_prime(2*starting_cells)
        
        self.mem_length = next_prime
        
        self.lookup_fd, HOLE_LOOKUPMEM_BUFFER_PATH = tempfile.mkstemp(prefix="voidfinder", 
                                                                      dir=resource_dir, 
                                                                      text=False)
        
        hole_lookup_buffer_length = self.mem_length*7
        
        os.ftruncate(self.lookup_fd, hole_lookup_buffer_length)
        
        self.hole_lookup_buffer = mmap.mmap(self.lookup_fd, hole_lookup_buffer_length)
        
        os.unlink(HOLE_LOOKUPMEM_BUFFER_PATH)
        
        lookup_dtype = [("filled_flag", np.uint8, ()), #() indicates scalar, or length 1 shape
                        ("i", np.int16, ()),
                        ("j", np.int16, ()),
                        ("k", np.int16, ())]
        
        self.numpy_dtype = np.dtype(lookup_dtype, align=False)
        
        self.lookup_memory = np.frombuffer(self.hole_lookup_buffer, dtype=self.numpy_dtype)
        
        ################################################################################
        # Writing a bunch of 0's directly to the self.hole_lookup_buffer did not work
        # I do not know why, but instead we must use the array broadcasting with
        # an element whose flag has been 0'd to properly zero out the hash table
        ################################################################################
        curr_element.filled_flag = 0
        
        self.lookup_memory[:] = curr_element
        
        self.num_elements = 0
        
        self.num_collisions = 0
        
        
        
        
            
    def __len__(self):
        return self.num_elements
            
    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.cdivision(True)
    cdef DTYPE_INT64_t custom_hash(self, 
                                   CELL_ID_t i, 
                                   CELL_ID_t j, 
                                   CELL_ID_t k):
        """
        Given a cell ID (i,j,k), calculate its hash address in our
        memory array.  Uses the natural grid cell ordering
        0->(0,0,0), 1->(0,0,1), 2->(0,0,2) etc, to calculate the sequential
        grid cell index, then takes that value modulus of the number of slots
        in the memory array.
        """
        
        cdef DTYPE_INT64_t index
        cdef DTYPE_INT64_t hash_addr
        
        index = self.jk_mod * <DTYPE_INT64_t>i + \
                self.k_dim * <DTYPE_INT64_t>j + \
                <DTYPE_INT64_t>k
        
        hash_addr = index % self.mem_length
        
        ############################################################
        # the @cython.cdivision(True) directive, while providing
        # speedup benefits, means that -1 % 5 = -1, not 4 as in
        # python, so we have to explicitly check for negative vals
        ############################################################
        if hash_addr < 0:
            
            hash_addr += self.mem_length
        
        return hash_addr
    
    
    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.cdivision(True)
    cpdef DTYPE_B_t contains(self,
                             CELL_ID_t i, 
                             CELL_ID_t j, 
                             CELL_ID_t k):
        
        cdef DTYPE_INT64_t hash_addr
        
        cdef DTYPE_INT64_t hash_offset
        
        cdef DTYPE_INT64_t curr_hash_addr
        
        cdef HOLE_LOOKUPMEM_t curr_element
        
        hash_addr = self.custom_hash(i, j, k)
        
        for hash_offset in range(self.mem_length):
            
            curr_hash_addr = (hash_addr + hash_offset) % self.mem_length
            
            curr_element = self.lookup_memory[curr_hash_addr]
        
            if not curr_element.filled_flag:
                
                return False
            
            else:
                
                if curr_element.key_i == i and \
                   curr_element.key_j == j and \
                   curr_element.key_k == k:
                    
                    return True
                        
        return False
    
    
    
    
    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.cdivision(True)
    cpdef void setitem(self, 
                       CELL_ID_t i,
                       CELL_ID_t j,
                       CELL_ID_t k, 
                       ):
        """
        we initialize the length of
        self.lookup_memory to be longer than the number of items
        """
        
        cdef DTYPE_INT64_t hash_addr
        
        cdef DTYPE_INT64_t hash_offset
        
        cdef DTYPE_INT64_t curr_hash_addr
        
        cdef HOLE_LOOKUPMEM_t curr_element
        
        cdef HOLE_LOOKUPMEM_t out_element
        
        cdef DTYPE_B_t first_try = True
        
        out_element.filled_flag = 1
        out_element.key_i = i
        out_element.key_j = j
        out_element.key_k = k
        
        hash_addr = self.custom_hash(i, j, k)
        
        for hash_offset in range(self.mem_length):
            
            curr_hash_addr = (hash_addr + hash_offset) % self.mem_length
            
            curr_element = self.lookup_memory[curr_hash_addr]
            
            if not curr_element.filled_flag:
                
                if not first_try:
                    
                    self.num_collisions += 1
                
                self.lookup_memory[curr_hash_addr] = out_element
                
                self.num_elements += 1
                
                break
            
            else:
                
                if curr_element.key_i == i and \
                   curr_element.key_j == j and \
                   curr_element.key_k == k:
                    
                    if not first_try:
                    
                        self.num_collisions += 1
                    
                    self.lookup_memory[curr_hash_addr] = out_element
                    
                    # Dont increment num_elements here because this is overwriting
                    # an existing key
                
                    break
                
            first_try = False
    
        if self.num_elements >= (<DTYPE_INT64_t>(0.60*self.mem_length)):
            
            self.resize(2*self.mem_length)
            
        return
            
            
    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.cdivision(True)
    cdef void resize(self, DTYPE_INT64_t new_mem_length):
    
        cdef HOLE_LOOKUPMEM_t curr_element
        
        cdef ITYPE_t idx
        
        cdef HOLE_LOOKUPMEM_t[:] old_lookup_mem
        
        ################################################################################
        # There may be a better way, but for now, we're just going to make a copy
        # of the 'old memory' since the key-value pairs it is storing need to be
        # re-distributed into new locations.  There might be a way to do this in-place
        # but that's an optimization for future-Steve to look into
        ################################################################################
        old_elements = np.frombuffer(self.hole_lookup_buffer, dtype=self.numpy_dtype)
        
        old_elements = old_elements.copy()
        
        old_lookup_mem = old_elements
        
        old_mem_length = self.mem_length
        
        
        ################################################################################
        # Resize our existing memory to the first prime number larger than the
        # requested value given by new_mem_length
        ################################################################################
        next_prime = find_next_prime(new_mem_length)
    
        self.mem_length = next_prime
        
        hole_lookup_buffer_length = self.mem_length*7 #7 bytes per element
        
        os.ftruncate(self.lookup_fd, hole_lookup_buffer_length)
        
        ################################################################################
        # Close the old mmap and re-map it since we changed the size of our memory file
        # pointed to by self.lookup_fd.  Then point our self.lookup_memory memoryview
        # object to the extended version of where it was already pointing
        ################################################################################
        self.hole_lookup_buffer.close()
        
        self.hole_lookup_buffer = mmap.mmap(self.lookup_fd, hole_lookup_buffer_length)
        
        self.lookup_memory = np.frombuffer(self.hole_lookup_buffer, dtype=self.numpy_dtype)
        
        ################################################################################
        # Zero out the hash table before re-filling it
        ################################################################################
        curr_element.filled_flag = 0
        
        self.lookup_memory[:] = curr_element
        
        self.num_elements = 0
        
        self.num_collisions = 0
        
        
        ################################################################################
        # Re-fill our hash table using the copy of the old one
        ################################################################################
        #num_filled = 0
        
        for idx in range(old_mem_length):
        
            curr_element = old_lookup_mem[idx]
            
            if curr_element.filled_flag:
                
                self.setitem(curr_element.key_i, curr_element.key_j, curr_element.key_k)
                
                #num_filled += 1

        #print("Num filled: ", num_filled)

        #print("HoleGridCustomDict resized to: "+str(self.mem_length)+" "+str(self.num_elements), flush=True)
        
        return



    def close(self):
        
        self.hole_lookup_buffer.close()
        
        #print("Closing file descriptor: ", self.lookup_fd, flush=True)
        
        os.close(self.lookup_fd)


cdef class GalaxyMapCustomDict:
    """
    Description
    ===========
    
    NOTE: i-j-k names are used below, these should be converted to p-q-r names
    
    A dictionary-like object whose purpose is to provide index information
    into the galaxy_map_array object, where we can find the indices of the
    galaxies which correspond to a given p-q-r galaxy map grid cell.
    
    Provides setitem() getitem() and contains() methods for assistance in 
    efficiently retrieving the indexes for a given p-q-r grid cell.
    
    
    Parameters
    ==========
    
    grid_dimensions : length 3 tuple
        the number of cells in the p, q, and r dimensions of the galaxy map grid
        
    lookup_memory : numpy.ndarray of shape (N,)
        Each element of this array is of type _voidfinder_cython_find_next.pxd 
        LOOKUPMEM_t and contains a flag for if the array element is filled,
        p-q-r values for key comparisons, and 2 integers - the offset into the
        galaxy_map_array object corresponding to the p-q-r cell of interest, and
        the number of elements at that offset for the current p-q-r cell.
    
        The value of (N,) is chosen to be the first prime number larger than 2 times
        the number of elements to be inserted into this dictionary, to help reduce
        hash-table collisions.
    
    
    """
  
    def __init__(self, grid_dimensions, resource_dir, starting_cells=8):
        
        cdef LOOKUPMEM_t curr_element
        
        
        ################################################################################
        # First, set up some parameters corresponding to the VoidFinder hole-growing
        # grid, which we use in the hash function to calculate hash table indices
        ################################################################################
        self.i_dim = grid_dimensions[0]
        
        self.j_dim = grid_dimensions[1]
        
        self.k_dim = grid_dimensions[2]
        
        self.jk_mod = self.j_dim*self.k_dim
        
        
        ################################################################################
        # Next, create a memory-mapped file we can use as the shared memory
        # to share among our processes.  Since we're using fork(), the file descriptor
        # which we create here will point to the same file in the child processes
        # meaning we can just pass this whole class across the fork()
        ################################################################################
        next_prime = find_next_prime(2*starting_cells)
        
        self.mem_length = next_prime
        
        self.lookup_fd, LOOKUPMEM_BUFFER_PATH = tempfile.mkstemp(prefix="voidfinder", 
                                                                 dir=resource_dir, 
                                                                 text=False)
        
        lookup_buffer_length = self.mem_length*23
        
        os.ftruncate(self.lookup_fd, lookup_buffer_length)
        
        self.lookup_buffer = mmap.mmap(self.lookup_fd, lookup_buffer_length)
        
        os.unlink(LOOKUPMEM_BUFFER_PATH)
        
        lookup_dtype = [("filled_flag", np.uint8, ()), #() indicates scalar, or length 1 shape
                        ("p", np.int16, ()),
                        ("q", np.int16, ()),
                        ("r", np.int16, ()),
                        ("offset", np.int64, ()),
                        ("num_elements", np.int64, ())]
        # 1 + 2 + 2 + 2 + 8 + 8 = 23 bytes per element
        
        self.numpy_dtype = np.dtype(lookup_dtype, align=False)
        
        self.lookup_memory = np.frombuffer(self.lookup_buffer, dtype=self.numpy_dtype)
        
        ################################################################################
        # Writing a bunch of 0's directly to the self.hole_lookup_buffer did not work
        # I do not know why, but instead we must use the array broadcasting with
        # an element whose flag has been 0'd to properly zero out the hash table
        ################################################################################
        curr_element.filled_flag = 0
        
        self.lookup_memory[:] = curr_element
        
        #self.num_elements = 0
        self.num_elements = Value(c_int64, 0, lock=False)
        
        self.process_local_num_elements = 0
        
        self.num_collisions = 0
            
            
    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.cdivision(True)
    cdef void refresh(self):
        """
        If someone has updated the underlying shared memory, need to refresh the size of our 
        mappings.  Don't need to call the expensive resize() method, just refresh.
        """
        
    
        self.lookup_buffer.resize(self.lookup_buffer.size()) #resize to full file length
        
        self.lookup_memory = np.frombuffer(self.lookup_buffer, dtype=self.numpy_dtype)
        
        self.mem_length = self.lookup_buffer.size() // 23
        
        self.process_local_num_elements = self.num_elements.value
        
        # Whoever updated the memory needs to have also updated the self.num_elements value
        # so we do not update it here
        
        
        
    
    def __len__(self):
        return self.num_elements.value
            
    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.cdivision(True)
    cdef DTYPE_INT64_t custom_hash(self, 
                                   CELL_ID_t i, 
                                   CELL_ID_t j, 
                                   CELL_ID_t k):
        """
        TODO: Update names to p-q-r
        
        Given a cell ID (i,j,k), calculate its hash address in our
        memory array.  Uses the natural grid cell ordering
        0->(0,0,0), 1->(0,0,1), 2->(0,0,2) etc, to calculate the sequential
        grid cell index, then takes that value modulus of the number of slots
        in the memory array.
        """
        
        cdef DTYPE_INT64_t index
        cdef DTYPE_INT64_t hash_addr
        
        index = self.jk_mod * <DTYPE_INT64_t>i + \
                self.k_dim * <DTYPE_INT64_t>j + \
                <DTYPE_INT64_t>k
        
        hash_addr = index % self.mem_length
        
        ############################################################
        # the @cython.cdivision(True) directive, while providing
        # speedup benefits, means that -1 % 5 = -1, not 4 as in
        # python, so we have to explicitly check for negative vals
        ############################################################
        if hash_addr < 0:
            
            hash_addr += self.mem_length
        
        return hash_addr
    
    
    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.cdivision(True)
    cpdef DTYPE_B_t contains(self,
                             CELL_ID_t i, 
                             CELL_ID_t j, 
                             CELL_ID_t k):
        """
        TODO: update names to p-q-r
        """
        
        cdef DTYPE_INT64_t hash_addr
        
        cdef DTYPE_INT64_t hash_offset
        
        cdef DTYPE_INT64_t curr_hash_addr
        
        cdef LOOKUPMEM_t curr_element
        
        hash_addr = self.custom_hash(i, j, k)
        
        for hash_offset in range(self.mem_length):
            
            curr_hash_addr = (hash_addr + hash_offset) % self.mem_length
            
            curr_element = self.lookup_memory[curr_hash_addr]
            
            if not curr_element.filled_flag:
                
                return False
            
            else:
                
                if curr_element.key_i == i and \
                   curr_element.key_j == j and \
                   curr_element.key_k == k:
                    
                    return True
                        
        return False
    
    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.cdivision(True)
    cpdef OffsetNumPair getitem(self,
                                CELL_ID_t i, 
                                CELL_ID_t j, 
                                CELL_ID_t k) except *:
        """
        TODO: update names to p-q-r
        """
        
        cdef DTYPE_INT64_t hash_addr
        
        cdef DTYPE_INT64_t hash_offset
        
        cdef DTYPE_INT64_t curr_hash_addr
        
        cdef LOOKUPMEM_t curr_element
        
        cdef OffsetNumPair out
        
        hash_addr = self.custom_hash(i, j, k)
        
        for hash_offset in range(self.mem_length):
            
            curr_hash_addr = (hash_addr + hash_offset) % self.mem_length
            
            curr_element = self.lookup_memory[curr_hash_addr]
            
            if not curr_element.filled_flag:
                
                raise KeyError("key: ", i, j, k, " not in dictionary")
            
            else:
                
                if curr_element.key_i == i and \
                   curr_element.key_j == j and \
                   curr_element.key_k == k:
                    
                    out.offset = curr_element.offset
                    out.num_elements = curr_element.num_elements
                    
                    return out
                
        raise KeyError("key: ", i, j, k, " not in dictionary")
    
    
    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.cdivision(True)
    cpdef void setitem(self, 
                       CELL_ID_t i,
                       CELL_ID_t j,
                       CELL_ID_t k, 
                       DTYPE_INT64_t offset,
                       DTYPE_INT64_t num_elements):
        """
        TODO: update names to p-q-r
        
        Will always succeed since we initialize the length of
        self.lookup_memory to be longer than the number of items
        """
        
        cdef DTYPE_INT64_t hash_addr
        
        cdef DTYPE_INT64_t hash_offset
        
        cdef DTYPE_INT64_t curr_hash_addr
        
        cdef LOOKUPMEM_t curr_element
        
        cdef LOOKUPMEM_t out_element
        
        cdef DTYPE_B_t first_try = True
        
        out_element.filled_flag = 1
        out_element.key_i = i
        out_element.key_j = j
        out_element.key_k = k
        out_element.offset = offset
        out_element.num_elements = num_elements
                
        hash_addr = self.custom_hash(i, j, k)
        
        for hash_offset in range(self.mem_length):
            
            curr_hash_addr = (hash_addr + hash_offset) % self.mem_length
            
            curr_element = self.lookup_memory[curr_hash_addr]
            
            if not curr_element.filled_flag:
                
                if not first_try:
                    
                    self.num_collisions += 1
                
                self.lookup_memory[curr_hash_addr] = out_element
                
                self.num_elements.value += 1
                
                self.process_local_num_elements += 1
                
                break
            
            else:
                
                if curr_element.key_i == i and \
                   curr_element.key_j == j and \
                   curr_element.key_k == k:
                    
                    if not first_try:
                    
                        self.num_collisions += 1
                    
                    self.lookup_memory[curr_hash_addr] = out_element
                    
                    # Don't increment num_elements here because we are
                    # overwriting an existing key
                
                    break
                
            first_try = False
    
        if self.num_elements.value >= (<DTYPE_INT64_t>(0.60*self.mem_length)):
            
            self.resize(2*self.mem_length)
            
        return
    
    
          
    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.cdivision(True)
    cdef void resize(self, DTYPE_INT64_t new_mem_length):
    
        cdef LOOKUPMEM_t curr_element
        
        cdef ITYPE_t idx
        
        cdef LOOKUPMEM_t[:] old_lookup_mem
        
        ########################################################################
        # There may be a better way, but for now, we're just going to make a 
        # copy of the 'old memory' since the key-value pairs it is storing need 
        # to be re-distributed into new locations.  There might be a way to do 
        # this in-place but that's an optimization for future-Steve to look into
        ########################################################################
        old_elements = np.frombuffer(self.lookup_buffer, dtype=self.numpy_dtype)
        
        old_elements = old_elements.copy()
        
        old_lookup_mem = old_elements
        
        old_mem_length = self.mem_length
        
        
        ########################################################################
        # Resize our existing memory to the first prime number larger than the
        # requested value given by new_mem_length
        ########################################################################
        next_prime = find_next_prime(new_mem_length)
    
        self.mem_length = next_prime
        
        lookup_buffer_length = self.mem_length*23 #23 bytes per element
        
        os.ftruncate(self.lookup_fd, lookup_buffer_length)
        
        ########################################################################
        # Close the old mmap and re-map it since we changed the size of our 
        # memory file pointed to by self.lookup_fd.  Then point our 
        # self.lookup_memory memoryviewobject to the extended version of where 
        # it was already pointing
        ########################################################################
        curr_size = self.lookup_buffer.size()

        self.lookup_buffer.close()
        
        self.lookup_buffer = mmap.mmap(self.lookup_fd, curr_size)#lookup_buffer_length)
        #self.lookup_buffer.resize(self.lookup_buffer.size()) #resize to full file length
        
        self.lookup_memory = np.frombuffer(self.lookup_buffer, dtype=self.numpy_dtype)
        
        ########################################################################
        # Zero out the hash table before re-filling it
        ########################################################################
        curr_element.filled_flag = 0
        
        self.lookup_memory[:] = curr_element
        
        self.num_elements.value = 0
        
        self.process_local_num_elements = 0
        
        self.num_collisions = 0
        
        
        ########################################################################
        # Re-fill our hash table using the copy of the old one
        ########################################################################
        #num_filled = 0
        
        for idx in range(old_mem_length):
        
            curr_element = old_lookup_mem[idx]
            
            if curr_element.filled_flag:
                
                self.setitem(curr_element.key_i, 
                             curr_element.key_j, 
                             curr_element.key_k,
                             curr_element.offset,
                             curr_element.num_elements)
                
                #num_filled += 1

        #print("Num filled: ", num_filled)

        #print("HoleGridCustomDict resized to: "+str(self.mem_length)+" "+str(self.num_elements), flush=True)
        
        return



    def close(self):
        
        self.lookup_buffer.close()
        
        #print("Closing file descriptor: ", self.lookup_fd, flush=True)
        
        os.close(self.lookup_fd)
    
    
    
    
    
cdef class GalaxyMap:
    '''
    Right now this is a glorified container class for passing around a handful of
    object references in memory.
    
    Changing this so that it implements the logic for regular mode versus
    periodic mode where we need to be able to resize our arrays in shared 
    memory.
    
    This will also be the interface of the GalaxyMapCustomDict to the
    _gen_shell and _gen_cube methods where the logic of filtering out
    cell IDs is implemented, so that when mask_mode==periodic, we
    store off even the cells whose num_elements is 0 so that we can
    be sure that we have created them if they didn't yet exist.
    
    
    '''
    
    def __init__(self,
                 resource_dir,
                 mask_mode,
                 galaxy_coords, 
                 coord_min, 
                 dl,
                 galaxy_map,
                 galaxy_map_array):
        """
        TODO:  Appropriately address the existing reliance on casting to
               an integer. I believe python always rounds towards 0, and we
               are implicitly relying on this behavior.  Let's make this explicit
               so I can do it correctly in cython
        """
        self.mask_mode = mask_mode
        
        self.coord_min = coord_min
        
        #This value is the galaxy_map_grid_edge_length, not hole_grid_edge_length
        self.dl = dl
        
        self.galaxy_map = galaxy_map
        
        self.reference_point_ijk = np.empty((1,3), dtype=np.int16)
            
        self.shell_boundaries_xyz = np.empty((2,3), dtype=np.float64)
        
        self.cell_center_xyz = np.empty((1,3), dtype=np.float64)
        
        self.update_lock = RLock()
        
        ################################################################################
        # First, memmap the galaxy coordinate array
        ################################################################################
        wall_galaxies_coords_fd, WCOORD_BUFFER_PATH = tempfile.mkstemp(prefix="voidfinder", 
                                                                       dir=resource_dir, 
                                                                       text=False)
        
        
        num_galaxies = galaxy_coords.shape[0]
    
        w_coord_buffer_length = num_galaxies*3*8 # 3 for xyz and 8 for float64
        
        os.ftruncate(wall_galaxies_coords_fd, w_coord_buffer_length)
        
        w_coord_buffer = mmap.mmap(wall_galaxies_coords_fd, w_coord_buffer_length)
        
        w_coord_buffer.write(galaxy_coords.astype(np.float64).tobytes())
        
        del galaxy_coords
        
        galaxy_coords = np.frombuffer(w_coord_buffer, dtype=np.float64)
        
        galaxy_coords.shape = (num_galaxies, 3)
        
        os.unlink(WCOORD_BUFFER_PATH)
        
        self.wall_galaxy_coords = galaxy_coords
        
        self.wall_galaxy_buffer = w_coord_buffer
        
        self.num_wall_galaxies = num_galaxies
        
        self.wall_galaxies_coords_fd = wall_galaxies_coords_fd
        
        
        
        ################################################################################
        # Next, memmap the galaxy array helper
        ################################################################################
        gma_fd, GMA_BUFFER_PATH = tempfile.mkstemp(prefix="voidfinder", 
                                                   dir=resource_dir, 
                                                   text=False)
    
        
        num_gma_indices = galaxy_map_array.shape[0]
        
        gma_buffer_length = num_gma_indices*8 # 8 for int64
        
        os.ftruncate(gma_fd, gma_buffer_length)
        
        gma_buffer = mmap.mmap(gma_fd, gma_buffer_length)
        
        gma_buffer.write(galaxy_map_array.astype(np.int64).tobytes())
        
        del galaxy_map_array
        
        os.unlink(GMA_BUFFER_PATH)
        
        galaxy_map_array = np.frombuffer(gma_buffer, dtype=np.int64)
        
        galaxy_map_array.shape = (num_gma_indices,)
        
        self.galaxy_map_array = galaxy_map_array
        
        self.galaxy_map_array_buffer = gma_buffer
        
        self.num_gma_indices = num_gma_indices
        
        self.gma_fd = gma_fd
        
        
        ################################################################################
        # In periodic mode, create a secondary galaxy map for the dynamic portion
        # of cell lookups
        ################################################################################
        
        if self.mask_mode == 2:
            
            grid_dims = (self.galaxy_map.i_dim, self.galaxy_map.j_dim, self.galaxy_map.k_dim)
            
            grid_dims_inner = (self.galaxy_map.i_dim -1, self.galaxy_map.j_dim - 1, self.galaxy_map.k_dim - 1)
            
            starting_cells = np.prod(grid_dims) - np.prod(grid_dims_inner)
            
            self.galaxy_map_2 = GalaxyMapCustomDict(grid_dims, resource_dir, 2*starting_cells)
        
        
        
        
        
    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.cdivision(True)
    cpdef DTYPE_B_t contains(self,
                             CELL_ID_t i, 
                             CELL_ID_t j, 
                             CELL_ID_t k):
        """
        Return True when a cell actually contains galaxies.
        Mostly for periodic mode - a cell will always exist
        but we need to return False if it doesn't contain 
        any galaxies, we can't just return True all the time.
        
        However, I believe we will always call contains on a cell
        before we call getitem on that cell, which simplifies
        getitem()
        """
        
        cdef OffsetNumPair curr_item
        
        cdef DTYPE_B_t in_bounds
        
        if self.mask_mode == 0 or self.mask_mode == 1:
            
            return self.galaxy_map.contains(i, j, k)
        
        elif self.mask_mode == 2:
            '''
            self.update_lock.acquire()
            
            if self.galaxy_map.process_local_num_elements != self.galaxy_map.num_elements.value:
                self.galaxy_map.refresh()
                self.refresh()
                
            try:
                curr_item = self.galaxy_map.getitem(i, j, k)
                
            except KeyError:
                
                self.add_cell_periodic(i, j, k)
                
                curr_item = self.galaxy_map.getitem(i, j, k)
                
                
            self.update_lock.release()
                
            if curr_item.num_elements > 0:
                
                return True
            else:
                
                return False
            '''
            
            in_bounds = self.cell_in_source(i, j, k)
            
            if in_bounds:
                
                
                try:
                
                    return self.galaxy_map.contains(i,j,k)
                except KeyError:
                    print("Should not be getting keyerror: ", i,j,k)
                    
            else:
                self.update_lock.acquire()
                if self.galaxy_map_2.process_local_num_elements != self.galaxy_map_2.num_elements.value:
                    self.galaxy_map_2.refresh()
                    self.refresh()
                try:
                    curr_item = self.galaxy_map_2.getitem(i, j, k)
                    
                except KeyError:
                    
                    self.add_cell_periodic(i, j, k)
                    
                    curr_item = self.galaxy_map_2.getitem(i, j, k)
                    
                    
                self.update_lock.release()
                    
                if curr_item.num_elements > 0:
                    
                    return True
                else:
                    
                    return False
            
        
        
        
        
        
    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.cdivision(True)
    cdef OffsetNumPair getitem(self,
                               CELL_ID_t i, 
                               CELL_ID_t j, 
                               CELL_ID_t k):
                               
        """
        I believe we will always call contains() on a cell
        before we call getitem() on that cell, so it should be
        simple here
        """
        
        cdef OffsetNumPair curr_item
        
        cdef DTYPE_B_t in_bounds
        
        
        if self.mask_mode == 0 or self.mask_mode == 1:
        
            return self.galaxy_map.getitem(i, j, k)
        
        elif self.mask_mode == 2:
            
            '''
            self.update_lock.acquire()
            
            if self.galaxy_map.process_local_num_elements != self.galaxy_map.num_elements.value:
                self.galaxy_map.refresh()
                self.refresh()
            
            curr_item = self.galaxy_map.getitem(i, j, k)
        
            self.update_lock.release()
            
            return curr_item
            '''
            
            in_bounds = self.cell_in_source(i, j, k)
            
            if in_bounds:
                return self.galaxy_map.getitem(i, j, k)
            else:
                self.update_lock.acquire()
            
                if self.galaxy_map_2.process_local_num_elements != self.galaxy_map_2.num_elements.value:
                    self.galaxy_map_2.refresh()
                    self.refresh()
                
                curr_item = self.galaxy_map_2.getitem(i, j, k)
            
                self.update_lock.release()
                
                return curr_item
            
        
            
            
        
    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.cdivision(True)
    cdef void setitem(self, 
                      CELL_ID_t i,
                      CELL_ID_t j,
                      CELL_ID_t k, 
                      DTYPE_INT64_t offset,
                      DTYPE_INT64_t num_elements):
                       
        # Right now we don't have any multiprocessing synchronization
        # on setitem() because its only being used in single-threaded
        # or already-locked locations
        self.galaxy_map.setitem(i, j, k, offset, num_elements)
        
        
        
    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.cdivision(True)
    cdef DTYPE_B_t cell_in_source(self, CELL_ID_t i, CELL_ID_t j, CELL_ID_t k):
        """
        Return True if the cell is in the xyz_limits and false if its
        just part of the periodic infinity
        """
    
        cdef DTYPE_B_t in_bounds = 0
        
        if i >= 0 and i < self.galaxy_map.i_dim and \
           j >= 0 and j < self.galaxy_map.j_dim and \
           k >= 0 and k < self.galaxy_map.k_dim:
            in_bounds = 1
            
        return in_bounds
    
    
        
    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.cdivision(True)
    cdef void add_cell_periodic(self,
                                CELL_ID_t i,
                                CELL_ID_t j,
                                CELL_ID_t k):
        """
        This method is assumed to be called within the context
        of the update_lock for synchronization purposes.
        
        We just called galaxy_map.getitem(i,j,k) and it raised a KeyError meaning
        the cell didn't exist.
        
        Its now our job to add that cell to the galaxy map, add 0 or more galaxies
        to the wall_galaxies_coords array (with appropriate SHIFT), and add new idxs 
        to the galaxy_map_array and then the appropriate (offset, num_elements) 
        into the galaxy_map_cell_dict
        """
        
        cdef CELL_ID_t source_i, source_j, source_k
        
        cdef DTYPE_B_t in_bounds = 0
        
        cdef OffsetNumPair curr_item
        
        cdef DTYPE_INT64_t n_bytes, new_offset, num_new_indices, old_num_gals
        
        cdef ITYPE_t idx, kdx, gma_idx, curr_gal_idx, out_gma_idx
        
        cdef DTYPE_F64_t[:] source_center_xyz, curr_center_xyz, shift_xyz
        
        
        
        
        
        ################################################################################
        # We can figure out if the cell requested is part of the source data 
        # (aka within the xyz_limits) or a virtual/external cell
        # Figure that out and also map the requested i,j,k to its source i,j,k
        ################################################################################
        '''
        if i >= 0 and i < self.galaxy_map.i_dim and \
           j >= 0 and j < self.galaxy_map.j_dim and \
           k >= 0 and k < self.galaxy_map.k_dim:
            in_bounds = 1
        '''
        in_bounds = self.cell_in_source(i,j,k)
        
        #Since cython.cdivision(True) have to check for less than 0
        source_i = i % self.galaxy_map.i_dim
        if source_i < 0:
            source_i += self.galaxy_map.i_dim
            
        source_j = j % self.galaxy_map.j_dim
        if source_j < 0:
            source_j += self.galaxy_map.j_dim
            
        source_k = k % self.galaxy_map.k_dim
        if source_k < 0:
            source_k += self.galaxy_map.k_dim
        
        
        ################################################################################
        # Next, make sure our own internal references/buffers are up to date
        # self.galaxy_map.refresh() has already been called within the context
        # of the current lock acquire so that doesn't need to be called
        ################################################################################
        
        self.refresh()
        
        
        ################################################################################
        # Now that all our shared memory values are current, we can
        # add the new rows (or 0 new rows)
        #
        # If in_bounds == 1, we are adding a cell that was in our original survey
        # but had no galaxies - we know this because if the cell we're currently
        # adding is within the bounds, we're only adding it because it wasn't added
        # in the creation of GalaxyMapCustomDict because it had 0 galaxies in it
        ################################################################################
        new_offset = self.num_gma_indices
        
        old_num_gals = self.wall_galaxy_coords.shape[0]
        
        if in_bounds == 1:
            
            num_new_indices = 0
            
        # We were asked to add a virtual cell
        elif in_bounds == 0:
            
        
            # Try to get the source i,j,k from the dict, if that fails we know
            # it also has 0 elements
            try:
                
                #Keep galaxy_map not galaxy_map_2 here because it contains all the source
                curr_item = self.galaxy_map.getitem(source_i, source_j, source_k)
                
            except KeyError:
            
                #If the source ijk lookup fails, we know it had 0 galaxies
                num_new_indices = 0
                
            else:
                
                # If we got a valid offset,num_elements then we need to calculate how much
                # space they take in the wall_galaxy_coords and galaxy_map_array and copy
                # over the new chunk from the old galaxies
                # WITH GALACTIC SHIFT
                
                #new_offset = curr_item.offset
                num_new_indices = curr_item.num_elements
                
                
                # If we picked a source cell that already existed but had 0 elements we
                # dont have to calculate all this stuff
                if num_new_indices > 0:
                    
                    wall_gal_nbytes = 3*8*num_new_indices
                    gal_array_nbytes = 8*num_new_indices
                    
                    
                    # Extend our shared memory, these guys are basically being "appended" to
                    
                    self.wall_galaxy_buffer.resize(self.wall_galaxy_buffer.size()+wall_gal_nbytes)
            
                    galaxy_coords = np.frombuffer(self.wall_galaxy_buffer, dtype=np.float64)
                    
                    self.num_wall_galaxies = self.wall_galaxy_buffer.size()/(3*8)
                    
                    galaxy_coords.shape = (self.num_wall_galaxies, 3)
                    
                    self.wall_galaxy_coords = galaxy_coords
                    
                    
                    
                    self.galaxy_map_array_buffer.resize(self.galaxy_map_array_buffer.size()+gal_array_nbytes)
                    
                    galaxy_map_array = np.frombuffer(self.galaxy_map_array_buffer, dtype=np.int64)
                    
                    self.num_gma_indices = self.galaxy_map_array_buffer.size()/8
                    
                    galaxy_map_array.shape = (self.num_gma_indices,)
                    
                    self.galaxy_map_array = galaxy_map_array
                    
                    # Now we need to calculate the shift in position between the source ijk 
                    # and the current ijk we're being asked to fill to add it to all the galaxies 
                    # we are copying over
                    
                    source_center_xyz = np.zeros(3, dtype=np.float64)
                    curr_center_xyz = np.zeros(3, dtype=np.float64)
                    shift_xyz = np.zeros(3, dtype=np.float64)
                    
                    
                    
                    
                    source_center_xyz[0] = (<DTYPE_F64_t>source_i + 0.5)*self.dl + self.coord_min[0,0]
                    source_center_xyz[1] = (<DTYPE_F64_t>source_j + 0.5)*self.dl + self.coord_min[0,1]
                    source_center_xyz[2] = (<DTYPE_F64_t>source_k + 0.5)*self.dl + self.coord_min[0,2]
                    
                    curr_center_xyz[0] = (<DTYPE_F64_t>i + 0.5)*self.dl + self.coord_min[0,0]
                    curr_center_xyz[1] = (<DTYPE_F64_t>j + 0.5)*self.dl + self.coord_min[0,1]
                    curr_center_xyz[2] = (<DTYPE_F64_t>k + 0.5)*self.dl + self.coord_min[0,2]
                    
                    
                    
                    shift_xyz[0] = curr_center_xyz[0] - source_center_xyz[0]
                    shift_xyz[1] = curr_center_xyz[1] - source_center_xyz[1]
                    shift_xyz[2] = curr_center_xyz[2] - source_center_xyz[2]
                    
                    
                    # We now have the new space, but we need to fill it in
                    # Since we're appending to both the wall_galaxy_coords and galaxy_map_array
                    # the output index for both will be the same now
                    
                    for idx, gma_idx in enumerate(range(curr_item.offset, curr_item.offset+num_new_indices)):
                    
                        curr_gal_idx = self.galaxy_map_array[gma_idx]
                        
                        out_gma_idx = new_offset + idx
                        
                        for kdx in range(3):
                        
                            self.wall_galaxy_coords[out_gma_idx,kdx] = self.wall_galaxy_coords[curr_gal_idx,kdx] + shift_xyz[kdx] # PLUS SHIFT
                        
                        self.galaxy_map_array[out_gma_idx] = out_gma_idx
                        
                
        self.galaxy_map_2.setitem(i, j, k, new_offset, num_new_indices)
        
        return
        
        
        
        
    cpdef void refresh(self):
        """
        Resize the shared memory for our wall galaxy buffer and
        galaxy map array
        """
        
        n_bytes = self.wall_galaxy_buffer.size()
        
        self.wall_galaxy_buffer.resize(n_bytes)
        
        galaxy_coords = np.frombuffer(self.wall_galaxy_buffer, dtype=np.float64)
        
        self.num_wall_galaxies = n_bytes/(3*8)
        
        galaxy_coords.shape = (self.num_wall_galaxies, 3)
        
        self.wall_galaxy_coords = galaxy_coords
        
        
        
        
        n_bytes = self.galaxy_map_array_buffer.size()
        
        self.galaxy_map_array_buffer.resize(n_bytes)
        
        galaxy_map_array = np.frombuffer(self.galaxy_map_array_buffer, dtype=np.int64)
        
        self.num_gma_indices = n_bytes/8
        
        galaxy_map_array.shape = (self.num_gma_indices,)
        
        self.galaxy_map_array = galaxy_map_array
        
        
        
        
    def close(self):
        
        self.wall_galaxy_buffer.close()
        
        os.close(self.wall_galaxies_coords_fd)
        
        self.galaxy_map_array_buffer.close()
        
        os.close(self.gma_fd)
        
        
        
        

cdef class NeighborMemory:
    """
    Description
    ===========
    
    This class represents a bundle of memory arrays which are used in the process
    of finding nearest neighbors in the sphere growing process.  Since we don't know
    the maximum number of neighbors returned from a query at the beginning of the code,
    we need a class which is capable of resizing to be larger if necessary, but we also
    don't want to re-allocate memory on every loop.  The arrays in this object are
    intended to work like the Cell_ID_Memory class - they get overwritten on each 
    call to the _query_shell_radius() functions and can resize themselves
    if necessary.  This memory is not used in _query_first() because _query_first()
    always returns exactly 1 neighbor.
    
    Before each call to _query_shell_radius(), the attribute self.next_neigh_idx gets
    reset to 0, and as the append() method is called when new neighbors are found,
    the self.next_neigh_idx gets incremented to represent how many neighbor
    indices are currently in the self.i_nearest attribute.
    
    Parameters
    ==========
    
    max_num_neighbors : int
        maximum number of neighbors we need memory for
        
    """

    def __cinit__(self, size_t max_num_neighbors):
    
        self.max_num_neighbors = max_num_neighbors
        
        self.next_neigh_idx = 0
        
        ################################################################################
        # 1 memory for the neighbor indices
        ################################################################################
        self.i_nearest = <DTYPE_INT64_t*> PyMem_Malloc(max_num_neighbors * sizeof(DTYPE_INT64_t))
        
        if not self.i_nearest:
            
            raise MemoryError()
        
        ################################################################################
        # 2
        ################################################################################
        self.boolean_nearest = <DTYPE_B_t*> PyMem_Malloc(max_num_neighbors * sizeof(DTYPE_B_t))
        
        if not self.boolean_nearest:
            
            raise MemoryError()
        
        
        ################################################################################
        # 3
        ################################################################################
        self.i_nearest_reduced = <ITYPE_t*> PyMem_Malloc(max_num_neighbors * sizeof(ITYPE_t))
        
        if not self.i_nearest_reduced:
            
            raise MemoryError()
        
        ################################################################################
        # 4
        ################################################################################
        self.candidate_minus_A = <DTYPE_F64_t*> PyMem_Malloc(max_num_neighbors * 3 * sizeof(DTYPE_F64_t))
        
        if not self.candidate_minus_A:
            
            raise MemoryError()
        
        
        ################################################################################
        # 5
        ################################################################################
        self.candidate_minus_center = <DTYPE_F64_t*> PyMem_Malloc(max_num_neighbors * 3 * sizeof(DTYPE_F64_t))
        
        if not self.candidate_minus_center:
            
            raise MemoryError()
        
        
        ################################################################################
        # 6
        ################################################################################
        self.bot_ratio = <DTYPE_F64_t*> PyMem_Malloc(max_num_neighbors * sizeof(DTYPE_F64_t))
        
        if not self.bot_ratio:
            
            raise MemoryError()
        
        ################################################################################
        # 7
        ################################################################################
        self.top_ratio = <DTYPE_F64_t*> PyMem_Malloc(max_num_neighbors * sizeof(DTYPE_F64_t))
        
        if not self.top_ratio:
            
            raise MemoryError()
        
        ################################################################################
        # 8
        ################################################################################
        self.x_ratio = <DTYPE_F64_t*> PyMem_Malloc(max_num_neighbors * sizeof(DTYPE_F64_t))
        
        if not self.x_ratio:
            
            raise MemoryError()

        
    cdef void resize(self, size_t max_num_neighbors):
    
    
        #print("Neighbor mem resizing: "+str(max_num_neighbors), flush=True)
    
        self.max_num_neighbors = max_num_neighbors
        
        
        ################################################################################
        # 1
        ################################################################################
        mem = <DTYPE_INT64_t*> PyMem_Realloc(self.i_nearest, max_num_neighbors * sizeof(DTYPE_INT64_t))
            
        if not mem:
            
            raise MemoryError()
        
        # Only overwrite the pointer if the memory was really reallocated.
        # On error (mem is NULL), the original memory has not been freed.
        self.i_nearest = mem
    
    
        ################################################################################
        # 2
        ################################################################################
        mem2 = <DTYPE_B_t*> PyMem_Realloc(self.boolean_nearest, max_num_neighbors * sizeof(DTYPE_B_t))
            
        if not mem2:
            
            raise MemoryError()
        
        self.boolean_nearest = mem2
    
    
        ################################################################################
        # 3
        ################################################################################
        mem3 = <ITYPE_t*> PyMem_Realloc(self.i_nearest_reduced, max_num_neighbors * sizeof(ITYPE_t))
            
        if not mem3:
            
            raise MemoryError()
        
        self.i_nearest_reduced = mem3
        
        ################################################################################
        # 4
        ################################################################################
        mem4 = <DTYPE_F64_t*> PyMem_Realloc(self.candidate_minus_A, max_num_neighbors * 3 * sizeof(DTYPE_F64_t))
            
        if not mem4:
            
            raise MemoryError()
        
        self.candidate_minus_A = mem4
        
        
        ################################################################################
        # 5
        ################################################################################
        mem5 = <DTYPE_F64_t*> PyMem_Realloc(self.candidate_minus_center, max_num_neighbors * 3 * sizeof(DTYPE_F64_t))
            
        if not mem5:
            
            raise MemoryError()
        
        self.candidate_minus_center = mem5
        
        
        ################################################################################
        # 6
        ################################################################################
        mem6 = <DTYPE_F64_t*> PyMem_Realloc(self.bot_ratio, max_num_neighbors * sizeof(DTYPE_F64_t))
            
        if not mem6:
            
            raise MemoryError()
        
        self.bot_ratio = mem6
        
        ################################################################################
        # 7
        ################################################################################
        mem7 = <DTYPE_F64_t*> PyMem_Realloc(self.top_ratio, max_num_neighbors * sizeof(DTYPE_F64_t))
            
        if not mem7:
            
            raise MemoryError()
        
        self.top_ratio = mem7
        
        ################################################################################
        # 8
        ################################################################################
        mem8 = <DTYPE_F64_t*> PyMem_Realloc(self.x_ratio, max_num_neighbors * sizeof(DTYPE_F64_t))
            
        if not mem8:
            
            raise MemoryError()
        
        self.x_ratio = mem8
    
    
    cdef void append(self, DTYPE_INT64_t neigh_idx_val):
    
        if self.next_neigh_idx >= self.max_num_neighbors:
            
            self.resize(50+self.max_num_neighbors)
            
        self.i_nearest[self.next_neigh_idx] = neigh_idx_val
        
        self.next_neigh_idx += 1
        
        
        
        
    
    

    
cdef class Cell_ID_Memory:
    """
    Description
    ===========
    
    TODO: update naming scheme to p-q-r, currently incorrectly using i-j-k
    
    This class is essentially a wrapper to the memory that holds the 
    (p,q,r) cell IDs that need to be searched.
    
    Through the course of running VoidFinder on an algorithm, searching adjacent
    grid cells at the various (p,q,r) locations will require unknown amounts of memory,
    so this class comes with a resize() method.  During a run, the memory held by this class
    will grow larger and larger, but ultimately there has to be 1 galaxy in the survey whose
    grid search for the next nearest neighbor is the farthest.  If this distance possible to 
    compute ahead of time, we would just pre-allocate that many rows, but that would be a 
    very expensive overhead so instead we provide the resize() method.  In practice the number
    of resizes is very small and quickly converges to a reasonable limit.
    
    For example, if the largest grid size that needs to be searched is 50 cells wide we need
    50^3 = 125,000 slots (times 3 for pqr and times sizeof(CELL_ID_t) total bytes, so 750kb for
    a 50x50x50 grid search)
    
    
    Parameters
    ==========
    
    level : int
        level represents the maximum grid "level" that we need memory for.  Since we are in
        3D space, we use a cubic formula.  For level 0 we need 1 cell, for level
        2 we need the 3x3x3 = 27 cell IDs, for level 3 we need the 5x5x5 = 125 cell IDs, and
        in general for level n > 0 we need (2*n + 1)^3 rows of cell ID memory.
        
    """
    def __cinit__(self, size_t level):
        
        
        num_rows = (2*level + 1)**3
        
        self.max_level_mem = level
        
        self.max_level_available = 0
        
        ################################################################################
        # allocate some memory (uninitialised, may contain arbitrary data)
        ################################################################################
        self.data = <CELL_ID_t*> PyMem_Malloc(num_rows * 3 * sizeof(CELL_ID_t))
        
        if not self.data:
            
            raise MemoryError()
        
        ################################################################################
        # 2
        ################################################################################
        self.level_start_idx = <DTYPE_INT64_t*> PyMem_Malloc((level+1)*sizeof(DTYPE_INT64_t))
        
        if not self.level_start_idx:
            
            raise MemoryError()
        
        ################################################################################
        # 3
        ################################################################################
        self.level_stop_idx = <DTYPE_INT64_t*> PyMem_Malloc((level+1)*sizeof(DTYPE_INT64_t))
        
        if not self.level_stop_idx:
            
            raise MemoryError()
        
        ################################################################################
        # 4
        # this ones a little different, we need to store the p-q-r coordinates of the
        # last cell that has been queried, so we only need 3 CELL_ID_t slots.  For
        # initialization, we use the value 2^15-1 since we're using 16-bit integers
        # for the Cell IDs, so the maximum grid size we could support would be
        # (2^15 - 1)^3 = (32767,32767,32767)
        # As it's unlikely we'll ever be running on a grid of size (32767,32767,32767)
        # (which is a grid of 3.51*10^13 elements), we use 32767 as the initialization
        # value for the current pqr cell so it won't match whatever the first
        # cell ID check is.
        ################################################################################
        self.curr_ijk = <CELL_ID_t*> PyMem_Malloc(3 * sizeof(CELL_ID_t))
        
        if not self.curr_ijk:
            
            raise MemoryError()
        
        self.curr_ijk[0] = 32767
        self.curr_ijk[1] = 32767
        self.curr_ijk[2] = 32767
        
        self.total_num_rows = num_rows
        self.num_available_rows = num_rows
        self.next_unused_row_idx = 0
        

    def resize(self, size_t level):
        """
        Allocates num_rows * 3 * sizeof(CELL_ID_t) bytes,
        -PRESERVING THE CURRENT CONTENT- and making a best-effort to
        re-use the original data location.
        """
        
        #print("Cell ID Mem resizing: ", self.max_level_mem, level, flush=True)

        
        if level > <size_t>self.max_level_mem:
            
            num_rows = (2*level+1)**3
            
            ################################################################################
            # 1
            ################################################################################
            mem = <CELL_ID_t*> PyMem_Realloc(self.data, num_rows * 3 * sizeof(CELL_ID_t))
            
            if not mem:
                
                raise MemoryError()
            
            # Only overwrite the pointer if the memory was really reallocated.
            # On error (mem is NULL), the original memory has not been freed.
            self.data = mem
            
            ################################################################################
            # 2
            ################################################################################
            mem2 = <DTYPE_INT64_t*> PyMem_Realloc(self.level_start_idx, (level+1)*sizeof(DTYPE_INT64_t))
            
            if not mem2:
                
                raise MemoryError()
            
            self.level_start_idx = mem2
            
            ################################################################################
            # 3
            ################################################################################
            mem3 = <DTYPE_INT64_t*> PyMem_Realloc(self.level_stop_idx, (level+1)*sizeof(DTYPE_INT64_t))
            
            if not mem3:
                
                raise MemoryError()
            
            self.level_stop_idx = mem3
            
            ################################################################################
            # update counter variables as well
            ################################################################################
            self.total_num_rows = num_rows
            
            self.max_level_mem = level
            

    def __dealloc__(self):
        
        PyMem_Free(self.data)  # no-op if self.data is NULL
        PyMem_Free(self.level_start_idx)
        PyMem_Free(self.level_stop_idx)







@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
@cython.profile(False)
cdef DistIdxPair _query_first(CELL_ID_t[:,:] reference_point_ijk,
                              DTYPE_F64_t[:,:] coord_min,
                              DTYPE_F64_t dl,
                              DTYPE_F64_t[:,:] shell_boundaries_xyz,
                              DTYPE_F64_t[:,:] cell_center_xyz,
                              GalaxyMap galaxy_map,
                              Cell_ID_Memory cell_ID_mem,
                              DTYPE_F64_t[:,:] reference_point_xyz
                              ):
    """
    
    TODO: update names from ijk to pqr
    
    Description
    ===========
    
    Only called once in _voidfinder_cython.main_algorithm()
    
    Finds first nearest neighbor for the given reference point
    
    NOTE:  This function is OK as a "find first only" setup because
    we're only ever going to give it data points which are Cell centers
    and not data points from w_coord, if we gave it a point from w_coord
    it would always just return that same point which would be dumb, but
    we're ok cause we're not gonna do that.
    
    
    Parameters
    ==========
    
    reference_point_xyz : ndarray of shape (1,3)
        the point in xyz coordinates of whom we would like to find
        the nearest neighbors for
        
    """
    
    
    ################################################################################
    # Convert our query point from xyz to pqr space
    ################################################################################
    
    reference_point_ijk[0,0] = <CELL_ID_t>((reference_point_xyz[0,0] - coord_min[0,0])/dl)
    reference_point_ijk[0,1] = <CELL_ID_t>((reference_point_xyz[0,1] - coord_min[0,1])/dl)
    reference_point_ijk[0,2] = <CELL_ID_t>((reference_point_xyz[0,2] - coord_min[0,2])/dl)
    
    ################################################################################
    # All the variables we need cdef'd
    ################################################################################
    cdef DTYPE_INT32_t current_shell = -1
    
    cdef DTYPE_B_t check_next_shell = True
    
    cdef ITYPE_t neighbor_idx = 0
    
    cdef DTYPE_F64_t neighbor_dist_xyz_sq = INFINITY
    
    cdef DTYPE_F64_t neighbor_dist_xyz = INFINITY
    
    cdef DTYPE_F64_t min_containing_radius_xyz
    
    cdef ITYPE_t cell_ID_idx
    
    cdef ITYPE_t offset, num_elements
    
    cdef OffsetNumPair curr_offset_num_pair
    
    cdef DistIdxPair return_vals
    
    cdef ITYPE_t idx
    
    cdef DTYPE_F64_t dist_sq
    
    cdef DTYPE_F64_t temp1
    
    cdef DTYPE_F64_t temp2
    
    cdef DTYPE_F64_t temp3
    
    cdef ITYPE_t potential_neighbor_idx
    
    cdef DTYPE_F64_t[:] potential_neighbor_xyz
    
    cdef DTYPE_INT64_t num_cell_IDs
    
    cdef DTYPE_INT64_t cell_start_row, cell_end_row
    
    cdef CELL_ID_t id1, id2, id3
    
    ################################################################################
    # Iterate through the grid cells shell-by-shell growing outwards until we find
    # the nearest neighbor to our reference_point_xyz
    ################################################################################
    while check_next_shell:
        
        current_shell += 1
        
        _gen_shell_boundaries(shell_boundaries_xyz,
                              cell_center_xyz,
                              coord_min,
                              dl,
                              reference_point_ijk,
                              current_shell)
        
        min_containing_radius_xyz = _min_contain_radius(shell_boundaries_xyz, 
                                                        reference_point_xyz)
        
        cell_start_row, cell_end_row = _gen_shell(reference_point_ijk, 
                                                  current_shell,
                                                  cell_ID_mem,
                                                  galaxy_map)
        
        ################################################################################
        # When we iterate through the cell IDs below, we won't get any non-existent
        # ones because the cell_ID_mem object has already checked the cell IDs against
        # the galaxy map to confirm they are populated with galaxies
        ################################################################################
        for cell_ID_idx in range(<ITYPE_t>cell_start_row, <ITYPE_t>cell_end_row):
            
            id1 = cell_ID_mem.data[3*cell_ID_idx]
            id2 = cell_ID_mem.data[3*cell_ID_idx+1]
            id3 = cell_ID_mem.data[3*cell_ID_idx+2]
            
            curr_offset_num_pair = galaxy_map.getitem(id1, id2, id3)
            
            offset = curr_offset_num_pair.offset
            
            num_elements = curr_offset_num_pair.num_elements
            
            for idx in range(num_elements):
                
                potential_neighbor_idx = <ITYPE_t>(galaxy_map.galaxy_map_array[offset+idx])
                
                potential_neighbor_xyz = galaxy_map.wall_galaxy_coords[potential_neighbor_idx]
                
                temp1 = potential_neighbor_xyz[0] - reference_point_xyz[0,0]
                temp2 = potential_neighbor_xyz[1] - reference_point_xyz[0,1]
                temp3 = potential_neighbor_xyz[2] - reference_point_xyz[0,2]
                
                dist_sq = temp1*temp1 + temp2*temp2 + temp3*temp3
                
                if dist_sq < neighbor_dist_xyz_sq:
                    
                    neighbor_idx = potential_neighbor_idx
                    
                    neighbor_dist_xyz_sq = dist_sq
                    
                    neighbor_dist_xyz = sqrt(dist_sq)
                    
                    ################################################################################
                    # Don't need to check against the min_containing_radius here because we want to 
                    # check everybody in this shell, since even if this guy matches our criteria
                    # for the min_containing_radius_xyz, someone else in this batch could be closer
                    # and therefore be the true result
                    ################################################################################
                            
        if neighbor_dist_xyz < min_containing_radius_xyz:
            
            check_next_shell = False

    return_vals.idx = neighbor_idx
    
    return_vals.dist = neighbor_dist_xyz
    
    return return_vals
                    






@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
@cython.profile(False)
cdef void _query_shell_radius(CELL_ID_t[:,:] reference_point_ijk,
                              DTYPE_F64_t[:,:] coord_min, 
                              DTYPE_F64_t dl,
                              GalaxyMap galaxy_map,
                              Cell_ID_Memory cell_ID_mem,
                              NeighborMemory neighbor_mem,
                              DTYPE_F64_t[:,:] reference_point_xyz, 
                              DTYPE_F64_t search_radius_xyz
                              ):
    """
    
    TODO: update names from ijk to pqr
    
    Description
    ===========
    
    Find all the neighbors within a given radius of a reference point.
    
    Since we're finding galaxies, we are using the parameters of the GalaxyMap
    search space (PQR), not the hole grid space (IJK)
    
    Only used in find_next_galaxy() (called once in a loop)
    When find_next_galaxy() propagates the center out of a search grid cell
    it will have to reset the memory in the cell_ID_mem, but while it remains
    in that cell the _gen_cube and _gen_shell functions will re-use that memory
    
    
    
    Returns
    =======
    
    Fills in neighbor candidate indices into the neighbor_mem.i_nearest
    array using the neighbor_mem.append() method
    
    
    """
    ################################################################################
    # Convert from xyz space to pqr space
    ################################################################################
    reference_point_ijk[0,0] = <CELL_ID_t>((reference_point_xyz[0,0] - coord_min[0,0])/dl)
    reference_point_ijk[0,1] = <CELL_ID_t>((reference_point_xyz[0,1] - coord_min[0,1])/dl)
    reference_point_ijk[0,2] = <CELL_ID_t>((reference_point_xyz[0,2] - coord_min[0,2])/dl)
         
    ################################################################################
    # declare our working variables
    ################################################################################
    cdef DTYPE_INT32_t max_shell
    
    cdef ITYPE_t cell_ID_idx
    
    cdef ITYPE_t idx
    
    cdef ITYPE_t curr_galaxy_idx
    
    cdef ITYPE_t offset, num_elements
    
    cdef OffsetNumPair curr_offset_num_pair
    
    cdef DTYPE_INT64_t num_cell_IDs
    
    cdef CELL_ID_t id1, id2, id3
    
    cdef DTYPE_F64_t search_radius_xyz_sq = search_radius_xyz*search_radius_xyz
    
    cdef DTYPE_INT64_t[:] curr_galaxies_idxs
    
    cdef DTYPE_F64_t[:] galaxy_xyz
    
    cdef DTYPE_F64_t temp1, temp2, temp3, temp4
    
    cdef DTYPE_F64_t dist_sq
    
    cdef DTYPE_F64_t[3] cell_ijk_in_xyz
    
    ################################################################################
    # Calculate the max shell needed to search
    # fill in implementation details here, using component-wise max for something
    ################################################################################
    cell_ijk_in_xyz[0] = (<DTYPE_F64_t>reference_point_ijk[0,0] + 0.5)*dl + coord_min[0,0]
    cell_ijk_in_xyz[1] = (<DTYPE_F64_t>reference_point_ijk[0,1] + 0.5)*dl + coord_min[0,1]
    cell_ijk_in_xyz[2] = (<DTYPE_F64_t>reference_point_ijk[0,2] + 0.5)*dl + coord_min[0,2]
    
    
    temp1 = fabs((cell_ijk_in_xyz[0] - reference_point_xyz[0,0])/dl)
    temp4 = temp1
    temp2 = fabs((cell_ijk_in_xyz[1] - reference_point_xyz[0,1])/dl)
    if temp2 > temp4:
        temp4 = temp2
    temp3 = fabs((cell_ijk_in_xyz[2] - reference_point_xyz[0,2])/dl)
    if temp3 > temp4:
        temp4 = temp3
    
    
    max_shell = <DTYPE_INT32_t>ceil((search_radius_xyz - (0.5-temp4)*dl)/dl)
        

    ################################################################################
    # Since we are querying based on radius, we can calculate the maximum grid
    # shape we are going to search this time, so use _gen_cube to generate all
    # the grid cells.  Just like in _query_first(), the cell_ID_mem object
    # checks all the grid cells against the galaxy map so that we will only iterate
    # through cell IDs which actually exist here.
    ################################################################################
    num_cell_IDs = _gen_cube(reference_point_ijk, 
                             max_shell,
                             cell_ID_mem,
                             galaxy_map)
    
    for cell_ID_idx in range(<ITYPE_t>num_cell_IDs):
        
        id1 = cell_ID_mem.data[3*cell_ID_idx]
        id2 = cell_ID_mem.data[3*cell_ID_idx+1]
        id3 = cell_ID_mem.data[3*cell_ID_idx+2]
        
        curr_offset_num_pair = galaxy_map.getitem(id1, id2, id3)
            
        offset = curr_offset_num_pair.offset
        
        num_elements = curr_offset_num_pair.num_elements
        
        for idx in range(num_elements):
            
            curr_galaxy_idx = <ITYPE_t>galaxy_map.galaxy_map_array[offset+idx]
            
            galaxy_xyz = galaxy_map.wall_galaxy_coords[curr_galaxy_idx]
            
            temp1 = galaxy_xyz[0] - reference_point_xyz[0,0]
            temp2 = galaxy_xyz[1] - reference_point_xyz[0,1]
            temp3 = galaxy_xyz[2] - reference_point_xyz[0,2]
            
            dist_sq = temp1*temp1 + temp2*temp2 + temp3*temp3
            
            if dist_sq < search_radius_xyz_sq:
                
                neighbor_mem.append(curr_galaxy_idx)
                
    return      







@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
@cython.profile(False)
cdef void _gen_shell_boundaries(DTYPE_F64_t[:,:] shell_boundaries_xyz, 
                                DTYPE_F64_t[:,:] cell_center_xyz,
                                DTYPE_F64_t[:,:] coord_min,
                                DTYPE_F64_t dl,
                                CELL_ID_t[:,:] center_ijk, 
                                DTYPE_INT64_t level
                                ):
    """
    TODO: update names from ijk to pqr
    
    Description
    ===========
    Calculate the xyz center of the cell given the ijk center, calculate the xyz arm length
    of the distance from the center of shell 0 to the edge of shell 'level', then add
    that arm length to the xyz in the first row and subtract it in the 2nd row to get
    a pair of maximal and minimal bounding points for comparison
    """
    
    cell_center_xyz[0,0] = (<DTYPE_F64_t>center_ijk[0,0] + 0.5)*dl + coord_min[0,0]
    cell_center_xyz[0,1] = (<DTYPE_F64_t>center_ijk[0,1] + 0.5)*dl + coord_min[0,1]
    cell_center_xyz[0,2] = (<DTYPE_F64_t>center_ijk[0,2] + 0.5)*dl + coord_min[0,2]
    
    cdef DTYPE_F64_t temp1 = 0.5*dl + <DTYPE_F64_t>level*dl
    
    shell_boundaries_xyz[0,0] = cell_center_xyz[0,0] + temp1
    shell_boundaries_xyz[0,1] = cell_center_xyz[0,1] + temp1
    shell_boundaries_xyz[0,2] = cell_center_xyz[0,2] + temp1
    shell_boundaries_xyz[1,0] = cell_center_xyz[0,0] - temp1
    shell_boundaries_xyz[1,1] = cell_center_xyz[0,1] - temp1
    shell_boundaries_xyz[1,2] = cell_center_xyz[0,2] - temp1
    
    return
        
        
        
        

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
@cython.profile(False)
cdef DTYPE_F64_t _min_contain_radius(DTYPE_F64_t[:,:] shell_boundaries_xyz, 
                                     DTYPE_F64_t[:,:] reference_point_xyz
                                     ) except *:
    """
    Description
    ===========
    
    Find the minimum distance from our reference point to the existing shell boundary
    given by self.shell_boundaries_xyz
    """
    
    cdef DTYPE_F64_t running_min = INFINITY
    cdef DTYPE_F64_t temp
    
    running_min = fabs(shell_boundaries_xyz[0,0] - reference_point_xyz[0,0])
    
    temp = fabs(shell_boundaries_xyz[0,1] - reference_point_xyz[0,1])
    if temp < running_min:
        running_min = temp
        
    temp = fabs(shell_boundaries_xyz[0,2] - reference_point_xyz[0,2])
    if temp < running_min:
        running_min = temp
        
    temp = fabs(shell_boundaries_xyz[1,0] - reference_point_xyz[0,0])
    if temp < running_min:
        running_min = temp
        
    temp = fabs(shell_boundaries_xyz[1,1] - reference_point_xyz[0,1])
    if temp < running_min:
        running_min = temp
        
    temp = fabs(shell_boundaries_xyz[1,2] - reference_point_xyz[0,2])
    if temp < running_min:
        running_min = temp
        
    return running_min
    





    
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
@cython.profile(False)
cdef (DTYPE_INT64_t, DTYPE_INT64_t) _gen_shell(CELL_ID_t[:,:] center_ijk, 
                                               DTYPE_INT32_t level,
                                               Cell_ID_Memory cell_ID_mem,
                                               GalaxyMap galaxy_map,
                                               ):
    """
    TODO: update names from ijk to pqr
    
    Description
    ===========
    
    Generate all the possible locations in the "shell" defined by the level parameter.
    
    Given a cubic grid structure, level 0 is a single cell, level 1 is a shell of
    3x3x3 cells minus the 1 interior, level 2 is 5x5x5 minus the 3x3x3 interior, etc.
    
    
    Notes
    =====
    
    Only called once in _query_first() (_voidfinder_cython_find_next) in main_algorithm()
    (_voidfinder_cython), and now also by _gen_cube() in _query_shell_radius()
    
    This means in main_algorithm, we come upon an ijk cell.  We call _query_first on that ijk
    cell, and within _query_first we stay on that ijk cell and grow outward shells until we find
    the first neighbor.  Then _gen_shell won't be called again until the next _ijk shell, and instead
    we will use _gen_cube to find the 2nd 3rd and 4a/b-th neighbors
    
    ASSUMES THIS WILL ONLY BE CALLED SEQUENTIALLY AND IN-ORDER FOR A SINGLE IJK, WHICH BASED ON ABOVE
    USAGE WOULD SEEM TO BE TRUE (except for level 0, which should never be called, level 1, then 2, 3, 
    4,5, etc).  VIOLATING THIS WILL RESULT IN THE cell_ID_mem.max_level_available possibly being 
    set incorrectly
    
    """
    
    ################################################################################
    # variable declarations
    ################################################################################
    cdef ITYPE_t out_idx = 0
    
    cdef DTYPE_INT32_t i, j, k, temp
    
    cdef CELL_ID_t out_i, out_j, out_k
    
    cdef DTYPE_INT32_t i_lower, i_upper, j_lower, j_upper, k_lower, k_upper
    
    cdef CELL_ID_t center_i, center_j, center_k
    
    cdef DTYPE_INT64_t num_written = 0
    
    
    
    
    
    ################################################################################
    # We can precompute the maximum number of rows we might need for this shell 
    # generation to be successful, so if necessary resize the cell_ID_mem to be big 
    # enough to hold the data.
    #
    # Note that we may use less than this number of rows due to the
    # galaxy_map.contains() filtering out some grid cells
    #
    # Also - cool trick, turns out PyMem_Realloc PRESERVES THE EXISTING DATA!
    # So gonna use this to optimize the cell_ID_mem cell generation
    ################################################################################
    
    if cell_ID_mem.max_level_mem < <DTYPE_INT64_t>level:
        
        cell_ID_mem.resize(<size_t>level)
    
    
    
    ################################################################################
    # Next, use the cell_ID_mem object to check if we're still working from the
    # same p-q-r grid cell as the last time this function was called.
    #
    # If we match on the p-q-r location from last time, we're working from the same 
    # pqr cell as last time this was called, so check the level parameter against
    # the maximum shell level already stored.  If the stored one is greater, we've
    # already got all the cell IDs stored that we need.
    #
    # If we matched on p-q-r but the current query is asking for a higher level
    # than what we have stored, we need to calculate that level, but not the levels
    # prior, so we start from the cell_ID_mem.next_unused_row_idx 
    ################################################################################
    if center_ijk[0,0] == cell_ID_mem.curr_ijk[0] and \
       center_ijk[0,1] == cell_ID_mem.curr_ijk[1] and \
       center_ijk[0,2] == cell_ID_mem.curr_ijk[2]:
        
        if <DTYPE_INT64_t>level <= cell_ID_mem.max_level_available:
            
            
            #print("_gen_cell shortcutting on cell ", center_ijk[0,0], center_ijk[0,1], center_ijk[0,2], level, flush=True)
            
            return cell_ID_mem.level_start_idx[level], cell_ID_mem.level_stop_idx[level]
    
        else:
            
            out_idx = 3*cell_ID_mem.next_unused_row_idx
            
            cell_ID_mem.level_start_idx[level] = cell_ID_mem.next_unused_row_idx
            
            cell_ID_mem.max_level_available = <DTYPE_INT64_t>level
            
    else:
        
        ################################################################################
        # If we didn't match on the p-q-r check, we have to start over from 0
        ################################################################################
        
        cell_ID_mem.max_level_available = <DTYPE_INT64_t>level
        
        cell_ID_mem.next_unused_row_idx = 0
        
        cell_ID_mem.curr_ijk[0] = center_ijk[0,0]
        cell_ID_mem.curr_ijk[1] = center_ijk[0,1]
        cell_ID_mem.curr_ijk[2] = center_ijk[0,2]
        
        cell_ID_mem.level_start_idx[level] = 0
        
    
    
    
    ################################################################################
    # For level 0, the algorithm below actually would write out the original
    # cell ijk twice, but we only want to write it once so have a special block
    # here to handle that single special case and return early
    #
    # Kind of annoying - maybe we can think up a better shell filling algorithm
    # that doesn't need special handling for level 0
    ################################################################################
    if level == 0:
        
        if galaxy_map.contains(center_ijk[0,0], center_ijk[0,1], center_ijk[0,2]):
            
            cell_ID_mem.data[out_idx] = center_ijk[0,0]
            cell_ID_mem.data[out_idx+1] = center_ijk[0,1]
            cell_ID_mem.data[out_idx+2] = center_ijk[0,2]
            
            num_written = 1
            
        else:
            
            num_written = 0
        
        cell_ID_mem.next_unused_row_idx += num_written
    
        cell_ID_mem.level_stop_idx[level] = cell_ID_mem.next_unused_row_idx
        
        
        #print("_gen_cell level 0 shortcut on cell ", center_ijk[0,0], center_ijk[0,1], center_ijk[0,2], level, flush=True)
        
        return cell_ID_mem.level_start_idx[level], cell_ID_mem.level_stop_idx[level]
    
    ################################################################################
    # Technically not necessary but made the code below look a tad cleaner
    ################################################################################
    center_i = center_ijk[0,0]
    center_j = center_ijk[0,1]
    center_k = center_ijk[0,2]
    
    
    ################################################################################
    # i dimension first
    # Iterate through the possible shell locations, starting with the i dimension
    # this iteration does all the cells in the 2 "planes" at the "i +/- level"
    # grid coordinate 
    ################################################################################
    for j in range(-level, level+1):
        
        for k in range(-level, level+1):
            
            out_i = <CELL_ID_t>level + center_i
            out_j = <CELL_ID_t>j + center_j
            out_k = <CELL_ID_t>k + center_k
            
            if galaxy_map.contains(out_i, out_j, out_k):
            
                cell_ID_mem.data[out_idx] = out_i
                cell_ID_mem.data[out_idx+1] = out_j
                cell_ID_mem.data[out_idx+2] = out_k
                
                out_idx += 3
                num_written += 1
                
            out_i = <CELL_ID_t>(-level) + center_i
            
            if galaxy_map.contains(out_i, out_j, out_k):
            
                cell_ID_mem.data[out_idx] = out_i
                cell_ID_mem.data[out_idx+1] = out_j
                cell_ID_mem.data[out_idx+2] = out_k
                
                out_idx += 3
                num_written += 1
                
    ################################################################################
    # j dimension
    # Next do the 2 "planes" at the "j +/- level" coordinates, except for the edges
    # which have already been done by doing the "i +/- level" planes, so the i
    # parameter below runs from (-level+1, level) instead of (-level, level+1)
    ################################################################################
    for i in range(-level+1, level):
        for k in range(-level, level+1):
            
            out_i = <CELL_ID_t>i + center_i
            out_j = <CELL_ID_t>level + center_j
            out_k = <CELL_ID_t>k + center_k
            
            if galaxy_map.contains(out_i, out_j, out_k):
                
                cell_ID_mem.data[out_idx] = out_i
                cell_ID_mem.data[out_idx+1] = out_j
                cell_ID_mem.data[out_idx+2] = out_k
                
                out_idx += 3
                num_written += 1
                
            out_j = <CELL_ID_t>(-level) + center_j
            
            if galaxy_map.contains(out_i, out_j, out_k):
                
                cell_ID_mem.data[out_idx] = out_i
                cell_ID_mem.data[out_idx+1] = out_j
                cell_ID_mem.data[out_idx+2] = out_k
                
                out_idx += 3
                num_written += 1
                
    
    ################################################################################
    # k dimension
    # Lastly do the 2 "planes" at the "k +/- level" coordinates, noting that since
    # we are in 3D and have already done i and j, the border cells around the
    # positive and negative k planes have already been checked, so both i and j
    # run from (-level+1, level) instead of (-level, level+1)
    ################################################################################
    for i in range(-level+1, level):
        for j in range(-level+1, level):
            
            out_i = <CELL_ID_t>i + center_i
            out_j = <CELL_ID_t>j + center_j
            out_k = <CELL_ID_t>level + center_k
            
            if galaxy_map.contains(out_i, out_j, out_k):
                
                cell_ID_mem.data[out_idx] = out_i
                cell_ID_mem.data[out_idx+1] = out_j
                cell_ID_mem.data[out_idx+2] = out_k
                
                out_idx += 3
                num_written += 1
                
            out_k = <CELL_ID_t>(-level) + center_k
            
            if galaxy_map.contains(out_i, out_j, out_k):
                
                cell_ID_mem.data[out_idx] = out_i
                cell_ID_mem.data[out_idx+1] = out_j
                cell_ID_mem.data[out_idx+2] = out_k
                
                out_idx += 3
                num_written += 1
            
            
    #print("_gen_shell called on ", center_i, center_j, center_k, " with level", level, "written cells ", num_written, flush=True)
            
            
    cell_ID_mem.next_unused_row_idx += num_written
    
    cell_ID_mem.level_stop_idx[level] = cell_ID_mem.next_unused_row_idx
    
    return cell_ID_mem.level_start_idx[level], cell_ID_mem.level_stop_idx[level]
    
        
        



@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
@cython.profile(False)
cdef DTYPE_INT64_t _gen_cube(CELL_ID_t[:,:] center_ijk, 
                             DTYPE_INT32_t level,
                             Cell_ID_Memory cell_ID_mem,
                             GalaxyMap galaxy_map
                             ):
    """
    Description
    ===========
    
    Only called once in _query_shell_radius() (this file) which is only called once in
    find_next_galaxy() (also this file)
    
    Take advantage of the optimizations in _gen_shell() - this function is really
    just a wrapper to sequentially call _gen_shell()
    
    
    Parameters
    ==========
    
    TODO: describe
    
    Returns
    =======
    
    Number of cell IDs loaded into cell_ID_mem
    
    """
    
    
    
    cdef DTYPE_INT32_t curr_level
    
    cdef DTYPE_INT64_t row_start, row_stop
    
    for curr_level in range(level+1): #level 5 means include 0,1,2,3,4,5 so do level+1 for range
        
        row_start, row_stop = _gen_shell(center_ijk,
                                         curr_level,
                                         cell_ID_mem,
                                         galaxy_map)
        
                
    return cell_ID_mem.next_unused_row_idx
    
        
        


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
@cython.profile(False)
cpdef DTYPE_INT64_t find_next_prime(DTYPE_INT64_t threshold_value):
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
    
    cdef DTYPE_INT64_t check_val, j, sqrt_check
    
    cdef DTYPE_B_t at_least_one_divisor
    
    
    for check_val in range(threshold_value+1, 2*threshold_value):
        
        if check_val%2 == 0:
            continue
        
        sqrt_check = <DTYPE_INT64_t>sqrt(<DTYPE_F64_t>check_val) + 1
        
        at_least_one_divisor = 0
        
        for j in range(3, sqrt_check):
            
            if check_val % j == 0:
                
                at_least_one_divisor = 1
                
                break
            
        if not at_least_one_divisor:
            
            return check_val




