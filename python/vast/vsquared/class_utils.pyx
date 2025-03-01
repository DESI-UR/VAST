#cython: language_level=3
#cython: initializedcheck=False
#cython: boundscheck=False
#cython: wraparound=False
#cython: cdivision=True
#cython: nonecheck=False
#cython: profile=False


from __future__ import print_function

cimport cython

import numpy as np

cimport numpy as np

np.import_array()  # required in order to use C-API

from vast.voidfinder.typedefs cimport DTYPE_CP128_t, \
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

from libc.math cimport fabs, sqrt, asin, atan#, exp, pow, cos, sin, asin


from scipy.spatial import ConvexHull



cpdef void calculate_region_volume(ITYPE_t idx,
                                   list region,
                                   DTYPE_F64_t[:] output_volume,
                                   np.ndarray vertices,
                                   DTYPE_F64_t r_max,
                                   DTYPE_F64_t r_min,
                                   DTYPE_F64_t[:] vrh,
                                   DTYPE_B_t[:] in_mask):
    """
    Description
    ===========
    
    Check for validity, and if valid, calculate the volume of a 
    voronoi cell and write it to the output_volume array
    
    Parameters
    ==========
    
    idx : int
        index to the current cell in the arrays:
        output_volume, 
        
    region : list of ints
        indices to the verticies of the current voronoi cell
        
    output_volume : array of shape (N,)
        array to write final output values into 
        
    vertices : array of shape (K, 3)
        array of the verticies of the voronoi cells
    
    r_max, r_min : float
        max and min radial values to compare against from the
        galaxy coordinates
        
    vrh : array of shape (K,)
        radii of the vertices to check against r_max and r_min
    
    """
    
    #convert to numpy array with specific dtype so we can leverage typing
    #in loops below
    #cdef DTYPE_F64_t[:,:] vertices_memview = verticies
    cdef ITYPE_t[:] region_memview = np.array(region, dtype=np.intp)
    cdef ITYPE_t index
    cdef DTYPE_F64_t curr_radius

    ################################################################################
    # First cut - if there is a -1 in the region index list, that means that 
    # region extends out to infinity so we exclude it from volume calculations
    ################################################################################

    if -1 in region_memview:
        return

    ################################################################################
    # Second & Third cuts - make sure the verticies all fall within the r_min and 
    # r_max values, and also that all verticies are in the mask
    ################################################################################

    for index in region_memview:
        
        curr_radius = vrh[index]

        #using <= and >= since original code inversely checked just > and <
        if curr_radius <= r_min or curr_radius >= r_max:
            return

        if in_mask[index] == 0:
            return

    ################################################################################
    # Now get the volume and write out
    ################################################################################

    voronoi_cell_verticies = vertices[region] #using the list not the memview since we cant fancy index with a memview
    
    try:
        hull = ConvexHull(voronoi_cell_verticies)
    except:
        #If failed,
        #Retry creating the convex hull using "Joggling" which is
        #a pertubation of the points which may result in better
        #numerical stability
        hull = ConvexHull(voronoi_cell_verticies, qhull_options='QJ')
        
    output_volume[idx] = hull.volume
    
    return



