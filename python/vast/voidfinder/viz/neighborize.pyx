#cython: language_level=3

cimport cython
import numpy as np
cimport numpy as np
np.import_array()  # required in order to use C-API


from ..typedefs cimport DTYPE_CP128_t, \
                        DTYPE_CP64_t, \
                        DTYPE_F64_t, \
                        DTYPE_F32_t, \
                        DTYPE_B_t, \
                        ITYPE_t, \
                        DTYPE_INT32_t, \
                        DTYPE_INT64_t

from numpy.math cimport NAN, INFINITY

from libc.math cimport fabs, sqrt, asin, atan#, exp, pow, cos, sin, asin



@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cpdef list build_neighbor_index(neighbor_tree, #need int64 types
                                DTYPE_INT64_t[:] hole_order,
                                DTYPE_F32_t[:,:] holes_xyz,
                                DTYPE_F32_t[:] holes_radii):
                                  
    cdef list neighbor_index = []
    
    cdef ITYPE_t curr_idx
    cdef ITYPE_t num_holes = hole_order.shape[0]
    cdef DTYPE_F32_t[:,:] query_memory = np.empty((1,3), dtype=np.float32)
    cdef DTYPE_F32_t curr_radius
    
    for curr_idx in range(num_holes):
        
        query_memory[0,0] = holes_xyz[curr_idx, 0]
        query_memory[0,1] = holes_xyz[curr_idx, 1]
        query_memory[0,2] = holes_xyz[curr_idx, 2]
        
        curr_radius = 2.0*holes_radii[curr_idx]
        
        curr_neighbors = neighbor_tree.query_radius(query_memory, curr_radius)
        
        neighbor_index.append(curr_neighbors[0])
    
    return neighbor_index



@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cpdef list build_grouped_neighbor_index(neighbor_tree, #need int64 types
                                        DTYPE_INT64_t[:] hole_order,
                                        DTYPE_F32_t[:,:] holes_xyz,
                                        DTYPE_F32_t[:] holes_radii,
                                        DTYPE_INT32_t[:] holes_group_IDs):
                                  
    cdef list neighbor_index = []
    
    cdef ITYPE_t curr_idx
    cdef ITYPE_t jdx
    cdef ITYPE_t curr_num_neighbors
    cdef DTYPE_INT32_t curr_group_ID
    cdef DTYPE_INT32_t neighbor_group_ID
    cdef ITYPE_t num_holes = hole_order.shape[0]
    cdef DTYPE_F32_t[:,:] query_memory = np.empty((1,3), dtype=np.float32)
    cdef DTYPE_F32_t curr_radius
    
    for curr_idx in range(num_holes):
        
        query_memory[0,0] = holes_xyz[curr_idx, 0]
        query_memory[0,1] = holes_xyz[curr_idx, 1]
        query_memory[0,2] = holes_xyz[curr_idx, 2]
        
        curr_radius = 2.0*holes_radii[curr_idx]
        
        curr_group_ID = holes_group_IDs[curr_idx]
        
        curr_neighbors = neighbor_tree.query_radius(query_memory, curr_radius)[0]
        
        curr_num_neighbors = <ITYPE_t>curr_neighbors.shape[0]
        
        valid_idx = np.ones(curr_num_neighbors, dtype=np.bool)
        
        for jdx in range(curr_num_neighbors):
            
            neighbor_group_ID = holes_group_IDs[<ITYPE_t>curr_neighbors[jdx]]
            
            if neighbor_group_ID != curr_group_ID:
                
                valid_idx[jdx] = 0
                
        group_neighbors = curr_neighbors[valid_idx]
            
        neighbor_index.append(group_neighbors)
    
    return neighbor_index



