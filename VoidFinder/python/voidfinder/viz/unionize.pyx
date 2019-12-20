

cimport cython
import numpy as np
cimport numpy as np
np.import_array()  # required in order to use C-API


from typedefs cimport DTYPE_CP128_t, DTYPE_CP64_t, DTYPE_F64_t, DTYPE_F32_t, DTYPE_B_t, ITYPE_t, DTYPE_INT32_t, DTYPE_INT64_t

from numpy.math cimport NAN, INFINITY

from libc.math cimport fabs, sqrt, asin, atan#, exp, pow, cos, sin, asin



@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cpdef void union_vertex_selection(neighbor_index, #need int64 types
                                  DTYPE_B_t[:] valid_vertex_idx, #output uint8
                                  DTYPE_F32_t[:,:] holes_xyz,
                                  DTYPE_F32_t[:] holes_radii,
                                  DTYPE_F32_t[:,:] sphere_coord_data,
                                  ITYPE_t vert_per_sphere
                                  ) except *:
    """
    neighbor_index : ndarray shape (num_holes, ?)
        unknown number of neighbors per hole
        
    valid_vertex_idx : ndarray shape (num_vertex, )
        output set to 0 to say a vertex should be dropped
        
    holes_xyz : ndarray shape (num_holes, 3)
        xyz coords of hole centers
        
    holes_radii : ndarray shape (num_holes, )
        radii of holes
        
    sphere_coord_data : ndarray shape (num_vertex, 4)
        xyz coordinates of sphere vertices and 4th column is w dimension set to 1.0
        3 consecutive rows form a triangle which is part of the current sphere
        
    vert_per_sphere : int
        number of vertices per sphere (divide by 3 to get number of triangles)
    
    
    """
                                  
    
    #cdef DTYPE_B_t[:] already_invalid = np.zeros(sphere_coord_data.shape[0], dtype=np.uint8)                              
    
    cdef ITYPE_t num_rows = holes_xyz.shape[0]
    
    cdef ITYPE_t curr_idx
    
    cdef ITYPE_t start_idx
    cdef ITYPE_t end_idx
    cdef ITYPE_t neigh_idx
    cdef ITYPE_t num_neighbor
    cdef ITYPE_t curr_offset
    cdef ITYPE_t triangles_per_sphere = vert_per_sphere/3
    
    cdef ITYPE_t vertex_idx
    
    cdef DTYPE_F32_t[:] hole_xyz
    #cdef DTYPE_F32_t[:,:] hole_xyz_tile = np.empty((3,3), dtype=np.float32)
    cdef DTYPE_F32_t[:,:] vertex_dist_components = np.empty((3,3), dtype=np.float32)
    
    cdef DTYPE_F32_t hole_radius
    cdef DTYPE_F32_t hole_radius_sq
    cdef DTYPE_F32_t[:] hole_radius_compare = np.zeros((3,), dtype=np.float32)
    cdef DTYPE_F32_t[:] vertex_dist_sq = np.zeros((3,), dtype=np.float32)
    
    cdef DTYPE_INT64_t[:] close_idx
    cdef DTYPE_INT64_t neighbor_idx
    
    cdef DTYPE_B_t[:] vertex_valid = np.zeros((3,), dtype=np.uint8)
                                  
    for curr_idx in range(num_rows):
        
        #if curr_idx % 100 == 0:
        #    print("Working: ", curr_idx)
        
        hole_radius = holes_radii[curr_idx]
        
        hole_radius_sq = hole_radius*hole_radius
        
        hole_xyz = holes_xyz[curr_idx]
        
        close_idx = neighbor_index[curr_idx]
        
        num_neighbor = close_idx.shape[0]
        
        for neigh_idx in range(num_neighbor):
            
            neighbor_idx = close_idx[neigh_idx]
            
            if neighbor_idx == curr_idx:
                
                continue
            
            start_idx = (<ITYPE_t>neighbor_idx)*vert_per_sphere
            
            #end_idx = ((<ITYPE_t>neighbor_idx)+1)*vert_per_sphere
            
            for curr_offset in range(triangles_per_sphere): #working on sets of 3 vertices per loop
        
                vertex_idx = start_idx + curr_offset*3
                
                if valid_vertex_idx[vertex_idx] == 0: #don't need to do any calculation for vertices which have already been invalidated
                    
                    continue
        
        
                #vertex_dist_components = np.subtract(sphere_coord_data[vertex_idx:(vertex_idx+3), 0:3], hole_xyz)
                
                
                vertex_dist_components[0,0] = sphere_coord_data[vertex_idx, 0] - hole_xyz[0]
                vertex_dist_components[0,1] = sphere_coord_data[vertex_idx, 1] - hole_xyz[1]
                vertex_dist_components[0,2] = sphere_coord_data[vertex_idx, 2] - hole_xyz[2]
                
                vertex_dist_components[1,0] = sphere_coord_data[vertex_idx+1, 0] - hole_xyz[0]
                vertex_dist_components[1,1] = sphere_coord_data[vertex_idx+1, 1] - hole_xyz[1]
                vertex_dist_components[1,2] = sphere_coord_data[vertex_idx+1, 2] - hole_xyz[2]
                
                vertex_dist_components[2,0] = sphere_coord_data[vertex_idx+2, 0] - hole_xyz[0]
                vertex_dist_components[2,1] = sphere_coord_data[vertex_idx+2, 1] - hole_xyz[1]
                vertex_dist_components[2,2] = sphere_coord_data[vertex_idx+2, 2] - hole_xyz[2]
                
                
                
                #vertex_dist_sq[:] = np.sum(np.multiply(vertex_dist_components, vertex_dist_components), axis=1)
                
                
                vertex_dist_sq[0] = vertex_dist_components[0,0]*vertex_dist_components[0,0] + \
                                    vertex_dist_components[0,1]*vertex_dist_components[0,1] + \
                                    vertex_dist_components[0,2]*vertex_dist_components[0,2]
                                    
                vertex_dist_sq[1] = vertex_dist_components[1,0]*vertex_dist_components[1,0] + \
                                    vertex_dist_components[1,1]*vertex_dist_components[1,1] + \
                                    vertex_dist_components[1,2]*vertex_dist_components[1,2]
                                    
                vertex_dist_sq[2] = vertex_dist_components[2,0]*vertex_dist_components[2,0] + \
                                    vertex_dist_components[2,1]*vertex_dist_components[2,1] + \
                                    vertex_dist_components[2,2]*vertex_dist_components[2,2]
                
                
                #vertex_valid = np.greater_equal(vertex_dist_sq, hole_radius_compare)
                
                vertex_valid[0] = vertex_dist_sq[0] >= hole_radius_sq
                vertex_valid[1] = vertex_dist_sq[1] >= hole_radius_sq
                vertex_valid[2] = vertex_dist_sq[2] >= hole_radius_sq
                
                #if np.all(np.logical_not(vertex_valid)):
                if (vertex_valid[0] == 0) and \
                   (vertex_valid[1] == 0) and \
                   (vertex_valid[2] == 0):
                    
                    #already_invalid[vertex_idx:(vertex_idx+3)] = 1
        
                    valid_vertex_idx[vertex_idx:(vertex_idx+3)] = 0
        
        
        
        
        
        
        
        