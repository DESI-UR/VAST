

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
                                  ITYPE_t vert_per_sphere,
                                  ) except *:
    """
    Description
    -----------
    
    Given a list of neighbor void spheres which may possibly intersect, remove all the
    interior triangles from the sphere triangularizations of this voids.  Alternately
    described: create a union triangularization by removing intersections.
    
    Given the list of all triangle vertices in the sphere_coord_data input, fill in the
    correct values of valid_vertex_idx where True indicates that we should keep the
    vertex, and False indicates the vertex should be removed.  The caller can then
    use the valid_vertex_idx to filter out the unwanted vertices.
    
    
    Parameters
    ----------
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
    
    smooth_seams : bool
        if true, attempt to correct vertex data for triangles which live on intersection
        between two spheres but which are not removed completely
    
    DEPRECATED - aggressive : bool
        If True, indicates to remove sets of 3 vertices, any of the 3 which fail to
        meet the inclusion criteria.
        If False, indicates to remove sets of 3 vertices, ONLY when all 3 fail to
        meet the inclusion criteria.
        When aggressive was implemented and set to True, the void objects ended up with big unruly
        sets of missing vertices, replaced it with the smooth_seams parameter above
        
    Output
    ------
    
    Fills in correct values of the valid_vertex_idx input vector.
    
    Modifies sphere_coord_data if smooth_seams==True
    
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
    
    #cdef ITYPE_t vertex_idx_00
    cdef ITYPE_t vertex_idx_0
    cdef ITYPE_t vertex_idx_1
    cdef ITYPE_t vertex_idx_2
    
    #cdef ITYPE_t num_valid
    
    cdef DTYPE_F32_t[:] hole_xyz
    #cdef DTYPE_F32_t[:,:] hole_xyz_tile = np.empty((3,3), dtype=np.float32)
    cdef DTYPE_F32_t[:,:] vertex_dist_components = np.empty((3,3), dtype=np.float32)
    
    cdef DTYPE_F32_t hole_radius
    cdef DTYPE_F32_t hole_radius_sq
    cdef DTYPE_F32_t[:] hole_radius_compare = np.zeros((3,), dtype=np.float32)
    cdef DTYPE_F32_t[:] vertex_dist_sq = np.zeros((3,), dtype=np.float32)
    
    
    cdef DTYPE_F32_t neighbor_hole_radius
    cdef DTYPE_F32_t[:] neighbor_hole_xyz
    
    
    cdef DTYPE_F32_t temp1
    cdef DTYPE_F32_t temp2
    cdef DTYPE_F32_t temp3
    #cdef DTYPE_F32_t temp4
    #cdef DTYPE_F32_t temp5
    
    cdef DTYPE_INT64_t[:] close_idx
    cdef DTYPE_INT64_t neighbor_idx
    
    cdef DTYPE_B_t[:] vertex_valid = np.zeros((3,), dtype=np.uint8)
    
    
    
    #cdef DTYPE_F32_t[:] edge_1 = np.zeros((3,), dtype=np.float32)
    #cdef DTYPE_F32_t[:] edge_2 = np.zeros((3,), dtype=np.float32)
    #cdef DTYPE_F32_t[:] proj_dir = np.zeros((3,), dtype=np.float32)
    
    #cdef DTYPE_F32_t[:] new_vertex_0 = np.zeros((3,), dtype=np.float32)
    #cdef DTYPE_F32_t[:] new_vertex_1 = np.zeros((3,), dtype=np.float32)
    #cdef DTYPE_F32_t[:] new_vertex_2 = np.zeros((3,), dtype=np.float32)
    
    
    #################################################################
    # Iterate through each void sphere
    #################################################################
    for curr_idx in range(num_rows):
        
        #if curr_idx % 100 == 0:
        #    print("Working: ", curr_idx)
        
        hole_radius = holes_radii[curr_idx]
        
        hole_radius_sq = hole_radius*hole_radius
        
        hole_xyz = holes_xyz[curr_idx]
        
        close_idx = neighbor_index[curr_idx]
        
        num_neighbor = close_idx.shape[0]
        
        #################################################################
        # Iterate through all the neighbors of the current void sphere
        #################################################################
        for neigh_idx in range(num_neighbor):
            
            neighbor_idx = close_idx[neigh_idx]
            
            if neighbor_idx == curr_idx:
                
                continue
            
            ############################################################
            # Add a quick distance check on hole radii.  If the
            # distance between the current void and the neighbor void
            # is greater than the sum of the two radii, they don't
            # actually intersect so we don't need to check any more
            # things about this pair.
            ############################################################
            neighbor_hole_radius = holes_radii[neighbor_idx]
            
            temp1 = neighbor_hole_radius + hole_radius
            
            temp2 = temp1*temp1
            
            neighbor_hole_xyz = holes_xyz[neighbor_idx]
            
            hole_radius_compare[0] = hole_xyz[0] - neighbor_hole_xyz[0]
            hole_radius_compare[1] = hole_xyz[1] - neighbor_hole_xyz[1]
            hole_radius_compare[2] = hole_xyz[2] - neighbor_hole_xyz[2]
            
            temp3 = hole_radius_compare[0]*hole_radius_compare[0] + \
                    hole_radius_compare[1]*hole_radius_compare[1] + \
                    hole_radius_compare[2]*hole_radius_compare[2]
                    
            if temp3 > temp2:
                
                continue
            
            
            ######################################################################
            # Now that we know the current void and its neighbor intersect,
            # iterate through all the triangles in the neighbor's section of 
            # the sphere coordinate data.
            #
            # We actually iterate 3 vertices at a time since there are 3 per
            # triangle.
            # 
            # If the valid_vertix_idx for this triangle has already been
            # invalidated, we can skip it since it's already going to be removed
            ######################################################################
            
            start_idx = (<ITYPE_t>neighbor_idx)*vert_per_sphere
            
            for curr_offset in range(triangles_per_sphere): #working on sets of 3 vertices per loop
        
                vertex_idx_0 = start_idx + curr_offset*3
                
                if valid_vertex_idx[vertex_idx_0] == 0: #don't need to do any calculation for vertices which have already been invalidated
                    
                    continue
        
                #vertex_idx_0 = vertex_idx_00
                vertex_idx_1 = vertex_idx_0 + 1
                vertex_idx_2 = vertex_idx_0 + 2
        
        
                vertex_dist_components[0,0] = sphere_coord_data[vertex_idx_0, 0] - hole_xyz[0]
                vertex_dist_components[0,1] = sphere_coord_data[vertex_idx_0, 1] - hole_xyz[1]
                vertex_dist_components[0,2] = sphere_coord_data[vertex_idx_0, 2] - hole_xyz[2]
                
                vertex_dist_components[1,0] = sphere_coord_data[vertex_idx_1, 0] - hole_xyz[0]
                vertex_dist_components[1,1] = sphere_coord_data[vertex_idx_1, 1] - hole_xyz[1]
                vertex_dist_components[1,2] = sphere_coord_data[vertex_idx_1, 2] - hole_xyz[2]
                
                vertex_dist_components[2,0] = sphere_coord_data[vertex_idx_2, 0] - hole_xyz[0]
                vertex_dist_components[2,1] = sphere_coord_data[vertex_idx_2, 1] - hole_xyz[1]
                vertex_dist_components[2,2] = sphere_coord_data[vertex_idx_2, 2] - hole_xyz[2]
                
                
                vertex_dist_sq[0] = vertex_dist_components[0,0]*vertex_dist_components[0,0] + \
                                    vertex_dist_components[0,1]*vertex_dist_components[0,1] + \
                                    vertex_dist_components[0,2]*vertex_dist_components[0,2]
                                    
                vertex_dist_sq[1] = vertex_dist_components[1,0]*vertex_dist_components[1,0] + \
                                    vertex_dist_components[1,1]*vertex_dist_components[1,1] + \
                                    vertex_dist_components[1,2]*vertex_dist_components[1,2]
                                    
                vertex_dist_sq[2] = vertex_dist_components[2,0]*vertex_dist_components[2,0] + \
                                    vertex_dist_components[2,1]*vertex_dist_components[2,1] + \
                                    vertex_dist_components[2,2]*vertex_dist_components[2,2]
                
                
                vertex_valid[0] = vertex_dist_sq[0] >= hole_radius_sq
                vertex_valid[1] = vertex_dist_sq[1] >= hole_radius_sq
                vertex_valid[2] = vertex_dist_sq[2] >= hole_radius_sq
                
                ######################################################################
                # For fully invalid triangles, set the output to 0 to remove the 
                # whole thing
                ######################################################################
                if (vertex_valid[0] == 0) and \
                   (vertex_valid[1] == 0) and \
                   (vertex_valid[2] == 0):
                    
                    valid_vertex_idx[vertex_idx_0] = 0
                    valid_vertex_idx[vertex_idx_1] = 0
                    valid_vertex_idx[vertex_idx_2] = 0
                
                
                
                

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cpdef void seam_vertex_adjustment(neighbor_index, #need int64 types
                                  DTYPE_INT64_t[:, :] hole_vertex_map, 
                                  DTYPE_F32_t[:,:] holes_xyz,
                                  DTYPE_F32_t[:] holes_radii,
                                  DTYPE_F32_t[:,:] sphere_coord_data,
                                  ) except *:
                                  
    
    ######################################################################
    # For partially invalid triangles, re-adjust them so vertices only 
    # touch sphere borders and not interior
    ######################################################################
    
    
    cdef ITYPE_t num_rows = holes_xyz.shape[0]
    
    cdef ITYPE_t curr_idx
    cdef ITYPE_t start_idx
    cdef ITYPE_t end_idx
    cdef ITYPE_t neigh_idx
    cdef ITYPE_t num_neighbor
    cdef ITYPE_t curr_offset
    cdef ITYPE_t triangles_per_sphere
    
    cdef ITYPE_t vertex_idx_00
    cdef ITYPE_t vertex_idx_0
    cdef ITYPE_t vertex_idx_1
    cdef ITYPE_t vertex_idx_2
    
    cdef ITYPE_t num_valid
    
    cdef DTYPE_F32_t[:] hole_xyz
    #cdef DTYPE_F32_t[:,:] hole_xyz_tile = np.empty((3,3), dtype=np.float32)
    cdef DTYPE_F32_t[:,:] vertex_dist_components = np.empty((3,3), dtype=np.float32)
    
    cdef DTYPE_F32_t hole_radius
    cdef DTYPE_F32_t hole_radius_sq
    cdef DTYPE_F32_t[:] hole_radius_compare = np.zeros((3,), dtype=np.float32)
    cdef DTYPE_F32_t[:] vertex_dist_sq = np.zeros((3,), dtype=np.float32)
    
    
    cdef DTYPE_F32_t neighbor_hole_radius
    cdef DTYPE_F32_t[:] neighbor_hole_xyz
    
    
    cdef DTYPE_F32_t temp1
    cdef DTYPE_F32_t temp2
    cdef DTYPE_F32_t temp3
    cdef DTYPE_F32_t temp4
    cdef DTYPE_F32_t temp5
    
    cdef DTYPE_INT64_t[:] close_idx
    cdef DTYPE_INT64_t neighbor_idx
    
    cdef DTYPE_B_t[:] vertex_valid = np.zeros((3,), dtype=np.uint8)
    
    
    
    #cdef DTYPE_F32_t[:] edge_1 = np.zeros((3,), dtype=np.float32)
    #cdef DTYPE_F32_t[:] edge_2 = np.zeros((3,), dtype=np.float32)
    cdef DTYPE_F32_t[:] proj_dir = np.zeros((3,), dtype=np.float32)
    
    #cdef DTYPE_F32_t[:] new_vertex_0 = np.zeros((3,), dtype=np.float32)
    #cdef DTYPE_F32_t[:] new_vertex_1 = np.zeros((3,), dtype=np.float32)
    #cdef DTYPE_F32_t[:] new_vertex_2 = np.zeros((3,), dtype=np.float32)
    
    
    #################################################################
    # Iterate through each void sphere
    #################################################################
    for curr_idx in range(num_rows):
        
        #if curr_idx % 100 == 0:
        #    print("Working: ", curr_idx)
        
        hole_radius = holes_radii[curr_idx]
        
        hole_radius_sq = hole_radius*hole_radius
        
        hole_xyz = holes_xyz[curr_idx]
        
        close_idx = neighbor_index[curr_idx]
        
        num_neighbor = close_idx.shape[0]
        
        #################################################################
        # Iterate through all the neighbors of the current void sphere
        #################################################################
        for neigh_idx in range(num_neighbor):
            
            neighbor_idx = close_idx[neigh_idx]
            
            if neighbor_idx == curr_idx:
                
                continue
            
            ############################################################
            # Add a quick distance check on hole radii.  If the
            # distance between the current void and the neighbor void
            # is greater than the sum of the two radii, they don't
            # actually intersect so we don't need to check any more
            # things about this pair.
            ############################################################
            neighbor_hole_radius = holes_radii[neighbor_idx]
            
            temp1 = neighbor_hole_radius + hole_radius
            
            temp2 = temp1*temp1
            
            neighbor_hole_xyz = holes_xyz[neighbor_idx]
            
            hole_radius_compare[0] = hole_xyz[0] - neighbor_hole_xyz[0]
            hole_radius_compare[1] = hole_xyz[1] - neighbor_hole_xyz[1]
            hole_radius_compare[2] = hole_xyz[2] - neighbor_hole_xyz[2]
            
            temp3 = hole_radius_compare[0]*hole_radius_compare[0] + \
                    hole_radius_compare[1]*hole_radius_compare[1] + \
                    hole_radius_compare[2]*hole_radius_compare[2]
                    
            if temp3 > temp2:
                
                continue
            
            
            ######################################################################
            # Now that we know the current void and its neighbor intersect,
            # iterate through all the triangles in the neighbor's section of 
            # the sphere coordinate data.
            #
            # We actually iterate 3 vertices at a time since there are 3 per
            # triangle.
            # 
            # If the valid_vertix_idx for this triangle has already been
            # invalidated, we can skip it since it's already going to be removed
            ######################################################################
            
            #start_idx = (<ITYPE_t>neighbor_idx)*vert_per_sphere
            
            start_idx = <ITYPE_t>hole_vertex_map[<ITYPE_t>neighbor_idx,0]
            
            triangles_per_sphere = <ITYPE_t>hole_vertex_map[<ITYPE_t>neighbor_idx,1]
            
            for curr_offset in range(triangles_per_sphere): #working on sets of 3 vertices per loop
        
                vertex_idx_00 = start_idx + curr_offset*3
                
                #if valid_vertex_idx[vertex_idx_0] == 0: #don't need to do any calculation for vertices which have already been invalidated
                    
                #    continue
        
                vertex_idx_0 = vertex_idx_00
                vertex_idx_1 = vertex_idx_0 + 1
                vertex_idx_2 = vertex_idx_0 + 2
        
        
                vertex_dist_components[0,0] = sphere_coord_data[vertex_idx_0, 0] - hole_xyz[0]
                vertex_dist_components[0,1] = sphere_coord_data[vertex_idx_0, 1] - hole_xyz[1]
                vertex_dist_components[0,2] = sphere_coord_data[vertex_idx_0, 2] - hole_xyz[2]
                
                vertex_dist_components[1,0] = sphere_coord_data[vertex_idx_1, 0] - hole_xyz[0]
                vertex_dist_components[1,1] = sphere_coord_data[vertex_idx_1, 1] - hole_xyz[1]
                vertex_dist_components[1,2] = sphere_coord_data[vertex_idx_1, 2] - hole_xyz[2]
                
                vertex_dist_components[2,0] = sphere_coord_data[vertex_idx_2, 0] - hole_xyz[0]
                vertex_dist_components[2,1] = sphere_coord_data[vertex_idx_2, 1] - hole_xyz[1]
                vertex_dist_components[2,2] = sphere_coord_data[vertex_idx_2, 2] - hole_xyz[2]
                
                
                vertex_dist_sq[0] = vertex_dist_components[0,0]*vertex_dist_components[0,0] + \
                                    vertex_dist_components[0,1]*vertex_dist_components[0,1] + \
                                    vertex_dist_components[0,2]*vertex_dist_components[0,2]
                                    
                vertex_dist_sq[1] = vertex_dist_components[1,0]*vertex_dist_components[1,0] + \
                                    vertex_dist_components[1,1]*vertex_dist_components[1,1] + \
                                    vertex_dist_components[1,2]*vertex_dist_components[1,2]
                                    
                vertex_dist_sq[2] = vertex_dist_components[2,0]*vertex_dist_components[2,0] + \
                                    vertex_dist_components[2,1]*vertex_dist_components[2,1] + \
                                    vertex_dist_components[2,2]*vertex_dist_components[2,2]
                
                
                vertex_valid[0] = vertex_dist_sq[0] >= hole_radius_sq
                vertex_valid[1] = vertex_dist_sq[1] >= hole_radius_sq
                vertex_valid[2] = vertex_dist_sq[2] >= hole_radius_sq
                
    
    
                num_valid = <ITYPE_t>vertex_valid[0] + \
                            <ITYPE_t>vertex_valid[1] + \
                            <ITYPE_t>vertex_valid[2]
    
    
                if num_valid > 0 and num_valid < 3:
                    
                    if num_valid == 1:
                        pass
                        
                    elif num_valid == 2:
                        
                        ######################################################################
                        # Vertex 0 will be the 'bad' one, push vertex 0 back towards vertex 1
                        # since num_valid == 2, there is only 1 bad one to fix in this case
                        # if Vertex 0 is already bad, do nothing, but if Vertex 1 or 2 is the
                        # bad one, swap it with Vertex 0
                        ######################################################################
                        
                        if not vertex_valid[0]:
                            pass
                            
                        elif not vertex_valid[1]:
                            
                            vertex_idx_0 = vertex_idx_00 + 1
                            vertex_idx_1 = vertex_idx_00
                            
                        elif not vertex_valid[2]:
                            
                            vertex_idx_0 = vertex_idx_00 + 2
                            vertex_idx_2 = vertex_idx_00
                            
                        ######################################################################
                        # Set up the projection vector as the vector pointing from vertex 0
                        # to vertex 1, and normalize it to unit vector
                        ######################################################################
                        proj_dir[0] = sphere_coord_data[vertex_idx_1, 0] - sphere_coord_data[vertex_idx_0, 0]
                        proj_dir[1] = sphere_coord_data[vertex_idx_1, 1] - sphere_coord_data[vertex_idx_0, 1]
                        proj_dir[2] = sphere_coord_data[vertex_idx_1, 2] - sphere_coord_data[vertex_idx_0, 2]
                        
                        
                        temp1 = proj_dir[0]*proj_dir[0] + proj_dir[1]*proj_dir[1] + proj_dir[2]*proj_dir[2]
            
                        temp2 = sqrt(temp1)
                        
                        proj_dir[0] /= temp2
                        proj_dir[1] /= temp2
                        proj_dir[2] /= temp2
                        
                        
                        ######################################################################
                        # Calculate the (s-c) vector
                        ######################################################################
                        temp1 = sphere_coord_data[vertex_idx_0, 0] - hole_xyz[0]
                        temp2 = sphere_coord_data[vertex_idx_0, 1] - hole_xyz[1]
                        temp3 = sphere_coord_data[vertex_idx_0, 2] - hole_xyz[2]
                        
                        ######################################################################
                        #Calculate v.(s-c)
                        ######################################################################
                        temp4 = proj_dir[0]*temp1 + proj_dir[1]*temp2 + proj_dir[2]*temp3
                        
                        ######################################################################
                        #Calculate ||s-c||^2 - r^2
                        ######################################################################
                        
                        temp5 = temp1*temp1 + temp2*temp2 + temp3*temp3 - hole_radius_sq
                        
                        
                        
                        ######################################################################
                        #  POSSIBLE ERROR - If triangle orientation is bad, number inside
                        # SQRT could be negative and we segfault here?
                        ######################################################################
                        
                        if temp4*temp4 - 4.0*temp5 < 0.0:
                            print("BAD PROPERTY")
                        
                        
                        temp1 = -2.0*temp4 + sqrt(temp4*temp4 - 4.0*temp5)
                        temp2 = -2.0*temp4 - sqrt(temp4*temp4 - 4.0*temp5)
                        
                        if temp1 < 0.0:
                            
                            #print("BAD PROPERTY 2", temp1, temp2)
                            
                            pass
                            
                        else:
                            
                            
            
                            sphere_coord_data[vertex_idx_0, 0] = sphere_coord_data[vertex_idx_0, 0] + temp1*proj_dir[0]
                            sphere_coord_data[vertex_idx_0, 1] = sphere_coord_data[vertex_idx_0, 1] + temp1*proj_dir[1]
                            sphere_coord_data[vertex_idx_0, 2] = sphere_coord_data[vertex_idx_0, 2] + temp1*proj_dir[2]
                        
                        
    
                                  
    
                
                
                
                
                
                
                
                
cdef void placeholer_code(smooth_seams, num_valid) except *:
                
    if smooth_seams and (num_valid > 0 and num_valid < 3):
     
     
        pass
    '''   
        if num_valid == 1:
            pass
            
        elif num_valid == 2:
        
            #Vertex 0 will be the good one, push vertices 1 and 2 back towards 0
            
            if not valid_vertex_idx[vertex_idx_0]:
                pass
                
            elif not valid_vertex_idx[vertex_idx_1]:
                
                vertex_idx_0 = vertex_idx_1
                vertex_idx_1 = vertex_idx_00
                
            elif not valid_vertex_idx[vertex_idx_2]:
                
                vertex_idx_0 = vertex_idx_2
                vertex_idx_2 = vertex_idx_00
                
                
            proj_dir[0] = sphere_coord_data[vertex_idx_1, 0] - sphere_coord_data[vertex_idx_0, 0]
            proj_dir[1] = sphere_coord_data[vertex_idx_1, 1] - sphere_coord_data[vertex_idx_0, 1]
            proj_dir[2] = sphere_coord_data[vertex_idx_1, 2] - sphere_coord_data[vertex_idx_0, 2]
                
                
            temp1 = proj_dir[0]*proj_dir[0] + proj_dir[1]*proj_dir[1] + proj_dir[2]*proj_dir[2]
            
            temp2 = sqrt(temp1)
            
            proj_dir[0] /= temp2
            proj_dir[1] /= temp2
            proj_dir[2] /= temp2
            
            temp1 = sphere_coord_data[vertex_idx_0, 0] - hole_xyz[0]
            temp2 = sphere_coord_data[vertex_idx_0, 1] - hole_xyz[1]
            temp3 = sphere_coord_data[vertex_idx_0, 2] - hole_xyz[2]

            #v.(s-c)
            temp4 = proj_dir[0]*temp1 + proj_dir[1]*temp2 + proj_dir[2]*temp3
            
            #||s-c||^2 - r^2
            
            temp5 = temp1*temp1 + temp2*temp2 + temp3*temp3 - hole_radius_sq
            
            ######################################################################
            #  POSSIBLE ERROR - If triangle orientation is bad, number inside
            # SQRT could be negative and we segfault here?
            ######################################################################
            
            if temp4*temp4 - 4.0*temp5 < 0.0:
                print("BAD PROPERTY")
            
            
            temp1 = -2.0*temp4 + sqrt(temp4*temp4 - 4.0*temp5)

            sphere_coord_data[vertex_idx_0, 0] = sphere_coord_data[vertex_idx_0, 0] + temp1*proj_dir[0]
            sphere_coord_data[vertex_idx_0, 0] = sphere_coord_data[vertex_idx_0, 1] + temp1*proj_dir[1]
            sphere_coord_data[vertex_idx_0, 0] = sphere_coord_data[vertex_idx_0, 2] + temp1*proj_dir[2]
            
            
                        
                
                        
                        
        

    
    if vertex_valid[0] == 0:


        edge_1[0] = sphere_coord_data[vertex_idx_1, 0] - sphere_coord_data[vertex_idx_0, 0]
        edge_1[1] = sphere_coord_data[vertex_idx_1, 1] - sphere_coord_data[vertex_idx_0, 1]
        edge_1[2] = sphere_coord_data[vertex_idx_1, 2] - sphere_coord_data[vertex_idx_0, 2]
        
        
        edge_2[0] = sphere_coord_data[vertex_idx_2, 0] - sphere_coord_data[vertex_idx_0, 0]
        edge_2[1] = sphere_coord_data[vertex_idx_2, 1] - sphere_coord_data[vertex_idx_0, 1]
        edge_2[2] = sphere_coord_data[vertex_idx_2, 2] - sphere_coord_data[vertex_idx_0, 2]


        proj_dir[0] = (edge_1[0] +  edge_2[0])/2.0
        proj_dir[1] = (edge_1[1] +  edge_2[1])/2.0
        proj_dir[2] = (edge_1[2] +  edge_2[2])/2.0
        
        
        temp1 = proj_dir[0]*proj_dir[0] + proj_dir[1]*proj_dir[1] + proj_dir[2]*proj_dir[2]
        
        temp2 = sqrt(temp1)
        
        proj_dir[0] /= temp2
        proj_dir[1] /= temp2
        proj_dir[2] /= temp2
        
        #print(proj_dir[0]*proj_dir[0] + proj_dir[1]*proj_dir[1] + proj_dir[2]*proj_dir[2])
        
        
        ######################################################################
        # Some fancy math involving quadratic formula, check notebook for 
        # particulars.
        #
        # We're iterating through the NEIGHBOR'S triangles, so we're moving
        # vertices onto the edge of the current selected void, so use
        # hole_radius_sq for comparison
        ######################################################################
        
        # r^2 = hole_radius_sq
        
        # (s-c)
        
        temp1 = sphere_coord_data[vertex_idx_0, 0] - hole_xyz[0]
        temp2 = sphere_coord_data[vertex_idx_0, 1] - hole_xyz[1]
        temp3 = sphere_coord_data[vertex_idx_0, 2] - hole_xyz[2]

        #v.(s-c)
        temp4 = proj_dir[0]*temp1 + proj_dir[1]*temp2 + proj_dir[2]*temp3
        
        #||s-c||^2 - r^2
        
        temp5 = temp1*temp1 + temp2*temp2 + temp3*temp3 - hole_radius_sq
        
        ######################################################################
        #  POSSIBLE ERROR - If triangle orientation is bad, number inside
        # SQRT could be negative and we segfault here?
        ######################################################################
        
        if temp4*temp4 - 4.0*temp5 < 0.0:
            print("BAD PROPERTY")
        
        
        temp1 = -2.0*temp4 + sqrt(temp4*temp4 - 4.0*temp5)

        new_vertex_0[0] = sphere_coord_data[vertex_idx_0, 0] + temp1*proj_dir[0]
        new_vertex_0[1] = sphere_coord_data[vertex_idx_0, 1] + temp1*proj_dir[1]
        new_vertex_0[2] = sphere_coord_data[vertex_idx_0, 2] + temp1*proj_dir[2]
    
        
    
    if vertex_valid[1] == 0:


        edge_1[0] = sphere_coord_data[vertex_idx_0, 0] - sphere_coord_data[vertex_idx_1, 0]
        edge_1[1] = sphere_coord_data[vertex_idx_0, 1] - sphere_coord_data[vertex_idx_1, 1]
        edge_1[2] = sphere_coord_data[vertex_idx_0, 2] - sphere_coord_data[vertex_idx_1, 2]
        
        
        edge_2[0] = sphere_coord_data[vertex_idx_2, 0] - sphere_coord_data[vertex_idx_1, 0]
        edge_2[1] = sphere_coord_data[vertex_idx_2, 1] - sphere_coord_data[vertex_idx_1, 1]
        edge_2[2] = sphere_coord_data[vertex_idx_2, 2] - sphere_coord_data[vertex_idx_1, 2]


        proj_dir[0] = edge_1[0] +  edge_2[0]
        proj_dir[1] = edge_1[1] +  edge_2[1]
        proj_dir[2] = edge_1[2] +  edge_2[2]
        
        
        temp1 = proj_dir[0]*proj_dir[0] + proj_dir[1]*proj_dir[1] + proj_dir[2]*proj_dir[2]
        
        temp2 = sqrt(temp1)
        
        proj_dir[0] /= temp2
        proj_dir[1] /= temp2
        proj_dir[2] /= temp2
        
        
        ######################################################################
        # Some fancy math involving quadratic formula, check notebook for 
        # particulars.
        #
        # We're iterating through the NEIGHBOR'S triangles, so we're moving
        # vertices onto the edge of the current selected void, so use
        # hole_radius_sq for comparison
        ######################################################################
        
        # r^2 = hole_radius_sq
        
        # (s-c)
        
        temp1 = sphere_coord_data[vertex_idx_1, 0] - hole_xyz[0]
        temp2 = sphere_coord_data[vertex_idx_1, 1] - hole_xyz[1]
        temp3 = sphere_coord_data[vertex_idx_1, 2] - hole_xyz[2]

        #v.(s-c)
        temp4 = proj_dir[0]*temp1 + proj_dir[1]*temp2 + proj_dir[2]*temp3
        
        #||s-c||^2 - r^2
        
        temp5 = temp1*temp1 + temp2*temp2 + temp3*temp3 - hole_radius_sq
        
        ######################################################################
        #  POSSIBLE ERROR - If triangle orientation is bad, number inside
        # SQRT could be negative and we segfault here?
        ######################################################################
        
        temp1 = -2.0*temp4 + sqrt(temp4*temp4 - 4.0*temp5)

        new_vertex_1[0] = sphere_coord_data[vertex_idx_1, 0] + temp1*proj_dir[0]
        new_vertex_1[1] = sphere_coord_data[vertex_idx_1, 1] + temp1*proj_dir[1]
        new_vertex_1[2] = sphere_coord_data[vertex_idx_1, 2] + temp1*proj_dir[2]



    if vertex_valid[2] == 0:


        edge_1[0] = sphere_coord_data[vertex_idx_1, 0] - sphere_coord_data[vertex_idx_2, 0]
        edge_1[1] = sphere_coord_data[vertex_idx_1, 1] - sphere_coord_data[vertex_idx_2, 1]
        edge_1[2] = sphere_coord_data[vertex_idx_1, 2] - sphere_coord_data[vertex_idx_2, 2]
        
        
        edge_2[0] = sphere_coord_data[vertex_idx_0, 0] - sphere_coord_data[vertex_idx_2, 0]
        edge_2[1] = sphere_coord_data[vertex_idx_0, 1] - sphere_coord_data[vertex_idx_2, 1]
        edge_2[2] = sphere_coord_data[vertex_idx_0, 2] - sphere_coord_data[vertex_idx_2, 2]


        proj_dir[0] = edge_1[0] +  edge_2[0]
        proj_dir[1] = edge_1[1] +  edge_2[1]
        proj_dir[2] = edge_1[2] +  edge_2[2]
        
        
        temp1 = proj_dir[0]*proj_dir[0] + proj_dir[1]*proj_dir[1] + proj_dir[2]*proj_dir[2]
        
        temp2 = sqrt(temp1)
        
        proj_dir[0] /= temp2
        proj_dir[1] /= temp2
        proj_dir[2] /= temp2
        
        
        ######################################################################
        # Some fancy math involving quadratic formula, check notebook for 
        # particulars.
        #
        # We're iterating through the NEIGHBOR'S triangles, so we're moving
        # vertices onto the edge of the current selected void, so use
        # hole_radius_sq for comparison
        ######################################################################
        
        # r^2 = hole_radius_sq
        
        # (s-c)
        
        temp1 = sphere_coord_data[vertex_idx_2, 0] - hole_xyz[0]
        temp2 = sphere_coord_data[vertex_idx_2, 1] - hole_xyz[1]
        temp3 = sphere_coord_data[vertex_idx_2, 2] - hole_xyz[2]

        #v.(s-c)
        temp4 = proj_dir[0]*temp1 + proj_dir[1]*temp2 + proj_dir[2]*temp3
        
        #||s-c||^2 - r^2
        
        temp5 = temp1*temp1 + temp2*temp2 + temp3*temp3 - hole_radius_sq
        
        ######################################################################
        #  POSSIBLE ERROR - If triangle orientation is bad, number inside
        # SQRT could be negative and we segfault here?
        ######################################################################
        
        temp1 = -2.0*temp4 + sqrt(temp4*temp4 - 4.0*temp5)

        new_vertex_2[0] = sphere_coord_data[vertex_idx_2, 0] + temp1*proj_dir[0]
        new_vertex_2[1] = sphere_coord_data[vertex_idx_2, 1] + temp1*proj_dir[1]
        new_vertex_2[2] = sphere_coord_data[vertex_idx_2, 2] + temp1*proj_dir[2]


    
        
        
    

    if vertex_valid[0] == 0:
        
        sphere_coord_data[vertex_idx_0, 0] = new_vertex_0[0]
        sphere_coord_data[vertex_idx_0, 1] = new_vertex_0[1]
        sphere_coord_data[vertex_idx_0, 2] = new_vertex_0[2]
        
    
        
    
    if vertex_valid[1] == 0:
        
        sphere_coord_data[vertex_idx_1, 0] = new_vertex_1[0]
        sphere_coord_data[vertex_idx_1, 1] = new_vertex_1[1]
        sphere_coord_data[vertex_idx_1, 2] = new_vertex_1[2]
        
    if vertex_valid[2] == 0:
        
        sphere_coord_data[vertex_idx_2, 0] = new_vertex_2[0]
        sphere_coord_data[vertex_idx_2, 1] = new_vertex_2[1]
        sphere_coord_data[vertex_idx_2, 2] = new_vertex_2[2]
    '''

    
    '''
    curr_idx = valid_vertex_idx.shape[0]/3
    
    for curr_offset in range(curr_idx): #working on sets of 3 vertices per loop
        
        vertex_idx_0 = curr_offset*3
        
        vertex_idx_1 = vertex_idx_0 + 1
        
        vertex_idx_2 = vertex_idx_0 + 2
        
        
        if valid_vertex_idx[vertex_idx_0] or \
           valid_vertex_idx[vertex_idx_1] or \
           valid_vertex_idx[vertex_idx_2]:
            
            valid_vertex_idx[vertex_idx_0] = 1
            valid_vertex_idx[vertex_idx_1] = 1
            valid_vertex_idx[vertex_idx_2] = 1
    '''
    
    
        
        
        
        
        
        
        
        