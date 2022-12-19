#cython: language_level=3



cimport numpy as np
cimport cython

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




cdef class MaskChecker:

    cdef int mode

    cdef DTYPE_B_t[:,:] survey_mask_ra_dec
    
    cdef DTYPE_INT32_t n #mask resolution factor
    
    cdef DTYPE_F64_t rmin
    
    cdef DTYPE_F64_t rmax
    
    cdef DTYPE_F64_t[:,:] xyz_limits
    
    cpdef DTYPE_B_t not_in_mask(self, DTYPE_F64_t[:] coordinates)



cdef struct FindNextReturnVal:
    ITYPE_t nearest_neighbor_index
    DTYPE_F64_t min_x_val
    DTYPE_B_t failed
    
    
cdef struct DistIdxPair:
    ITYPE_t idx
    DTYPE_F64_t dist
                                  
                                         
cdef packed struct LOOKUPMEM_t:
    DTYPE_B_t filled_flag
    CELL_ID_t key_i
    CELL_ID_t key_j
    CELL_ID_t key_k
    DTYPE_INT64_t offset
    DTYPE_INT64_t num_elements                   
     

cdef packed struct HOLE_LOOKUPMEM_t:
    DTYPE_B_t filled_flag
    CELL_ID_t key_i
    CELL_ID_t key_j
    CELL_ID_t key_k
    

cdef struct OffsetNumPair:
    DTYPE_INT64_t offset, num_elements      
    
    
'''
cdef FindNextReturnVal find_next_galaxy(DTYPE_F64_t[:,:] hole_center_memview, \
                                        DTYPE_F64_t[:,:] moving_hole_center_memview, \
                                        DTYPE_F64_t hole_radius, \
                                        DTYPE_F64_t dr, \
                                        DTYPE_F64_t direction_mod,\
                                        DTYPE_F64_t[:] unit_vector_memview, \
                                        SpatialMap galaxy_tree, \
                                        DTYPE_INT64_t[:] nearest_gal_index_list, \
                                        ITYPE_t num_neighbors, \
                                        MaskChecker mask_checker, \
                                        DTYPE_F64_t[:] Bcenter_memview, \
                                        Cell_ID_Memory cell_ID_mem, \
                                        NeighborMemory neighbor_mem, \
                                        )
'''

cpdef DTYPE_B_t not_in_mask(DTYPE_F64_t[:] coordinates, \
                  DTYPE_B_t[:,:] survey_mask_ra_dec, \
                  DTYPE_INT32_t n, \
                  DTYPE_F64_t rmin, \
                  DTYPE_F64_t rmax)

#cpdef DTYPE_B_t not_in_mask(DTYPE_F64_t[:,:] coordinates, \
#                  DTYPE_B_t[:,:] survey_mask_ra_dec, \
#                  DTYPE_INT32_t n, \
#                  DTYPE_F64_t rmin, \
#                  DTYPE_F64_t rmax)








cdef class HoleGridCustomDict:

    cdef object hole_lookup_buffer
    
    cdef object numpy_dtype
    

    cdef DTYPE_INT64_t i_dim, j_dim, k_dim, jk_mod, num_elements, lookup_fd
    
    cdef public HOLE_LOOKUPMEM_t[:] lookup_memory
    
    cdef public object dummy_arr
    
    cdef public DTYPE_INT64_t num_collisions
    
    cdef public DTYPE_INT64_t mem_length

    cdef public DTYPE_INT64_t custom_hash(self, 
                                          CELL_ID_t i, 
                                          CELL_ID_t j, 
                                          CELL_ID_t k)
    
    cpdef public DTYPE_B_t contains(self,
                                    CELL_ID_t i, 
                                    CELL_ID_t j, 
                                    CELL_ID_t k)

    cpdef public void setitem(self, 
                              CELL_ID_t i,
                              CELL_ID_t j,
                              CELL_ID_t k)

    cdef public void resize(self, DTYPE_INT64_t new_mem_length)



cdef class GalaxyMapCustomDict:

    cdef object lookup_buffer
    
    cdef object numpy_dtype
         
    cdef object num_elements
         
    cdef DTYPE_INT64_t i_dim, j_dim, k_dim, jk_mod, lookup_fd, process_local_num_elements
    
    cdef public LOOKUPMEM_t[:] lookup_memory
    
    cdef public object dummy_arr
    
    cdef public DTYPE_INT64_t num_collisions
    
    cdef public DTYPE_INT64_t mem_length
    
    cdef public void refresh(self)

    cdef public DTYPE_INT64_t custom_hash(self, 
                                          CELL_ID_t i, 
                                          CELL_ID_t j, 
                                          CELL_ID_t k)
    
    cpdef public DTYPE_B_t contains(self,
                                    CELL_ID_t i, 
                                    CELL_ID_t j, 
                                    CELL_ID_t k)
    
    cpdef public OffsetNumPair getitem(self,
                                       CELL_ID_t i, 
                                       CELL_ID_t j, 
                                       CELL_ID_t k)

    cpdef public void setitem(self, 
                              CELL_ID_t i,
                              CELL_ID_t j,
                              CELL_ID_t k, 
                              DTYPE_INT64_t offset,
                              DTYPE_INT64_t num_elements)
    
    cdef public void resize(self, DTYPE_INT64_t new_mem_length)


cdef class SpatialMap:

    cdef public Cell_ID_Memory cell_ID_mem
    
    cdef public NeighborMemory neighbor_mem

    cdef public object update_lock
    
    cdef public DTYPE_INT32_t mask_mode
    
    cdef public DTYPE_F64_t[:,:] points_xyz
    
    cdef public object points_buffer
    
    cdef public DTYPE_INT64_t num_points, points_xyz_fd
    
    cdef public DTYPE_F64_t[:] coord_min
    
    cdef public DTYPE_F64_t hole_grid_edge_length
    
    cdef public DTYPE_F64_t dl
    
    cdef public CELL_ID_t[:] reference_point_ijk
    
    cdef public DTYPE_F64_t[:,:] shell_boundaries_xyz
    
    cdef public DTYPE_F64_t[:,:] cell_center_xyz
    
    cdef DTYPE_F64_t temp1
    
    cdef DTYPE_F64_t temp2
    
    cdef public dict nonvoid_cell_ID_dict
    
    cdef public GalaxyMapCustomDict galaxy_map
    
    cdef public GalaxyMapCustomDict galaxy_map_2
    
    cdef public DTYPE_INT64_t[:] galaxy_map_array
    
    cdef public object galaxy_map_array_buffer
    
    cdef public DTYPE_INT64_t num_gma_indices
    
    cdef public DTYPE_INT64_t gma_fd



    cdef public DTYPE_F64_t[:] temp_vector
    
    cdef public DTYPE_F64_t temp_f64_accum, temp_f64_val
    
    cdef public DTYPE_F64_t[:] temp_hole_center



    #DEBUGGING
    cdef public object kdtree
    
    
    cpdef public DTYPE_B_t contains(self,
                                   CELL_ID_t i, 
                                   CELL_ID_t j, 
                                   CELL_ID_t k)
    
    cdef public OffsetNumPair getitem(self,
                                      CELL_ID_t i, 
                                      CELL_ID_t j, 
                                      CELL_ID_t k)
    
    cdef public void setitem(self, 
                             CELL_ID_t i,
                             CELL_ID_t j,
                             CELL_ID_t k, 
                             DTYPE_INT64_t offset,
                             DTYPE_INT64_t num_elements)


    cdef DTYPE_B_t cell_in_source(self, CELL_ID_t i, CELL_ID_t j, CELL_ID_t k)
    
    cdef void add_cell_periodic(self,
                                CELL_ID_t i,
                                CELL_ID_t j,
                                CELL_ID_t k)
 
    cpdef void refresh(self)
    
    cdef public ITYPE_t find_first_neighbor(self, DTYPE_F64_t[:])
    
    cpdef public FindNextReturnVal find_next_bounding_point(self, 
                                                            DTYPE_F64_t[:] start_hole_center,
                                                            DTYPE_F64_t[:] search_unit_vector,
                                                            ITYPE_t[:] existing_bounding_idxs,
                                                            ITYPE_t num_neighbors,
                                                            MaskChecker mask_checker)
    
    cpdef public FindNextReturnVal find_next_bounding_point_2(SpatialMap self, 
                                                            DTYPE_F64_t[:] start_hole_center,
                                                            DTYPE_F64_t[:] search_unit_vector,
                                                            ITYPE_t[:] existing_bounding_idxs,
                                                            ITYPE_t num_neighbors,
                                                            MaskChecker mask_checker)
    
    
    cdef public DTYPE_F64_t calculate_x_val(self, 
                                            ITYPE_t gal_idx, 
                                            ITYPE_t k1g_idx, 
                                            DTYPE_F64_t[:] start_hole_center, 
                                            DTYPE_F64_t[:] search_unit_vector)
    
    cdef public void ijk_to_xyz(self, 
                                DTYPE_INT64_t[:] ijk_location, 
                                DTYPE_F64_t[:] output_xyz)
    
    cdef public void xyz_to_pqr(self,
                                DTYPE_F64_t[:] input_xyz,
                                CELL_ID_t[:] output_pqr)
 


cdef class NeighborMemory:
    
    cdef size_t max_num_neighbors
    
    cdef size_t next_neigh_idx
    
    cdef DTYPE_INT64_t* i_nearest
    
    cdef DTYPE_F64_t* candidate_minus_A
    
    cdef DTYPE_F64_t* candidate_minus_center
    
    cdef DTYPE_F64_t* A_minus_center
    
    cdef DTYPE_F64_t* bot_ratio
    
    cdef DTYPE_F64_t* top_ratio
    
    cdef DTYPE_F64_t* x_vals
    
    cdef void resize(self, size_t max_num_neighbors)
    
    cdef void append(self, DTYPE_INT64_t neigh_idx_val)

           
cdef class Cell_ID_Memory:

    cdef CELL_ID_t* data     
    
    cdef DTYPE_INT64_t* level_start_idx
    
    cdef DTYPE_INT64_t* level_stop_idx
    
    cdef CELL_ID_t* curr_ijk  
    
    cdef DTYPE_INT64_t total_num_rows    
    
    cdef DTYPE_INT64_t num_available_rows   
     
    cdef DTYPE_INT64_t next_unused_row_idx
    
    cdef DTYPE_INT64_t max_level_mem
    
    cdef DTYPE_INT64_t max_level_available
    
    
    

cdef class SphereGrower:

    cdef public DTYPE_F64_t[:] sphere_center_xyz
        
    cdef public DTYPE_F64_t[:] search_unit_vector
    
    cdef public ITYPE_t[:] existing_bounding_idxs
        
    cdef public DTYPE_F64_t[:] midpoint_memview
        
    cdef public DTYPE_F64_t[:] AB_memview
        
    cdef public DTYPE_F64_t[:] BC_memview
        
    cdef public DTYPE_F64_t[:] v3_memview
        
    cdef public DTYPE_F64_t[:] hole_center_to_3plane_memview
    
    
    cdef public DTYPE_F64_t[:] hole_center_k4g1
    
    cdef public DTYPE_F64_t[:] hole_center_k4g2
    
    
    cdef public DTYPE_F64_t[:] temp_vector
    
    cdef public DTYPE_F64_t vector_modulus, temp_f64_val, temp_f64_accum
    
    cdef public ITYPE_t idx, jdx
    
    
    cdef public void update_hole_center(self, DTYPE_F64_t x_val)
    
    cdef public void calculate_search_unit_vector_after_k1g(self, DTYPE_F64_t[:] k1g_xyz)
    
    cdef public void calculate_search_unit_vector_after_k2g(self, 
                                                            DTYPE_F64_t[:] k1g_xyz, 
                                                            DTYPE_F64_t[:] k2g_xyz)
    
    cdef public DTYPE_B_t calculate_search_unit_vector_after_k3g(self, 
                                                                 DTYPE_F64_t[:] k1g_xyz, 
                                                                 DTYPE_F64_t[:] k2g_xyz, 
                                                                 DTYPE_F64_t[:] k3g_xyz)
    
    
    
    
              
cpdef DistIdxPair _query_first(CELL_ID_t[:] reference_point_ijk, \
                               DTYPE_F64_t[:] coord_min, \
                               DTYPE_F64_t dl, \
                               DTYPE_F64_t[:,:] shell_boundaries_xyz, \
                               DTYPE_F64_t[:,:] cell_center_xyz, \
                               SpatialMap galaxy_map, \
                               Cell_ID_Memory cell_ID_mem, \
                               DTYPE_F64_t[:] reference_point_xyz)
                                         
                                         
                                         
                                         
                                         
cdef DTYPE_INT64_t _gen_cube(CELL_ID_t[:] center_ijk, \
                             DTYPE_INT32_t level, \
                             Cell_ID_Memory cell_ID_mem, \
                             SpatialMap galaxy_map)
                                         
                                         
                                         
                                         
cpdef DTYPE_INT64_t find_next_prime(DTYPE_INT64_t threshold_value)
                                         
