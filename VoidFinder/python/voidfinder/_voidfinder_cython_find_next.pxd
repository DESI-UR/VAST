



cimport numpy as np
cimport cython

from typedefs cimport DTYPE_CP128_t, \
                      DTYPE_CP64_t, \
                      DTYPE_F64_t, \
                      DTYPE_F32_t, \
                      DTYPE_B_t, \
                      ITYPE_t, \
                      DTYPE_INT32_t, \
                      DTYPE_INT64_t, \
                      DTYPE_UINT16_t, \
                      CELL_ID_t


cdef struct FindNextReturnVal:
    ITYPE_t nearest_neighbor_index
    DTYPE_F64_t min_x_ratio
    DTYPE_B_t in_mask
    
    
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
    
    

cdef FindNextReturnVal find_next_galaxy(DTYPE_F64_t[:,:] hole_center_memview, \
                                        DTYPE_F64_t[:,:] moving_hole_center_memview, \
                                        DTYPE_F64_t hole_radius, \
                                        DTYPE_F64_t dr, \
                                        DTYPE_F64_t direction_mod,\
                                        DTYPE_F64_t[:] unit_vector_memview, \
                                        galaxy_tree, \
                                        DTYPE_INT64_t[:] nearest_gal_index_list, \
                                        ITYPE_t num_neighbors, \
                                        DTYPE_F64_t[:,:] w_coord, \
                                        DTYPE_B_t[:,:] mask, \
                                        DTYPE_INT32_t mask_resolution, \
                                        DTYPE_F64_t min_dist, \
                                        DTYPE_F64_t max_dist, \
                                        DTYPE_F64_t[:] Bcenter_memview, \
                                        Cell_ID_Memory cell_ID_mem, \
                                        NeighborMemory neighbor_mem, \
                                        )
                            

cdef DTYPE_B_t not_in_mask(DTYPE_F64_t[:,:] coordinates, \
                  DTYPE_B_t[:,:] survey_mask_ra_dec, \
                  DTYPE_INT32_t n, \
                  DTYPE_F64_t rmin, \
                  DTYPE_F64_t rmax)


cdef class HoleGridCustomDict:

    cdef DTYPE_INT64_t i_dim, j_dim, k_dim, jk_mod
    
    cdef public HOLE_LOOKUPMEM_t[:] lookup_memory
    
    cdef public DTYPE_INT64_t num_collisions
    
    cdef DTYPE_INT64_t mem_length

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


cdef class GalaxyMapCustomDict:
         
    cdef DTYPE_INT64_t i_dim, j_dim, k_dim, jk_mod
    
    cdef public LOOKUPMEM_t[:] lookup_memory
    
    cdef public DTYPE_INT64_t num_collisions
    
    cdef DTYPE_INT64_t mem_length

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


cdef class GalaxyMap:
    
    cdef public DTYPE_F64_t[:,:] w_coord
    
    cdef public DTYPE_F64_t[:,:] coord_min
    
    cdef public DTYPE_F64_t dl
    
    cdef public CELL_ID_t[:,:] reference_point_ijk
    
    cdef public DTYPE_F64_t[:,:] shell_boundaries_xyz
    
    cdef public DTYPE_F64_t[:,:] cell_center_xyz
    
    cdef DTYPE_F64_t temp1
    
    cdef DTYPE_F64_t temp2
    
    cdef public dict nonvoid_cell_ID_dict
    
    cdef public GalaxyMapCustomDict galaxy_map
    
    cdef public DTYPE_INT64_t[:] galaxy_map_array


cdef class NeighborMemory:
    
    cdef size_t max_num_neighbors
    
    cdef size_t next_neigh_idx
    
    cdef DTYPE_INT64_t* i_nearest
    
    cdef DTYPE_B_t* boolean_nearest
    
    cdef ITYPE_t* i_nearest_reduced
    
    cdef DTYPE_F64_t* candidate_minus_A
    
    cdef DTYPE_F64_t* candidate_minus_center
    
    cdef DTYPE_F64_t* bot_ratio
    
    cdef DTYPE_F64_t* top_ratio
    
    cdef DTYPE_F64_t* x_ratio
    
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
    
              
cdef DistIdxPair _query_first(CELL_ID_t[:,:] reference_point_ijk, \
                              DTYPE_F64_t[:,:] coord_min, \
                              DTYPE_F64_t dl, \
                              DTYPE_F64_t[:,:] shell_boundaries_xyz, \
                              DTYPE_F64_t[:,:] cell_center_xyz, \
                              GalaxyMapCustomDict galaxy_map, \
                              DTYPE_INT64_t[:] galaxy_map_array, \
                              DTYPE_F64_t[:,:] w_coord, \
                              Cell_ID_Memory cell_ID_mem, \
                              DTYPE_F64_t[:,:] reference_point_xyz)
                                         
                                         
                                         
                                         
                                         
                                         
                                         
                                         
                                         
                                         
                                         
                                         
