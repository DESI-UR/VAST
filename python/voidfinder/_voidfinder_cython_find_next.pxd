



cimport numpy as np
cimport cython

from typedefs cimport DTYPE_CP128_t, DTYPE_CP64_t, DTYPE_F64_t, DTYPE_F32_t, DTYPE_B_t, ITYPE_t, DTYPE_INT32_t



cdef void find_next_galaxy(DTYPE_F64_t[:,:] hole_center_memview, \
                            DTYPE_F64_t hole_radius, \
                            DTYPE_F64_t dr, \
                            DTYPE_F64_t direction_mod,\
                            DTYPE_F64_t[:] unit_vector_memview, \
                            galaxy_tree, \
                            ITYPE_t[:] nearest_gal_index_list, \
                            DTYPE_F64_t[:,:] w_coord, \
                            DTYPE_B_t[:,:,:] mask, \
                            DTYPE_F64_t min_dist, \
                            DTYPE_F64_t max_dist, \
                            ITYPE_t[:] nearest_neighbor_x_ratio_index, \
                            ITYPE_t[:] nearest_neighbor_index, \
                            DTYPE_F64_t[:] min_x_ratio, \
                            DTYPE_B_t[:] in_mask) except *




cdef DTYPE_B_t not_in_mask(DTYPE_F64_t[:,:] coordinates, \
                  DTYPE_B_t[:,:,:] survey_mask_ra_dec, \
                  DTYPE_F64_t rmin, \
                  DTYPE_F64_t rmax)