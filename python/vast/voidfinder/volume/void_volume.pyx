#cython: language_level=3


cimport cython
import numpy as np
cimport numpy as np
np.import_array()  # required in order to use C-API


from ..typedefs cimport DTYPE_F32_t, \
                        DTYPE_INT32_t

#from numpy.math cimport NAN, INFINITY

#from libc.math cimport fabs, sqrt, asin, atan#, exp, pow, cos, sin, asin



@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cpdef DTYPE_F32_t calculate_void_volume(hole_centers_np, 
                                        hole_radii_np, 
                                        DTYPE_F32_t step_size):
    '''
    Calculate the volume of a void (union of some number of spheres) via a 
    finite-element cube approximation.
    
    
    PARAMETERS
    ==========
    
    hole_centers : ndarray of shape (N,3)
        Hole center coordinates (x, y, z) in units of Mpc/h
        
    hole_radii : ndarray of shape (N,)
        Hole radii in units of Mpc/h
        
    step_size : float
        Distance (in Mpc/h) in a given direction (x, y, and z) between grid 
        points used to estimate void volume
        
        
    RETURNS
    =======
    
    vol : float
        Estimated volume of void in units of (Mpc/h)^3
    '''
    
    
    ############################################################################
    # Find the bounds of the box enclosing the entire void
    #---------------------------------------------------------------------------
    cdef DTYPE_F32_t x_max = np.max(hole_centers_np[:,0] + hole_radii_np)
    cdef DTYPE_F32_t x_min = np.min(hole_centers_np[:,0] - hole_radii_np)
    cdef DTYPE_F32_t y_max = np.max(hole_centers_np[:,1] + hole_radii_np)
    cdef DTYPE_F32_t y_min = np.min(hole_centers_np[:,1] - hole_radii_np)
    cdef DTYPE_F32_t z_max = np.max(hole_centers_np[:,2] + hole_radii_np)
    cdef DTYPE_F32_t z_min = np.min(hole_centers_np[:,2] - hole_radii_np)
    ############################################################################
    
    
    ############################################################################
    # Calculate the number of points in each direction
    #---------------------------------------------------------------------------
    cdef DTYPE_INT32_t x_size = <DTYPE_INT32_t>((x_max - x_min) / step_size)
    cdef DTYPE_INT32_t y_size = <DTYPE_INT32_t>((y_max - y_min) / step_size)
    cdef DTYPE_INT32_t z_size = <DTYPE_INT32_t>((z_max - z_min) / step_size)
    ############################################################################
    
    
    ############################################################################
    # Calculate total number of points in the box
    #---------------------------------------------------------------------------
    cdef DTYPE_F32_t[:,:] hole_centers = hole_centers_np
    cdef DTYPE_F32_t[:] hole_radii = hole_radii_np
    
    cdef DTYPE_INT32_t N_holes = hole_radii_np.shape[0]
    
    cdef DTYPE_INT32_t points_in = 0
    
    cdef DTYPE_INT32_t i,j,k,hole_idx
    cdef DTYPE_F32_t x,y,z
    cdef DTYPE_F32_t diffx,diffy,diffz,d
    
    for i in range(x_size):
        x = x_min + i*step_size
        for j in range(y_size):
            y = y_min + j*step_size
            for k in range(z_size):
                z = z_min + k*step_size
                
                for hole_idx in range(N_holes):
                    diffx = x - hole_centers[hole_idx, 0]
                    diffy = y - hole_centers[hole_idx, 1]
                    diffz = z - hole_centers[hole_idx, 2]
                    
                    d = diffx*diffx + diffy*diffy + diffz*diffz
                    
                    if d < hole_radii[hole_idx]*hole_radii[hole_idx]:
                        points_in += 1
                        break
    ############################################################################
    
    
    ############################################################################
    # Calculate approximate void volume
    #---------------------------------------------------------------------------
    cdef DTYPE_F32_t vol = points_in*step_size*step_size*step_size
    ############################################################################
    
    
    return vol
                    
    
    
    
    
    
    
    
    
    
    
    
    
    
    