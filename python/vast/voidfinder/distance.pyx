#cython: language_level=3


"""
VoidFinder module for some fast cythonized distance calculations.

"""
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
                      DTYPE_INT64_t

from libc.math cimport NAN, INFINITY, fabs, sqrt, asin, atan#, exp, pow, cos, sin, asin

from scipy.integrate import _quadpack

from .constants import c as speed_of_light

cdef DTYPE_F32_t c = <DTYPE_F32_t>(speed_of_light)



@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef DTYPE_F32_t R_W_interval(DTYPE_F32_t a, 
                              DTYPE_F32_t omega_M):
    """
    Function for the Robertson-Walker Interval
    """

    return 1.0/(sqrt(a*omega_M*(1.0+((1.0-omega_M)*a*a*a/omega_M))))




@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cpdef np.ndarray z_to_comoving_dist(DTYPE_F32_t[:] z_input, 
                                    DTYPE_F32_t omega_M, 
                                    DTYPE_F32_t h):
    """
    Convert redshift values into the comoving distance cosmology using the 
    integral of the Robertson-Walker metric.
    
    
    Parameters
    ==========
    
    z_input : numpy.ndarray of shape (N,)
        redshift values to compute distances for
        
    omega_M : float
        Cosmological matter energy density
        
    h : float
        Hubble constant factor
        
        
    Returns
    =======
    
    output_comov_dists : numpy.ndarray of shape (N,)
        the comoving distance values in units of Mpc/h
    """
    
    
    cdef ITYPE_t num_redshifts = z_input.shape[0]
    
    #create a python object for return from this function
    output_comov_dists = np.ones(num_redshifts, dtype=np.float32)
    
    cdef DTYPE_F32_t[:] out_dists_array = output_comov_dists
    
    cdef DTYPE_F32_t H0 = 100.0*h
    
    cdef ITYPE_t idx
    
    cdef DTYPE_F32_t curr_redshift
    
    cdef DTYPE_F32_t a_start
    
    cdef tuple retval
    
    for idx in range(num_redshifts):
        
        curr_redshift = z_input[idx]
        
        a_start = 1.0/(1.0+curr_redshift)
        
        ########################################################################
        # This function is the python scipy wrapper/interface around the 
        # _quadpack FORTRAN library.  Included here for reference since we are 
        # being sneaky and skipping the main scipy interface for a lower-level 
        # scipy wrapper.
        #
        # def quad(func, a, b, args=(), full_output=0, epsabs=1.49e-8, 
        #          epsrel=1.49e-8, limit=50, points=None, weight=None, 
        #          wvar=None, wopts=None, maxp1=50, limlst=50):
        ########################################################################
        
        retval = _quadpack._qagse(R_W_interval, a_start, 1.0, (omega_M,), 0, 1.49e-8, 1.49e-8, 50)
                                
        out_dists_array[idx] = (<DTYPE_F32_t>retval[0])*(c/H0)
        
    return output_comov_dists
        
        
                                    
                                    
                                    
'''                           
def Distance(z,omega_m,h):
    dist = np.ones(len(z))
    H0 = 100*h
    for i,redshift in enumerate(z):
        a_start = 1/(1+redshift)
        I = quad(f,a_start,1,args=omega_m)
        dist[i] = I[0]*(c/H0)
    return dist


def f(a,omega_m):
     return 1/(np.sqrt(a*omega_m*(1+((1-omega_m)*a**3/omega_m))))
'''




