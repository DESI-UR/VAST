"""

This module houses the cython type definitions for the various cythonized functions
within VoidFinder

"""

cimport cython
import numpy as np
cimport numpy as np
np.import_array()

cdef extern from "complex.h" nogil:
    float crealf(float complex)
    double creal(double complex)
    long double creall(long double complex)
    
    
    

ctypedef np.complex128_t DTYPE_CP128_t
ctypedef np.complex64_t DTYPE_CP64_t
ctypedef np.float64_t DTYPE_F64_t  
ctypedef np.float32_t DTYPE_F32_t
ctypedef np.uint8_t DTYPE_B_t
ctypedef np.intp_t ITYPE_t  
ctypedef np.int32_t DTYPE_INT32_t
ctypedef np.int64_t DTYPE_INT64_t
ctypedef np.uint16_t DTYPE_UINT16_t
ctypedef np.uint16_t CELL_ID_t
ctypedef np.int8_t DTYPE_INT8_t