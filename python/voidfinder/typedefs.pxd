

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