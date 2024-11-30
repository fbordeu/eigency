cimport numpy as np

# If this is changed, array_type_t in core.pxd needs to be changed too
# Also, the handling of the types needs to be changed in eigency_cpp.h for NDAC and NDAV
ctypedef fused array_type_t:
    signed char
    unsigned char
    short
    unsigned short
    int
    unsigned int
    long
    unsigned long
    long long
    unsigned long long
    float
    double
    long double
    float complex
    double complex
    long double complex


# Array with limit 2D
cdef api np.ndarray[array_type_t, ndim=2] ndarray(const array_type_t *data)
cdef api np.ndarray[array_type_t, ndim=2] ndarray_C(array_type_t *data, long rows, long cols, long outer_stride, long inner_stride)
cdef api np.ndarray[array_type_t, ndim=2] ndarray_F(array_type_t *data, long rows, long cols, long outer_stride, long inner_stride)
cdef api np.ndarray[array_type_t, ndim=2] ndarray_copy_C(const array_type_t *data, long rows, long cols, long outer_stride, long inner_stride)
cdef api np.ndarray[array_type_t, ndim=2] ndarray_copy_F(const array_type_t *data, long rows, long cols, long outer_stride, long inner_stride)
