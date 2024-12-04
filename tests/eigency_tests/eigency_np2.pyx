# distutils: language = c++
# distutils: sources = eigency_tests/eigency_tests_cpp.cpp
import numpy as np

from eigency_tests cimport *

from eigency.core cimport *


cdef extern from "eigency_tests/eigency_tests_cpp.h":
     cdef PlainObjectBase _function_type_long "function_type_long"(FlattenedMap[Array, long, Dynamic, Dynamic] &)
     cdef PlainObjectBase _function_type_ulong "function_type_ulong"(FlattenedMap[Array, unsigned long, Dynamic, Dynamic] &)

# Functions with different matrix types: unsigned long
def function_type_long(np.ndarray[long, ndim=2] array):
    return ndarray(_function_type_long(FlattenedMap[Array, long, Dynamic, Dynamic](array)))

# Functions with different matrix types: unsigned long
def function_type_ulong(np.ndarray[u_long, ndim=2] array):
    return ndarray(_function_type_ulong(FlattenedMap[Array, u_long, Dynamic, Dynamic](array)))
