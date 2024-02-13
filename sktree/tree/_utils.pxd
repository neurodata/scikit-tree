import numpy as np

cimport numpy as cnp
from libcpp.vector cimport vector

cnp.import_array()

from .._lib.sklearn.tree._splitter cimport SplitRecord
from .._lib.sklearn.tree._utils cimport UINT32_t
from .._lib.sklearn.utils._typedefs cimport float32_t, float64_t, int32_t, intp_t


cdef int rand_weighted_binary(float64_t p0, UINT32_t* random_state) noexcept nogil

cpdef unravel_index(
    intp_t index, cnp.ndarray[intp_t, ndim=1] shape
)

cpdef ravel_multi_index(vector[intp_t] coords, const intp_t[:] shape)

cdef vector[intp_t] unravel_index_cython(intp_t index, const intp_t[:] shape) noexcept nogil

cdef intp_t ravel_multi_index_cython(vector[intp_t] coords, const intp_t[:] shape) noexcept nogil

cdef vector[vector[intp_t]] cartesian_cython(vector[vector[intp_t]] sequences) noexcept nogil

cpdef cartesian_python(vector[vector[intp_t]]& sequences)
