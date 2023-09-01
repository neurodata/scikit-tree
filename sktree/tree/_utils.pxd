import numpy as np

cimport numpy as cnp

from libcpp.vector cimport vector

cnp.import_array()

from sktree._lib.sklearn.tree._splitter cimport SplitRecord
from sktree._lib.sklearn.tree._tree cimport DOUBLE_t  # Type of y, sample_weight
from sktree._lib.sklearn.tree._tree cimport DTYPE_t  # Type of X
from sktree._lib.sklearn.tree._tree cimport INT32_t  # Signed 32 bit integer
from sktree._lib.sklearn.tree._tree cimport SIZE_t  # Type for indices and counters
from sktree._lib.sklearn.tree._tree cimport UINT32_t  # Unsigned 32 bit integer


cdef int rand_weighted_binary(double p0, UINT32_t* random_state) noexcept nogil

cpdef unravel_index(
    SIZE_t index, cnp.ndarray[SIZE_t, ndim=1] shape
)

cpdef ravel_multi_index(SIZE_t[:] coords, const SIZE_t[:] shape)

cdef void unravel_index_cython(SIZE_t index, const SIZE_t[:] shape, SIZE_t[:] coords) noexcept nogil

cdef SIZE_t ravel_multi_index_cython(SIZE_t[:] coords, const SIZE_t[:] shape) nogil

cdef SIZE_t vector_hash(
    const vector[SIZE_t]& v
) noexcept nogil