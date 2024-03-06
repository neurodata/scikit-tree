from .._lib.sklearn.utils._typedefs cimport float32_t, float64_t, int32_t, intp_t, uint32_t


cpdef sample_projection_matrix(intp_t max_features, ):
    # Extract input
    cdef float32_t[:] X_data = X.data
    cdef int32_t[:] X_indices = X.indices
    cdef int32_t[:] X_indptr = X.indptr
