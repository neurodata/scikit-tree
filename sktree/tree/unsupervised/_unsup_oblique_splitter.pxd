import numpy as np

cimport numpy as cnp
from libcpp.vector cimport vector

from ..._lib.sklearn.tree._splitter cimport SplitRecord
from ..._lib.sklearn.tree._utils cimport UINT32_t
from ..._lib.sklearn.utils._typedefs cimport float32_t, float64_t, intp_t
from ._unsup_splitter cimport UnsupervisedSplitter


cdef struct ObliqueSplitRecord:
    # Data to track sample split
    intp_t feature              # Which feature to split on.
    intp_t pos                  # Split samples array at the given position,
    #                           # i.e. count of samples below threshold for feature.
    #                           # pos is >= end if the node is a leaf.
    float64_t threshold            # Threshold to split at.
    float64_t improvement          # Impurity improvement given parent node.
    float64_t impurity_left        # Impurity of the left split.
    float64_t impurity_right       # Impurity of the right split.

    vector[float32_t] proj_vec_weights   # weights of the vector (max_features,)
    vector[intp_t] proj_vec_indices    # indices of the features (max_features,)


cdef class UnsupervisedObliqueSplitter(UnsupervisedSplitter):
    """
    Notable changes wrt scikit-learn:
    1. `weighted_n_node_samples` is used as a stopping criterion and just used to
    keep count of the "number of samples (weighted)". All samples have a default weight
    of '1'.
    2. `X` array instead of `y` array is stored as the criterions are computed over the X
    array.
    """

    # Oblique Splitting extra parameters
    cdef public float64_t feature_combinations             # Number of features to combine
    cdef intp_t n_non_zeros                             # Number of non-zero features
    cdef vector[vector[float32_t]] proj_mat_weights       # nonzero weights of sparse proj_mat matrix
    cdef vector[vector[intp_t]] proj_mat_indices        # nonzero indices of sparse proj_mat matrix
    cdef intp_t[::1] indices_to_sample                  # an array of indices to sample of size mtry X n_features

    # All oblique splitters (i.e. non-axis aligned splitters) require a
    # function to sample a projection matrix that is applied to the feature matrix
    # to quickly obtain the sampled projections for candidate splits.
    cdef void sample_proj_mat(self,
                              vector[vector[float32_t]]& proj_mat_weights,
                              vector[vector[intp_t]]& proj_mat_indices) noexcept nogil

    # Redefined here since the new logic requires calling sample_proj_mat
    cdef int node_reset(self, intp_t start, intp_t end,
                        float64_t* weighted_n_node_samples) except -1 nogil

    cdef int node_split(
        self,
        float64_t impurity,   # Impurity of the node
        SplitRecord* split,
        intp_t* n_constant_features,
        float64_t lower_bound,
        float64_t upper_bound
    ) except -1 nogil
    cdef int init(
        self,
        const float32_t[:, :] X,
        const float64_t[:] sample_weight
    ) except -1
    cdef intp_t pointer_size(self) noexcept nogil

    cdef void compute_features_over_samples(
        self,
        intp_t start,
        intp_t end,
        const intp_t[:] samples,
        float32_t[:] feature_values,
        vector[float32_t]& proj_vec_weights,  # weights of the vector (max_features,)
        vector[intp_t]& proj_vec_indices    # indices of the features (max_features,)
    ) noexcept nogil
