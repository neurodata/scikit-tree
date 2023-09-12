import numpy as np

cimport numpy as cnp
from libcpp.unordered_map cimport unordered_map
from libcpp.vector cimport vector

from ..._lib.sklearn.tree._splitter cimport SplitRecord
from ..._lib.sklearn.tree._tree cimport DOUBLE_t  # Type of y, sample_weight
from ..._lib.sklearn.tree._tree cimport DTYPE_t  # Type of X
from ..._lib.sklearn.tree._tree cimport INT32_t  # Signed 32 bit integer
from ..._lib.sklearn.tree._tree cimport SIZE_t  # Type for indices and counters
from ..._lib.sklearn.tree._tree cimport UINT32_t  # Unsigned 32 bit integer
from ._unsup_splitter cimport UnsupervisedSplitter


cdef struct ObliqueSplitRecord:
    # Data to track sample split
    SIZE_t feature              # Which feature to split on.
    SIZE_t pos                  # Split samples array at the given position,
    #                           # i.e. count of samples below threshold for feature.
    #                           # pos is >= end if the node is a leaf.
    double threshold            # Threshold to split at.
    double improvement          # Impurity improvement given parent node.
    double impurity_left        # Impurity of the left split.
    double impurity_right       # Impurity of the right split.

    vector[DTYPE_t]* proj_vec_weights   # weights of the vector (max_features,)
    vector[SIZE_t]* proj_vec_indices    # indices of the features (max_features,)


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
    cdef public double feature_combinations             # Number of features to combine
    cdef SIZE_t n_non_zeros                             # Number of non-zero features
    cdef vector[vector[DTYPE_t]] proj_mat_weights       # nonzero weights of sparse proj_mat matrix
    cdef vector[vector[SIZE_t]] proj_mat_indices        # nonzero indices of sparse proj_mat matrix
    cdef SIZE_t[::1] indices_to_sample                  # an array of indices to sample of size mtry X n_features

    # keep a hashmap of every projection vector indices sampled
    cdef unordered_map[size_t, bint] proj_vec_hash

    cdef unordered_map[SIZE_t, DTYPE_t] min_val_map
    cdef unordered_map[SIZE_t, DTYPE_t] max_val_map
    cdef unordered_map[SIZE_t, bint] constant_col_map

    # All oblique splitters (i.e. non-axis aligned splitters) require a
    # function to sample a projection matrix that is applied to the feature matrix
    # to quickly obtain the sampled projections for candidate splits.
    cdef void sample_proj_mat(
        self,
        vector[vector[DTYPE_t]]& proj_mat_weights,
        vector[vector[SIZE_t]]& proj_mat_indices,
        SIZE_t n_known_constants
    ) noexcept nogil

    # Redefined here since the new logic requires calling sample_proj_mat
    cdef int node_reset(
        self,
        SIZE_t start,
        SIZE_t end,
        double* weighted_n_node_samples
    ) except -1 nogil

    cdef int node_split(
        self,
        double impurity,   # Impurity of the node
        SplitRecord* split,
        SIZE_t* n_constant_features,
        double lower_bound,
        double upper_bound
    ) except -1 nogil
    cdef int init(
        self,
        const DTYPE_t[:, :] X,
        const DOUBLE_t[:] sample_weight
    ) except -1
    cdef int pointer_size(self) noexcept nogil

    cdef void compute_features_over_samples(
        self,
        SIZE_t start,
        SIZE_t end,
        const SIZE_t[:] samples,
        DTYPE_t[:] feature_values,
        vector[DTYPE_t]* proj_vec_weights,  # weights of the vector (max_features,)
        vector[SIZE_t]* proj_vec_indices,    # indices of the features (max_features,)
        SIZE_t* n_known_constants
    ) noexcept nogil
