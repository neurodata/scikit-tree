# distutils: language = c++

# Authors: Adam Li <adam2392@gmail.com>
#          Chester Huynh <chester.huynh924@gmail.com>
#          Parth Vora <pvora4@jhu.edu>
#
# License: BSD 3 clause

# See _oblique_splitter.pyx for details.

import numpy as np

cimport numpy as cnp
from libcpp.vector cimport vector

from .._lib.sklearn.tree._criterion cimport Criterion
from .._lib.sklearn.tree._splitter cimport SplitRecord, Splitter
from .._lib.sklearn.tree._utils cimport DOUBLE_t  # Type of y, sample_weight
from .._lib.sklearn.tree._utils cimport DTYPE_t  # Type of X
from .._lib.sklearn.tree._utils cimport INT32_t  # Signed 32 bit integer
from .._lib.sklearn.tree._utils cimport SIZE_t  # Type for indices and counters
from .._lib.sklearn.tree._utils cimport UINT32_t  # Unsigned 32 bit integer
from ._sklearn_splitter cimport sort


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


cdef class BaseObliqueSplitter(Splitter):
    # Base class for oblique splitting, where additional data structures and API is defined.
    #

    # Oblique Splitting extra parameters (mtry, n_dims) matrix
    cdef vector[vector[DTYPE_t]] proj_mat_weights       # nonzero weights of sparse proj_mat matrix
    cdef vector[vector[SIZE_t]] proj_mat_indices        # nonzero indices of sparse proj_mat matrix

    # TODO: assumes all oblique splitters only work with dense data
    cdef const DTYPE_t[:, :] X

    # feature weights across (n_dims,)
    cdef DTYPE_t[:] feature_weights

    # All oblique splitters (i.e. non-axis aligned splitters) require a
    # function to sample a projection matrix that is applied to the feature matrix
    # to quickly obtain the sampled projections for candidate splits.
    cdef void sample_proj_mat(
        self,
        vector[vector[DTYPE_t]]& proj_mat_weights,
        vector[vector[SIZE_t]]& proj_mat_indices
    ) noexcept nogil

    # Redefined here since the new logic requires calling sample_proj_mat
    cdef int node_reset(
        self,
        SIZE_t start,
        SIZE_t end,
        double* weighted_n_node_samples
    ) except -1 nogil

    cdef void compute_features_over_samples(
        self,
        SIZE_t start,
        SIZE_t end,
        const SIZE_t[:] samples,
        DTYPE_t[:] feature_values,
        vector[DTYPE_t]* proj_vec_weights,  # weights of the vector (max_features,)
        vector[SIZE_t]* proj_vec_indices    # indices of the features (max_features,)
    ) noexcept nogil

    cdef int node_split(
        self,
        double impurity,   # Impurity of the node
        SplitRecord* split,
        SIZE_t* n_constant_features,
        double lower_bound,
        double upper_bound,
    ) except -1 nogil


cdef class ObliqueSplitter(BaseObliqueSplitter):
    # The splitter searches in the input space for a linear combination of features and a threshold
    # to split the samples samples[start:end].

    # Oblique Splitting extra parameters
    cdef public double feature_combinations             # Number of features to combine
    cdef SIZE_t n_non_zeros                             # Number of non-zero features
    cdef SIZE_t[::1] indices_to_sample                  # an array of indices to sample of size mtry X n_features

    # All oblique splitters (i.e. non-axis aligned splitters) require a
    # function to sample a projection matrix that is applied to the feature matrix
    # to quickly obtain the sampled projections for candidate splits.
    cdef void sample_proj_mat(
        self,
        vector[vector[DTYPE_t]]& proj_mat_weights,
        vector[vector[SIZE_t]]& proj_mat_indices
    ) noexcept nogil
