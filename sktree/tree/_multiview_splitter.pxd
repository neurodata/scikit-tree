import numpy as np

cimport numpy as cnp
from libcpp.vector cimport vector

from .._lib.sklearn.tree._criterion cimport Criterion
from .._lib.sklearn.tree._splitter cimport SplitRecord, Splitter
from .._lib.sklearn.tree._utils cimport UINT32_t
from .._lib.sklearn.utils._typedefs cimport float32_t, float64_t, intp_t
from ._sklearn_splitter cimport sort


cdef struct MultiViewSplitRecord:
    # Data to track sample split
    intp_t feature              # Which feature to split on.
    intp_t pos                  # Split samples array at the given position,
    #                           # i.e. count of samples below threshold for feature.
    #                           # pos is >= end if the node is a leaf.
    float64_t threshold            # Threshold to split at.
    float64_t improvement          # Impurity improvement given parent node.
    float64_t impurity_left        # Impurity of the left split.
    float64_t impurity_right       # Impurity of the right split.
    float64_t lower_bound       # Lower bound on value of both children for monotonicity
    float64_t upper_bound       # Upper bound on value of both children for monotonicity
    unsigned char missing_go_to_left  # Controls if missing values go to the left node.
    intp_t n_missing            # Number of missing values for the feature being split on
    intp_t n_constant_features  # Number of constant features in the split

    # could maybe be optimized
    vector[intp_t] vec_n_constant_features     # Number of constant features in the split for each feature view


# XXX: This splitter is experimental. Expect changes frequently.
cdef class MultiViewSplitter(Splitter):
    cdef const intp_t[:] feature_set_ends   # an array indicating the column indices of the end of each feature set
    cdef intp_t n_feature_sets                  # the number of feature sets is the length of feature_set_ends + 1

    cdef const intp_t[:] max_features_per_set  # the maximum number of features to sample from each feature set

    # The following are used to track per feature set:
    # - the number of visited features
    # - the number of found constants in this split search
    # - the number of drawn constants in this split search
    cdef intp_t[:] vec_n_visited_features
    cdef intp_t[:] vec_n_found_constants
    cdef intp_t[:] vec_n_drawn_constants

    # XXX: moved from partitioner to this class
    cdef const float32_t[:, :] X
    cdef const unsigned char[::1] missing_values_in_feature_mask
    cdef intp_t n_missing
    cdef void sort_samples_and_feature_values(
        self, intp_t current_feature
    ) noexcept nogil

    cdef void next_p(self, intp_t* p_prev, intp_t* p) noexcept nogil
    cdef intp_t partition_samples(self, float64_t current_threshold) noexcept nogil
    cdef void partition_samples_final(
        self,
        intp_t best_pos,
        float64_t best_threshold,
        intp_t best_feature,
        intp_t best_n_missing,
    ) noexcept nogil
