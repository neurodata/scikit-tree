# distutils: language = c++

# Authors: Adam Li <adam2392@gmail.com>
#          Chester Huynh <chester.huynh924@gmail.com>
#          Parth Vora <pvora4@jhu.edu>
#
# License: BSD 3 clause

# See _oblique_tree.pyx for details.

import numpy as np

cimport numpy as cnp
from libcpp.vector cimport vector

from .._lib.sklearn.tree._splitter cimport SplitRecord
from .._lib.sklearn.tree._tree cimport Node, Tree, TreeBuilder
from .._lib.sklearn.utils._typedefs cimport float32_t, float64_t, intp_t
from ._oblique_splitter cimport ObliqueSplitRecord


cdef class ObliqueTree(Tree):
    cdef vector[vector[float32_t]] proj_vec_weights  # (capacity, n_features) array of projection vectors
    cdef vector[vector[intp_t]] proj_vec_indices     # (capacity, n_features) array of projection vectors

    # overridden methods
    cdef int _resize_c(
        self,
        intp_t capacity=*
    ) except -1 nogil
    cdef int _set_split_node(
        self,
        SplitRecord* split_node,
        Node *node,
        intp_t node_id
    )  nogil except -1
    cdef float32_t _compute_feature(
        self,
        const float32_t[:, :] X_ndarray,
        intp_t sample_index,
        Node *node
    ) noexcept nogil
    cdef void _compute_feature_importances(
        self,
        float64_t[:] importances,
        Node* node
    ) noexcept nogil

    cpdef cnp.ndarray get_projection_matrix(self)
