# distutils: language = c++

# Authors: Adam Li <adam2392@gmail.com>
#
# License: BSD 3 clause

# See _unsup_oblique_tree.pyx for details.

import numpy as np

cimport numpy as cnp
from libcpp.vector cimport vector

from ..._lib.sklearn.tree._splitter cimport SplitRecord
from ..._lib.sklearn.tree._tree cimport Node
from ..._lib.sklearn.tree._utils cimport DOUBLE_t  # Type of y, sample_weight
from ..._lib.sklearn.tree._utils cimport DTYPE_t  # Type of X
from ..._lib.sklearn.tree._utils cimport INT32_t  # Signed 32 bit integer
from ..._lib.sklearn.tree._utils cimport SIZE_t  # Type for indices and counters
from ..._lib.sklearn.tree._utils cimport UINT32_t  # Unsigned 32 bit integer
from .._oblique_splitter cimport ObliqueSplitRecord
from ._unsup_tree cimport UnsupervisedTree


cdef class UnsupervisedObliqueTree(UnsupervisedTree):
    cdef vector[vector[DTYPE_t]] proj_vec_weights  # (capacity, n_features) array of projection vectors
    cdef vector[vector[SIZE_t]] proj_vec_indices   # (capacity, n_features) array of projection vectors

    # overridden methods
    cdef int _resize_c(
        self,
        SIZE_t capacity=*
    ) except -1 nogil
    cdef int _set_split_node(
        self,
        SplitRecord* split_node,
        Node *node,
        SIZE_t node_id
    )  nogil except -1
    cdef DTYPE_t _compute_feature(
        self,
        const DTYPE_t[:, :] X_ndarray,
        SIZE_t sample_index,
        Node *node
    ) noexcept nogil
    cdef void _compute_feature_importances(
        self,
        cnp.float64_t[:] importances,
        Node* node
    ) noexcept nogil

    cpdef cnp.ndarray get_projection_matrix(self)
