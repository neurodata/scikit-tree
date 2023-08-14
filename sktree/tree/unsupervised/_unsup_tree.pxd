# Authors: Adam Li <adam2392@gmail.com>
#          Jong Shin <jshinm@gmail.com>
#

# License: BSD 3 clause

# See _unsup_tree.pyx for details.

import numpy as np

cimport numpy as cnp

from ..._lib.sklearn.tree._splitter cimport SplitRecord
from ..._lib.sklearn.tree._tree cimport BaseTree, Node
from ..._lib.sklearn.tree._utils cimport DOUBLE_t  # Type of y, sample_weight
from ..._lib.sklearn.tree._utils cimport DTYPE_t  # Type of X
from ..._lib.sklearn.tree._utils cimport INT32_t  # Signed 32 bit integer
from ..._lib.sklearn.tree._utils cimport SIZE_t  # Type for indices and counters
from ..._lib.sklearn.tree._utils cimport UINT32_t  # Unsigned 32 bit integer
from ._unsup_splitter cimport UnsupervisedSplitter


# TODO: copy changes from https://github.com/scikit-learn/scikit-learn/pull/25540/files
cdef class UnsupervisedTree(BaseTree):
    # The Tree object is a binary tree structure constructed by the
    # TreeBuilder. The tree structure is used for predictions and
    # feature importances.
    #
    # Inner structures: values are stored separately from node structure,
    # since size is determined at runtime.
    # cdef double* value                   # (capacity) array of values
    # cdef SIZE_t value_stride             # = 1

    # Input/Output layout
    cdef public SIZE_t n_features        # Number of features in X

    # cdef INT32_t* n_categories          # (n_features,) array of number of categories per feature
    #                                   # is <0 for non-categorial (i.e. -1)

    # Methods
    cdef cnp.ndarray _get_value_ndarray(self)
    cdef cnp.ndarray _get_node_ndarray(self)

    # Overridden Methods
    cdef int _set_split_node(
        self,
        SplitRecord* split_node,
        Node* node,
        SIZE_t node_id
    ) except -1 nogil
    cdef int _set_leaf_node(
        self,
        SplitRecord* split_node,
        Node* node
    ) except -1 nogil
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

# =============================================================================
# Tree builder
# =============================================================================

cdef class UnsupervisedTreeBuilder:
    # The TreeBuilder recursively builds a Tree object from training samples,
    # using a Splitter object for splitting internal nodes and assigning
    # values to leaves.
    #
    # This class controls the various stopping criteria and the node splitting
    # evaluation order, e.g. depth-first or best-first.

    cdef UnsupervisedSplitter splitter  # Splitting algorithm

    cdef SIZE_t min_samples_split       # Minimum number of samples in an internal node
    cdef SIZE_t min_samples_leaf        # Minimum number of samples in a leaf
    cdef double min_weight_leaf         # Minimum weight in a leaf
    cdef SIZE_t max_depth               # Maximal tree depth
    cdef double min_impurity_decrease   # Impurity threshold for early stopping

    cpdef build(
        self,
        UnsupervisedTree tree,
        object X,
        const DOUBLE_t[:] sample_weight=*
    )
    cdef _check_input(
        self,
        object X,
        const DOUBLE_t[:] sample_weight
    )
