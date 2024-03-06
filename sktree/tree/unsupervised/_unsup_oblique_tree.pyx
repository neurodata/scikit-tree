# cython: cdivision=True
# cython: boundscheck=False
# cython: wraparound=False
# cython: initializedcheck=False

# Authors: Adam Li <adam2392@gmail.com>
#
# License: BSD 3 clause

from libc.stdint cimport SIZE_MAX
from libc.string cimport memcpy, memset

import numpy as np

cimport numpy as cnp

cnp.import_array()

from cython.operator cimport dereference as deref

from ..._lib.sklearn.tree._utils cimport safe_realloc


# Gets Node dtype exposed inside oblique_tree.
# See "_tree.pyx" for more details.
cdef Node dummy
NODE_DTYPE = np.asarray(<Node[:1]>(&dummy)).dtype

# Mitigate precision differences between 32 bit and 64 bit
cdef float32_t FEATURE_THRESHOLD = 1e-7

# =============================================================================
# ObliqueTree
# =============================================================================

cdef class UnsupervisedObliqueTree(UnsupervisedTree):
    """Array-based representation of a binary oblique decision tree.

    The oblique decision tree is represented as a number of parallel arrays. The i-th
    element of each array holds information about the node `i`. Node 0 is the
    tree's root. You can find a detailed description of all arrays in
    `_tree.pxd`. NOTE: Some of the arrays only apply to either leaves or split
    nodes, resp. In this case the values of nodes of the other type are
    arbitrary!

    Attributes
    ----------
    node_count : intp_t
        The number of nodes (internal nodes + leaves) in the tree.

    capacity : intp_t
        The current capacity (i.e., size) of the arrays, which is at least as
        great as `node_count`.

    max_depth : intp_t
        The depth of the tree, i.e. the maximum depth of its leaves.

    children_left : array of intp_t, shape [node_count]
        children_left[i] holds the node id of the left child of node i.
        For leaves, children_left[i] == TREE_LEAF. Otherwise,
        children_left[i] > i. This child handles the case where
        X[:, feature[i]] <= threshold[i].

    children_right : array of intp_t, shape [node_count]
        children_right[i] holds the node id of the right child of node i.
        For leaves, children_right[i] == TREE_LEAF. Otherwise,
        children_right[i] > i. This child handles the case where
        X[:, feature[i]] > threshold[i].

    feature : array of intp_t, shape [node_count]
        feature[i] holds the feature to split on, for the internal node i.

    threshold : array of float64_t, shape [node_count]
        threshold[i] holds the threshold for the internal node i.

    value : array of float64_t, shape [node_count, n_outputs, max_n_classes]
        Contains the constant prediction value of each node.

    impurity : array of float64_t, shape [node_count]
        impurity[i] holds the impurity (i.e., the value of the splitting
        criterion) at node i.

    n_node_samples : array of intp_t, shape [node_count]
        n_node_samples[i] holds the number of training samples reaching node i.

    weighted_n_node_samples : array of intp_t, shape [node_count]
        weighted_n_node_samples[i] holds the weighted number of training samples
        reaching node i.
    """
    def __cinit__(self, intp_t n_features):
        """Constructor."""
        # Input/Output layout
        self.n_features = n_features
        self.value_stride = 1

        # Inner structures
        self.max_depth = 0
        self.node_count = 0
        self.capacity = 0
        self.value = NULL
        self.nodes = NULL

        self.proj_vec_weights = vector[vector[float32_t]](self.capacity)
        self.proj_vec_indices = vector[vector[intp_t]](self.capacity)

    def __reduce__(self):
        """Reduce re-implementation, for pickling."""
        return (UnsupervisedObliqueTree, (self.n_features,), self.__getstate__())

    def __getstate__(self):
        """Getstate re-implementation, for pickling."""
        d = {}
        print("Setting state...")
        # capacity is inferred during the __setstate__ using nodes
        d["max_depth"] = self.max_depth
        d["node_count"] = self.node_count
        d["nodes"] = self._get_node_ndarray()
        d["values"] = self._get_value_ndarray()

        proj_vecs = self.get_projection_matrix()
        d["proj_vecs"] = proj_vecs
        print("Finished Setting state...")
        return d

    def __setstate__(self, d):
        """Setstate re-implementation, for unpickling."""
        self.max_depth = d["max_depth"]
        self.node_count = d["node_count"]

        if "nodes" not in d:
            raise ValueError("You have loaded ObliqueTree version which "
                             "cannot be imported")
        print("Getting state...")
        node_ndarray = d["nodes"]
        value_ndarray = d["values"]

        value_shape = (node_ndarray.shape[0],)
        if (node_ndarray.ndim != 1 or
                node_ndarray.dtype != NODE_DTYPE or
                not node_ndarray.flags.c_contiguous or
                value_ndarray.shape != value_shape or
                not value_ndarray.flags.c_contiguous or
                value_ndarray.dtype != np.float64):
            raise ValueError("Did not recognise loaded array layout")

        self.capacity = node_ndarray.shape[0]
        if self._resize_c(self.capacity) != 0:
            raise MemoryError("resizing tree to %d" % self.capacity)

        # now set the projection vector weights and indices
        proj_vecs = d["proj_vecs"]
        self.n_features = proj_vecs.shape[1]
        for i in range(0, self.node_count):
            for j in range(0, self.n_features):
                weight = proj_vecs[i, j]
                if weight == 0:
                    continue
                self.proj_vec_weights[i].push_back(weight)
                self.proj_vec_indices[i].push_back(j)

        print("Finsihed getting state...")
        memcpy(self.nodes, cnp.PyArray_DATA(node_ndarray),
               self.capacity * sizeof(Node))
        memcpy(self.value, cnp.PyArray_DATA(value_ndarray),
               self.capacity * self.value_stride * sizeof(float64_t))

    cpdef cnp.ndarray get_projection_matrix(self):
        """Get the projection matrix of shape (node_count, n_features)."""
        print("getting proj.")
        proj_vecs = np.zeros((self.node_count, self.n_features), dtype=np.float64)
        for i in range(0, self.node_count):
            for j in range(0, self.proj_vec_weights[i].size()):
                weight = self.proj_vec_weights[i][j]
                feat = self.proj_vec_indices[i][j]
                proj_vecs[i, feat] = weight
        print("finished getting proj.")
        return proj_vecs

    cdef int _resize_c(self, intp_t capacity=SIZE_MAX) except -1 nogil:
        """Guts of _resize.

        Additionally resizes the projection indices and weights.

        Returns -1 in case of failure to allocate memory (and raise MemoryError)
        or 0 otherwise.
        """
        if capacity == self.capacity and self.nodes != NULL:
            return 0

        if capacity == SIZE_MAX:
            if self.capacity == 0:
                capacity = 3  # default initial value
            else:
                capacity = 2 * self.capacity

        safe_realloc(&self.nodes, capacity)
        safe_realloc(&self.value, capacity * self.value_stride)

        # only thing added for oblique trees
        # TODO: this could possibly be removed if we can add projection indices and
        # weights to Node
        self.proj_vec_weights.resize(capacity)
        self.proj_vec_indices.resize(capacity)

        if capacity > self.capacity:
            # value memory is initialised to 0 to enable classifier argmax
            memset(<void*>(self.value + self.capacity * self.value_stride), 0,
                   (capacity - self.capacity) * self.value_stride * sizeof(float64_t))

            # node memory is initialised to 0 to ensure deterministic pickle (padding in Node struct)
            memset(<void*>(self.nodes + self.capacity), 0, (capacity - self.capacity) * sizeof(Node))

        # if capacity smaller than node_count, adjust the counter
        if capacity < self.node_count:
            self.node_count = capacity

        self.capacity = capacity
        return 0

    cdef int _set_split_node(self, SplitRecord* split_node, Node *node, intp_t node_id) except -1 nogil:
        """Set node data.
        """
        # Cython type cast split record into its inherited split record
        # For reference, see:
        # https://www.codementor.io/@arpitbhayani/powering-inheritance-in-c-using-structure-composition-176sygr724
        cdef ObliqueSplitRecord* oblique_split_node = <ObliqueSplitRecord*>(split_node)
        node_id = self.node_count
        with gil:
            print("Trying to set split nodes...")

        node.feature = deref(oblique_split_node).feature
        node.threshold = deref(oblique_split_node).threshold

        # oblique trees store the projection indices and weights
        # inside the tree itself
        self.proj_vec_weights[node_id] = deref(oblique_split_node).proj_vec_weights
        self.proj_vec_indices[node_id] = deref(oblique_split_node).proj_vec_indices

        with gil:
            print("Finished setting for ", node_id)
        return 1

    cdef float32_t _compute_feature(
        self,
        const float32_t[:, :] X_ndarray,
        intp_t sample_index,
        Node *node
    ) noexcept nogil:
        """Compute feature from a given data matrix, X.

        In oblique-aligned trees, this is the projection of X.
        In this case, we take a simple linear combination of some columns of X.
        """
        cdef float32_t proj_feat = 0.0
        cdef float32_t weight = 0.0
        cdef intp_t j = 0
        cdef intp_t feature_index

        # get the index of the node
        cdef intp_t node_id = node - self.nodes

        # cdef intp_t n_projections = proj_vec_indices.size()
        # compute projection of the data based on trained tree
        # proj_vec_weights = self.proj_vec_weights[node_id]
        # proj_vec_indices = self.proj_vec_indices[node_id]
        for j in range(0, self.proj_vec_indices[node_id].size()):
            feature_index = self.proj_vec_indices[node_id][j]
            weight = self.proj_vec_weights[node_id][j]

            # skip a multiplication step if there is nothing to be done
            if weight == 0:
                continue
            proj_feat += X_ndarray[sample_index, feature_index] * weight

        return proj_feat

    cdef void _compute_feature_importances(
        self,
        float64_t[:] importances,
        Node* node
    ) noexcept nogil:
        """Compute feature importances from a Node in the Tree.

        Wrapped in a private function to allow subclassing that
        computes feature importances.
        """
        cdef Node* nodes = self.nodes
        cdef Node* left
        cdef Node* right

        # get the index of the node
        cdef intp_t node_id = node - self.nodes

        left = &nodes[node.left_child]
        right = &nodes[node.right_child]

        cdef intp_t i, feature_index
        cdef float32_t weight
        for i in range(0, self.proj_vec_indices[node_id].size()):
            feature_index = self.proj_vec_indices[node_id][i]
            weight = self.proj_vec_weights[node_id][i]
            if weight < 0:
                weight *= -1

            importances[feature_index] += weight * (
                node.weighted_n_node_samples * node.impurity -
                left.weighted_n_node_samples * left.impurity -
                right.weighted_n_node_samples * right.impurity)
