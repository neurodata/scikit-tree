# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False, initializedcheck=False, cdivision=True

# Authors: Adam Li <adam2392@gmail.com>
#          Jong Shin <jshinm@gmail.com>
#

# License: BSD 3 clause

from cpython cimport Py_INCREF, PyObject, PyTypeObject
from cython.operator cimport dereference as deref
from libc.stdint cimport INTPTR_MAX
from libc.stdlib cimport free, malloc
from libc.string cimport memcpy
from libcpp cimport bool
from libcpp.algorithm cimport pop_heap, push_heap
from libcpp.vector cimport vector

import struct

import numpy as np
from scipy.sparse import issparse

cimport numpy as cnp

cnp.import_array()


cdef extern from "numpy/arrayobject.h":
    object PyArray_NewFromDescr(PyTypeObject* subtype, cnp.dtype descr,
                                int nd, cnp.npy_intp* dims,
                                cnp.npy_intp* strides,
                                void* data, int flags, object obj)
    int PyArray_SetBaseObject(cnp.ndarray arr, PyObject* obj)

cdef extern from "<stack>" namespace "std" nogil:
    cdef cppclass stack[T]:
        ctypedef T value_type
        stack() except +
        bint empty()
        void pop()
        void push(T&) except +  # Raise c++ exception for bad_alloc -> MemoryError
        T& top()

# =============================================================================
# Types and constants
# =============================================================================

from numpy import float32 as DTYPE
from numpy import float64 as DOUBLE


cdef double INFINITY = np.inf
cdef double EPSILON = np.finfo('double').eps

# Some handy constants (BestFirstTreeBuilder)
cdef int IS_FIRST = 1
cdef int IS_NOT_FIRST = 0
cdef int IS_LEFT = 1
cdef int IS_NOT_LEFT = 0

TREE_LEAF = -1
TREE_UNDEFINED = -2
cdef SIZE_t _TREE_LEAF = TREE_LEAF
cdef SIZE_t _TREE_UNDEFINED = TREE_UNDEFINED

# Build the corresponding numpy dtype for Node.
# This works by casting `dummy` to an array of Node of length 1, which numpy
# can construct a `dtype`-object for. See https://stackoverflow.com/q/62448946
# for a more detailed explanation.
cdef Node dummy
NODE_DTYPE = np.asarray(<Node[:1]>(&dummy)).dtype

# =============================================================================
# Unsupervised TreeBuilder
# =============================================================================

cdef class UnsupervisedTreeBuilder:
    """Interface for different tree building strategies."""

    cpdef build(
        self,
        UnsupervisedTree tree,
        object X,
        const DOUBLE_t[:] sample_weight=None
    ):
        """Build a decision tree from the training set X."""
        pass

    cdef inline _check_input(
        self,
        object X,
        const DOUBLE_t[:] sample_weight,
    ):
        """Check input dtype, layout and format"""
        if issparse(X):
            X = X.tocsc()
            X.sort_indices()

            if X.data.dtype != DTYPE:
                X.data = np.ascontiguousarray(X.data, dtype=DTYPE)

            if X.indices.dtype != np.int32 or X.indptr.dtype != np.int32:
                raise ValueError("No support for np.int64 index based "
                                 "sparse matrices")

        elif X.dtype != DTYPE:
            # since we have to copy we will make it fortran for efficiency
            X = np.asfortranarray(X, dtype=DTYPE)

        if (sample_weight is not None and
            (sample_weight.base.dtype != DOUBLE or not
             sample_weight.base.flags.contiguous)):
            sample_weight = np.asarray(sample_weight, dtype=DOUBLE, order="C")

        return X, sample_weight

# Best first builder ----------------------------------------------------------
cdef struct FrontierRecord:
    # Record of information of a Node, the frontier for a split. Those records are
    # maintained in a heap to access the Node with the best improvement in impurity,
    # allowing growing trees greedily on this improvement.
    SIZE_t node_id
    SIZE_t start
    SIZE_t end
    SIZE_t pos
    SIZE_t depth
    bint is_leaf
    double impurity
    double impurity_left
    double impurity_right
    double improvement

# Depth first builder ---------------------------------------------------------
# A record on the stack for depth-first tree growing
cdef struct StackRecord:
    SIZE_t start
    SIZE_t end
    SIZE_t depth
    SIZE_t parent
    bint is_left
    double impurity
    SIZE_t n_constant_features

cdef inline bool _compare_records(
    const FrontierRecord& left,
    const FrontierRecord& right,
):
    return left.improvement < right.improvement

cdef inline void _add_to_frontier(
    FrontierRecord rec,
    vector[FrontierRecord]& frontier,
) noexcept nogil:
    """Adds record `rec` to the priority queue `frontier`."""
    frontier.push_back(rec)
    push_heap(frontier.begin(), frontier.end(), &_compare_records)


cdef class UnsupervisedBestFirstTreeBuilder(UnsupervisedTreeBuilder):
    """Build an unsupervised decision tree in best-first fashion.

    The best node to expand is given by the node at the frontier that has the
    highest impurity improvement.
    """
    cdef SIZE_t max_leaf_nodes

    def __cinit__(
        self,
        UnsupervisedSplitter splitter,
        SIZE_t min_samples_split,
        SIZE_t min_samples_leaf,
        double min_weight_leaf,
        SIZE_t max_depth,
        SIZE_t max_leaf_nodes,
        double min_impurity_decrease
    ):
        self.splitter = splitter
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.min_weight_leaf = min_weight_leaf
        self.max_depth = max_depth
        self.max_leaf_nodes = max_leaf_nodes
        self.min_impurity_decrease = min_impurity_decrease

    cpdef build(
        self,
        UnsupervisedTree tree,
        object X,
        const DOUBLE_t[:] sample_weight=None
    ):
        """Build a decision tree from the training set X."""
        # check input
        X, sample_weight = self._check_input(X, sample_weight)

        # Parameters
        cdef UnsupervisedSplitter splitter = self.splitter
        cdef SIZE_t max_leaf_nodes = self.max_leaf_nodes

        # Recursive partition (without actual recursion)
        splitter.init(X, sample_weight)

        cdef vector[FrontierRecord] frontier
        cdef FrontierRecord record
        cdef FrontierRecord split_node_left
        cdef FrontierRecord split_node_right

        cdef SIZE_t n_node_samples = splitter.n_samples
        cdef SIZE_t max_split_nodes = max_leaf_nodes - 1
        cdef bint is_leaf
        cdef SIZE_t max_depth_seen = -1
        cdef int rc = 0
        cdef Node* node

        # Initial capacity
        cdef SIZE_t init_capacity = max_split_nodes + max_leaf_nodes
        tree._resize(init_capacity)

        with nogil:
            # add root to frontier
            rc = self._add_split_node(splitter, tree, 0, n_node_samples,
                                      INFINITY, IS_FIRST, IS_LEFT, NULL, 0,
                                      &split_node_left)
            if rc >= 0:
                _add_to_frontier(split_node_left, frontier)

            while not frontier.empty():
                pop_heap(frontier.begin(), frontier.end(), &_compare_records)
                record = frontier.back()
                frontier.pop_back()

                node = &tree.nodes[record.node_id]
                is_leaf = (record.is_leaf or max_split_nodes <= 0)

                if is_leaf:
                    # Node is not expandable; set node as leaf
                    node.left_child = _TREE_LEAF
                    node.right_child = _TREE_LEAF
                    node.feature = _TREE_UNDEFINED
                    node.threshold = _TREE_UNDEFINED

                else:
                    # Node is expandable

                    # Decrement number of split nodes available
                    max_split_nodes -= 1

                    # Compute left split node
                    rc = self._add_split_node(splitter, tree,
                                              record.start, record.pos,
                                              record.impurity_left,
                                              IS_NOT_FIRST, IS_LEFT, node,
                                              record.depth + 1,
                                              &split_node_left)
                    if rc == -1:
                        break

                    # tree.nodes may have changed
                    node = &tree.nodes[record.node_id]

                    # Compute right split node
                    rc = self._add_split_node(splitter, tree, record.pos,
                                              record.end,
                                              record.impurity_right,
                                              IS_NOT_FIRST, IS_NOT_LEFT, node,
                                              record.depth + 1,
                                              &split_node_right)
                    if rc == -1:
                        break

                    # Add nodes to queue
                    _add_to_frontier(split_node_left, frontier)
                    _add_to_frontier(split_node_right, frontier)

                if record.depth > max_depth_seen:
                    max_depth_seen = record.depth

            if rc >= 0:
                rc = tree._resize_c(tree.node_count)

            if rc >= 0:
                tree.max_depth = max_depth_seen

        if rc == -1:
            raise MemoryError()

    cdef inline int _add_split_node(
        self,
        UnsupervisedSplitter splitter,
        UnsupervisedTree tree,
        SIZE_t start,
        SIZE_t end,
        double impurity,
        bint is_first,
        bint is_left,
        Node* parent,
        SIZE_t depth,
        FrontierRecord* res
    ) except -1 nogil:
        """Adds node w/ partition ``[start, end)`` to the frontier. """
        # initialize record to keep track of split node data and a pointer to the
        # memory address containing the split node
        # Note: the pointer allows us to modularly define different split records
        cdef SplitRecord split
        cdef SplitRecord* split_ptr = <SplitRecord *>malloc(splitter.pointer_size())

        cdef SIZE_t node_id
        cdef SIZE_t n_node_samples
        cdef SIZE_t n_constant_features = 0
        cdef double min_impurity_decrease = self.min_impurity_decrease
        cdef double weighted_n_node_samples
        cdef bint is_leaf

        splitter.node_reset(start, end, &weighted_n_node_samples)

        if is_first:
            impurity = splitter.node_impurity()

        n_node_samples = end - start
        is_leaf = (depth >= self.max_depth or
                   n_node_samples < self.min_samples_split or
                   n_node_samples < 2 * self.min_samples_leaf or
                   weighted_n_node_samples < 2 * self.min_weight_leaf or
                   impurity <= EPSILON  # impurity == 0 with tolerance
                   )

        if not is_leaf:
            splitter.node_split(impurity, split_ptr, &n_constant_features, 0., 0.)

            # assign local copy of SplitRecord to assign
            # pos, improvement, and impurity scores
            split = deref(split_ptr)

            # If EPSILON=0 in the below comparison, float precision issues stop
            # splitting early, producing trees that are dissimilar to v0.18
            is_leaf = (is_leaf or split.pos >= end or
                       split.improvement + EPSILON < min_impurity_decrease)

        node_id = tree._add_node(parent - tree.nodes
                                 if parent != NULL
                                 else _TREE_UNDEFINED,
                                 is_left, is_leaf,
                                 split_ptr,
                                 impurity, n_node_samples,
                                 weighted_n_node_samples, 0)
        if node_id == INTPTR_MAX:
            return -1

        # compute values also for split nodes (might become leafs later).
        splitter.node_value(tree.value + node_id * tree.value_stride)

        res.node_id = node_id
        res.start = start
        res.end = end
        res.depth = depth
        res.impurity = impurity

        if not is_leaf:
            # is split node
            res.pos = split.pos
            res.is_leaf = 0
            res.improvement = split.improvement
            res.impurity_left = split.impurity_left
            res.impurity_right = split.impurity_right

        else:
            # is leaf => 0 improvement
            res.pos = end
            res.is_leaf = 1
            res.improvement = 0.0
            res.impurity_left = impurity
            res.impurity_right = impurity

        # free up memory containing the pointer to the split record
        free(split_ptr)
        return 0

cdef class UnsupervisedDepthFirstTreeBuilder(UnsupervisedTreeBuilder):
    """Build a decision tree in depth-first fashion."""

    def __cinit__(
        self,
        UnsupervisedSplitter splitter,
        SIZE_t min_samples_split,
        SIZE_t min_samples_leaf,
        double min_weight_leaf,
        SIZE_t max_depth,
        double min_impurity_decrease
    ):
        self.splitter = splitter
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.min_weight_leaf = min_weight_leaf
        self.max_depth = max_depth
        self.min_impurity_decrease = min_impurity_decrease

    cpdef build(
        self,
        UnsupervisedTree tree,
        object X,
        const DOUBLE_t[:] sample_weight=None
    ):
        """Build a decision tree from the training set (X, y)."""

        # check input
        X, sample_weight = self._check_input(X, sample_weight)

        # Initial capacity
        cdef int init_capacity

        if tree.max_depth <= 10:
            init_capacity = <int>(2 ** (tree.max_depth + 1)) - 1
        else:
            init_capacity = 2047

        tree._resize(init_capacity)

        # Parameters
        cdef UnsupervisedSplitter splitter = self.splitter
        cdef SIZE_t max_depth = self.max_depth
        cdef SIZE_t min_samples_leaf = self.min_samples_leaf
        cdef double min_weight_leaf = self.min_weight_leaf
        cdef SIZE_t min_samples_split = self.min_samples_split
        cdef double min_impurity_decrease = self.min_impurity_decrease

        # Recursive partition (without actual recursion)
        splitter.init(X, sample_weight)

        cdef SIZE_t start
        cdef SIZE_t end
        cdef SIZE_t depth
        cdef SIZE_t parent
        cdef bint is_left
        cdef SIZE_t n_node_samples = splitter.n_samples
        cdef double weighted_n_node_samples
        cdef SIZE_t node_id

        # initialize record to keep track of split node data and a pointer to the
        # memory address containing the split node
        # Note: the pointer allows us to modularly define different split records
        cdef SplitRecord split
        cdef SplitRecord* split_ptr = <SplitRecord *>malloc(splitter.pointer_size())

        cdef double impurity = INFINITY
        cdef SIZE_t n_constant_features
        cdef bint is_leaf
        cdef bint first = 1
        cdef SIZE_t max_depth_seen = -1
        cdef int rc = 0

        cdef stack[StackRecord] builder_stack
        cdef StackRecord stack_record

        with nogil:
            # push root node onto stack
            builder_stack.push({
                "start": 0,
                "end": n_node_samples,
                "depth": 0,
                "parent": _TREE_UNDEFINED,
                "is_left": 0,
                "impurity": INFINITY,
                "n_constant_features": 0})

            while not builder_stack.empty():
                stack_record = builder_stack.top()
                builder_stack.pop()

                start = stack_record.start
                end = stack_record.end
                depth = stack_record.depth
                parent = stack_record.parent
                is_left = stack_record.is_left
                impurity = stack_record.impurity
                n_constant_features = stack_record.n_constant_features

                n_node_samples = end - start
                splitter.node_reset(start, end, &weighted_n_node_samples)

                is_leaf = (depth >= max_depth or
                           n_node_samples < min_samples_split or
                           n_node_samples < 2 * min_samples_leaf or
                           weighted_n_node_samples < 2 * min_weight_leaf)

                if first:
                    impurity = 1.0e8
                    # impurity = splitter.node_impurity()
                    first = 0

                # impurity == 0 with tolerance due to rounding errors
                is_leaf = is_leaf or impurity <= EPSILON

                if not is_leaf:
                    splitter.node_split(impurity, split_ptr, &n_constant_features, 0., 0.)

                    # assign local copy of SplitRecord to assign
                    # pos, improvement, and impurity scores
                    split = deref(split_ptr)

                    # If EPSILON=0 in the below comparison, float precision
                    # issues stop splitting, producing trees that are
                    # dissimilar to v0.18
                    is_leaf = (is_leaf or split.pos >= end or
                               (split.improvement + EPSILON <
                                min_impurity_decrease))

                node_id = tree._add_node(parent, is_left, is_leaf, split_ptr,
                                         impurity, n_node_samples,
                                         weighted_n_node_samples, 0)
                if node_id == INTPTR_MAX:
                    rc = -1
                    break

                # Store value for all nodes, to facilitate tree/model
                # inspection and interpretation
                splitter.node_value(tree.value + node_id * tree.value_stride)

                if not is_leaf:
                    # Push right child on stack
                    builder_stack.push({
                        "start": split.pos,
                        "end": end,
                        "depth": depth + 1,
                        "parent": node_id,
                        "is_left": 0,
                        "impurity": split.impurity_right,
                        "n_constant_features": n_constant_features})

                    # Push left child on stack
                    builder_stack.push({
                        "start": start,
                        "end": split.pos,
                        "depth": depth + 1,
                        "parent": node_id,
                        "is_left": 1,
                        "impurity": split.impurity_left,
                        "n_constant_features": n_constant_features})

                if depth > max_depth_seen:
                    max_depth_seen = depth

            if rc >= 0:
                rc = tree._resize_c(tree.node_count)

            if rc >= 0:
                tree.max_depth = max_depth_seen

        # free the memory created for the SplitRecord pointer
        free(split_ptr)
        if rc == -1:
            raise MemoryError()

# =============================================================================
# Unsupervised Tree
# =============================================================================

cdef class UnsupervisedTree(BaseTree):
    """Array-based representation of a binary decision tree for unsupervised learning.

    This is essentially an exact copy of the corresponding Tree class in
    scikit-learn with the exception that there is no ``y``. Therefore,
    any reference to ``n_classes``, ``n_outputs`` are removed. There is
    no ``predict`` function.

    The binary tree is represented as a number of parallel arrays. The i-th
    element of each array holds information about the node `i`. Node 0 is the
    tree's root. You can find a detailed description of all arrays in
    `_tree.pxd`. NOTE: Some of the arrays only apply to either leaves or split
    nodes, resp. In this case the values of nodes of the other type are
    arbitrary!

    Attributes
    ----------
    node_count : int
        The number of nodes (internal nodes + leaves) in the tree.

    capacity : int
        The current capacity (i.e., size) of the arrays, which is at least as
        great as `node_count`.

    max_depth : int
        The depth of the tree, i.e. the maximum depth of its leaves.

    children_left : array of int, shape [node_count]
        children_left[i] holds the node id of the left child of node i.
        For leaves, children_left[i] == TREE_LEAF. Otherwise,
        children_left[i] > i. This child handles the case where
        X[:, feature[i]] <= threshold[i].

    children_right : array of int, shape [node_count]
        children_right[i] holds the node id of the right child of node i.
        For leaves, children_right[i] == TREE_LEAF. Otherwise,
        children_right[i] > i. This child handles the case where
        X[:, feature[i]] > threshold[i].

    feature : array of int, shape [node_count]
        feature[i] holds the feature to split on, for the internal node i.

    threshold : array of double, shape [node_count]
        threshold[i] holds the threshold for the internal node i.

    value : array of double, shape [node_count]
        Contains the constant prediction value of each node.

    impurity : array of double, shape [node_count]
        impurity[i] holds the impurity (i.e., the value of the splitting
        criterion) at node i.

    n_node_samples : array of int, shape [node_count]
        n_node_samples[i] holds the number of training samples reaching node i.

    weighted_n_node_samples : array of double, shape [node_count]
        weighted_n_node_samples[i] holds the weighted number of training samples
        reaching node i.
    """
    # Wrap for outside world.
    # WARNING: these reference the current `nodes` and `value` buffers, which
    # must not be freed by a subsequent memory allocation.
    # (i.e. through `_resize` or `__setstate__`)
    @property
    def children_left(self):
        return self._get_node_ndarray()['left_child'][:self.node_count]

    @property
    def children_right(self):
        return self._get_node_ndarray()['right_child'][:self.node_count]

    def n_leaves(self):
        return np.sum(np.logical_and(
            self.children_left == -1,
            self.children_right == -1))

    @property
    def feature(self):
        return self._get_node_ndarray()['feature'][:self.node_count]

    @property
    def threshold(self):
        return self._get_node_ndarray()['threshold'][:self.node_count]

    @property
    def impurity(self):
        return self._get_node_ndarray()['impurity'][:self.node_count]

    @property
    def n_node_samples(self):
        return self._get_node_ndarray()['n_node_samples'][:self.node_count]

    @property
    def weighted_n_node_samples(self):
        return self._get_node_ndarray()['weighted_n_node_samples'][:self.node_count]

    @property
    def value(self):
        return self._get_value_ndarray()[:self.node_count]

    # @property
    # def n_categories(self):
    #     return int32_ptr_to_ndarray(self.n_categories, self.n_features)

    def __cinit__(self, int n_features):
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

        # self.n_categories = NULL
        # cdef SIZE_t k
        # safe_realloc(&self.n_categories, n_features)
        # if n_categories is not None:
        #     for k in range(n_features):
        #         self.n_categories[k] = n_categories[k]

    def __dealloc__(self):
        """Destructor."""
        # Free all inner structures
        free(self.value)
        free(self.nodes)
        # free(self.n_categories)

    def __reduce__(self):
        """Reduce re-implementation, for pickling."""
        return (UnsupervisedTree, (self.n_features,), self.__getstate__())

    def __getstate__(self):
        """Getstate re-implementation, for pickling."""
        d = {}
        # capacity is inferred during the __setstate__ using nodes
        d["max_depth"] = self.max_depth
        d["node_count"] = self.node_count
        d["nodes"] = self._get_node_ndarray()
        d["values"] = self._get_value_ndarray()
        return d

    def __setstate__(self, d):
        """Setstate re-implementation, for unpickling."""
        self.max_depth = d["max_depth"]
        self.node_count = d["node_count"]

        if 'nodes' not in d:
            raise ValueError('You have loaded Tree version which '
                             'cannot be imported')

        node_ndarray = d['nodes']
        value_ndarray = d['values']

        value_shape = (node_ndarray.shape[0],)

        node_ndarray = _check_node_ndarray(node_ndarray, expected_dtype=NODE_DTYPE)
        value_ndarray = _check_value_ndarray(
            value_ndarray,
            expected_dtype=np.dtype(np.float64),
            expected_shape=value_shape
        )

        self.capacity = node_ndarray.shape[0]
        if self._resize_c(self.capacity) != 0:
            raise MemoryError("resizing tree to %d" % self.capacity)

        memcpy(self.nodes, cnp.PyArray_DATA(node_ndarray),
               self.capacity * sizeof(Node))
        memcpy(self.value, cnp.PyArray_DATA(value_ndarray),
               self.capacity * self.value_stride * sizeof(double))

    cdef cnp.ndarray _get_value_ndarray(self):
        """Wraps value as a 3-d NumPy array.

        The array keeps a reference to this Tree, which manages the underlying
        memory.
        """
        cdef cnp.npy_intp shape[1]
        shape[0] = <cnp.npy_intp> self.node_count
        cdef cnp.ndarray arr
        arr = cnp.PyArray_SimpleNewFromData(1, shape, cnp.NPY_DOUBLE, self.value)
        Py_INCREF(self)
        if PyArray_SetBaseObject(arr, <PyObject*> self) < 0:
            raise ValueError("Can't initialize array.")
        return arr

    cdef cnp.ndarray _get_node_ndarray(self):
        """Wraps nodes as a NumPy struct array.

        The array keeps a reference to this Tree, which manages the underlying
        memory. Individual fields are publicly accessible as properties of the
        Tree.
        """
        cdef cnp.npy_intp shape[1]
        shape[0] = <cnp.npy_intp> self.node_count
        cdef cnp.npy_intp strides[1]
        strides[0] = sizeof(Node)
        cdef cnp.ndarray arr
        Py_INCREF(NODE_DTYPE)
        arr = PyArray_NewFromDescr(<PyTypeObject *> cnp.ndarray,
                                   <cnp.dtype> NODE_DTYPE, 1, shape,
                                   strides, <void*> self.nodes,
                                   cnp.NPY_ARRAY_DEFAULT, None)
        Py_INCREF(self)
        if PyArray_SetBaseObject(arr, <PyObject*> self) < 0:
            raise ValueError("Can't initialize array.")
        return arr


def _check_value_ndarray(value_ndarray, expected_dtype, expected_shape):
    if value_ndarray.shape != expected_shape:
        raise ValueError(
            "Wrong shape for value array from the pickle: "
            f"expected {expected_shape}, got {value_ndarray.shape}"
        )

    if not value_ndarray.flags.c_contiguous:
        raise ValueError(
            "value array from the pickle should be a C-contiguous array"
        )

    if value_ndarray.dtype == expected_dtype:
        return value_ndarray

    # Handles different endianness
    if value_ndarray.dtype.str.endswith('f8'):
        return value_ndarray.astype(expected_dtype, casting='equiv')

    raise ValueError(
        "value array from the pickle has an incompatible dtype:\n"
        f"- expected: {expected_dtype}\n"
        f"- got:      {value_ndarray.dtype}"
    )


def _dtype_to_dict(dtype):
    return {name: dt.str for name, (dt, *rest) in dtype.fields.items()}


def _dtype_dict_with_modified_bitness(dtype_dict):
    # field names in Node struct with SIZE_t types (see sklearn/tree/_tree.pxd)
    indexing_field_names = ["left_child", "right_child", "feature", "n_node_samples"]

    expected_dtype_size = str(struct.calcsize("P"))
    allowed_dtype_size = "8" if expected_dtype_size == "4" else "4"

    allowed_dtype_dict = dtype_dict.copy()
    for name in indexing_field_names:
        allowed_dtype_dict[name] = allowed_dtype_dict[name].replace(
            expected_dtype_size, allowed_dtype_size
        )

    return allowed_dtype_dict


def _all_compatible_dtype_dicts(dtype):
    # The Cython code for decision trees uses platform-specific SIZE_t
    # typed indexing fields that correspond to either i4 or i8 dtypes for
    # the matching fields in the numpy array depending on the bitness of
    # the platform (32 bit or 64 bit respectively).
    #
    # We need to cast the indexing fields of the NODE_DTYPE-dtyped array at
    # pickle load time to enable cross-bitness deployment scenarios. We
    # typically want to make it possible to run the expensive fit method of
    # a tree estimator on a 64 bit server platform, pickle the estimator
    # for deployment and run the predict method of a low power 32 bit edge
    # platform.
    #
    # A similar thing happens for endianness, the machine where the pickle was
    # saved can have a different endianness than the machine where the pickle
    # is loaded

    dtype_dict = _dtype_to_dict(dtype)
    dtype_dict_with_modified_bitness = _dtype_dict_with_modified_bitness(dtype_dict)
    dtype_dict_with_modified_endianness = _dtype_to_dict(dtype.newbyteorder())
    dtype_dict_with_modified_bitness_and_endianness = _dtype_dict_with_modified_bitness(
        dtype_dict_with_modified_endianness
    )

    return [
        dtype_dict,
        dtype_dict_with_modified_bitness,
        dtype_dict_with_modified_endianness,
        dtype_dict_with_modified_bitness_and_endianness,
    ]


def _check_node_ndarray(node_ndarray, expected_dtype):
    if node_ndarray.ndim != 1:
        raise ValueError(
            "Wrong dimensions for node array from the pickle: "
            f"expected 1, got {node_ndarray.ndim}"
        )

    if not node_ndarray.flags.c_contiguous:
        raise ValueError(
            "node array from the pickle should be a C-contiguous array"
        )

    node_ndarray_dtype = node_ndarray.dtype
    if node_ndarray_dtype == expected_dtype:
        return node_ndarray

    node_ndarray_dtype_dict = _dtype_to_dict(node_ndarray_dtype)
    all_compatible_dtype_dicts = _all_compatible_dtype_dicts(expected_dtype)

    if node_ndarray_dtype_dict not in all_compatible_dtype_dicts:
        raise ValueError(
            "node array from the pickle has an incompatible dtype:\n"
            f"- expected: {expected_dtype}\n"
            f"- got     : {node_ndarray_dtype}"
        )

    return node_ndarray.astype(expected_dtype, casting="same_kind")
