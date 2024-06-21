import numpy as np

cimport numpy as cnp

cnp.import_array()

from libc.math cimport isnan
from libc.stdlib cimport free, malloc
from libcpp.stack cimport stack

TREE_LEAF = -1
TREE_UNDEFINED = -2
cdef intp_t _TREE_LEAF = TREE_LEAF
cdef intp_t _TREE_UNDEFINED = TREE_UNDEFINED


def _build_pruned_tree_honesty(
    Tree tree,
    Tree orig_tree,
    HonestPruner pruner,
    object X,
    const float64_t[:, ::1] y,
    const float64_t[:] sample_weight,
    const unsigned char[::1] missing_values_in_feature_mask=None,
):
    cdef:
        intp_t n_nodes = orig_tree.node_count
        unsigned char[:] leaves_in_subtree = np.zeros(
            shape=n_nodes, dtype=np.uint8)

    # initialize the pruner/splitter
    pruner.init(X, y, sample_weight, missing_values_in_feature_mask)

    # apply pruning to the tree
    _honest_prune(leaves_in_subtree, orig_tree, pruner)

    _build_pruned_tree(tree, orig_tree, leaves_in_subtree,
                       pruner.capacity)


cdef class HonestPruner(Splitter):
    """Pruning to enforce honest splits are non-degenerate."""

    def __cinit__(
        self,
        Criterion criterion,
        intp_t max_features,
        intp_t min_samples_leaf,
        float64_t min_weight_leaf,
        object random_state,
        const int8_t[:] monotonic_cst,
        Tree orig_tree,
        *argv
    ):
        """
        Parameters
        ----------
        criterion : Criterion
            The criterion to measure the quality of a split.

        max_features : intp_t
            The maximal number of randomly selected features which can be
            considered for a split.

        min_samples_leaf : intp_t
            The minimal number of samples each leaf can have, where splits
            which would result in having less samples in a leaf are not
            considered.

        min_weight_leaf : float64_t
            The minimal weight each leaf can have, where the weight is the sum
            of the weights of each sample in it.

        random_state : object
            The user inputted random state to be used for pseudo-randomness

        monotonic_cst : const int8_t[:]
            Monotonicity constraints

        orig_tree : Tree
            The original tree to be pruned.
        """
        self.tree = orig_tree
        self.capacity = 0

    cdef int init(
        self,
        object X,
        const float64_t[:, ::1] y,
        const float64_t[:] sample_weight,
        const unsigned char[::1] missing_values_in_feature_mask,
    ) except -1:
        Splitter.init(self, X, y, sample_weight, missing_values_in_feature_mask)
        self.X = X

    cdef int partition_samples(
        self,
        intp_t node_idx,
    ) noexcept nogil:
        """Partition samples for X at the threshold and feature index.

        If missing values are present, this method partitions `samples`
        so that the `best_n_missing` missing values' indices are in the
        right-most end of `samples`, that is `samples[end_non_missing:end]`.
        """
        cdef float64_t threshold = self.tree.nodes[node_idx].feature
        cdef intp_t feature = self.tree.nodes[node_idx].feature
        cdef intp_t n_missing = 0
        cdef intp_t pos = self.start
        cdef intp_t p
        cdef intp_t sample_idx
        cdef intp_t current_end = self.end - n_missing
        cdef const float32_t[:, :] X_ndarray = self.X

        # TODO
        # partition the samples one by one
        for p in range(self.start, self.end):
            sample_idx = self.samples[p]

            if isnan(X_ndarray[sample_idx, feature]):
                self.samples[p], self.samples[current_end] = \
                    self.samples[current_end], self.samples[p]

                n_missing += 1
                current_end -= 1
            elif p > pos and X_ndarray[sample_idx, feature] <= threshold:
                self.samples[p], self.samples[pos] = \
                    self.samples[pos], self.samples[p]
                pos += 1

        # this is the split point for left/right children
        self.pos = pos
        self.n_missing = n_missing
        self.missing_go_to_left = self.tree.nodes[node_idx].missing_go_to_left

    cdef bint check_node_partition_conditions(
        self,
        SplitRecord* current_split,
    ) noexcept nogil:
        """Check that the current node satisfies paritioning conditions.

        Parameters
        ----------
        current_split : SplitRecord pointer
            A pointer to a memory-allocated SplitRecord object which will be filled with the
            split chosen.
        """
        # update the criterion if we are checking split conditions
        self.criterion.init_missing(self.n_missing)
        self.criterion.reset()
        self.criterion.update(self.pos)

        current_split.pos = self.pos
        current_split.n_missing = self.n_missing
        current_split.missing_go_to_left = self.missing_go_to_left

        shift_missing_values_to_left_if_required(
            current_split,
            self.samples,
            self.end
        )

        # first check the presplit conditions
        cdef bint valid_split = self.check_presplit_conditions(
            current_split,
            self.n_missing,
            self.missing_go_to_left
        )

        if not valid_split:
            return 0

        # TODO: make work with lower/upper bound
        # Reject if monotonicity constraints are not satisfied
        # if (
        #     self.with_monotonic_cst and
        #     self.monotonic_cst[current_split.feature] != 0 and
        #     not self.criterion.check_monotonicity(
        #         self.monotonic_cst[current_split.feature],
        #         lower_bound,
        #         upper_bound,
        #     )
        # ):
        #     return 0

        # next check the postsplit conditions that leverages the criterion
        valid_split = self.check_postsplit_conditions()
        return valid_split

    cdef inline intp_t n_left_samples(
        self
    ) noexcept nogil:
        cdef intp_t n_left

        if self.missing_go_to_left:
            n_left = self.pos - self.start + self.n_missing
        else:
            n_left = self.pos - self.start
        return n_left

    cdef inline intp_t n_right_samples(
        self
    ) noexcept nogil:
        cdef intp_t n_right
        cdef intp_t end_non_missing = self.end - self.n_missing
        if self.missing_go_to_left:
            n_right = end_non_missing - self.pos
        else:
            n_right = end_non_missing - self.pos + self.n_missing
        return n_right

    cdef int node_split(
        self,
        ParentInfo* parent_record,
        SplitRecord* split,
    ) except -1 nogil:
        """Split nodes using the already constructed tree.

        Returns 0 if a split cannot be done, 1 if a split can be done
        and -1 in case of failure to allocate memory (and raise MemoryError).
        """
        pass


cdef _honest_prune(
    unsigned char[:] leaves_in_subtree,
    Tree orig_tree,
    HonestPruner pruner,
):
    """Perform honest pruning of the tree.

    Iterates through the original tree in a BFS fashion using the pruner
    and tracks at each node:

    - the parent node id
    - the number of samples in the parent node
    - the number of samples in the node

    Until one of three stopping conditions are met:

    1. The orig_node is a leaf node.
    2. The orig_node is a non-leaf node and the split is degenerate.
    3. Stopping criterion is met based on the samples that reach the node.
        These are the stopping conditions implemented in a Splitter/Pruner.

    Parameters
    ----------
    leaves_in_subtree : array of shape (n_nodes,), dtype=np.uint8
        Array of booleans indicating whether the node is in the subtree.
    orig_tree : Tree
        The original tree.
    pruner : HonestPruner
        The input samples to be used for computing the split of samples
        in the nodes.
    """
    cdef:
        intp_t n_nodes = orig_tree.node_count

        # get the left child, right child and parents of every node
        intp_t[:] child_l = orig_tree.children_left
        intp_t[:] child_r = orig_tree.children_right
        intp_t[:] parents = np.zeros(shape=n_nodes, dtype=np.intp)

        # stack to keep track of the nodes to be pruned such that BFS is done
        stack[PruningRecord] pruning_stack
        PruningRecord stack_record

        intp_t node_idx
        SplitRecord* split_ptr = <SplitRecord *>malloc(pruner.pointer_size())

        bint is_leaf_in_origtree
        bint valid_split
        bint split_is_degenerate

        intp_t start = 0
        intp_t end = 0
        float64_t weighted_n_node_samples

    # find parent node ids and leaves
    with nogil:
        # Push the root node
        pruning_stack.push({
            "node_idx": 0,
            "parent": _TREE_UNDEFINED,
            "start": 0,
            "end": pruner.n_samples
        })

        while not pruning_stack.empty():
            stack_record = pruning_stack.top()
            pruning_stack.pop()
            start = stack_record.start
            end = stack_record.end
            # node index and parent pointer to actual node within the orig_tree
            parents[node_idx] = stack_record.parent
            node_idx = stack_record.node_idx

            # reset what is the samples considered at this split node
            pruner.node_reset(start, end, &weighted_n_node_samples)

            # first partition samples into left/right child
            pruner.partition_samples(node_idx)
            # check end conditions
            valid_split = pruner.check_node_partition_conditions(
                split_ptr
            )
            split_is_degenerate = (
                pruner.n_left_samples() == 0 or pruner.n_right_samples() == 0
            )
            is_leaf_in_origtree = child_l[node_idx] == _TREE_LEAF

            if not valid_split or split_is_degenerate or is_leaf_in_origtree:
                # ... and child_r[node_idx] == _TREE_LEAF:
                #
                # 1) if node is not degenerate, that means there are still honest-samples in
                # both left/right children of the proposed split, or the node itself is a leaf
                # or 2) there are still nodes to split on, but the honest-samples have been
                # used up so the "parent" should be the new leaf
                leaves_in_subtree[node_idx] = 1
            else:
                pruning_stack.push({
                    "node_idx": child_l[node_idx],
                    "parent": node_idx,
                    "start": pruner.start,
                    "end": pruner.pos,
                })
                pruning_stack.push({
                    "node_idx": child_r[node_idx],
                    "parent": node_idx,
                    "start": pruner.pos,
                    "end": pruner.end,
                })

    # free the memory created for the SplitRecord pointer
    free(split_ptr)


from libc.stdint cimport INTPTR_MAX
from libc.string cimport memcpy


cdef struct BuildPrunedRecord:
    intp_t start
    intp_t depth
    intp_t parent
    bint is_left


cdef _build_pruned_tree(
    Tree tree,  # OUT
    Tree orig_tree,
    const unsigned char[:] leaves_in_subtree,
    intp_t capacity
):
    """Build a pruned tree.

    Build a pruned tree from the original tree by transforming the nodes in
    ``leaves_in_subtree`` into leaves.

    Parameters
    ----------
    tree : Tree
        Location to place the pruned tree
    orig_tree : Tree
        Original tree
    leaves_in_subtree : unsigned char memoryview, shape=(node_count, )
        Boolean mask for leaves to include in subtree
    capacity : intp_t
        Number of nodes to initially allocate in pruned tree
    """
    tree._resize(capacity)

    cdef:
        intp_t orig_node_id
        intp_t new_node_id
        intp_t depth
        intp_t parent
        bint is_left
        bint is_leaf

        # value_stride for original tree and new tree are the same
        intp_t value_stride = orig_tree.value_stride
        intp_t max_depth_seen = -1
        intp_t rc = 0
        Node* node
        float64_t* orig_value_ptr
        float64_t* new_value_ptr

        stack[BuildPrunedRecord] prune_stack
        BuildPrunedRecord stack_record

        SplitRecord split

    with nogil:
        # push root node onto stack
        prune_stack.push({"start": 0, "depth": 0, "parent": _TREE_UNDEFINED, "is_left": 0})

        while not prune_stack.empty():
            stack_record = prune_stack.top()
            prune_stack.pop()

            orig_node_id = stack_record.start
            depth = stack_record.depth
            parent = stack_record.parent
            is_left = stack_record.is_left

            is_leaf = leaves_in_subtree[orig_node_id]
            node = &orig_tree.nodes[orig_node_id]

            # redefine to a SplitRecord to pass into _add_node
            split.feature = node.feature
            split.threshold = node.threshold

            new_node_id = tree._add_node(
                parent, is_left, is_leaf, &split,
                node.impurity, node.n_node_samples,
                node.weighted_n_node_samples, node.missing_go_to_left)

            if new_node_id == INTPTR_MAX:
                rc = -1
                break

            # copy value from original tree to new tree
            orig_value_ptr = orig_tree.value + value_stride * orig_node_id
            new_value_ptr = tree.value + value_stride * new_node_id
            memcpy(new_value_ptr, orig_value_ptr, sizeof(float64_t) * value_stride)

            if not is_leaf:
                # Push right child on stack
                prune_stack.push({"start": node.right_child, "depth": depth + 1,
                                  "parent": new_node_id, "is_left": 0})
                # push left child on stack
                prune_stack.push({"start": node.left_child, "depth": depth + 1,
                                  "parent": new_node_id, "is_left": 1})

            if depth > max_depth_seen:
                max_depth_seen = depth

        if rc >= 0:
            tree.max_depth = max_depth_seen
    if rc == -1:
        raise MemoryError("pruning tree")
