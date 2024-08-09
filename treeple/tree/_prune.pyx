# cython: boundscheck=False
# cython: wraparound=False
# cython: initializedcheck=False
from libcpp.stack cimport stack

from .._lib.sklearn.tree._splitter cimport SplitRecord

TREE_UNDEFINED = -2
cdef intp_t _TREE_UNDEFINED = TREE_UNDEFINED

TREE_LEAF = -1
cdef intp_t _TREE_LEAF = TREE_LEAF

cdef void _build_pruned_tree(
    Tree tree,  # OUT
    Tree orig_tree,
    const uint8_t[:] leaves_in_subtree,
    intp_t capacity
) noexcept:
    """Build a pruned tree.

    Build a pruned tree from the original tree by transforming the nodes in
    ``leaves_in_subtree`` into leaves.

    Parameters
    ----------
    tree : Tree
        Location to place the pruned tree
    orig_tree : Tree
        Original tree
    leaves_in_subtree : uint8_t memoryview, shape=(node_count, )
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

            # protect against an infinite loop as a runtime error, when leaves_in_subtree
            # are improperly set where a node is not marked as a leaf, but is a node
            # in the original tree. Thus, it violates the assumption that the node
            # is a leaf in the pruned tree, or has a descendant that will be pruned.
            if (not is_leaf and node.left_child == _TREE_LEAF
                    and node.right_child == _TREE_LEAF):
                raise ValueError(
                    "Node has reached a leaf in the original tree, but is not "
                    "marked as a leaf in the leaves_in_subtree mask."
                )

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
