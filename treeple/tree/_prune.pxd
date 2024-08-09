# Copied from scikit-learn/tree/_tree.pyx

from libc.stdint cimport INTPTR_MAX
from libc.string cimport memcpy

from .._lib.sklearn.tree._tree cimport Node, Tree
from .._lib.sklearn.utils._typedefs cimport float64_t, intp_t, uint8_t


cdef struct BuildPrunedRecord:
    intp_t start
    intp_t depth
    intp_t parent
    bint is_left


cdef void _build_pruned_tree(
    Tree tree,  # OUT
    Tree orig_tree,
    const uint8_t[:] leaves_in_subtree,
    intp_t capacity
) noexcept
