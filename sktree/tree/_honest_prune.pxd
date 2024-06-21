# from .._lib.sklearn.tree._tree import _build_pruned_tree

from .._lib.sklearn.tree._criterion cimport Criterion
from .._lib.sklearn.tree._splitter cimport (
    SplitRecord,
    Splitter,
    shift_missing_values_to_left_if_required,
)
from .._lib.sklearn.tree._tree cimport Node, ParentInfo, Tree
from .._lib.sklearn.utils._typedefs cimport float32_t, float64_t, int8_t, intp_t, uint32_t


# for each node, keep track of the node index and the parent index
# within the tree's node array
cdef struct PruningRecord:
    intp_t node_idx
    intp_t parent
    intp_t start
    intp_t end

cdef class HonestPruner(Splitter):
    cdef Tree tree          # The tree to be pruned
    cdef intp_t capacity    # The maximum number of nodes in the pruned tree
    cdef intp_t pos         # The current position to split left/right children
    cdef intp_t n_missing   # The number of missing values in the feature currently considered
    cdef unsigned char missing_go_to_left
    cdef const float32_t[:, :] X

    cdef int init(
        self,
        object X,
        const float64_t[:, ::1] y,
        const float64_t[:] sample_weight,
        const unsigned char[::1] missing_values_in_feature_mask,
    ) except -1

    cdef int node_split(
        self,
        ParentInfo* parent_record,
        SplitRecord* split,
    ) except -1 nogil

    cdef bint check_node_partition_conditions(
        self,
        SplitRecord* current_split,
    ) noexcept nogil

    cdef inline intp_t n_left_samples(
        self
    ) noexcept nogil
    cdef inline intp_t n_right_samples(
        self
    ) noexcept nogil

    cdef int partition_samples(
        self,
        intp_t node_idx,
    ) noexcept nogil
