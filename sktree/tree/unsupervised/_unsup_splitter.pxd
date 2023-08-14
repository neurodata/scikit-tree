from ..._lib.sklearn.tree._splitter cimport BaseSplitter, SplitRecord
from ..._lib.sklearn.tree._utils cimport DOUBLE_t  # Type of y, sample_weight
from ..._lib.sklearn.tree._utils cimport DTYPE_t  # Type of X
from ..._lib.sklearn.tree._utils cimport INT32_t  # Signed 32 bit integer
from ..._lib.sklearn.tree._utils cimport SIZE_t  # Type for indices and counters
from ..._lib.sklearn.tree._utils cimport UINT32_t  # Unsigned 32 bit integer
from ._unsup_criterion cimport UnsupervisedCriterion


cdef class UnsupervisedSplitter(BaseSplitter):
    """
    Notable changes wrt scikit-learn:
    1. `weighted_n_node_samples` is used as a stopping criterion and just used to
    keep count of the "number of samples (weighted)". All samples have a default weight
    of '1'.
    2. `X` array instead of `y` array is stored as the criterions are computed over the X
    array.
    """

    # XXX: requires BaseSplitter to not define "criterion"
    cdef public UnsupervisedCriterion criterion         # criterion computer
    cdef const DTYPE_t[:, :] X                                       # feature matrix
    cdef SIZE_t n_total_samples                         # store the total number of samples

    # Initialization method for unsupervised splitters
    cdef int init(
        self,
        const DTYPE_t[:, :] X,
        const DOUBLE_t[:] sample_weight
    ) except -1

    # Overridden Methods from base class
    cdef int node_reset(
        self,
        SIZE_t start,
        SIZE_t end,
        double* weighted_n_node_samples
    ) except -1 nogil
    cdef int node_split(
        self,
        double impurity,   # Impurity of the node
        SplitRecord* split,
        SIZE_t* n_constant_features,
        double lower_bound,
        double upper_bound,
    ) except -1 nogil
    cdef void node_value(self, double* dest) noexcept nogil
    cdef double node_impurity(self) noexcept nogil
