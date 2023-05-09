# distutils: language = c++

# Authors: Adam Li <adam2392@gmail.com>
#
# License: BSD 3 clause

# See _oblique_splitter.pyx for details.

import numpy as np

cimport numpy as cnp
from libcpp.vector cimport vector
from sklearn.utils._sorting cimport simultaneous_sort
from sklearn_fork.tree._splitter cimport SplitRecord
from sklearn_fork.tree._tree cimport DOUBLE_t  # Type of y, sample_weight
from sklearn_fork.tree._tree cimport DTYPE_t  # Type of X
from sklearn_fork.tree._tree cimport INT32_t  # Signed 32 bit integer
from sklearn_fork.tree._tree cimport SIZE_t  # Type for indices and counters
from sklearn_fork.tree._tree cimport UINT32_t  # Unsigned 32 bit integer

from .._oblique_splitter cimport BaseObliqueSplitter, ObliqueSplitRecord

# https://github.com/cython/cython/blob/master/Cython/Includes/libcpp/algorithm.pxd
# shows how to include standard library functions in Cython
# This includes the discrete_distribution C++ class, which can be used
# to generate samples from a discrete distribution with non-uniform probabilities.
# cdef extern from "<discrete_distribution>" namespace "std" nogil:
#     cdef cppclass discrete_distribution[T]
#         ctypedef T int_type
#         ctypedef G generator_type
#         discrete_distribution(T first, T last) except +
#         operator()(&G) except +

# XXX: replace with from libcpp.algorithm cimport swap
# when Cython 3.0 is released
cdef extern from "<algorithm>" namespace "std" nogil:
    void swap[T](T& a, T& b) except +  # array overload also works

ctypedef DTYPE_t* DTYPE_t_ptr
ctypedef SIZE_t* SIZE_t_ptr

cdef class PatchSplitter(BaseObliqueSplitter):
    # The PatchSplitter creates candidate feature values by sampling 2D patches from
    # an input data vector. The input data is vectorized, so `data_height` and
    # `data_width` are used to determine the vectorized indices corresponding to
    # (x,y) coordinates in the original un-vectorized data.

    cdef public SIZE_t max_patch_height                 # Maximum height of the patch to sample
    cdef public SIZE_t max_patch_width                  # Maximum width of the patch to sample
    cdef public SIZE_t min_patch_height                 # Minimum height of the patch to sample
    cdef public SIZE_t min_patch_width                  # Minimum width of the patch to sample
    cdef public SIZE_t data_height                      # Height of the input data
    cdef public SIZE_t data_width                       # Width of the input data

    cdef public SIZE_t ndim                       # The number of dimensions of the input data

    cdef SIZE_t[:] data_dims                      # The dimensions of the input data
    cdef SIZE_t[:] min_patch_dims                 # The minimum size of the patch to sample in each dimension
    cdef SIZE_t[:] max_patch_dims                 # The maximum size of the patch to sample in each dimension
    cdef cnp.uint8_t[:] dim_contiguous            # A boolean array indicating whether each dimension is contiguous

    # TODO: check if this works and is necessary for discontiguous data
    # cdef SIZE_t[:] stride_offsets                # The stride offsets for each dimension
    cdef bint _discontiguous

    cdef str boundary                            # how to sample the patch with boundary in mind
    cdef DTYPE_t[:, :] feature_weight               # Whether or not to normalize each column of X when adding in a patch

    cdef SIZE_t[::1] _index_data_buffer
    cdef SIZE_t[::1] _index_patch_buffer
    cdef SIZE_t[:] patch_dims_buff                # A buffer to store the dimensions of the sampled patch
    cdef SIZE_t[:] unraveled_patch_point          # A buffer to store the unraveled patch point

    # All oblique splitters (i.e. non-axis aligned splitters) require a
    # function to sample a projection matrix that is applied to the feature matrix
    # to quickly obtain the sampled projections for candidate splits.
    cdef (SIZE_t, SIZE_t) sample_top_left_seed(
        self
    ) noexcept nogil

    cdef void sample_proj_mat(
        self,
        vector[vector[DTYPE_t]]& proj_mat_weights,
        vector[vector[SIZE_t]]& proj_mat_indices
    ) nogil


cdef class UserKernelSplitter(PatchSplitter):
    """A class to hold user-specified kernels."""

    cdef vector[DTYPE_t*] kernel_dictionary  # A list of C-contiguous 2D kernels
    cdef vector[SIZE_t*] kernel_dims         # A list of arrays storing the dimensions of each kernel in `kernel_dictionary`
