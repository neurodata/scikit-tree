# distutils: language = c++

# Authors: Adam Li <adam2392@gmail.com>
#          Chester Huynh <chester.huynh924@gmail.com>
#          Parth Vora <pvora4@jhu.edu>
#
# License: BSD 3 clause

# See _oblique_splitter.pyx for details.

import numpy as np

from libcpp.vector cimport vector

from ..._lib.sklearn.tree._splitter cimport SplitRecord
from ..._lib.sklearn.utils._typedefs cimport float32_t, float64_t, int8_t, intp_t, uint8_t, uint32_t
from .._oblique_splitter cimport BestObliqueSplitter, ObliqueSplitRecord

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

cdef class PatchSplitter(BestObliqueSplitter):
    # The PatchSplitter creates candidate feature values by sampling 2D patches from
    # an input data vector. The input data is vectorized, so `data_height` and
    # `data_width` are used to determine the vectorized indices corresponding to
    # (x,y) coordinates in the original un-vectorized data.

    cdef public intp_t max_patch_height                 # Maximum height of the patch to sample
    cdef public intp_t max_patch_width                  # Maximum width of the patch to sample
    cdef public intp_t min_patch_height                 # Minimum height of the patch to sample
    cdef public intp_t min_patch_width                  # Minimum width of the patch to sample
    cdef public intp_t data_height                      # Height of the input data
    cdef public intp_t data_width                       # Width of the input data

    cdef public intp_t ndim                       # The number of dimensions of the input data

    cdef const intp_t[:] data_dims                      # The dimensions of the input data
    cdef const intp_t[:] min_patch_dims                 # The minimum size of the patch to sample in each dimension
    cdef const intp_t[:] max_patch_dims                 # The maximum size of the patch to sample in each dimension
    cdef const uint8_t[:] dim_contiguous            # A boolean array indicating whether each dimension is contiguous

    # TODO: check if this works and is necessary for discontiguous data
    # cdef intp_t[:] stride_offsets                # The stride offsets for each dimension
    cdef bint _discontiguous

    cdef bytes boundary                               # how to sample the patch with boundary in mind
    cdef const float32_t[:, :] feature_weight               # Whether or not to normalize each column of X when adding in a patch

    cdef intp_t[::1] _index_data_buffer
    cdef intp_t[::1] _index_patch_buffer
    cdef intp_t[:] patch_dims_buff                # A buffer to store the dimensions of the sampled patch
    cdef intp_t[:] unraveled_patch_point          # A buffer to store the unraveled patch point

    # All oblique splitters (i.e. non-axis aligned splitters) require a
    # function to sample a projection matrix that is applied to the feature matrix
    # to quickly obtain the sampled projections for candidate splits.
    cdef (intp_t, intp_t) sample_top_left_seed(
        self
    ) noexcept nogil

    cdef void sample_proj_mat(
        self,
        vector[vector[float32_t]]& proj_mat_weights,
        vector[vector[intp_t]]& proj_mat_indices
    ) noexcept nogil


ctypedef float32_t* float32_t_ptr
ctypedef intp_t* inpt_t_ptr

cdef class UserKernelSplitter(PatchSplitter):
    """A class to hold user-specified kernels."""
    # cdef vector[float32_t[:, ::1]] kernel_dictionary  # A list of C-contiguous 2D kernels
    cdef vector[float32_t*] kernel_dictionary  # A list of C-contiguous 2D kernels
    cdef vector[intp_t*] kernel_dims         # A list of arrays storing the dimensions of each kernel in `kernel_dictionary`


cdef class GaussianKernelSplitter(PatchSplitter):
    """A class to hold Gaussian kernels.

    Overrides the weights that are generated to be sampled from a Gaussian distribution.
    See: https://www.tutorialspoint.com/gaussian-filter-generation-in-cplusplus
    See: https://gist.github.com/thomasaarholt/267ec4fff40ca9dff1106490ea3b7567
    """

    cdef void sample_proj_mat(
        self,
        vector[vector[float32_t]]& proj_mat_weights,
        vector[vector[intp_t]]& proj_mat_indices
    ) noexcept nogil
