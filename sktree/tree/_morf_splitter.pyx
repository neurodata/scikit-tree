# cython: cdivision=True
# distutils: language=c++
# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: profile=True

import numpy as np
cimport numpy as cnp

cnp.import_array()

from libcpp.vector cimport vector
from sklearn.tree._criterion cimport Criterion
from sklearn.tree._utils cimport rand_int


cdef class PatchSplitter(BaseObliqueSplitter):
    """Patch splitter.

    A convolutional 2D patch splitter.
    """
    def __cinit__(
        self,
        Criterion criterion,
        SIZE_t max_features,
        SIZE_t min_samples_leaf,
        double min_weight_leaf,
        object random_state,
        SIZE_t min_patch_height,
        SIZE_t max_patch_height,
        SIZE_t min_patch_width,
        SIZE_t max_patch_width,
        SIZE_t data_height,
        SIZE_t data_width,
        *argv
    ):
        self.criterion = criterion

        self.n_samples = 0
        self.n_features = 0

        # Max features = output dimensionality of projection vectors
        self.max_features = max_features
        self.min_samples_leaf = min_samples_leaf
        self.min_weight_leaf = min_weight_leaf
        self.random_state = random_state

        # Sparse max_features x n_features projection matrix
        self.proj_mat_weights = vector[vector[DTYPE_t]](self.max_features)
        self.proj_mat_indices = vector[vector[SIZE_t]](self.max_features)

        # TODO: remove these when we generalize this to higher-dimensions
        # refactor to a tuple of indices that are passed in
        self.min_patch_height = min_patch_height
        self.max_patch_height = max_patch_height
        self.min_patch_width = min_patch_width
        self.max_patch_width = max_patch_width
        self.data_height = data_height
        self.data_width = data_width

        # initialize state to allow generalization to higher-dimensional tensors
        self.ndim = 2
        self.data_dims = np.zeros(self.ndim, dtype=np.intp)
        self.data_dims[1] = data_width
        self.data_dims[0] = data_height

        # create a buffer for storing the patch dimensions sampled per projection matrix
        self.patch_dims_buff = np.zeros(self.ndim, dtype=np.intp)

        # store the min and max patch dimension constraints
        self.min_patch_dims = np.zeros(self.ndim, dtype=np.intp)
        self.max_patch_dims = np.zeros(self.ndim, dtype=np.intp)
        self.min_patch_dims[1] = min_patch_width
        self.min_patch_dims[0] = min_patch_height
        self.max_patch_dims[1] = max_patch_width
        self.max_patch_dims[0] = max_patch_height

    def __getstate__(self):
        return {}

    def __setstate__(self, d):
        pass

    cdef int init(
        self,
        object X,
        const DOUBLE_t[:, ::1] y,
        const DOUBLE_t[:] sample_weight
    ) except -1:
        BaseObliqueSplitter.init(self, X, y, sample_weight)

        return 0

    cdef int node_reset(
        self,
        SIZE_t start,
        SIZE_t end,
        double* weighted_n_node_samples
    ) except -1 nogil:
        """Reset splitter on node samples[start:end].

        Returns -1 in case of failure to allocate memory (and raise MemoryError)
        or 0 otherwise.

        Parameters
        ----------
        start : SIZE_t
            The index of the first sample to consider
        end : SIZE_t
            The index of the last sample to consider
        weighted_n_node_samples : ndarray, dtype=double pointer
            The total weight of those samples
        """

        self.start = start
        self.end = end

        self.criterion.init(self.y,
                            self.sample_weight,
                            self.weighted_n_samples,
                            self.samples)
        self.criterion.set_sample_pointers(start, end)

        weighted_n_node_samples[0] = self.criterion.weighted_n_node_samples

        # Clear all projection vectors
        for i in range(self.max_features):
            self.proj_mat_weights[i].clear()
            self.proj_mat_indices[i].clear()
        return 0

    cdef void sample_proj_mat(
        self,
        vector[vector[DTYPE_t]]& proj_mat_weights,
        vector[vector[SIZE_t]]& proj_mat_indices
    ) noexcept nogil:
        """ Sample the projection vector.

        This is a placeholder method.

        """
        pass

    cdef int pointer_size(self) noexcept nogil:
        """Get size of a pointer to record for ObliqueSplitter."""

        return sizeof(ObliqueSplitRecord)


cdef class BaseDensePatchSplitter(PatchSplitter):
    # XXX: currently BaseOblique class defines this, which shouldn't be the case
    # cdef const DTYPE_t[:, :] X

    cdef int init(self,
                  object X,
                  const DOUBLE_t[:, ::1] y,
                  const DOUBLE_t[:] sample_weight) except -1:
        """Initialize the splitter

        Returns -1 in case of failure to allocate memory (and raise MemoryError)
        or 0 otherwise.
        """
        # Call parent init
        PatchSplitter.init(self, X, y, sample_weight)

        self.X = X
        return 0

cdef class BestPatchSplitter(BaseDensePatchSplitter):
    def __reduce__(self):
        """Enable pickling the splitter."""
        return (
            BestPatchSplitter,
            (
                self.criterion,
                self.max_features,
                self.min_samples_leaf,
                self.min_weight_leaf,
                self.random_state,
                self.min_patch_height,
                self.max_patch_height,
                self.min_patch_width,
                self.max_patch_width,
                self.data_height,
                self.data_width,
            ), self.__getstate__())

    # NOTE: vectors are passed by value, so & is needed to pass by reference
    cdef void sample_proj_mat(
        self,
        vector[vector[DTYPE_t]]& proj_mat_weights,
        vector[vector[SIZE_t]]& proj_mat_indices
    ) noexcept nogil:
        """Sample projection matrix using a contiguous patch.

        Randomly sample patches with weight of 1.
        """
        cdef SIZE_t max_features = self.max_features
        cdef UINT32_t* random_state = &self.rand_r_state

        cdef int proj_i, idx
        cdef int row_idx, col_idx

        # weights are default to 1
        cdef DTYPE_t weight = 1.

        # define parameters for the random patch
        cdef SIZE_t patch_dim
        cdef SIZE_t delta_patch_dim

        # define parameters for vectorized points in the original data shape
        # and top-left seed
        cdef SIZE_t vectorized_point
        cdef SIZE_t top_left_seed
        cdef SIZE_t top_left_patch_seed

        for proj_i in range(0, max_features):
            vectorized_point = 0

            # now get the top-left seed that is used to then determine the top-left
            # position in patch
            # compute top-left seed for the multi-dimensional patch
            top_left_patch_seed = 1
            for idx in range(self.ndim):
                # compute random patch width and height
                # Note: By constraining max patch height/width to be at least the min
                # patch height/width this ensures that the minimum value of
                # patch_height and patch_width is 1
                patch_dim = rand_int(
                    self.min_patch_dims[idx],
                    self.max_patch_dims[idx] + 1,
                    random_state
                )

                # write to buffer
                self.patch_dims_buff[idx] = patch_dim

                # compute the difference between the image dimensions and the current
                # random patch dimensions for sampling
                delta_patch_dim = self.data_dims[idx] - patch_dim + 1
                top_left_patch_seed *= delta_patch_dim
            top_left_seed = rand_int(0, top_left_patch_seed, random_state)

            # TODO: unravel this index
            # once the vectorized top-left-seed is unraveled, you can add the patch
            # points in the array structure and compute their vectorized (unraveled)
            # points, which are added to the projection vector

            # XXX: For now, we will only make this work for 2D C-contiguous arrays
            for row_idx in range(self.patch_dims_buff[0]):
                for col_idx in range(self.patch_dims_buff[1]):
                    vectorized_point = \
                        top_left_seed + col_idx + row_idx * self.data_dims[1]
                    if vectorized_point >= self.n_features:
                        with gil:
                            print("Vectorized point is greater ",
                                  vectorized_point, self.n_features)
                    proj_mat_indices[proj_i].push_back(vectorized_point)
                    proj_mat_weights[proj_i].push_back(weight)

            # Get the end-point of the patch
            # Note: The end-point of the dataset might be less than the patch.
            # This occurs if we sample a seed point at the edge of the "image".
            # Therefore, we take the minimum between the end-point, or the last
            # index in the vectorized image.
            # patch_end_seed = min(
            #     top_left_seed + delta_width * delta_height,
            #     self.n_features
            # )

            # need to extract an array of size of the sampled patch
            # for feat_seed in range(top_left_seed, patch_end_seed):
            #     # now compute vectorized point from
            #     # get the row-idx and col-idx of the patch in the original structured
            #     # 2D array
            #     vectorized_point = 0
            #     for idx in range(self.ndim):
            #         dim = self.data_dims[idx]
            #         dim_idx = <SIZE_t>floor(dim / feat_seed)
            #         vectorized_point += dim_idx * dim
            #     vectorized_point = cnp.unravel_index(feat_seed, self.data_dims)

            #     if vectorized_point > self.n_features:
            #         with gil:
            #             vectorized_point = 0
            #             for idx in range(self.ndim):
            #                 dim = self.data_dims[idx]
            #                 dim_idx = <SIZE_t>floor(dim / feat_seed)
            #                 vectorized_point += dim_idx * dim
            #             print('done')
            #         vectorized_point = 0

            #     # store the non-zero indices and non-zero weights of the data
            #     proj_mat_indices[proj_i].push_back(vectorized_point)
            #     proj_mat_weights[proj_i].push_back(weight)