# cython: cdivision=True
# distutils: language=c++
# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: initializedcheck=False

import numpy as np

cimport numpy as cnp

cnp.import_array()

from cython.operator cimport dereference as deref
from libcpp.vector cimport vector

from ..._lib.sklearn.tree._criterion cimport Criterion
from ..._lib.sklearn.tree._utils cimport rand_int
from .._utils cimport cartesian_cython, ravel_multi_index_cython, unravel_index_cython


cdef class PatchSplitter(BestObliqueSplitter):
    """Patch splitter.

    A convolutional 2D patch splitter.
    """
    def __getstate__(self):
        return {}

    def __setstate__(self, d):
        pass

    cdef int init(
        self,
        object X,
        const float64_t[:, ::1] y,
        const float64_t[:] sample_weight,
        const unsigned char[::1] missing_values_in_feature_mask,
    ) except -1:
        BestObliqueSplitter.init(self, X, y, sample_weight, missing_values_in_feature_mask)

        return 0

    cdef int node_reset(
        self,
        intp_t start,
        intp_t end,
        float64_t* weighted_n_node_samples
    ) except -1 nogil:
        """Reset splitter on node samples[start:end].

        Returns -1 in case of failure to allocate memory (and raise MemoryError)
        or 0 otherwise.

        Parameters
        ----------
        start : intp_t
            The index of the first sample to consider
        end : intp_t
            The index of the last sample to consider
        weighted_n_node_samples : ndarray, dtype=float64_t pointer
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
        vector[vector[float32_t]]& proj_mat_weights,
        vector[vector[intp_t]]& proj_mat_indices
    ) noexcept nogil:
        """ Sample the projection vector.

        This is a placeholder method.

        """
        pass

    cdef (intp_t, intp_t) sample_top_left_seed(self) noexcept nogil:
        pass


cdef class BaseDensePatchSplitter(PatchSplitter):
    cdef int init(
        self,
        object X,
        const float64_t[:, ::1] y,
        const float64_t[:] sample_weight,
        const unsigned char[::1] missing_values_in_feature_mask,
        # const int32_t[:] n_categories
    ) except -1:
        """Initialize the splitter

        Returns -1 in case of failure to allocate memory (and raise MemoryError)
        or 0 otherwise.
        """
        # Call parent init
        PatchSplitter.init(self, X, y, sample_weight, missing_values_in_feature_mask)

        self.X = X
        return 0


cdef class BestPatchSplitter(BaseDensePatchSplitter):
    def __cinit__(
        self,
        Criterion criterion,
        intp_t max_features,
        intp_t min_samples_leaf,
        float64_t min_weight_leaf,
        object random_state,
        const cnp.int8_t[:] monotonic_cst,
        float64_t feature_combinations,
        const intp_t[:] min_patch_dims,
        const intp_t[:] max_patch_dims,
        const cnp.uint8_t[:] dim_contiguous,
        const intp_t[:] data_dims,
        bytes boundary,
        const float32_t[:, :] feature_weight,
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
        self.monotonic_cst = monotonic_cst

        # Sparse max_features x n_features projection matrix
        self.proj_mat_weights = vector[vector[float32_t]](self.max_features)
        self.proj_mat_indices = vector[vector[intp_t]](self.max_features)

        # initialize state to allow generalization to higher-dimensional tensors
        self.ndim = data_dims.shape[0]
        self.data_dims = data_dims

        # create a buffer for storing the patch dimensions sampled per projection matrix
        self.patch_dims_buff = np.zeros(data_dims.shape[0], dtype=np.intp)

        # store the min and max patch dimension constraints
        self.min_patch_dims = min_patch_dims
        self.max_patch_dims = max_patch_dims
        self.dim_contiguous = dim_contiguous

        # whether or not to perform some discontinuous sampling
        if not all(self.dim_contiguous):
            self._discontiguous = True
        else:
            self._discontiguous = False

        self.boundary = boundary
        self.feature_weight = feature_weight

    def __reduce__(self):
        """Enable pickling the splitter."""
        return (
            type(self),
            (
                self.criterion,
                self.max_features,
                self.min_samples_leaf,
                self.min_weight_leaf,
                self.random_state,
                self.monotonic_cst.base if self.monotonic_cst is not None else None,
                self.feature_combinations,
                self.min_patch_dims.base if self.min_patch_dims is not None else None,
                self.max_patch_dims.base if self.max_patch_dims is not None else None,
                self.dim_contiguous.base if self.dim_contiguous is not None else None,
                self.data_dims.base if self.data_dims is not None else None,
                self.boundary,
                self.feature_weight.base if self.feature_weight is not None else None,
            ), self.__getstate__())

    cdef (intp_t, intp_t) sample_top_left_seed(self) noexcept nogil:
        """Sample the top-left seed for the n-dim patch.

        Returns
        -------
        top_left_seed : intp_t
            The top-left seed vectorized (i.e. raveled) for the n-dim patch.
        patch_size : intp_t
            The total size of the n-dim patch (i.e. the volume).
        """
        # now get the top-left seed that is used to then determine the top-left
        # position in patch
        # compute top-left seed for the multi-dimensional patch
        cdef intp_t top_left_patch_seed
        cdef intp_t patch_size = 1
        cdef vector[intp_t] unraveled_patch_point = vector[intp_t](self.ndim)

        cdef UINT32_t* random_state = &self.rand_r_state

        # define parameters for the random patch
        cdef intp_t patch_dim
        cdef intp_t delta_patch_dim

        cdef intp_t dim

        cdef intp_t idx

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
            # sample the top-left index and patch size for this dimension based on boundary effects
            if self.boundary is None:
                # compute the difference between the image dimensions and the current
                # random patch dimensions for sampling
                delta_patch_dim = (self.data_dims[idx] - patch_dim) + 1
                top_left_patch_seed = rand_int(0, delta_patch_dim, random_state)

                # write to buffer
                self.patch_dims_buff[idx] = patch_dim
                patch_size *= patch_dim
            elif self.boundary == "wrap":
                # add circular boundary conditions
                delta_patch_dim = self.data_dims[idx] + 2 * (patch_dim - 1)

                # sample the top left index for this dimension
                top_left_patch_seed = rand_int(0, delta_patch_dim, random_state)

                # resample the patch dimension due to padding
                dim = top_left_patch_seed % delta_patch_dim

                # resample the patch dimension due to padding
                patch_dim = min(patch_dim, min(dim+1, self.data_dims[idx] + patch_dim - dim - 1))
                self.patch_dims_buff[idx] = patch_dim
                patch_size *= patch_dim

                # TODO: make this work
                # Convert the top-left-seed value to it's appropriate index in the full image.
                top_left_patch_seed = max(0, dim - patch_dim + 1)

            unraveled_patch_point[idx] = top_left_patch_seed

        top_left_patch_seed = ravel_multi_index_cython(
            unraveled_patch_point,
            self.data_dims
        )
        return top_left_patch_seed, patch_size

    cdef void sample_proj_mat(
        self,
        vector[vector[float32_t]]& proj_mat_weights,
        vector[vector[intp_t]]& proj_mat_indices
    ) noexcept nogil:
        """Sample projection matrix using a contiguous patch.

        Randomly sample patches with weight of 1.
        """
        cdef intp_t max_features = self.max_features
        cdef intp_t proj_i

        # define parameters for vectorized points in the original data shape
        # and top-left seed
        cdef intp_t top_left_patch_seed

        # size of the sampled patch, which is just the size of the n-dim patch
        # (\prod_i self.patch_dims_buff[i])
        cdef intp_t patch_size

        for proj_i in range(0, max_features):
            # now get the top-left seed that is used to then determine the top-left
            # position in patch
            # compute top-left seed for the multi-dimensional patch
            top_left_patch_seed, patch_size = self.sample_top_left_seed()

            # sample a projection vector given the top-left seed point in n-dimensional space
            self.sample_proj_vec(
                proj_mat_weights,
                proj_mat_indices,
                proj_i,
                patch_size,
                top_left_patch_seed,
                self.patch_dims_buff
            )

    cdef void sample_proj_vec(
        self,
        vector[vector[float32_t]]& proj_mat_weights,
        vector[vector[intp_t]]& proj_mat_indices,
        intp_t proj_i,
        intp_t patch_size,
        intp_t top_left_patch_seed,
        const intp_t[:] patch_dims,
    ) noexcept nogil:
        """
        Samples projection vector.

        Parameters
        ----------
        proj_mat_weights : vector[vector[float32_t]]
            The weights of the projection matrix.
        proj_mat_indices : vector[vector[intp_t]]
            The indices of the projection matrix.
        proj_i : intp_t
            The index of feature.
        patch_size : intp_t
            The total size of the patch.
        top_left_patch_seed : intp_t
            The top-left seed of the patch raveled.
        patch_dims : array-like, shape (n_dims,)
            The dimensions of the patch.
        """
        # initialize a buffer to allow for Fisher-Yates
        cdef vector[intp_t] _index_data_buffer

        cdef UINT32_t* random_state = &self.rand_r_state
        cdef intp_t num_rows
        cdef int ndim = self.ndim
        cdef vector[vector[intp_t]] points = vector[vector[intp_t]](ndim)
        cdef intp_t patch_dim
        cdef intp_t idx
        cdef intp_t i

        # weights are default to 1
        cdef float32_t weight = 1.

        cdef vector[intp_t] unraveled_patch_point = vector[intp_t](self.ndim)
        unraveled_patch_point = unravel_index_cython(top_left_patch_seed, self.data_dims)

        for dim_idx in range(ndim):
            if self.dim_contiguous[dim_idx]:
                patch_dim = patch_dims[dim_idx]
                idx = patch_dim + unraveled_patch_point[dim_idx]
                for i in range(unraveled_patch_point[dim_idx], idx):
                    points[dim_idx].push_back(i)
            else:
                num_rows = self.data_dims[dim_idx]
                for i in range(0, num_rows):
                    _index_data_buffer.push_back(i)
                # then shuffle indices using Fisher-Yates
                for i in range(num_rows):
                    j = rand_int(0, num_rows - i, random_state)
                    _index_data_buffer[i], _index_data_buffer[j] = \
                        _index_data_buffer[j], _index_data_buffer[i]

                for i in range(0, patch_dims[dim_idx]):  # populate
                    points[dim_idx].push_back(_index_data_buffer[i])
                _index_data_buffer.clear()

        # make cartesian product of the points, ravel, then add to proj_mat_indices
        cdef vector[vector[intp_t]] products = cartesian_cython(points)
        for point in products:
            vectorized_point = ravel_multi_index_cython(point, self.data_dims)
            proj_mat_indices[proj_i].push_back(vectorized_point)
            proj_mat_weights[proj_i].push_back(weight)

    cdef void compute_features_over_samples(
        self,
        intp_t start,
        intp_t end,
        const intp_t[:] samples,
        float32_t[:] feature_values,
        vector[float32_t]* proj_vec_weights,  # weights of the vector (max_features,)
        vector[intp_t]* proj_vec_indices    # indices of the features (max_features,)
    ) noexcept nogil:
        """Compute the feature values for the samples[start:end] range.

        Returns -1 in case of failure to allocate memory (and raise MemoryError)
        or 0 otherwise.
        """
        cdef intp_t idx, jdx

        # initialize feature weight to normalize across patch
        cdef float32_t patch_weight

        # Compute linear combination of features and then
        # sort samples according to the feature values.
        for idx in range(start, end):
            patch_weight = 0.0

            # initialize the feature value to 0
            feature_values[idx] = 0
            for jdx in range(0, proj_vec_indices.size()):
                feature_values[idx] += self.X[
                    samples[idx], deref(proj_vec_indices)[jdx]
                ] * deref(proj_vec_weights)[jdx]

                if self.feature_weight is not None:
                    # gets the feature weight for this specific column from X
                    # the default of feature_weights[i] is (1/n_features) for all i
                    patch_weight += self.feature_weight[samples[idx], deref(proj_vec_indices)[jdx]]

            if self.feature_weight is not None:
                feature_values[idx] /= patch_weight


cdef class BestPatchSplitterTester(BestPatchSplitter):
    """A class to expose a Python interface for testing."""
    cpdef sample_top_left_seed_cpdef(self):
        top_left_patch_seed, patch_size = self.sample_top_left_seed()
        patch_dims = np.array(self.patch_dims_buff, dtype=np.intp)
        return top_left_patch_seed, patch_size, patch_dims

    cpdef sample_projection_vector(
        self,
        intp_t proj_i,
        intp_t patch_size,
        intp_t top_left_patch_seed,
        intp_t[:] patch_dims,
    ):
        cdef vector[vector[float32_t]] proj_mat_weights = vector[vector[float32_t]](self.max_features)
        cdef vector[vector[intp_t]] proj_mat_indices = vector[vector[intp_t]](self.max_features)
        cdef intp_t i, j

        # sample projection matrix in C/C++
        self.sample_proj_vec(
            proj_mat_weights,
            proj_mat_indices,
            proj_i,
            patch_size,
            top_left_patch_seed,
            patch_dims
        )

        # convert the projection matrix to something that can be used in Python
        proj_vecs = np.zeros((1, self.n_features), dtype=np.float64)
        for i in range(0, 1):
            for j in range(0, proj_mat_weights[i].size()):
                weight = proj_mat_weights[i][j]
                feat = proj_mat_indices[i][j]
                proj_vecs[i, feat] = weight
        return proj_vecs

    cpdef sample_projection_matrix_py(self):
        """Sample projection matrix using a patch.

        Used for testing purposes.

        Randomly sample patches with weight of 1.
        """
        cdef vector[vector[float32_t]] proj_mat_weights = vector[vector[float32_t]](self.max_features)
        cdef vector[vector[intp_t]] proj_mat_indices = vector[vector[intp_t]](self.max_features)
        cdef intp_t i, j

        # sample projection matrix in C/C++
        self.sample_proj_mat(proj_mat_weights, proj_mat_indices)

        # convert the projection matrix to something that can be used in Python
        proj_vecs = np.zeros((self.max_features, self.n_features), dtype=np.float64)
        for i in range(0, self.max_features):
            for j in range(0, proj_mat_weights[i].size()):
                weight = proj_mat_weights[i][j]
                feat = proj_mat_indices[i][j]

                proj_vecs[i, feat] = weight

        return proj_vecs

    cpdef init_test(self, X, y, sample_weight, missing_values_in_feature_mask=None):
        """Initializes the state of the splitter.

        Used for testing purposes.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            The input samples.
        y : array-like, shape (n_samples,)
            The target values (class labels in classification, real numbers in
            regression).
        sample_weight : array-like, shape (n_samples,)
            Sample weights.
        missing_values_in_feature_mask : array-like, shape (n_features,)
            Whether or not a feature has missing values.
        """
        self.init(X, y, sample_weight, missing_values_in_feature_mask)
