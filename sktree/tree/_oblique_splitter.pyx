# distutils: language=c++
# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False

import numpy as np

cimport numpy as cnp

cnp.import_array()

from cython.operator cimport dereference as deref
from libcpp.vector cimport vector
from sklearn.tree._utils cimport rand_int
from sklearn_fork.tree._criterion cimport Criterion


cdef double INFINITY = np.inf

# Mitigate precision differences between 32 bit and 64 bit
cdef DTYPE_t FEATURE_THRESHOLD = 1e-7

# Constant to switch between algorithm non zero value extract algorithm
# in SparseSplitter
cdef DTYPE_t EXTRACT_NNZ_SWITCH = 0.1


cdef inline void _init_split(ObliqueSplitRecord* self, SIZE_t start_pos) noexcept nogil:
    self.impurity_left = INFINITY
    self.impurity_right = INFINITY
    self.pos = start_pos
    self.feature = 0
    self.threshold = 0.
    self.improvement = -INFINITY

cdef class BaseObliqueSplitter(Splitter):
    """Abstract oblique splitter class.

    Splitters are called by tree builders to find the best_split splits on
    both sparse and dense data, one split at a time.
    """
    def __cinit__(
        self,
        Criterion criterion,
        SIZE_t max_features,
        SIZE_t min_samples_leaf,
        double min_weight_leaf,
        object random_state,
        *argv
    ):
        """
        Parameters
        ----------
        criterion : Criterion
            The criterion to measure the quality of a split.

        max_features : SIZE_t
            The maximal number of randomly selected features which can be
            considered for a split.

        min_samples_leaf : SIZE_t
            The minimal number of samples each leaf can have, where splits
            which would result in having less samples in a leaf are not
            considered.

        min_weight_leaf : double
            The minimal weight each leaf can have, where the weight is the sum
            of the weights of each sample in it.

        random_state : object
            The user inputted random state to be used for pseudo-randomness
        """
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

    def __getstate__(self):
        return {}

    def __setstate__(self, d):
        pass

    cdef int node_reset(self, SIZE_t start, SIZE_t end,
                        double* weighted_n_node_samples) except -1 nogil:
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

    cdef void compute_features_over_samples(
        self,
        SIZE_t start,
        SIZE_t end,
        const SIZE_t[:] samples,
        DTYPE_t[:] feature_values,
        vector[DTYPE_t]* proj_vec_weights,  # weights of the vector (max_features,)
        vector[SIZE_t]* proj_vec_indices    # indices of the features (max_features,)
    ) noexcept nogil:
        """Compute the feature values for the samples[start:end] range.

        Returns -1 in case of failure to allocate memory (and raise MemoryError)
        or 0 otherwise.
        """
        cdef SIZE_t idx, jdx

        # Compute linear combination of features and then
        # sort samples according to the feature values.
        for idx in range(start, end):
            # initialize the feature value to 0
            feature_values[idx] = 0.0
            for jdx in range(0, proj_vec_indices.size()):
                feature_values[idx] += self.X[
                    samples[idx], deref(proj_vec_indices)[jdx]
                ] * deref(proj_vec_weights)[jdx]

    cdef int node_split(
        self,
        double impurity,
        SplitRecord* split,
        SIZE_t* n_constant_features
    ) except -1 nogil:
        """Find the best_split split on node samples[start:end]

        Returns -1 in case of failure to allocate memory (and raise MemoryError)
        or 0 otherwise.
        """
        # typecast the pointer to an ObliqueSplitRecord
        cdef ObliqueSplitRecord* oblique_split = <ObliqueSplitRecord*>(split)

        # Draw random splits and pick the best_split
        cdef SIZE_t[::1] samples = self.samples
        cdef SIZE_t start = self.start
        cdef SIZE_t end = self.end

        # pointer array to store feature values to split on
        cdef DTYPE_t[::1]  feature_values = self.feature_values
        cdef SIZE_t max_features = self.max_features
        cdef SIZE_t min_samples_leaf = self.min_samples_leaf
        cdef double min_weight_leaf = self.min_weight_leaf

        # keep track of split record for current_split node and the best_split split
        # found among the sampled projection vectors
        cdef ObliqueSplitRecord best_split, current_split
        cdef double current_proxy_improvement = -INFINITY
        cdef double best_proxy_improvement = -INFINITY

        cdef SIZE_t feat_i, p       # index over computed features and start/end
        cdef SIZE_t partition_end
        cdef DTYPE_t temp_d         # to compute a projection feature value

        # instantiate the split records
        _init_split(&best_split, end)

        # Sample the projection matrix
        self.sample_proj_mat(self.proj_mat_weights, self.proj_mat_indices)

        # For every vector in the projection matrix
        for feat_i in range(max_features):
            # Projection vector has no nonzeros
            if self.proj_mat_weights[feat_i].empty():
                continue

            # XXX: 'feature' is not actually used in oblique split records
            # Just indicates which split was sampled
            current_split.feature = feat_i
            current_split.proj_vec_weights = &self.proj_mat_weights[feat_i]
            current_split.proj_vec_indices = &self.proj_mat_indices[feat_i]

            # Compute linear combination of features and then
            # sort samples according to the feature values.
            self.compute_features_over_samples(
                start,
                end,
                samples,
                feature_values,
                &self.proj_mat_weights[feat_i],
                &self.proj_mat_indices[feat_i]
            )

            # Sort the samples
            sort(&feature_values[start], &samples[start], end - start)

            # Evaluate all splits
            self.criterion.reset()
            p = start
            while p < end:
                while (p + 1 < end and feature_values[p + 1] <= feature_values[p] + FEATURE_THRESHOLD):
                    p += 1

                p += 1

                if p < end:
                    current_split.pos = p

                    # Reject if min_samples_leaf is not guaranteed
                    if (((current_split.pos - start) < min_samples_leaf) or
                            ((end - current_split.pos) < min_samples_leaf)):
                        continue

                    self.criterion.update(current_split.pos)
                    # Reject if min_weight_leaf is not satisfied
                    if ((self.criterion.weighted_n_left < min_weight_leaf) or
                            (self.criterion.weighted_n_right < min_weight_leaf)):
                        continue

                    current_proxy_improvement = \
                        self.criterion.proxy_impurity_improvement()

                    if current_proxy_improvement > best_proxy_improvement:
                        best_proxy_improvement = current_proxy_improvement
                        # sum of halves is used to avoid infinite value
                        current_split.threshold = feature_values[p - 1] / 2.0 + feature_values[p] / 2.0

                        if (
                            (current_split.threshold == feature_values[p]) or
                            (current_split.threshold == INFINITY) or
                            (current_split.threshold == -INFINITY)
                        ):
                            current_split.threshold = feature_values[p - 1]

                        best_split = current_split  # copy

        # Reorganize into samples[start:best_split.pos] + samples[best_split.pos:end]
        if best_split.pos < end:
            partition_end = end
            p = start

            while p < partition_end:
                # Account for projection vector
                temp_d = 0.0
                for j in range(best_split.proj_vec_indices.size()):
                    temp_d += self.X[samples[p], deref(best_split.proj_vec_indices)[j]] *\
                                deref(best_split.proj_vec_weights)[j]

                if temp_d <= best_split.threshold:
                    p += 1

                else:
                    partition_end -= 1
                    samples[p], samples[partition_end] = \
                        samples[partition_end], samples[p]

            self.criterion.reset()
            self.criterion.update(best_split.pos)
            self.criterion.children_impurity(&best_split.impurity_left,
                                             &best_split.impurity_right)
            best_split.improvement = self.criterion.impurity_improvement(
                impurity, best_split.impurity_left, best_split.impurity_right)

        # Return values
        deref(oblique_split).proj_vec_indices = best_split.proj_vec_indices
        deref(oblique_split).proj_vec_weights = best_split.proj_vec_weights
        deref(oblique_split).feature = best_split.feature
        deref(oblique_split).pos = best_split.pos
        deref(oblique_split).threshold = best_split.threshold
        deref(oblique_split).improvement = best_split.improvement
        deref(oblique_split).impurity_left = best_split.impurity_left
        deref(oblique_split).impurity_right = best_split.impurity_right
        return 0


cdef class ObliqueSplitter(BaseObliqueSplitter):
    def __cinit__(
        self,
        Criterion criterion,
        SIZE_t max_features,
        SIZE_t min_samples_leaf,
        double min_weight_leaf,
        object random_state,
        double feature_combinations,
        *argv
    ):
        """
        Parameters
        ----------
        criterion : Criterion
            The criterion to measure the quality of a split.

        max_features : SIZE_t
            The maximal number of randomly selected features which can be
            considered for a split.

        min_samples_leaf : SIZE_t
            The minimal number of samples each leaf can have, where splits
            which would result in having less samples in a leaf are not
            considered.

        min_weight_leaf : double
            The minimal weight each leaf can have, where the weight is the sum
            of the weights of each sample in it.

        feature_combinations : double
            The average number of features to combine in an oblique split.
            Each feature is independently included with probability
            ``feature_combination`` / ``n_features``.

        random_state : object
            The user inputted random state to be used for pseudo-randomness
        """
        # Oblique tree parameters
        self.feature_combinations = feature_combinations

        # or max w/ 1...
        self.n_non_zeros = max(int(self.max_features * self.feature_combinations), 1)

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
        Splitter.init(self, X, y, sample_weight)

        self.X = X

        # create a helper array for allowing efficient Fisher-Yates
        self.indices_to_sample = np.arange(self.max_features * self.n_features,
                                           dtype=np.intp)

        # XXX: Just to initialize stuff
        # self.feature_weights = np.ones((self.n_features,), dtype=DTYPE_t) / self.n_features
        return 0


cdef class BestObliqueSplitter(ObliqueSplitter):
    def __reduce__(self):
        """Enable pickling the splitter."""
        return (BestObliqueSplitter,
                (
                    self.criterion,
                    self.max_features,
                    self.min_samples_leaf,
                    self.min_weight_leaf,
                    self.feature_combinations,
                    self.random_state
                ), self.__getstate__())

    cdef void sample_proj_mat(
        self,
        vector[vector[DTYPE_t]]& proj_mat_weights,
        vector[vector[SIZE_t]]& proj_mat_indices
    ) noexcept nogil:
        """Sample oblique projection matrix.

        Randomly sample features to put in randomly sampled projection vectors
        weight = 1 or -1 with probability 0.5.

        Note: vectors are passed by value, so & is needed to pass by reference.

        Parameters
        ----------
        proj_mat_weights : vector of vectors reference
            The memory address of projection matrix non-zero weights.
        proj_mat_indices : vector of vectors reference
            The memory address of projection matrix non-zero indices.

        Notes
        -----
        Note that grid_size must be larger than or equal to n_non_zeros because
        it is assumed ``feature_combinations`` is forced to be smaller than
        ``n_features`` before instantiating an oblique splitter.
        """

        cdef SIZE_t n_features = self.n_features
        cdef SIZE_t n_non_zeros = self.n_non_zeros
        cdef UINT32_t* random_state = &self.rand_r_state

        cdef int i, feat_i, proj_i, rand_vec_index
        cdef DTYPE_t weight

        # construct an array to sample from mTry x n_features set of indices
        cdef SIZE_t[::1] indices_to_sample = self.indices_to_sample
        cdef SIZE_t grid_size = self.max_features * self.n_features

        # shuffle indices over the 2D grid to sample using Fisher-Yates
        for i in range(0, grid_size):
            j = rand_int(0, grid_size - i, random_state)
            indices_to_sample[j], indices_to_sample[i] = \
                indices_to_sample[i], indices_to_sample[j]

        # sample 'n_non_zeros' in a mtry X n_features projection matrix
        # which consists of +/- 1's chosen at a 1/2s rate
        for i in range(0, n_non_zeros):
            # get the next index from the shuffled index array
            rand_vec_index = indices_to_sample[i]

            # get the projection index and feature index
            proj_i = rand_vec_index // n_features
            feat_i = rand_vec_index % n_features

            # sample a random weight
            weight = 1 if (rand_int(0, 2, random_state) == 1) else -1

            proj_mat_indices[proj_i].push_back(feat_i)  # Store index of nonzero
            proj_mat_weights[proj_i].push_back(weight)  # Store weight of nonzero
