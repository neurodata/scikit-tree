# distutils: language=c++
# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: initializedcheck=False

import numpy as np

from cython.operator cimport dereference as deref
from libcpp.vector cimport vector

from .._lib.sklearn.tree._criterion cimport Criterion
from .._lib.sklearn.tree._utils cimport rand_int, rand_uniform


cdef float64_t INFINITY = np.inf

# Mitigate precision differences between 32 bit and 64 bit
cdef float32_t FEATURE_THRESHOLD = 1e-7

# Constant to switch between algorithm non zero value extract algorithm
# in SparseSplitter
cdef float32_t EXTRACT_NNZ_SWITCH = 0.1


cdef inline void _init_split(ObliqueSplitRecord* self, intp_t start_pos) noexcept nogil:
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
    def __getstate__(self):
        return {}

    def __setstate__(self, d):
        pass

    cdef int node_reset(self, intp_t start, intp_t end,
                        float64_t* weighted_n_node_samples) except -1 nogil:
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

    cdef intp_t pointer_size(self) noexcept nogil:
        """Get size of a pointer to record for ObliqueSplitter."""

        return sizeof(ObliqueSplitRecord)

    cdef inline void compute_features_over_samples(
        self,
        intp_t start,
        intp_t end,
        const intp_t[:] samples,
        float32_t[:] feature_values,
        vector[float32_t]& proj_vec_weights,  # weights of the vector (max_features,)
        vector[intp_t]& proj_vec_indices      # indices of the features (max_features,)
    ) noexcept nogil:
        """Compute the feature values for the samples[start:end] range.

        Returns -1 in case of failure to allocate memory (and raise MemoryError)
        or 0 otherwise.
        """
        cdef intp_t idx, jdx
        cdef intp_t col_idx
        cdef float32_t col_weight

        # Compute linear combination of features and then
        # sort samples according to the feature values.
        for jdx in range(0, proj_vec_indices.size()):
            col_idx = proj_vec_indices[jdx]
            col_weight = proj_vec_weights[jdx]

            for idx in range(start, end):
                # initialize the feature value to 0
                if jdx == 0:
                    feature_values[idx] = 0.0
                feature_values[idx] += self.X[samples[idx], col_idx] * col_weight

    cdef inline void fisher_yates_shuffle_memview(
        self,
        intp_t[::1] indices_to_sample,
        intp_t grid_size,
        UINT32_t* random_state,
    ) noexcept nogil:
        cdef intp_t i, j

        # XXX: should this be `i` or `i+1`? for valid Fisher-Yates?
        for i in range(0, grid_size - 1):
            j = rand_int(i, grid_size, random_state)
            indices_to_sample[j], indices_to_sample[i] = \
                indices_to_sample[i], indices_to_sample[j]

cdef class ObliqueSplitter(BaseObliqueSplitter):
    def __cinit__(
        self,
        Criterion criterion,
        intp_t max_features,
        intp_t min_samples_leaf,
        float64_t min_weight_leaf,
        object random_state,
        const cnp.int8_t[:] monotonic_cst,
        float64_t feature_combinations,
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

        feature_combinations : float64_t
            The average number of features to combine in an oblique split.
            Each feature is independently included with probability
            ``feature_combination`` / ``n_features``.

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
        self.monotonic_cst = monotonic_cst

        # Sparse max_features x n_features projection matrix
        self.proj_mat_weights = vector[vector[float32_t]](self.max_features)
        self.proj_mat_indices = vector[vector[intp_t]](self.max_features)

        # Oblique tree parameters
        self.feature_combinations = feature_combinations

        # or max w/ 1...
        self.n_non_zeros = max(<intp_t>(self.max_features * self.feature_combinations), 1)

    cdef int init(
        self,
        object X,
        const float64_t[:, ::1] y,
        const float64_t[:] sample_weight,
        const unsigned char[::1] missing_values_in_feature_mask,
    ) except -1:
        Splitter.init(self, X, y, sample_weight, missing_values_in_feature_mask)

        self.X = X

        # create a helper array for allowing efficient Fisher-Yates
        self.indices_to_sample = np.arange(self.max_features * self.n_features,
                                           dtype=np.intp)

        # XXX: Just to initialize stuff
        # self.feature_weights = np.ones((self.n_features,), dtype=float32_t) / self.n_features
        return 0

    cdef void sample_proj_mat(
        self,
        vector[vector[float32_t]]& proj_mat_weights,
        vector[vector[intp_t]]& proj_mat_indices
    ) noexcept nogil:
        """Sample oblique projection matrix.

        Randomly sample features to put in randomly sampled projection vectors
        weight = 1 or -1 with probability 0.5.

        Note: vectors are passed by value, so & is needed to pass by reference.

        Parameters
        ----------
        proj_mat_weights : vector of vectors reference of shape (mtry, n_features)
            The memory address of projection matrix non-zero weights.
        proj_mat_indices : vector of vectors reference of shape (mtry, n_features)
            The memory address of projection matrix non-zero indices.

        Notes
        -----
        Note that grid_size must be larger than or equal to n_non_zeros because
        it is assumed ``feature_combinations`` is forced to be smaller than
        ``n_features`` before instantiating an oblique splitter.
        """
        cdef intp_t n_features = self.n_features
        cdef intp_t n_non_zeros = self.n_non_zeros
        cdef UINT32_t* random_state = &self.rand_r_state

        cdef intp_t i, feat_i, proj_i, rand_vec_index
        cdef float32_t weight

        # construct an array to sample from mTry x n_features set of indices
        cdef intp_t[::1] indices_to_sample = self.indices_to_sample
        cdef intp_t grid_size = self.max_features * self.n_features

        # shuffle indices over the 2D grid to sample using Fisher-Yates
        self.fisher_yates_shuffle_memview(indices_to_sample, grid_size, random_state)

        # sample 'n_non_zeros' in a mtry X n_features projection matrix
        # which consists of +/- 1's chosen at a 1/2s rate
        for i in range(0, n_non_zeros):
            # get the next index from the shuffled index array
            rand_vec_index = indices_to_sample[i]

            # get the projection index (i.e. row of the projection matrix) and
            # feature index (i.e. column of the projection matrix)
            proj_i = rand_vec_index // n_features
            feat_i = rand_vec_index % n_features

            # sample a random weight
            weight = 1 if (rand_int(0, 2, random_state) == 1) else -1

            proj_mat_indices[proj_i].push_back(feat_i)  # Store index of nonzero
            proj_mat_weights[proj_i].push_back(weight)  # Store weight of nonzero

cdef class BestObliqueSplitter(ObliqueSplitter):
    def __reduce__(self):
        """Enable pickling the splitter."""
        return (type(self),
                (
                    self.criterion,
                    self.max_features,
                    self.min_samples_leaf,
                    self.min_weight_leaf,
                    self.random_state,
                    self.monotonic_cst.base if self.monotonic_cst is not None else None,
                    self.feature_combinations,
                ), self.__getstate__())

    cdef int node_split(
        self,
        float64_t impurity,
        SplitRecord* split,
        intp_t* n_constant_features,
        float64_t lower_bound,
        float64_t upper_bound,
    ) except -1 nogil:
        """Find the best_split split on node samples[start:end]

        Returns -1 in case of failure to allocate memory (and raise MemoryError)
        or 0 otherwise.
        """
        # typecast the pointer to an ObliqueSplitRecord
        cdef ObliqueSplitRecord* oblique_split = <ObliqueSplitRecord*>(split)

        # Draw random splits and pick the best_split
        cdef intp_t[::1] samples = self.samples
        cdef intp_t start = self.start
        cdef intp_t end = self.end

        # pointer array to store feature values to split on
        cdef float32_t[::1]  feature_values = self.feature_values
        cdef intp_t max_features = self.max_features
        cdef intp_t min_samples_leaf = self.min_samples_leaf

        # keep track of split record for current_split node and the best_split split
        # found among the sampled projection vectors
        cdef ObliqueSplitRecord best_split, current_split
        cdef float64_t current_proxy_improvement = -INFINITY
        cdef float64_t best_proxy_improvement = -INFINITY

        cdef intp_t feat_i, p       # index over computed features and start/end
        cdef intp_t partition_end
        cdef float32_t temp_d       # to compute a projection feature value

        # instantiate the split records
        _init_split(&best_split, end)

        # Sample the projection matrix by passing in a reference to the underlying
        # vector of vectors for weights and indices
        self.sample_proj_mat(self.proj_mat_weights, self.proj_mat_indices)

        # For every vector in the projection matrix
        for feat_i in range(max_features):
            # Projection vector has no nonzeros
            if self.proj_mat_weights[feat_i].empty():
                continue

            # Note: 'feature' is not actually used in oblique split records
            # Just indicates which index was sampled in the sampled projection matrix
            current_split.feature = feat_i

            # Compute linear combination of features and then
            # sort samples according to the feature values.
            self.compute_features_over_samples(
                start,
                end,
                samples,
                feature_values,
                self.proj_mat_weights[feat_i],
                self.proj_mat_indices[feat_i]
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
                    if self.check_postsplit_conditions() == 1:
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

                        # Note: we do not make a copy above if we are not going to use it
                        # as the candidate best split, so here we create a pointer to the
                        # copy of the projection vectors
                        # best_split.proj_vec_weights = self.proj_mat_weights[feat_i]
                        # best_split.proj_vec_indices = self.proj_mat_indices[feat_i]
                        # best_proj_vec_weights = &self.proj_mat_weights[feat_i]
                        # best_proj_vec_indices = &self.proj_mat_indices[feat_i]

        # Reorganize into samples[start:best_split.pos] + samples[best_split.pos:end]
        if best_split.pos < end:
            partition_end = end
            p = start

            while p < partition_end:
                # Account for projection vector
                temp_d = 0.0
                # for j in range(best_split.proj_vec_indices.size()):
                #    # temp_d += self.X[samples[p], best_split.proj_vec_indices[j]] *\
                #    #             best_split.proj_vec_weights[j]
                for j in range(self.proj_mat_weights[best_split.feature].size()):
                    temp_d += self.X[samples[p], self.proj_mat_indices[best_split.feature][j]] *\
                                self.proj_mat_weights[best_split.feature][j]

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

        # Ensure that the projection vectors are copied into the underlying split record that is
        # seen by the tree builder
        # deref(oblique_split).proj_vec_indices = deref(best_proj_vec_indices) # best_split.proj_vec_indices
        # deref(oblique_split).proj_vec_weights = deref(best_proj_vec_weights) # best_split.proj_vec_weights
        deref(oblique_split).proj_vec_indices = self.proj_mat_indices[best_split.feature]
        deref(oblique_split).proj_vec_weights = self.proj_mat_weights[best_split.feature]

        # Dereference the pointer to the split record and set the values here
        deref(oblique_split).feature = best_split.feature
        deref(oblique_split).pos = best_split.pos
        deref(oblique_split).threshold = best_split.threshold
        deref(oblique_split).improvement = best_split.improvement
        deref(oblique_split).impurity_left = best_split.impurity_left
        deref(oblique_split).impurity_right = best_split.impurity_right
        return 0

cdef class RandomObliqueSplitter(ObliqueSplitter):
    def __reduce__(self):
        """Enable pickling the splitter."""
        return (type(self),
                (
                    self.criterion,
                    self.max_features,
                    self.min_samples_leaf,
                    self.min_weight_leaf,
                    self.random_state,
                    self.monotonic_cst.base if self.monotonic_cst is not None else None,
                    self.feature_combinations,
                ), self.__getstate__())

    cdef inline void find_min_max(
        self,
        float32_t[::1] feature_values,
        float32_t* min_feature_value_out,
        float32_t* max_feature_value_out,
    ) noexcept nogil:
        """Find the minimum and maximum value for current_feature."""
        cdef:
            float32_t current_feature_value
            float32_t min_feature_value = INFINITY
            float32_t max_feature_value = -INFINITY
            intp_t start = self.start
            intp_t end = self.end
            intp_t p

        for p in range(start, end):
            current_feature_value = feature_values[p]
            if current_feature_value < min_feature_value:
                min_feature_value = current_feature_value
            elif current_feature_value > max_feature_value:
                max_feature_value = current_feature_value

        min_feature_value_out[0] = min_feature_value
        max_feature_value_out[0] = max_feature_value

    cdef inline intp_t partition_samples(self, float64_t current_threshold) noexcept nogil:
        """Partition samples for feature_values at the current_threshold."""
        cdef:
            intp_t p = self.start
            intp_t partition_end = self.end
            intp_t[::1] samples = self.samples
            float32_t[::1] feature_values = self.feature_values

        while p < partition_end:
            if feature_values[p] <= current_threshold:
                p += 1
            else:
                partition_end -= 1

                feature_values[p], feature_values[partition_end] = (
                    feature_values[partition_end], feature_values[p]
                )
                samples[p], samples[partition_end] = samples[partition_end], samples[p]

        return partition_end

    # overwrite the node_split method with random threshold selection
    cdef int node_split(
        self,
        float64_t impurity,
        SplitRecord* split,
        intp_t* n_constant_features,
        float64_t lower_bound,
        float64_t upper_bound,
    ) except -1 nogil:
        """Find the best_split split on node samples[start:end]

        Returns -1 in case of failure to allocate memory (and raise MemoryError)
        or 0 otherwise.
        """
        # typecast the pointer to an ObliqueSplitRecord
        cdef ObliqueSplitRecord* oblique_split = <ObliqueSplitRecord*>(split)

        # Draw random splits and pick the best_split
        cdef intp_t[::1] samples = self.samples
        cdef intp_t start = self.start
        cdef intp_t end = self.end
        cdef UINT32_t* random_state = &self.rand_r_state

        # pointer array to store feature values to split on
        cdef float32_t[::1] feature_values = self.feature_values
        cdef intp_t max_features = self.max_features
        cdef intp_t min_samples_leaf = self.min_samples_leaf
        cdef float64_t min_weight_leaf = self.min_weight_leaf

        # keep track of split record for current_split node and the best_split split
        # found among the sampled projection vectors
        cdef ObliqueSplitRecord best_split, current_split
        cdef float64_t current_proxy_improvement = -INFINITY
        cdef float64_t best_proxy_improvement = -INFINITY

        cdef intp_t p
        cdef intp_t feat_i
        cdef intp_t partition_end
        cdef float32_t temp_d         # to compute a projection feature value
        cdef float32_t min_feature_value
        cdef float32_t max_feature_value

        # Number of features discovered to be constant during the split search
        # cdef intp_t n_found_constants = 0
        # cdef intp_t n_known_constants = n_constant_features[0]
        # n_total_constants = n_known_constants + n_found_constants
        # cdef intp_t n_total_constants = n_known_constants
        cdef intp_t n_visited_features = 0

        # instantiate the split records
        _init_split(&best_split, end)

        # Sample the projection matrix
        self.sample_proj_mat(self.proj_mat_weights, self.proj_mat_indices)

        # For every vector in the projection matrix
        for feat_i in range(max_features):
            # Break if already reached max_features
            if n_visited_features >= max_features:
                break
            # Skip features known to be constant
            # if feat_i < n_total_constants:
            #     continue
            # Projection vector has no nonzeros
            if self.proj_mat_weights[feat_i].empty():
                continue

            # Note: 'feature' is not actually used in oblique split records
            # Just indicates which index was sampled in the sampled projection matrix
            current_split.feature = feat_i

            # Compute linear combination of features
            self.compute_features_over_samples(
                start,
                end,
                samples,
                feature_values,
                self.proj_mat_weights[feat_i],
                self.proj_mat_indices[feat_i]
            )

            # find min, max of the feature_values
            self.find_min_max(feature_values, &min_feature_value, &max_feature_value)

            # XXX: Add logic to keep track of constant features if they exist
            # if max_feature_value <= min_feature_value + FEATURE_THRESHOLD:
            #     n_found_constants += 1
            #     n_total_constants += 1
            #     continue

            # Draw a random threshold
            current_split.threshold = rand_uniform(
                min_feature_value,
                max_feature_value,
                random_state,
            )

            if current_split.threshold == max_feature_value:
                current_split.threshold = min_feature_value

            # Partition
            current_split.pos = self.partition_samples(current_split.threshold)

            # Reject if min_samples_leaf is not guaranteed
            if (((current_split.pos - start) < min_samples_leaf) or
                    ((end - current_split.pos) < min_samples_leaf)):
                continue

            # evaluate split
            self.criterion.reset()
            self.criterion.update(current_split.pos)

            # Reject if min_weight_leaf is not satisfied
            if ((self.criterion.weighted_n_left < min_weight_leaf) or
                    (self.criterion.weighted_n_right < min_weight_leaf)):
                continue

            current_proxy_improvement = self.criterion.proxy_impurity_improvement()

            if current_proxy_improvement > best_proxy_improvement:
                best_proxy_improvement = current_proxy_improvement
                best_split = current_split  # copy

                # Note: we do not make a copy above if we are not going to use it
                # as the candidate best split
                # The self.proj_mat_weights and self.proj_mat_indices already contain
                # the best projection vector found at `best_split.feature` (i.e. feat_i).
                # best_split.proj_vec_weights = self.proj_mat_weights[feat_i]
                # best_split.proj_vec_indices = self.proj_mat_indices[feat_i]
                # best_proj_vec_indices = &self.proj_mat_indices[feat_i]
                # best_proj_vec_weights = &self.proj_mat_weights[feat_i]

            n_visited_features += 1

        # Reorganize into samples[start:best_split.pos] + samples[best_split.pos:end]
        if best_split.pos < end:
            partition_end = end
            p = start

            while p < partition_end:
                # Account for projection vector
                temp_d = 0.0
                # for j in range(best_split.proj_vec_indices.size()):
                #     temp_d += self.X[samples[p], best_split.proj_vec_indices[j]] *\
                #                 best_split.proj_vec_weights[j]
                for j in range(self.proj_mat_indices[best_split.feature].size()):
                    temp_d += self.X[samples[p], self.proj_mat_indices[best_split.feature][j]] *\
                                self.proj_mat_weights[best_split.feature][j]

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

        # deref(oblique_split).proj_vec_indices = deref(best_proj_vec_indices) # best_split.proj_vec_indices
        # deref(oblique_split).proj_vec_weights = deref(best_proj_vec_weights) # best_split.proj_vec_weights
        deref(oblique_split).proj_vec_indices = self.proj_mat_indices[best_split.feature]
        deref(oblique_split).proj_vec_weights = self.proj_mat_weights[best_split.feature]

        # Return values
        deref(oblique_split).feature = best_split.feature
        deref(oblique_split).pos = best_split.pos
        deref(oblique_split).threshold = best_split.threshold
        deref(oblique_split).improvement = best_split.improvement
        deref(oblique_split).impurity_left = best_split.impurity_left
        deref(oblique_split).impurity_right = best_split.impurity_right
        return 0


cdef class MultiViewSplitter(BestObliqueSplitter):
    def __cinit__(
        self,
        Criterion criterion,
        intp_t max_features,
        intp_t min_samples_leaf,
        float64_t min_weight_leaf,
        object random_state,
        const cnp.int8_t[:] monotonic_cst,
        float64_t feature_combinations,
        const intp_t[:] feature_set_ends,
        intp_t n_feature_sets,
        const intp_t[:] max_features_per_set,
        *argv
    ):
        self.feature_set_ends = feature_set_ends

        # infer the number of feature sets
        self.n_feature_sets = n_feature_sets

        # replaces usage of max_features
        self.max_features_per_set = max_features_per_set

    def __getstate__(self):
        return {}

    def __setstate__(self, d):
        pass

    def __reduce__(self):
        """Enable pickling the splitter."""
        return (type(self),
                (
                    self.criterion,
                    self.max_features,
                    self.min_samples_leaf,
                    self.min_weight_leaf,
                    self.random_state,
                    self.monotonic_cst.base if self.monotonic_cst is not None else None,
                    self.feature_combinations,
                    self.feature_set_ends.base if self.feature_set_ends is not None else None,
                    self.n_feature_sets,
                    self.max_features_per_set.base if self.max_features_per_set is not None else None,
                ), self.__getstate__())

    cdef int init(
        self,
        object X,
        const float64_t[:, ::1] y,
        const float64_t[:] sample_weight,
        const unsigned char[::1] missing_values_in_feature_mask,
    ) except -1:
        Splitter.init(self, X, y, sample_weight, missing_values_in_feature_mask)

        self.X = X

        # create a helper array for allowing efficient Fisher-Yates
        self.multi_indices_to_sample = vector[vector[intp_t]](self.n_feature_sets)

        cdef intp_t i_feature = 0
        cdef intp_t feature_set_begin = 0
        cdef intp_t size_of_feature_set
        cdef intp_t ifeat = 0
        for i_feature in range(self.n_feature_sets):
            size_of_feature_set = self.feature_set_ends[i_feature] - feature_set_begin
            for ifeat in range(size_of_feature_set):
                self.multi_indices_to_sample[i_feature].push_back(ifeat + feature_set_begin)

            feature_set_begin = self.feature_set_ends[i_feature]
        return 0

    cdef void sample_proj_mat(
        self,
        vector[vector[float32_t]]& proj_mat_weights,
        vector[vector[intp_t]]& proj_mat_indices
    ) noexcept nogil:
        """Sample projection matrix accounting for multi-views.

        This proceeds as a normal sampling projection matrix,
        but now also uniformly samples features from each feature set.
        """
        cdef UINT32_t* random_state = &self.rand_r_state
        cdef intp_t feat_i, proj_i
        cdef float32_t weight

        # keep track of the beginning and ending indices of each feature set
        cdef intp_t idx
        cdef intp_t ifeature = 0
        cdef intp_t grid_size

        cdef intp_t max_features

        # 01: Algorithm samples features from each set equally with the same number
        # of candidates, but if one feature set is exhausted, then that one is no longer sampled
        cdef intp_t finished_feature_set_count = 0
        cdef bint finished_feature_sets = False
        cdef intp_t i, j

        proj_i = 0

        if self.max_features_per_set is None:
            while proj_i < self.max_features and not finished_feature_sets:
                finished_feature_sets = False
                finished_feature_set_count = 0

                # sample from a feature set
                for idx in range(self.n_feature_sets):
                    # indices_to_sample = self.multi_indices_to_sample[idx]
                    grid_size = self.multi_indices_to_sample[idx].size()

                    # Note: a temporary variable must not be used, else a copy will be made
                    if proj_i == 0:
                        for i in range(0, self.multi_indices_to_sample[idx].size() - 1):
                            j = rand_int(i + 1, grid_size, random_state)
                            self.multi_indices_to_sample[idx][i], self.multi_indices_to_sample[idx][j] = \
                                self.multi_indices_to_sample[idx][j], self.multi_indices_to_sample[idx][i]

                    # keep track of which feature-sets are exhausted
                    if ifeature >= grid_size:
                        finished_feature_set_count += 1
                        continue

                    # sample random feature in this set
                    feat_i = self.multi_indices_to_sample[idx][ifeature]

                    # here, axis-aligned splits are entirely weights of 1
                    weight = 1  # if (rand_int(0, 2, random_state) == 1) else -1

                    proj_mat_indices[proj_i].push_back(feat_i)  # Store index of nonzero
                    proj_mat_weights[proj_i].push_back(weight)  # Store weight of nonzero

                    proj_i += 1
                    if proj_i >= self.max_features:
                        break

                if finished_feature_set_count == self.n_feature_sets:
                    finished_feature_sets = True

                ifeature += 1
        # 02: Algorithm samples a different number features from each set, but considers
        # each feature-set equally
        else:
            while proj_i < self.max_features:
                # sample from a feature set
                for idx in range(self.n_feature_sets):
                    # get the max-features for this feature-set
                    max_features = self.max_features_per_set[idx]

                    grid_size = self.multi_indices_to_sample[idx].size()
                    # Note: a temporary variable must not be used, else a copy will be made
                    for i in range(0, self.multi_indices_to_sample[idx].size() - 1):
                        j = rand_int(i + 1, grid_size, random_state)
                        self.multi_indices_to_sample[idx][i], self.multi_indices_to_sample[idx][j] = \
                            self.multi_indices_to_sample[idx][j], self.multi_indices_to_sample[idx][i]

                    for ifeature in range(max_features):
                        # sample random feature in this set
                        feat_i = self.multi_indices_to_sample[idx][ifeature]

                        # here, axis-aligned splits are entirely weights of 1
                        weight = 1  # if (rand_int(0, 2, random_state) == 1) else -1

                        proj_mat_indices[proj_i].push_back(feat_i)  # Store index of nonzero
                        proj_mat_weights[proj_i].push_back(weight)  # Store weight of nonzero

                        proj_i += 1
                        if proj_i >= self.max_features:
                            break
                    if proj_i >= self.max_features:
                        break


cdef class MultiViewSplitterTester(MultiViewSplitter):
    """A class to expose a Python interface for testing."""

    cpdef sample_projection_matrix_py(self):
        """Sample projection matrix using a patch.

        Used for testing purposes.

        Returns projection matrix of shape (max_features, n_features).
        """
        cdef vector[vector[float32_t]] proj_mat_weights = vector[vector[float32_t]](self.max_features)
        cdef vector[vector[intp_t]] proj_mat_indices = vector[vector[intp_t]](self.max_features)
        cdef intp_t i, j

        # sample projection matrix in C/C++
        self.sample_proj_mat(proj_mat_weights, proj_mat_indices)

        # convert the projection matrix to something that can be used in Python
        proj_vecs = np.zeros((self.max_features, self.n_features), dtype=np.float32)
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


cdef class BestObliqueSplitterTester(BestObliqueSplitter):
    """A class to expose a Python interface for testing."""

    cpdef sample_projection_matrix_py(self):
        """Sample projection matrix using a patch.

        Used for testing purposes.

        Returns projection matrix of shape (max_features, n_features).
        """
        cdef vector[vector[float32_t]] proj_mat_weights = vector[vector[float32_t]](self.max_features)
        cdef vector[vector[intp_t]] proj_mat_indices = vector[vector[intp_t]](self.max_features)
        cdef intp_t i, j

        # sample projection matrix in C/C++
        self.sample_proj_mat(proj_mat_weights, proj_mat_indices)

        # convert the projection matrix to something that can be used in Python
        proj_vecs = np.zeros((self.max_features, self.n_features), dtype=np.float32)
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
