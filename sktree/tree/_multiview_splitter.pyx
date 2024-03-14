# distutils: language=c++
# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: initializedcheck=False
import numpy as np

from cython.operator cimport dereference as deref
from libc.math cimport isnan
from libc.string cimport memcpy
from libcpp.algorithm cimport fill
from libcpp.numeric cimport accumulate

from .._lib.sklearn.tree._splitter cimport shift_missing_values_to_left_if_required
from .._lib.sklearn.tree._utils cimport rand_int


cdef float64_t INFINITY = np.inf

# Mitigate precision differences between 32 bit and 64 bit
cdef float32_t FEATURE_THRESHOLD = 1e-7

# Constant to switch between algorithm non zero value extract algorithm
# in SparseSplitter
cdef float32_t EXTRACT_NNZ_SWITCH = 0.1


cdef inline void _init_split(SplitRecord* self, intp_t start_pos) noexcept nogil:
    self.impurity_left = INFINITY
    self.impurity_right = INFINITY
    self.pos = start_pos
    self.feature = 0
    self.threshold = 0.
    self.improvement = -INFINITY
    self.missing_go_to_left = False
    self.n_missing = 0
    self.n_constant_features = 0


cdef class MultiViewSplitter(Splitter):
    def __cinit__(
        self,
        Criterion criterion,
        intp_t max_features,
        intp_t min_samples_leaf,
        float64_t min_weight_leaf,
        object random_state,
        const cnp.int8_t[:] monotonic_cst,
        const intp_t[:] feature_set_ends,
        intp_t n_feature_sets,
        const intp_t[:] max_features_per_set,
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

        random_state : object
            The user inputted random state to be used for pseudo-randomness

        monotonic_cst : const cnp.int8_t[:]
            Monotonicity constraints

        """
        self.feature_set_ends = feature_set_ends

        # infer the number of feature sets
        self.n_feature_sets = n_feature_sets

        # replaces usage of max_features
        self.max_features_per_set = max_features_per_set

        # initialize arrays to store the number of visited features, drawn constants and found constants
        # for each feature set during each iteration of the split search
        self.vec_n_visited_features = np.empty(n_feature_sets, dtype=np.intp)
        self.vec_n_drawn_constants = np.empty(n_feature_sets, dtype=np.intp)
        self.vec_n_found_constants = np.empty(n_feature_sets, dtype=np.intp)

        self.n_missing = 0

    def __reduce__(self):
        return (type(self),
                (
                    self.criterion,
                    self.max_features,
                    self.min_samples_leaf,
                    self.min_weight_leaf,
                    self.random_state,
                    self.monotonic_cst.base if self.monotonic_cst is not None else None,
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
        self.missing_values_in_feature_mask = missing_values_in_feature_mask

    cdef intp_t pointer_size(self) noexcept nogil:
        """Get size of a pointer to record for ObliqueSplitter."""

        return sizeof(MultiViewSplitRecord)

    cdef inline void sort_samples_and_feature_values(
        self, intp_t current_feature
    ) noexcept nogil:
        """Simultaneously sort based on the feature_values.

        Missing values are stored at the end of feature_values.
        The number of missing values observed in feature_values is stored
        in self.n_missing.
        """
        cdef:
            intp_t i, current_end
            float32_t[::1] feature_values = self.feature_values
            const float32_t[:, :] X = self.X
            intp_t[::1] samples = self.samples
            intp_t n_missing = 0
            const unsigned char[::1] missing_values_in_feature_mask = self.missing_values_in_feature_mask

        # Sort samples along that feature; by
        # copying the values into an array and
        # sorting the array in a manner which utilizes the cache more
        # effectively.
        if missing_values_in_feature_mask is not None and missing_values_in_feature_mask[current_feature]:
            i, current_end = self.start, self.end - 1
            # Missing values are placed at the end and do not participate in the sorting.
            while i <= current_end:
                # Finds the right-most value that is not missing so that
                # it can be swapped with missing values at its left.
                if isnan(X[samples[current_end], current_feature]):
                    n_missing += 1
                    current_end -= 1
                    continue

                # X[samples[current_end], current_feature] is a non-missing value
                if isnan(X[samples[i], current_feature]):
                    samples[i], samples[current_end] = samples[current_end], samples[i]
                    n_missing += 1
                    current_end -= 1

                feature_values[i] = X[samples[i], current_feature]
                i += 1
        else:
            # When there are no missing values, we only need to copy the data into
            # feature_values
            for i in range(self.start, self.end):
                feature_values[i] = X[samples[i], current_feature]

        sort(&feature_values[self.start], &samples[self.start], self.end - self.start - n_missing)
        self.n_missing = n_missing

    cdef inline void next_p(self, intp_t* p_prev, intp_t* p) noexcept nogil:
        """Compute the next p_prev and p for iteratiing over feature values.

        The missing values are not included when iterating through the feature values.
        """
        cdef:
            float32_t[::1] feature_values = self.feature_values
            intp_t end_non_missing = self.end - self.n_missing

        while (
            p[0] + 1 < end_non_missing and
            feature_values[p[0] + 1] <= feature_values[p[0]] + FEATURE_THRESHOLD
        ):
            p[0] += 1

        p_prev[0] = p[0]

        # By adding 1, we have
        # (feature_values[p] >= end) or (feature_values[p] > feature_values[p - 1])
        p[0] += 1

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

    cdef inline void partition_samples_final(
        self,
        intp_t best_pos,
        float64_t best_threshold,
        intp_t best_feature,
        intp_t best_n_missing,
    ) noexcept nogil:
        """Partition samples for X at the best_threshold and best_feature.

        If missing values are present, this method partitions `samples`
        so that the `best_n_missing` missing values' indices are in the
        right-most end of `samples`, that is `samples[end_non_missing:end]`.
        """
        cdef:
            # Local invariance: start <= p <= partition_end <= end
            intp_t start = self.start
            intp_t p = start
            intp_t end = self.end - 1
            intp_t partition_end = end - best_n_missing
            intp_t[::1] samples = self.samples
            const float32_t[:, :] X = self.X
            float32_t current_value

        if best_n_missing != 0:
            # Move samples with missing values to the end while partitioning the
            # non-missing samples
            while p < partition_end:
                # Keep samples with missing values at the end
                if isnan(X[samples[end], best_feature]):
                    end -= 1
                    continue

                # Swap sample with missing values with the sample at the end
                current_value = X[samples[p], best_feature]
                if isnan(current_value):
                    samples[p], samples[end] = samples[end], samples[p]
                    end -= 1

                    # The swapped sample at the end is always a non-missing value, so
                    # we can continue the algorithm without checking for missingness.
                    current_value = X[samples[p], best_feature]

                # Partition the non-missing samples
                if current_value <= best_threshold:
                    p += 1
                else:
                    samples[p], samples[partition_end] = samples[partition_end], samples[p]
                    partition_end -= 1
        else:
            # Partitioning routine when there are no missing values
            while p < partition_end:
                if X[samples[p], best_feature] <= best_threshold:
                    p += 1
                else:
                    samples[p], samples[partition_end] = samples[partition_end], samples[p]
                    partition_end -= 1


cdef class BestMultiViewSplitter(MultiViewSplitter):
    """Splitter for finding the best split on dense data."""
    cdef int node_split(
        self,
        float64_t impurity,   # Impurity of the node
        SplitRecord* split,
        float64_t lower_bound,
        float64_t upper_bound,
    ) except -1 nogil:
        """Find the best split on node samples[start:end]

        Note: this implementation differs from scikit-learn because

        Returns -1 in case of failure to allocate memory (and raise MemoryError)
        or 0 otherwise.
        """
        # typecast the pointer to an ObliqueSplitRecord
        cdef MultiViewSplitRecord* multiview_split = <MultiViewSplitRecord*>(split)

        # Find the best split
        cdef intp_t start = self.start
        cdef intp_t end = self.end
        cdef intp_t end_non_missing
        cdef intp_t n_missing = 0
        cdef bint has_missing = 0
        cdef intp_t n_searches
        cdef intp_t n_left, n_right
        cdef bint missing_go_to_left

        cdef intp_t[::1] samples = self.samples
        cdef intp_t[::1] features = self.features
        cdef intp_t[::1] constant_features = self.constant_features

        cdef float32_t[::1] feature_values = self.feature_values
        cdef intp_t max_features = self.max_features
        cdef intp_t min_samples_leaf = self.min_samples_leaf
        cdef float64_t min_weight_leaf = self.min_weight_leaf
        cdef UINT32_t* random_state = &self.rand_r_state

        cdef SplitRecord best_split, current_split
        cdef float64_t current_proxy_improvement = -INFINITY
        cdef float64_t best_proxy_improvement = -INFINITY

        cdef bint found_new_constants = False

        # pointer in the feature set
        cdef intp_t f_i

        cdef intp_t f_j
        cdef intp_t p
        cdef intp_t p_prev

        cdef intp_t ifeature
        cdef intp_t feature_set_begin = 0

        # Number of features discovered to be constant during the split search
        cdef intp_t[:] vec_n_found_constants = self.vec_n_found_constants
        # Number of features known to be constant and drawn without replacement
        cdef intp_t[:] vec_n_drawn_constants = self.vec_n_drawn_constants
        cdef intp_t[:] vec_n_visited_features = self.vec_n_visited_features

        # We reset the number of visited features, drawn constants and found constants
        # for each feature set to 0 at the beginning of the split search.
        for ifeature in range(self.n_feature_sets):
            vec_n_found_constants[ifeature] = 0
            vec_n_drawn_constants[ifeature] = 0
            vec_n_visited_features[ifeature] = 0

        if deref(multiview_split).vec_n_constant_features.size() == 0:
            deref(multiview_split).vec_n_constant_features.resize(self.n_feature_sets)
            fill(deref(multiview_split).vec_n_constant_features.begin(), deref(multiview_split).vec_n_constant_features.end(), 0)

        cdef vector[intp_t] n_known_constants_vec = deref(multiview_split).vec_n_constant_features

        # n_total_constants = n_known_constants + n_found_constants
        cdef vector[intp_t] n_total_constants_vec = n_known_constants_vec

        _init_split(&best_split, end)

        for ifeature in range(self.n_feature_sets):
            # get the max-features for this feature-set
            max_features = self.max_features_per_set[ifeature]
            f_i = self.feature_set_ends[ifeature]

            # Sample up to max_features without replacement using a
            # Fisher-Yates-based algorithm (using the local variables `f_i` and
            # `f_j` to compute a permutation of the `features` array).
            #
            # Skip the CPU intensive evaluation of the impurity criterion for
            # features that were already detected as constant (hence not suitable
            # for good splitting) by ancestor nodes and save the information on
            # newly discovered constant features to spare computation on descendant
            # nodes.
            while ((f_i - feature_set_begin) > n_total_constants_vec[ifeature] and  # Stop early if remaining features
                                                                                    # are constant within this feature set
                    (vec_n_visited_features[ifeature] < max_features or  # At least one drawn features must be non constant
                     vec_n_visited_features[ifeature] <= vec_n_found_constants[ifeature] + vec_n_drawn_constants[ifeature])):

                vec_n_visited_features[ifeature] += 1

                # The following is loop invariant per feature set:
                # [ --- view-one ---, --- view-two --- ]
                # within each view, the features are ordered as follows:
                # [constant, known constant, newly found constant, non-constant]

                # Loop invariant: elements of features in
                # - [:n_drawn_constant[ holds drawn and known constant features;
                # - [n_drawn_constant:n_known_constant[ holds known constant
                #   features that haven't been drawn yet;
                # - [n_known_constant:n_total_constant[ holds newly found constant
                #   features;
                # - [n_total_constant:f_i[ holds features that haven't been drawn
                #   yet and aren't constant apriori.
                # - [f_i:n_features[ holds features that have been drawn
                #   and aren't constant.

                # Draw a feature at random from the feature-set
                f_j = rand_int(vec_n_drawn_constants[ifeature] + feature_set_begin, f_i - vec_n_found_constants[ifeature],
                               random_state)

                # If the drawn feature is known to be constant, swap it with the
                # last known constant feature and update the number of drawn constant features
                if f_j < n_known_constants_vec[ifeature]:
                    # f_j in the interval [n_drawn_constants, n_known_constants[
                    features[vec_n_drawn_constants[ifeature]], features[f_j] = features[f_j], features[vec_n_drawn_constants[ifeature]]

                    vec_n_drawn_constants[ifeature] += 1
                    found_new_constants = True
                    continue

                # f_j in the interval [n_known_constants, f_i - n_found_constants[
                f_j += vec_n_found_constants[ifeature]
                # f_j in the interval [n_total_constants, f_i[
                current_split.feature = features[f_j]
                self.sort_samples_and_feature_values(current_split.feature)
                n_missing = self.n_missing
                end_non_missing = end - n_missing

                if (
                    # All values for this feature are missing, or
                    end_non_missing == start or
                    # This feature is considered constant (max - min <= FEATURE_THRESHOLD)
                    feature_values[end_non_missing - 1] <= feature_values[start] + FEATURE_THRESHOLD
                ):
                    # We consider this feature constant in this case.
                    # Since finding a split among constant feature is not valuable,
                    # we do not consider this feature for splitting.
                    features[f_j], features[n_total_constants_vec[ifeature]] = features[n_total_constants_vec[ifeature]], features[f_j]

                    vec_n_found_constants[ifeature] += 1
                    n_total_constants_vec[ifeature] += 1
                    found_new_constants = True
                    continue

                f_i -= 1
                features[f_i], features[f_j] = features[f_j], features[f_i]
                has_missing = n_missing != 0
                self.criterion.init_missing(n_missing)  # initialize even when n_missing == 0

                # Evaluate all splits
                # If there are missing values, then we search twice for the most optimal split.
                # The first search will have all the missing values going to the right node.
                # The second search will have all the missing values going to the left node.
                # If there are no missing values, then we search only once for the most
                # optimal split.
                n_searches = 2 if has_missing else 1

                for i in range(n_searches):
                    missing_go_to_left = i == 1
                    self.criterion.missing_go_to_left = missing_go_to_left
                    self.criterion.reset()

                    p = start

                    while p < end_non_missing:
                        self.next_p(&p_prev, &p)

                        if p >= end_non_missing:
                            continue

                        current_split.pos = p

                        # Reject if monotonicity constraints are not satisfied
                        if (
                            self.with_monotonic_cst and
                            self.monotonic_cst[current_split.feature] != 0 and
                            not self.criterion.check_monotonicity(
                                self.monotonic_cst[current_split.feature],
                                lower_bound,
                                upper_bound,
                            )
                        ):
                            continue

                        # Reject if min_samples_leaf is not guaranteed
                        if missing_go_to_left:
                            n_left = current_split.pos - self.start + n_missing
                            n_right = end_non_missing - current_split.pos
                        else:
                            n_left = current_split.pos - self.start
                            n_right = end_non_missing - current_split.pos + n_missing
                        if self.check_presplit_conditions(&current_split, n_missing, missing_go_to_left) == 1:
                            continue

                        self.criterion.update(current_split.pos)

                        # Reject if monotonicity constraints are not satisfied
                        if (
                            self.with_monotonic_cst and
                            self.monotonic_cst[current_split.feature] != 0 and
                            not self.criterion.check_monotonicity(
                                self.monotonic_cst[current_split.feature],
                                lower_bound,
                                upper_bound,
                            )
                        ):
                            continue

                        # Reject if min_weight_leaf is not satisfied
                        if self.check_postsplit_conditions() == 1:
                            continue

                        current_proxy_improvement = self.criterion.proxy_impurity_improvement()

                        if current_proxy_improvement > best_proxy_improvement:
                            best_proxy_improvement = current_proxy_improvement
                            # sum of halves is used to avoid infinite value
                            current_split.threshold = (
                                feature_values[p_prev] / 2.0 + feature_values[p] / 2.0
                            )

                            if (
                                current_split.threshold == feature_values[p] or
                                current_split.threshold == INFINITY or
                                current_split.threshold == -INFINITY
                            ):
                                current_split.threshold = feature_values[p_prev]

                            current_split.n_missing = n_missing
                            if n_missing == 0:
                                current_split.missing_go_to_left = n_left > n_right
                            else:
                                current_split.missing_go_to_left = missing_go_to_left

                            best_split = current_split  # copy

                # Evaluate when there are missing values and all missing values goes
                # to the right node and non-missing values goes to the left node.
                if has_missing:
                    n_left, n_right = end - start - n_missing, n_missing
                    p = end - n_missing
                    missing_go_to_left = 0

                    if not (n_left < min_samples_leaf or n_right < min_samples_leaf):
                        self.criterion.missing_go_to_left = missing_go_to_left
                        self.criterion.update(p)

                        if not ((self.criterion.weighted_n_left < min_weight_leaf) or
                                (self.criterion.weighted_n_right < min_weight_leaf)):
                            current_proxy_improvement = self.criterion.proxy_impurity_improvement()

                            if current_proxy_improvement > best_proxy_improvement:
                                best_proxy_improvement = current_proxy_improvement
                                current_split.threshold = INFINITY
                                current_split.missing_go_to_left = missing_go_to_left
                                current_split.n_missing = n_missing
                                current_split.pos = p
                                best_split = current_split

            # update the feature_set_begin for the next iteration
            feature_set_begin = self.feature_set_ends[ifeature]

        # Reorganize into samples[start:best_split.pos] + samples[best_split.pos:end]
        if best_split.pos < end:
            self.partition_samples_final(
                best_split.pos,
                best_split.threshold,
                best_split.feature,
                best_split.n_missing
            )
            self.criterion.init_missing(best_split.n_missing)
            self.criterion.missing_go_to_left = best_split.missing_go_to_left

            self.criterion.reset()
            self.criterion.update(best_split.pos)
            self.criterion.children_impurity(
                &best_split.impurity_left, &best_split.impurity_right
            )
            best_split.improvement = self.criterion.impurity_improvement(
                impurity,
                best_split.impurity_left,
                best_split.impurity_right
            )

            shift_missing_values_to_left_if_required(&best_split, samples, end)

        # Reorganize constant features per feature view
        if found_new_constants:
            feature_set_begin = 0
            for ifeature in range(self.n_feature_sets):
                # Respect invariant for constant features: the original order of
                # element in features[:n_known_constants] must be preserved for sibling
                # and child nodes
                memcpy(&features[feature_set_begin], &constant_features[feature_set_begin], sizeof(intp_t) * n_known_constants_vec[ifeature])

                # Copy newly found constant features starting from [n_known_constants:n_found_constants]
                memcpy(&constant_features[n_known_constants_vec[ifeature]],
                       &features[n_known_constants_vec[ifeature]],
                       sizeof(intp_t) * vec_n_found_constants[ifeature])

                feature_set_begin = self.feature_set_ends[ifeature]

        # Return values
        if found_new_constants:
            deref(multiview_split).n_constant_features = accumulate(
                n_known_constants_vec.begin(),
                n_known_constants_vec.end(),
                0
            )
            deref(multiview_split).vec_n_constant_features = n_known_constants_vec

        deref(multiview_split).feature = best_split.feature
        deref(multiview_split).pos = best_split.pos
        deref(multiview_split).threshold = best_split.threshold
        deref(multiview_split).improvement = best_split.improvement
        deref(multiview_split).impurity_left = best_split.impurity_left
        deref(multiview_split).impurity_right = best_split.impurity_right
        # multiview_split[0] = best_split
        # n_constant_features[0] = n_total_constants
        return 0
