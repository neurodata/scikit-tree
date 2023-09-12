# distutils: language = c++
# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: initializedcheck=False
# cython: cdivision=True

cimport numpy as cnp
import numpy as np
from libc.math cimport log

cnp.import_array()

cdef DTYPE_t PI = np.pi


cdef class UnsupervisedCriterion(BaseCriterion):
    """Abstract criterion for unsupervised learning.

    This object is a copy of the Criterion class of scikit-learn, but is used
    for unsupervised learning. However, ``Criterion`` in scikit-learn was
    designed for supervised learning, where the necessary
    ingredients to compute a split point is solely with y-labels. In
    this object, we subclass and instead rely on the X-data.

    This object stores methods on how to calculate how good a split is using
    different metrics for unsupervised splitting.
    """

    def __cinit__(self):
        """Initialize attributes for unsupervised criterion.
        """
        self.start = 0
        self.pos = 0
        self.end = 0

        self.n_node_samples = 0
        self.weighted_n_node_samples = 0.0
        self.weighted_n_left = 0.0
        self.weighted_n_right = 0.0

        # Initialize count metric for current, going left and going right
        self.sum_total = 0.0
        self.sum_left = 0.0
        self.sum_right = 0.0

        self.sumsq_total = 0.0
        self.sumsq_left = 0.0
        self.sumsq_right = 0.0

    def __reduce__(self):
        return (type(self), (), self.__getstate__())

    cdef void init_feature_vec(
        self,
    ) noexcept nogil:
        """Initialize the 1D feature vector, which is used for computing criteria.

        This function is used to set a read-only memoryview of a feature
        vector. The feature vector must be set in order for criteria to be
        computed. It then keeps a running total of the feature vector from
        samples[start:end] so that way it is efficient to compute the right and
        left sums and corresponding metrics.

        Parameters
        ----------
        Xf : array-like, dtype=DTYPE_t
            The read-only memoryview 1D feature vector with (n_samples,) shape.
        """
        # also compute the sum total
        self.sum_total = 0.0
        self.sumsq_total = 0.0
        self.weighted_n_node_samples = 0.0
        cdef SIZE_t s_idx
        cdef SIZE_t p_idx

        # XXX: this can be further optimized by computing a cumulative sum hash map of the sum_total and sumsq_total
        # and then update will never have to iterate through even
        cdef DOUBLE_t w = 1.0

        # cdef SIZE_t prev_s_idx = -1
        # self.cumsum_of_squares_map[prev_s_idx] = 0.0
        # self.cumsum_map[prev_s_idx] = 0.0
        # self.cumsum_weights_map[prev_s_idx] = 0.0

        for p_idx in range(self.start, self.end):
            s_idx = self.sample_indices[p_idx]

            # w is originally set to be 1.0, meaning that if no sample weights
            # are given, the default weight of each sample is 1.0.
            if self.sample_weight is not None:
                w = self.sample_weight[s_idx]

            self.sum_total += self.feature_values[s_idx] * w
            self.sumsq_total += self.feature_values[s_idx] * self.feature_values[s_idx] * w * w
            self.weighted_n_node_samples += w

        # Reset to pos=start
        self.reset()

    cdef int init(
        self,
        const DTYPE_t[:] feature_values,
        const DOUBLE_t[:] sample_weight,
        double weighted_n_samples,
        const SIZE_t[:] sample_indices,
    ) except -1 nogil:
        """Initialize the unsuperivsed criterion.

        Returns -1 in case of failure to allocate memory (and raise MemoryError)
        or 0 otherwise.

        Parameters
        ----------
        feature_values : array-like, dtype=DTYPE_t
            The memoryview 1D feature vector with (n_samples,) shape.
        sample_weight : array-like, dtype=DOUBLE_t
            The weight of each sample (i.e. row of X).
        weighted_n_samples : double
            The total weight of all sample_indices.
        sample_indices : array-like, dtype=SIZE_t
            A mask on the sample_indices, showing which ones we want to use
        """
        self.feature_values = feature_values
        self.sample_weight = sample_weight
        self.weighted_n_samples = weighted_n_samples
        self.sample_indices = sample_indices

        return 0

    cdef int reset(self) except -1 nogil:
        """Reset the criterion at pos=start.

        Returns -1 in case of failure to allocate memory (and raise MemoryError)
        or 0 otherwise.
        """
        self.pos = self.start

        self.weighted_n_left = 0.0
        self.weighted_n_right = self.weighted_n_node_samples
        self.sum_left = 0.0
        self.sum_right = self.sum_total

        self.sumsq_left = 0.0
        self.sumsq_right = self.sumsq_total
        return 0

    cdef int reverse_reset(self) except -1 nogil:
        """Reset the criterion at pos=end.

        Returns -1 in case of failure to allocate memory (and raise MemoryError)
        or 0 otherwise.
        """
        self.pos = self.end

        self.weighted_n_left = self.weighted_n_node_samples
        self.weighted_n_right = 0.0
        self.sum_right = 0.0
        self.sum_left = self.sum_total

        self.sumsq_right = 0.0
        self.sumsq_left = self.sumsq_total
        return 0

    cdef int update(
        self,
        SIZE_t new_pos
    ) except -1 nogil:
        """Updated statistics by moving sample_indices[pos:new_pos] to the left child.

        Returns -1 in case of failure to allocate memory (and raise MemoryError)
        or 0 otherwise.

        Parameters
        ----------
        new_pos : SIZE_t
            The new ending position for which to move sample_indices from the right
            child to the left child.
        """
        cdef SIZE_t pos = self.pos
        cdef SIZE_t end = self.end

        cdef const SIZE_t[:] sample_indices = self.sample_indices
        cdef const DOUBLE_t[:] sample_weight = self.sample_weight

        cdef SIZE_t i
        cdef SIZE_t p
        cdef DOUBLE_t w = 1.0

        # Update statistics up to new_pos
        #
        # Given that
        #   sum_left[x] +  sum_right[x] = sum_total[x]
        # and that sum_total is known, we are going to update
        # sum_left from the direction that require the least amount
        # of computations, i.e. from pos to new_pos or from end to new_pos.
        if (new_pos - pos) <= (end - new_pos):
            for p in range(pos, new_pos):
                i = sample_indices[p]

                if sample_weight is not None:
                    w = sample_weight[i]

                # accumulate the values of the feature vectors weighted
                # by the sample weight
                self.sum_left += self.feature_values[i] * w
                self.sumsq_left += self.feature_values[i] * self.feature_values[i] * w * w
                # keep track of the weighted count of each sample
                self.weighted_n_left += w
        else:
            self.reverse_reset()

            for p in range(end - 1, new_pos - 1, -1):
                i = sample_indices[p]

                if sample_weight is not None:
                    w = sample_weight[i]

                self.sum_left -= self.feature_values[i] * w
                self.sumsq_left -= self.feature_values[i] * self.feature_values[i] * w * w
                self.weighted_n_left -= w

        # Update right part statistics
        self.weighted_n_right = (self.weighted_n_node_samples -
                                 self.weighted_n_left)
        self.sum_right = self.sum_total - self.sum_left
        self.sumsq_right = self.sumsq_total - self.sumsq_left
        self.pos = new_pos
        return 0

    cdef void node_value(
        self,
        double* dest
    ) noexcept nogil:
        """Set the node value with sum_total and save it into dest.

        Parameters
        ----------
        dest : double pointer
            The memory address which we will save the node value into.
        """
        # set values at the address pointer is pointing to with the total value
        dest[0] = self.sum_total

    cdef void set_sample_pointers(
        self,
        SIZE_t start,
        SIZE_t end
    ) noexcept nogil:
        """Set sample pointers in the criterion.

        Set given start and end sample_indices for a given node.

        Parameters
        ----------
        start : SIZE_t
            The start sample pointer.
        end : SIZE_t
            The end sample pointer.
        """
        self.n_node_samples = end - start
        self.start = start
        self.end = end


cdef class TwoMeans(UnsupervisedCriterion):
    r"""Two means split impurity.

    The two means split finds the cutpoint that minimizes the one-dimensional
    2-means objective, which is finding the cutoff point where the total variance
    from cluster 1 and cluster 2 are minimal.

    The mathematical optimization problem is to find the cutoff index ``s``,
    which is called 'pos' in scikit-learn.

        \min_s \sum_{i=1}^s (x_i - \hat{\mu}_1)^2 + \sum_{i=s+1}^N (x_i - \hat{\mu}_2)^2

    where x is a N-dimensional feature vector, N is the number of sample_indices and
    the \mu terms are the estimated means of each cluster 1 and 2.

    The variance of the node, left child and right child is computed by keeping track of
    `sum_total`, `sum_left` and `sum_right`, which are the sums of a feature vector at
    varying split points.

    Weighted Mean and Variance
    --------------------------
    Since we allow `sample_weights` to be passed, then we optionally allow for us
    to compute a weighted sample mean and weighted sample variance. The weighted
    sample variance has two approaches to compute an unbiased estimate. The first
    is using "frequencies" and the second is using "reliability" weights. Currently,
    we have implemented the frequencies approach.

    # TODO: implement reliability weighting

    Node-Wise Feature Generation
    ----------------------------
    URerF doesn't choose split points in the original feature space
    It follows the random projection framework

    \tilde{X}= A^T X'

    where, A is p x d matrix distributed as f_A, where f_A is the
    projection distribution and d is the dimensionality of the
    projected space. A is generated by randomly sampling from
    {-1,+1} lpd times, then distributing these values uniformly
    at random in A. l parameter is used to control the sparsity of A
    and is set to 1/20.

    Each of the d rows \tilde{X}[i; :], i \in {1,2,...d} is then
    inspected for the best split point. The optimal split point and
    splitting dimension are chosen according to which point/dimension
    pair minimizes the splitting criteria described in the following
    section
    """
    cdef double node_impurity(
        self
    ) noexcept nogil:
        """Evaluate the impurity of the current node.

        Evaluate the TwoMeans criterion impurity as variance of the current node,
        i.e. the variance of Xf[sample_indices[start:end]]. The smaller the impurity the
        better.
        """
        cdef double impurity

        # then compute the impurity as the variance
        impurity = self.fast_variance(self.weighted_n_node_samples, self.sumsq_total, self.sum_total)
        return impurity

    cdef void children_impurity(
        self,
        double* impurity_left,
        double* impurity_right
    ) noexcept nogil:
        """Evaluate the impurity in children nodes.

        i.e. the impurity of the left child (sample_indices[start:pos]) and the
        impurity the right child (sample_indices[pos:end]).

        where the impurity of two children nodes are essentially the variances of
        two nodes:

        - left_variance = left_weight * left_impurity / n_sample_of_left_child
        - right_variance = right_weight * right_impurity / n_sample_of_right_child

        Parameters
        ----------
        impurity_left : double pointer
            The memory address to save the impurity of the left node
        impurity_right : double pointer
            The memory address to save the impurity of the right node
        """
        # set values at the address pointer is pointing to with the variance
        # of the left and right child
        impurity_left[0] = self.fast_variance(self.weighted_n_left, self.sumsq_left, self.sum_left)
        impurity_right[0] = self.fast_variance(self.weighted_n_right, self.sumsq_right, self.sum_right)

    cdef inline double fast_variance(
        self,
        double weighted_n_node_samples,
        double sumsq_total,
        double sum_total
    ) noexcept nogil:
        return (1. / weighted_n_node_samples) * \
            ((sumsq_total) - (1. / weighted_n_node_samples) * (sum_total * sum_total))


cdef class FastBIC(TwoMeans):
    r"""Fast-BIC split criterion

    The Bayesian Information Criterion (BIC) is a popular model seleciton
    criteria that is based on the log likelihood of the model given data.
    MClust-BIC is a method in the R-package 'mclust' that uses BIC as a splitting
    criterion.
    See: https://stats.stackexchange.com/questions/237220/mclust-model-selection

    Fast-BIC is a method that combines the speed of the two-means clustering
    method with the model flexibility of Mclust-BIC. It sorts data for each
    feature and tries all possible splits to assign data points to one of
    two Gaussian distributions based on their position relative to the split.
    The parameters for each cluster are estimated using maximum likelihood
    estimation (MLE). The method performs hard clustering rather than soft
    clustering like in GMM, resulting in a simpler calculation of the likelihood.

    \hat{L} = \sum_{n=1}^s[\log\hat{\pi}_1+\log{\mathcal{N}(x_n;\hat{\mu}_1,\hat{\sigma}_1^2)}]
    + \sum_{n=s+1}^N[\log\hat{\pi}_2+\log{\mathcal{N}(x_n;\hat{\mu}_2,\hat{\sigma}_2^2)}]

    where the prior, mean, and variance are defined as follows, respectively:

    - \hat{\pi} = \frac{s}{N}
    - \hat{\mu} = \frac{1}{s}\sum_{n\le s}{x_n},
    - \hat{\sigma}^2 = \frac{1}{s}\sum_{n\le s}{||x_n-\hat{\mu_j}||^2}

    Fast-BIC is guaranteed to obtain the global maximum likelihood estimator.
    Additionally, Fast-BIC is substantially faster than the traditional BIC method.

    Reference: https://arxiv.org/abs/1907.02844
    """
    cdef inline double bic_cluster(
        self,
        SIZE_t n_samples,
        double variance
    ) noexcept nogil:
        """Help compute the BIC from assigning to a specific cluster.

        Parameters
        ----------
        n_samples : SIZE_t
            The number of samples assigned cluster.
        variance : double
            The plug-in variance for assigning to specific cluster.

        Notes
        -----
        Computes the following:

        :math:`-2 * (n_i log(w_i) - n_i/2 log(2 \\pi \\sigma_i^2))

        where :math:`n_i` is the number of samples assigned to cluster i,
        :math:`w_i` is the probability of choosing cluster i at random (or also known
        as the prior) and :math:`\\sigma_i^2` is the variance estimate for cluster i.

        Note that :math:`\\sigma_i^2` in the Fast-BIC derivation may be the
        variance of the cluster itself, or the estimated combined variance
        from both clusters.
        """
        # chances of choosing the cluster based on how many samples are hard-assigned to cluster
        # i.e. the prior
        # cast to double, so we do not round to integers
        cdef double w_cluster = (n_samples + 0.0) / self.n_node_samples

        # add to prevent taking log of 0 when there is a degenerate cluster (i.e. single sample, or no variance)
        return -2. * (n_samples * log(w_cluster) + 0.5 * n_samples * log(2. * PI * variance + 1.e-7))

    cdef double node_impurity(
        self
    ) noexcept nogil:
        """Evaluate the impurity of the current node.

        Evaluate the FastBIC criterion impurity as estimated maximum log likelihood.
        This is the maximum likelihood given prior, mean, and variance at s number of samples
        Namely, this is the maximum likelihood of Xf[sample_indices[start:end]].
        The smaller the impurity the better.
        """
        cdef double variance
        cdef double impurity

        # then compute the variance of the cluster
        variance = self.fast_variance(self.weighted_n_node_samples, self.sumsq_total, self.sum_total)

        # Compute the BIC of the current set of samples
        # Note: we do not compute the BIC_diff_var and BIC_same_var because
        # they are equivalent in the single cluster setting
        impurity = self.bic_cluster(self.n_node_samples, variance)
        return impurity

    cdef void children_impurity(
        self,
        double* impurity_left,
        double* impurity_right
    ) noexcept nogil:
        """Evaluate the impurity in children nodes.

        i.e. the impurity of the left child (sample_indices[start:pos]) and the
        impurity the right child (sample_indices[pos:end]).

        Parameters
        ----------
        impurity_left : double pointer
            The memory address to save the impurity of the left node
        impurity_right : double pointer
            The memory address to save the impurity of the right node
        """
        cdef SIZE_t pos = self.pos
        cdef SIZE_t start = self.start
        cdef SIZE_t end = self.end
        cdef SIZE_t n_samples_left, n_samples_right

        cdef double variance_left, variance_right, variance_comb
        cdef double BIC_diff_var_left, BIC_diff_var_right
        cdef double BIC_same_var_left, BIC_same_var_right
        cdef double BIC_same_var, BIC_diff_var

        # number of samples of left and right
        n_samples_left = pos - start
        n_samples_right = end - pos

        # compute the estimated variance of the left and right children
        variance_left = self.fast_variance(self.weighted_n_left, self.sumsq_left, self.sum_left)
        variance_right = self.fast_variance(self.weighted_n_right, self.sumsq_right, self.sum_right)

        # compute the estimated combined variance
        variance_comb = (self.sumsq_left + self.sumsq_right) / (self.weighted_n_left + self.weighted_n_right)

        # Compute the BIC using different variances for left and right
        BIC_diff_var_left = self.bic_cluster(n_samples_left, variance_left)
        BIC_diff_var_right = self.bic_cluster(n_samples_right, variance_right)

        # Compute the BIC using different variances for left and right
        BIC_same_var_left = self.bic_cluster(n_samples_left, variance_comb)
        BIC_same_var_right = self.bic_cluster(n_samples_right, variance_comb)
        BIC_same_var = BIC_same_var_left - BIC_same_var_right
        BIC_diff_var = BIC_diff_var_left - BIC_diff_var_right

        # choose the BIC formulation that gives us the smallest values
        # (i.e. min of (BIC_diff, BIC_same) in the paper) and then
        # assign the left and right child BIC values by reference
        if BIC_diff_var < BIC_same_var:
            impurity_left[0] = -BIC_diff_var_left
            impurity_right[0] = -BIC_diff_var_right
        else:
            impurity_left[0] = -BIC_same_var_left
            impurity_right[0] = -BIC_same_var_right
