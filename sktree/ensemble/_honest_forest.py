# Authors: Ronan Perry, Sambit Panda, Haoyin Xu
# Adopted from: https://github.com/neurodata/honest-forests

import threading
from numbers import Integral
from warnings import catch_warnings, simplefilter, warn

import numpy as np
from joblib import Parallel, delayed
from scipy.sparse import issparse
from sklearn.base import _fit_context
from sklearn.ensemble._base import _partition_estimators
from sklearn.ensemble._hist_gradient_boosting.binning import _BinMapper
from sklearn.exceptions import DataConversionWarning
from sklearn.utils import check_random_state, compute_sample_weight
from sklearn.utils._openmp_helpers import _openmp_effective_n_threads
from sklearn.utils._param_validation import Interval, RealNotInt
from sklearn.utils.multiclass import type_of_target
from sklearn.utils.validation import _check_sample_weight, check_is_fitted

from .._lib.sklearn.ensemble._forest import ForestClassifier
from .._lib.sklearn.tree._tree import DOUBLE, DTYPE
from ..tree import HonestTreeClassifier
from ._extensions import ForestClassifierMixin, _generate_sample_indices, _get_n_samples_bootstrap


def _parallel_build_trees(
    tree,
    bootstrap,
    X,
    y,
    sample_weight,
    tree_idx,
    n_trees,
    verbose=0,
    class_weight=None,
    n_samples_bootstrap=None,
    missing_values_in_feature_mask=None,
    classes=None,
):
    """
    Private function used to fit a single tree in parallel.

    XXX: this is copied over from scikit-learn and modified to allow sampling with
    and without replacement given ``bootstrap``.
    """
    if verbose > 1:
        print("building tree %d of %d" % (tree_idx + 1, n_trees))

    n_samples = X.shape[0]
    if sample_weight is None:
        curr_sample_weight = np.ones((n_samples,), dtype=np.float64)
    else:
        curr_sample_weight = sample_weight.copy()

    indices = _generate_sample_indices(tree.random_state, n_samples, n_samples_bootstrap, bootstrap)
    sample_counts = np.bincount(indices, minlength=n_samples)
    curr_sample_weight *= sample_counts

    if class_weight == "subsample":
        with catch_warnings():
            simplefilter("ignore", DeprecationWarning)
            curr_sample_weight *= compute_sample_weight("auto", y, indices=indices)
    elif class_weight == "balanced_subsample":
        curr_sample_weight *= compute_sample_weight("balanced", y, indices=indices)

    tree._fit(
        X,
        y,
        sample_weight=curr_sample_weight,
        check_input=False,
        missing_values_in_feature_mask=missing_values_in_feature_mask,
        classes=classes,
    )

    return tree


class HonestForestClassifier(ForestClassifier, ForestClassifierMixin):
    """
    A forest classifier with honest leaf estimates.

    Parameters
    ----------
    n_estimators : int, default=100
        The number of trees in the forest.

    criterion : {"gini", "entropy"}, default="gini"
        The function to measure the quality of a split. Supported criteria are
        "gini" for the Gini impurity and "entropy" for the information gain.
        Note: this parameter is tree-specific.

    splitter : {"best", "random"}, default="best"
        The strategy used to choose the split at each node. Supported
        strategies are "best" to choose the best split and "random" to choose
        the best random split.

    max_depth : int, default=None
        The maximum depth of the tree. If None, then nodes are expanded until
        all leaves are pure or until all leaves contain less than
        min_samples_split samples.

    min_samples_split : int or float, default=2
        The minimum number of samples required to split an internal node:

        - If int, then consider `min_samples_split` as the minimum number.
        - If float, then `min_samples_split` is a fraction and
          `ceil(min_samples_split * n_samples)` are the minimum
          number of samples for each split.

    min_samples_leaf : int or float, default=1
        The minimum number of samples required to be at a leaf node.
        A split point at any depth will only be considered if it leaves at
        least ``min_samples_leaf`` training samples in each of the left and
        right branches.  This may have the effect of smoothing the model,
        especially in regression.

        - If int, then consider `min_samples_leaf` as the minimum number.
        - If float, then `min_samples_leaf` is a fraction and
          `ceil(min_samples_leaf * n_samples)` are the minimum
          number of samples for each node.

    min_weight_fraction_leaf : float, default=0.0
        The minimum weighted fraction of the sum total of weights (of all
        the input samples) required to be at a leaf node. Samples have
        equal weight when sample_weight is not provided.

    max_features : {"sqrt", "log2", None}, int or float, default=0.3
        The number of features to consider when looking for the best split:

        - If int, then consider `max_features` features at each split.
        - If float, then `max_features` is a fraction and
          `round(max_features * n_features)` features are considered at each
          split.
        - If "auto", then `max_features=sqrt(n_features)`.
        - If "sqrt", then `max_features=sqrt(n_features)`.
        - If "log2", then `max_features=log2(n_features)`.
        - If None, then `max_features=n_features`.

        Note: the search for a split does not stop until at least one
        valid partition of the node samples is found, even if it requires to
        effectively inspect more than ``max_features`` features.

    max_leaf_nodes : int, default=None
        Grow trees with ``max_leaf_nodes`` in best-first fashion.
        Best nodes are defined as relative reduction in impurity.
        If None then unlimited number of leaf nodes.

    min_impurity_decrease : float, default=0.0
        A node will be split if this split induces a decrease of the impurity
        greater than or equal to this value.

        The weighted impurity decrease equation is the following::

            N_t / N * (impurity - N_t_R / N_t * right_impurity
                                - N_t_L / N_t * left_impurity)

        where ``N`` is the total number of samples, ``N_t`` is the number of
        samples at the current node, ``N_t_L`` is the number of samples in the
        left child, and ``N_t_R`` is the number of samples in the right child.

        ``N``, ``N_t``, ``N_t_R`` and ``N_t_L`` all refer to the weighted sum,
        if ``sample_weight`` is passed.

    bootstrap : bool, default=True
        Whether bootstrap samples are used when building trees. If False, the
        whole dataset is used to build each tree.

        When bootstrap is True, each tree bootstrap samples the dataset, and then
        the unique indices are split in half, where one half is used to train
        the structure of the tree and one half is used to train the leaves of the tree.
        The remaining sample indices are considered "out of bag".

    oob_score : bool, default=False
        Whether to use out-of-bag samples to estimate the generalization score.
        Only available if bootstrap=True.

    n_jobs : int, default=None
        The number of jobs to run in parallel. :meth:`fit`, :meth:`predict`,
        :meth:`decision_path` and :meth:`apply` are all parallelized over the
        trees. ``None`` means 1 unless in a `joblib.parallel_backend`
        context. ``-1`` means using all processors. See :term:`Glossary
        <n_jobs>` for more details.

    random_state : int, RandomState instance or None, default=None
        Controls both the randomness of the bootstrapping of the samples used
        when building trees (if ``bootstrap=True``) and the sampling of the
        features to consider when looking for the best split at each node
        (if ``max_features < n_features``).
        See :term:`Glossary <random_state>` for details.

    verbose : int, default=0
        Controls the verbosity when fitting and predicting.

    warm_start : bool, default=False
        When set to ``True``, reuse the solution of the previous call to fit
        and add more estimators to the ensemble, otherwise, just fit a whole
        new forest. See :term:`the Glossary <warm_start>`.

    class_weight : {"balanced", "balanced_subsample"}, dict or list of dicts, \
            default=None
        Weights associated with classes in the form ``{class_label: weight}``.
        If not given, all classes are supposed to have weight one. For
        multi-output problems, a list of dicts can be provided in the same
        order as the columns of y.

        Note that for multioutput (including multilabel) weights should be
        defined for each class of every column in its own dict. For example,
        for four-class multilabel classification weights should be
        [{0: 1, 1: 1}, {0: 1, 1: 5}, {0: 1, 1: 1}, {0: 1, 1: 1}] instead of
        [{1:1}, {2:5}, {3:1}, {4:1}].

        The "balanced" mode uses the values of y to automatically adjust
        weights inversely proportional to class frequencies in the input data
        as ``n_samples / (n_classes * np.bincount(y))``

        The "balanced_subsample" mode is the same as "balanced" except that
        weights are computed based on the bootstrap sample for every tree
        grown.

        For multi-output, the weights of each column of y will be multiplied.

        Note that these weights will be multiplied with sample_weight (passed
        through the fit method) if sample_weight is specified.

    ccp_alpha : non-negative float, default=0.0
        Complexity parameter used for Minimal Cost-Complexity Pruning. The
        subtree with the largest cost complexity that is smaller than
        ``ccp_alpha`` will be chosen. By default, no pruning is performed. See
        :ref:`minimal_cost_complexity_pruning` for details.

    max_samples : int or float, default=1.6
        If bootstrap is True, the number of samples to draw from X
        to train each base tree estimator with replacement.
        If bootstrap is False, then this will subsample
        the dataset without replacement.

        - If None (default), then draw `X.shape[0]` samples.
        - If int, then draw `max_samples` samples.
        - If float, then draw `max_samples * X.shape[0]` samples.

    honest_prior : {"ignore", "uniform", "empirical"}, default="empirical"
        Method for dealing with empty leaves during evaluation of a test
        sample. If "ignore", the tree is ignored. If "uniform", the prior tree
        posterior is 1/(number of classes). If "empirical", the prior tree
        posterior is the relative class frequency in the voting subsample.
        If all trees are ignored, the empirical estimate is returned.

    honest_fraction : float, default=0.5
        Fraction of training samples used for estimates in the trees. The
        remaining samples will be used to learn the tree structure. A larger
        fraction creates shallower trees with lower variance estimates.

    tree_estimator : object, default=None
        Type of decision tree classifier to use. By default `None`, which
        defaults to `sktree.tree.DecisionTreeClassifier`. Note
        that one MUST use trees imported from the `sktree.tree`
        API namespace rather than from `sklearn.tree`.

    stratify : bool, default=True
        Whether or not to stratify sample when considering structure and leaf indices.
        By default False.

    Attributes
    ----------
    estimator : sktree.tree.HonestTreeClassifier
        The child estimator template used to create the collection of fitted
        sub-estimators.

    estimators_ : list of sktree.tree.HonestTreeClassifier
        The collection of fitted sub-estimators.

    classes_ : ndarray of shape (n_classes,) or a list of such arrays
        The classes labels (single output problem), or a list of arrays of
        class labels (multi-output problem).

    n_classes_ : int or list
        The number of classes (single output problem), or a list containing the
        number of classes for each output (multi-output problem).

    n_features_ : int
        The number of features when ``fit`` is performed.

    n_features_in_ : int
        Number of features seen during :term:`fit`.

    feature_names_in_ : ndarray of shape (`n_features_in_`,)
        Names of features seen during :term:`fit`. Defined only when `X`
        has feature names that are all strings.

    n_outputs_ : int
        The number of outputs when ``fit`` is performed.

    feature_importances_ : ndarray of shape (n_features,)
        The impurity-based feature importances.
        The higher, the more important the feature.
        The importance of a feature is computed as the (normalized)
        total reduction of the criterion brought by that feature.  It is also
        known as the Gini importance.

        Warning: impurity-based feature importances can be misleading for
        high cardinality features (many unique values). See
        :func:`sklearn.inspection.permutation_importance` as an alternative.

    oob_score_ : float
        Score of the training dataset obtained using an out-of-bag estimate.
        This attribute exists only when ``oob_score`` is True.

    oob_decision_function_ : ndarray of shape (n_samples, n_classes) or \
            (n_samples, n_classes, n_outputs)
        Decision function computed with out-of-bag estimate on the training
        set. If n_estimators is small it might be possible that a data point
        was never left out during the bootstrap. In this case,
        `oob_decision_function_` might contain NaN. This attribute exists
        only when ``oob_score`` is True.

    honest_decision_function_ : ndarray of shape (n_samples, n_classes) or \
            (n_samples, n_classes, n_outputs)
        Decision function computed on each sample, including only the trees
        for which it was in the honest subsample. It is possible that a sample
        is never in the honest subset in which case `honest_decision_function_`
        might contain NaN.

    structure_indices_ : list of lists, shape=(n_estimators, n_structure)
        Indices of training samples used to learn the structure.

    honest_indices_ : list of lists, shape=(n_estimators, n_honest)
        Indices of training samples used to learn leaf estimates.

    oob_samples_ : list of lists, shape=(n_estimators, n_samples_bootstrap)
        The indices of training samples that are "out-of-bag". Only used
        if ``bootstrap=True``.

    Notes
    -----
    The default values for the parameters controlling the size of the trees
    (e.g. ``max_depth``, ``min_samples_leaf``, etc.) lead to fully grown and
    unpruned trees which can potentially be very large on some data sets. To
    reduce memory consumption, the complexity and size of the trees should be
    controlled by setting those parameter values.

    The features are always randomly permuted at each split. Therefore,
    the best found split may vary, even with the same training data,
    ``max_features=n_features`` and ``bootstrap=False``, if the improvement
    of the criterion is identical for several splits enumerated during the
    search of the best split. To obtain a deterministic behaviour during
    fitting, ``random_state`` has to be fixed.

    Honesty is a feature of trees that enables unbiased estimates of confidence
    intervals. The default implementation here is using double sampling to
    implement honesty. The amount of samples used for learning split nodes vs
    leaf nodes is controlled by the ``honest_fraction`` parameter. In order to
    enforce honesty, but also enable the tree to have access to all y labels,
    we set sample_weight to 0 for a random subset of samples. This results in
    inefficiency when building trees using a greedy splitter as we still sort
    over all values of X. We recommend using propensity trees if you are
    computing causal effects.

    This forest classifier is a "meta-estimator" because any tree model can
    be used in the classification process, while enabling honesty separates
    the data used for split and leaf nodes.

    References
    ----------
    .. [1] L. Breiman, "Random Forests", Machine Learning, 45(1), 5-32, 2001.

    .. [2] S. Athey, J. Tibshirani, and S. Wager. "Generalized
            Random Forests", Annals of Statistics, 2019.

    Examples
    --------
    >>> from honest_forests.estimators import HonestForestClassifier
    >>> from sklearn.datasets import make_classification
    >>> X, y = make_classification(n_samples=1000, n_features=4,
    ...                            n_informative=2, n_redundant=0,
    ...                            random_state=0, shuffle=False)
    >>> clf = HonestForestClassifier(
    >>>        max_depth=2,
    >>>        random_state=0,
    >>>        tree_estimator=ObliqueDecisionTreeClassifier())
    >>> clf.fit(X, y)
    HonestForestClassifier(...)
    >>> print(clf.predict([[0, 0, 0, 0]]))
    [1]
    """

    _parameter_constraints: dict = {
        **ForestClassifier._parameter_constraints,
    }
    _parameter_constraints.pop("max_samples")
    _parameter_constraints["max_samples"] = [
        None,
        Interval(RealNotInt, 0.0, None, closed="right"),
        Interval(Integral, 1, None, closed="left"),
    ]

    def __init__(
        self,
        n_estimators=100,
        criterion="gini",
        splitter="best",
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        min_weight_fraction_leaf=0.0,
        max_features=0.3,
        max_leaf_nodes=None,
        min_impurity_decrease=0.0,
        bootstrap=True,
        oob_score=False,
        n_jobs=None,
        random_state=None,
        verbose=0,
        warm_start=False,
        class_weight=None,
        ccp_alpha=0.0,
        max_samples=1.6,
        honest_prior="empirical",
        honest_fraction=0.5,
        tree_estimator=None,
        stratify=True,
    ):
        super().__init__(
            estimator=HonestTreeClassifier(),
            n_estimators=n_estimators,
            estimator_params=(
                "criterion",
                "splitter",
                "max_depth",
                "min_samples_split",
                "min_samples_leaf",
                "min_weight_fraction_leaf",
                "max_features",
                "max_leaf_nodes",
                "min_impurity_decrease",
                "random_state",
                "ccp_alpha",
                "tree_estimator",
                "honest_fraction",
                "honest_prior",
                "stratify",
            ),
            bootstrap=bootstrap,
            oob_score=oob_score,
            n_jobs=n_jobs,
            random_state=random_state,
            verbose=verbose,
            warm_start=warm_start,
            class_weight=class_weight,
            max_samples=max_samples,
        )
        self.criterion = criterion
        self.splitter = splitter
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.min_weight_fraction_leaf = min_weight_fraction_leaf
        self.max_features = max_features
        self.max_leaf_nodes = max_leaf_nodes
        self.min_impurity_decrease = min_impurity_decrease
        self.ccp_alpha = ccp_alpha
        self.honest_fraction = honest_fraction
        self.honest_prior = honest_prior
        self.tree_estimator = tree_estimator
        self.stratify = stratify

    @_fit_context(prefer_skip_nested_validation=True)
    def fit(self, X, y, sample_weight=None, classes=None):
        """
        Build a forest of trees from the training set (X, y).

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            The training input samples. Internally, its dtype will be converted
            to ``dtype=np.float32``. If a sparse matrix is provided, it will be
            converted into a sparse ``csc_matrix``.

        y : array-like of shape (n_samples,) or (n_samples, n_outputs)
            The target values (class labels in classification, real numbers in
            regression).

        sample_weight : array-like of shape (n_samples,), default=None
            Sample weights. If None, then samples are equally weighted. Splits
            that would create child nodes with net zero or negative weight are
            ignored while searching for a split in each node. In the case of
            classification, splits are also ignored if they would result in any
            single class carrying a negative weight in either child node.

        classes : array-like of shape (n_classes,), default=None
            List of all the classes that can possibly appear in the y vector.

        Returns
        -------
        self : object
            Fitted estimator.
        """
        # XXX: This entire function is a copy of what is in scikit-learn
        # with the exception of:
        # - _get_n_samples_bootstrap is a re-defined function to allow higher
        #   max_samples

        MAX_INT = np.iinfo(np.int32).max
        # Validate or convert input data
        if issparse(y):
            raise ValueError("sparse multilabel-indicator for y is not supported.")

        X, y = self._validate_data(
            X,
            y,
            multi_output=True,
            accept_sparse="csc",
            dtype=DTYPE,
            force_all_finite=False,
        )
        # _compute_missing_values_in_feature_mask checks if X has missing values and
        # will raise an error if the underlying tree base estimator can't handle missing
        # values. Only the criterion is required to determine if the tree supports
        # missing values.
        estimator = type(self.estimator)(criterion=self.criterion)
        missing_values_in_feature_mask = estimator._compute_missing_values_in_feature_mask(
            X, estimator_name=self.__class__.__name__
        )

        if sample_weight is not None:
            sample_weight = _check_sample_weight(sample_weight, X)

        if issparse(X):
            # Pre-sort indices to avoid that each individual tree of the
            # ensemble sorts the indices.
            X.sort_indices()

        y = np.atleast_1d(y)
        if y.ndim == 2 and y.shape[1] == 1:
            warn(
                (
                    "A column-vector y was passed when a 1d array was"
                    " expected. Please change the shape of y to "
                    "(n_samples,), for example using ravel()."
                ),
                DataConversionWarning,
                stacklevel=2,
            )

        if y.ndim == 1:
            # reshape is necessary to preserve the data contiguity against vs
            # [:, np.newaxis] that does not.
            y = np.reshape(y, (-1, 1))

        if self.criterion == "poisson":
            if np.any(y < 0):
                raise ValueError(
                    "Some value(s) of y are negative which is "
                    "not allowed for Poisson regression."
                )
            if np.sum(y) <= 0:
                raise ValueError(
                    "Sum of y is not strictly positive which "
                    "is necessary for Poisson regression."
                )

        self._n_samples, self.n_outputs_ = y.shape

        y, expanded_class_weight = self._validate_y_class_weight(y, classes=classes)

        if getattr(y, "dtype", None) != DOUBLE or not y.flags.contiguous:
            y = np.ascontiguousarray(y, dtype=DOUBLE)

        if expanded_class_weight is not None:
            if sample_weight is not None:
                sample_weight = sample_weight * expanded_class_weight
            else:
                sample_weight = expanded_class_weight

        # compute the number of samples we want to sample for each tree
        # possibly without replacement
        n_samples_bootstrap = _get_n_samples_bootstrap(
            n_samples=X.shape[0], max_samples=self.max_samples
        )

        self._n_samples_bootstrap = n_samples_bootstrap

        self._validate_estimator()

        if not self.bootstrap and self.oob_score:
            raise ValueError("Out of bag estimation only available if bootstrap=True")

        random_state = check_random_state(self.random_state)

        if not self.warm_start or not hasattr(self, "estimators_"):
            # Free allocated memory, if any
            self.estimators_ = []

        n_more_estimators = self.n_estimators - len(self.estimators_)

        if self.max_bins is not None:
            # `_openmp_effective_n_threads` is used to take cgroups CPU quotes
            # into account when determine the maximum number of threads to use.
            n_threads = _openmp_effective_n_threads()

            # Bin the data
            # For ease of use of the API, the user-facing GBDT classes accept the
            # parameter max_bins, which doesn't take into account the bin for
            # missing values (which is always allocated). However, since max_bins
            # isn't the true maximal number of bins, all other private classes
            # (binmapper, histbuilder...) accept n_bins instead, which is the
            # actual total number of bins. Everywhere in the code, the
            # convention is that n_bins == max_bins + 1
            n_bins = self.max_bins + 1  # + 1 for missing values
            self._bin_mapper = _BinMapper(
                n_bins=n_bins,
                # is_categorical=self.is_categorical_,
                known_categories=None,
                random_state=random_state,
                n_threads=n_threads,
            )

            # XXX: in order for this to work with the underlying tree submodule's Cython
            # code, we need to convert this into the original data's DTYPE because
            # the Cython code assumes that `DTYPE` is used.
            # The proper implementation will be a lot more complicated and should be
            # tackled once scikit-learn has finalized their inclusion of missing data
            # and categorical support for decision trees
            X = self._bin_data(X, is_training_data=True)  # .astype(DTYPE)
        else:
            self._bin_mapper = None

        if n_more_estimators < 0:
            raise ValueError(
                "n_estimators=%d must be larger or equal to "
                "len(estimators_)=%d when warm_start==True"
                % (self.n_estimators, len(self.estimators_))
            )

        elif n_more_estimators == 0:
            warn("Warm-start fitting without increasing n_estimators does not " "fit new trees.")
        else:
            if self.warm_start and len(self.estimators_) > 0:
                # We draw from the random state to get the random state we
                # would have got if we hadn't used a warm_start.
                random_state.randint(MAX_INT, size=len(self.estimators_))

            trees = self._construct_trees(
                X,
                y,
                sample_weight,
                random_state,
                n_samples_bootstrap,
                missing_values_in_feature_mask,
                classes,
                n_more_estimators,
            )

            # Collect newly grown trees
            self.estimators_.extend(trees)

        if self.oob_score and (n_more_estimators > 0 or not hasattr(self, "oob_score_")):
            y_type = type_of_target(y)
            if y_type == "unknown" or (
                self._estimator_type == "classifier" and y_type == "multiclass-multioutput"
            ):
                # FIXME: we could consider to support multiclass-multioutput if
                # we introduce or reuse a constructor parameter (e.g.
                # oob_score) allowing our user to pass a callable defining the
                # scoring strategy on OOB sample.
                raise ValueError(
                    "The type of target cannot be used to compute OOB "
                    f"estimates. Got {y_type} while only the following are "
                    "supported: continuous, continuous-multioutput, binary, "
                    "multiclass, multilabel-indicator."
                )

            if callable(self.oob_score):
                self._set_oob_score_and_attributes(X, y, scoring_function=self.oob_score)
            else:
                self._set_oob_score_and_attributes(X, y)

        # Decapsulate classes_ attributes
        if hasattr(self, "classes_") and self.n_outputs_ == 1:
            self.n_classes_ = self.n_classes_[0]
            self.classes_ = self.classes_[0]

        # Compute honest decision function
        self.honest_decision_function_ = self._predict_proba(
            X, indices=self.honest_indices_, impute_missing=np.nan
        )

        return self

    def _construct_trees(
        self,
        X,
        y,
        sample_weight,
        random_state,
        n_samples_bootstrap,
        missing_values_in_feature_mask,
        classes,
        n_more_estimators,
    ):
        trees = [
            self._make_estimator(append=False, random_state=random_state)
            for i in range(n_more_estimators)
        ]

        # Parallel loop: we prefer the threading backend as the Cython code
        # for fitting the trees is internally releasing the Python GIL
        # making threading more efficient than multiprocessing in
        # that case. However, for joblib 0.12+ we respect any
        # parallel_backend contexts set at a higher level,
        # since correctness does not rely on using threads.
        trees = Parallel(
            n_jobs=self.n_jobs,
            verbose=self.verbose,
            prefer="threads",
        )(
            delayed(_parallel_build_trees)(
                t,
                self.bootstrap,
                X,
                y,
                sample_weight,
                i,
                len(trees),
                verbose=self.verbose,
                class_weight=self.class_weight,
                n_samples_bootstrap=n_samples_bootstrap,
                missing_values_in_feature_mask=missing_values_in_feature_mask,
                classes=classes,
            )
            for i, t in enumerate(trees)
        )
        return trees

    def predict_proba(self, X):
        """
        Predict class probabilities for X.

        The predicted class probabilities of an input sample are computed as
        the mean predicted class probabilities of the trees in the forest.
        The class probability of a single tree is the fraction of samples of
        the same class in a leaf.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            The input samples. Internally, its dtype will be converted to
            ``dtype=np.float32``. If a sparse matrix is provided, it will be
            converted into a sparse ``csr_matrix``.

        Returns
        -------
        p : ndarray of shape (n_samples, n_classes), or a list of such arrays
            The class probabilities of the input samples. The order of the
            classes corresponds to that in the attribute :term:`classes_`.
        """
        return self._predict_proba(X)

    def _predict_proba(self, X, indices=None, impute_missing=None):
        """predict_proba helper class"""
        check_is_fitted(self)
        X = self._validate_X_predict(X)
        n_jobs, _, _ = _partition_estimators(self.n_estimators, self.n_jobs)

        # avoid storing the output of every tree estimator by summing them here
        posteriors = [
            np.zeros((X.shape[0], j), dtype=np.float64) for j in np.atleast_1d(self.n_classes_)
        ]
        if indices is None:
            indices = [None] * self.n_estimators

        lock = threading.Lock()
        Parallel(n_jobs=n_jobs, verbose=self.verbose, require="sharedmem")(
            delayed(_accumulate_prediction)(tree.predict_proba, X, posteriors, lock, idx)
            for tree, idx in zip(self.estimators_, indices)
        )

        # Normalize to unit length, due to prior weighting
        posteriors = np.array(posteriors)
        zero_mask = posteriors.sum(2) == 0
        posteriors[~zero_mask] /= posteriors[~zero_mask].sum(1, keepdims=True)

        if impute_missing is None:
            pass
        else:
            posteriors[zero_mask] = impute_missing

        # preserve shape of multi-outputs
        if self.n_outputs_ > 1:
            posteriors = [post for post in posteriors]

        if len(posteriors) == 1:
            return posteriors[0]
        else:
            return posteriors

    @property
    def structure_indices_(self):
        """The indices used to learn the structure of the trees."""
        check_is_fitted(self)
        return [tree.structure_indices_ for tree in self.estimators_]

    @property
    def honest_indices_(self):
        """The indices used to fit the leaf nodes."""
        check_is_fitted(self)
        return [tree.honest_indices_ for tree in self.estimators_]

    @property
    def oob_samples_(self):
        """The sample indices that are out-of-bag.

        Only utilized if ``bootstrap=True``, otherwise, all samples are "in-bag".
        """
        if self.bootstrap is False and self._n_samples_bootstrap == self._n_samples:
            raise RuntimeError(
                "Cannot extract out-of-bag samples when bootstrap is False and "
                "n_samples == n_samples_bootstrap"
            )
        check_is_fitted(self)

        oob_samples = []

        possible_indices = np.arange(self._n_samples)
        for structure_idx, honest_idx in zip(self.structure_indices_, self.honest_indices_):
            _oob_samples = np.setdiff1d(
                possible_indices, np.concatenate((structure_idx, honest_idx))
            )
            oob_samples.append(_oob_samples)
        return oob_samples

    def _more_tags(self):
        return {"multioutput": False}

    def apply(self, X):
        """
        Apply trees in the forest to X, return leaf indices.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            The input samples. Internally, its dtype will be converted to
            ``dtype=np.float32``. If a sparse matrix is provided, it will be
            converted into a sparse ``csr_matrix``.

        Returns
        -------
        X_leaves : ndarray of shape (n_samples, n_estimators)
            For each datapoint x in X and for each tree in the forest,
            return the index of the leaf x ends up in.
        """
        return self.estimator_.apply(X)

    def decision_path(self, X):
        """
        Return the decision path in the forest.

        .. versionadded:: 0.18

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            The input samples. Internally, its dtype will be converted to
            ``dtype=np.float32``. If a sparse matrix is provided, it will be
            converted into a sparse ``csr_matrix``.

        Returns
        -------
        indicator : sparse matrix of shape (n_samples, n_nodes)
            Return a node indicator matrix where non zero elements indicates
            that the samples goes through the nodes. The matrix is of CSR
            format.

        n_nodes_ptr : ndarray of shape (n_estimators + 1,)
            The columns from indicator[n_nodes_ptr[i]:n_nodes_ptr[i+1]]
            gives the indicator value for the i-th estimator.
        """
        return self.estimator_.decision_path(X)

    def predict_quantiles(self, X, quantiles=0.5, method="nearest"):
        """Predict class or regression value for X at given quantiles.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            Input data.
        quantiles : float, optional
            The quantiles at which to evaluate, by default 0.5 (median).
        method : str, optional
            The method to interpolate, by default 'linear'. Can be any keyword
            argument accepted by :func:`~np.quantile`.

        Returns
        -------
        y : ndarray of shape (n_samples, n_quantiles, [n_outputs])
            The predicted values. The ``n_outputs`` dimension is present only
            for multi-output regressors.
        """
        return self.estimator_.predict_quantiles(X, quantiles, method)

    def get_leaf_node_samples(self, X):
        """Get samples in each leaf node across the forest.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The data array.

        Returns
        -------
        leaf_node_samples : array-like of shape (n_samples, n_estimators)
            Samples within each leaf node.
        """
        return self.estimator_.get_leaf_node_samples(X)

    def _get_estimators_indices(self):
        # Get drawn indices along both sample and feature axes
        for tree in self.estimators_:
            if not self.bootstrap and self._n_samples_bootstrap == self._n_samples:
                yield np.arange(self._n_samples, dtype=np.int32)
            else:
                # tree.random_state is actually an immutable integer seed rather
                # than a mutable RandomState instance, so it's safe to use it
                # repeatedly when calling this property.
                seed = tree.random_state
                # Operations accessing random_state must be performed identically
                # to those in `_parallel_build_trees()`
                yield _generate_sample_indices(
                    seed, self._n_samples, self._n_samples_bootstrap, self.bootstrap
                )


def _accumulate_prediction(predict, X, out, lock, indices=None):
    """
    See https://github.com/scikit-learn/scikit-learn/blob/
    95119c13af77c76e150b753485c662b7c52a41a2/sklearn/ensemble/_forest.py#L460
    This is a utility function for joblib's Parallel.
    It cannot be placed in ForestClassifier or ForestRegressor due to joblib's
    compatibility issue with pickle.
    """

    if indices is None:
        indices = np.arange(X.shape[0])
    proba = predict(X[indices], check_input=False)

    with lock:
        if len(out) == 1:
            out[0][indices] += proba
        else:
            for i in range(len(out)):
                out[i][indices] += proba[i]
