import numpy as np
from joblib import Parallel, delayed
from sklearn.base import _fit_context

from .._lib.sklearn.ensemble._forest import _parallel_build_trees
from ..ensemble._honest_forest import HonestForestClassifier


class PermutationHonestForestClassifier(HonestForestClassifier):
    """
    A forest classifier with a permutation over the dataset.

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

    max_features : {"sqrt", "log2", None}, int or float, default="sqrt"
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

    max_samples : int or float, default=None
        If bootstrap is True, the number of samples to draw from X
        to train each base tree estimator.

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

    honest_method : {"prune", "apply"}, default="prune"
        Method for enforcing honesty. If "prune", the tree is pruned to enforce
        honesty. If "apply", the tree is not pruned, but the leaf estimates are
        adjusted to enforce honesty.

    tree_estimator : object, default=None
        Type of decision tree classifier to use. By default `None`, which
        defaults to `treeple.tree.DecisionTreeClassifier`. Note
        that one MUST use trees imported from the `treeple.tree`
        API namespace rather than from `sklearn.tree`.

    stratify : bool
        Whether or not to stratify sample when considering structure and leaf indices.
        By default False.

    permute_per_tree : bool
        Whether or not to permute the dataset per tree. By default False.

    **tree_estimator_params : dict
        Parameters to pass to the underlying base tree estimators.
        These must be parameters for ``tree_estimator``.
        
    Attributes
    ----------
    estimator : treeple.tree.HonestTreeClassifier
        The child estimator template used to create the collection of fitted
        sub-estimators.

    estimators_ : list of treeple.tree.HonestTreeClassifier
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

    permutation_indices_ : list of lists, shape=(n_estimators, n_samples)
        The indices of the permutation used to fit each tree. I.e.
        which samples were shuffled.

    covariate_index_ : list of (n_features,) or None
        The index of the dataset to shuffle per tree. Will have up to
        ``n_features`` elements. By default None, which will shuffle all features.
    """

    def __init__(
        self,
        n_estimators=100,
        criterion="gini",
        splitter="best",
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        min_weight_fraction_leaf=0,
        max_features="sqrt",
        max_leaf_nodes=None,
        min_impurity_decrease=0,
        bootstrap=False,
        oob_score=False,
        n_jobs=None,
        random_state=None,
        verbose=0,
        warm_start=False,
        class_weight=None,
        ccp_alpha=0,
        max_samples=None,
        honest_prior="empirical",
        honest_fraction=0.5,
        honest_method="apply",
        tree_estimator=None,
        stratify=False,
        permute_per_tree=False,
        **tree_estimator_params,
    ):
        super().__init__(
            n_estimators=n_estimators,
            criterion=criterion,
            splitter=splitter,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            min_weight_fraction_leaf=min_weight_fraction_leaf,
            max_features=max_features,
            max_leaf_nodes=max_leaf_nodes,
            min_impurity_decrease=min_impurity_decrease,
            bootstrap=bootstrap,
            oob_score=oob_score,
            n_jobs=n_jobs,
            random_state=random_state,
            verbose=verbose,
            warm_start=warm_start,
            class_weight=class_weight,
            ccp_alpha=ccp_alpha,
            max_samples=max_samples,
            honest_prior=honest_prior,
            honest_fraction=honest_fraction,
            honest_method=honest_method,
            tree_estimator=tree_estimator,
            stratify=stratify,
            **tree_estimator_params,
        )
        self.permute_per_tree = permute_per_tree

    @_fit_context(prefer_skip_nested_validation=True)
    def fit(self, X, y, sample_weight=None, classes=None, covariate_index=None):
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

        covariate_index : list
            The indices of the dataset features (i.e. columns) to shuffle per tree.
            Will have up to ``n_features`` elements. By default None, which will
            shuffle all features.

        Returns
        -------
        self : object
            Fitted estimator.
        """
        if covariate_index is None:
            covariate_index = np.arange(X.shape[1], dtype=np.intp)

        if not isinstance(covariate_index, (list, tuple, np.ndarray)):
            raise RuntimeError("covariate_index must be an iterable of integer indices")
        else:
            if not all(isinstance(idx, (np.integer, int)) for idx in covariate_index):
                raise RuntimeError("Not all covariate_index are integer indices")

        if len(covariate_index) > X.shape[1]:
            raise ValueError(
                "The length of the covariate index must not be greater than the number of features."
            )
        self.covariate_index_ = covariate_index
        self = super().fit(X, y, sample_weight, classes)
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
        if self.permute_per_tree:
            # TODO: refactor to make this a more robust implementation
            # XXX: this does not currently allow permuting individual covariates.
            permutation_arr_per_tree = [
                random_state.choice(self._n_samples, size=self._n_samples, replace=False)
                for _ in range(self.n_estimators)
            ]
            if sample_weight is None:
                sample_weight = np.ones((self._n_samples,))

            # fitted array of what indices were used to fit each tree
            self.permutation_indices_ = permutation_arr_per_tree

            trees = Parallel(
                n_jobs=self.n_jobs,
                verbose=self.verbose,
                prefer="threads",
            )(
                delayed(_parallel_build_trees)(
                    t,
                    self.bootstrap,
                    X,
                    y[perm_idx],
                    sample_weight[perm_idx],
                    i,
                    len(trees),
                    verbose=self.verbose,
                    class_weight=self.class_weight,
                    n_samples_bootstrap=n_samples_bootstrap,
                    missing_values_in_feature_mask=missing_values_in_feature_mask,
                    classes=classes,
                )
                for i, (t, perm_idx) in enumerate(
                    zip(
                        trees,
                        permutation_arr_per_tree,
                    )
                )
            )
        else:
            perm_idx = np.array(
                random_state.choice(self._n_samples, size=(self._n_samples, 1), replace=False)
            )
            X[:, self.covariate_index_] = X[perm_idx, self.covariate_index_]
            self.permutation_indices_ = perm_idx

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

        self.estimators_.extend(trees)
