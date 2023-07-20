import copy
from numbers import Real

import numpy as np
from scipy.sparse import issparse
from sklearn.base import ClusterMixin, TransformerMixin
from sklearn.cluster import AgglomerativeClustering
from sklearn.utils._param_validation import Interval
from sklearn.utils.validation import check_is_fitted

from .._lib.sklearn.tree import (
    BaseDecisionTree,
    DecisionTreeClassifier,
    DecisionTreeRegressor,
    _criterion,
)
from .._lib.sklearn.tree import _tree as _sklearn_tree
from .._lib.sklearn.tree._criterion import BaseCriterion
from .._lib.sklearn.tree._tree import BestFirstTreeBuilder, DepthFirstTreeBuilder
from . import _oblique_splitter
from ._neighbors import SimMatrixMixin
from ._oblique_splitter import ObliqueSplitter
from ._oblique_tree import ObliqueTree
from .manifold import _morf_splitter
from .manifold._morf_splitter import PatchSplitter
from .unsupervised import _unsup_criterion, _unsup_oblique_splitter, _unsup_splitter
from .unsupervised._unsup_criterion import UnsupervisedCriterion
from .unsupervised._unsup_oblique_splitter import UnsupervisedObliqueSplitter
from .unsupervised._unsup_oblique_tree import UnsupervisedObliqueTree
from .unsupervised._unsup_splitter import UnsupervisedSplitter
from .unsupervised._unsup_tree import (
    UnsupervisedBestFirstTreeBuilder,
    UnsupervisedDepthFirstTreeBuilder,
    UnsupervisedTree,
)

DTYPE = _sklearn_tree.DTYPE
DOUBLE = _sklearn_tree.DOUBLE

CRITERIA_CLF = {
    "gini": _criterion.Gini,
    "log_loss": _criterion.Entropy,
    "entropy": _criterion.Entropy,
}
CRITERIA_REG = {
    "squared_error": _criterion.MSE,
    "friedman_mse": _criterion.FriedmanMSE,
    "absolute_error": _criterion.MAE,
    "poisson": _criterion.Poisson,
}

OBLIQUE_DENSE_SPLITTERS = {
    "best": _oblique_splitter.BestObliqueSplitter,
}

PATCH_DENSE_SPLITTERS = {
    "best": _morf_splitter.BestPatchSplitter,
}

UNSUPERVISED_CRITERIA = {"twomeans": _unsup_criterion.TwoMeans, "fastbic": _unsup_criterion.FastBIC}
UNSUPERVISED_SPLITTERS = {
    "best": _unsup_splitter.BestUnsupervisedSplitter,
}

UNSUPERVISED_OBLIQUE_SPLITTERS = {"best": _unsup_oblique_splitter.BestObliqueUnsupervisedSplitter}


class UnsupervisedDecisionTree(SimMatrixMixin, TransformerMixin, ClusterMixin, BaseDecisionTree):
    """Unsupervised decision tree.

    Parameters
    ----------
    criterion : {"twomeans", "fastbic"}, default="twomeans"
        The function to measure the quality of a split. Supported criteria are
        "twomeans" for the variance impurity and "fastbic" for the
        BIC criterion. If ``UnsupervisedCriterion`` instance is passed in, then
        the user must abide by the Cython internal API. See source code.

    splitter : {"best", "random"}, default="best"
        The strategy used to choose the split at each node. Supported
        strategies are "best" to choose the best split and "random" to choose
        the best random split. If ``UnsupervisedSplitter`` instance is passed in, then
        the user must abide by the Cython internal API. See source code.

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

    max_features : int, float or {"auto", "sqrt", "log2"}, default=None
        The number of features to consider when looking for the best split:

            - If int, then consider `max_features` features at each split.
            - If float, then `max_features` is a fraction and
              `max(1, int(max_features * n_features_in_))` features are considered at
              each split.
            - If "auto", then `max_features=sqrt(n_features)`.
            - If "sqrt", then `max_features=sqrt(n_features)`.
            - If "log2", then `max_features=log2(n_features)`.
            - If None, then `max_features=n_features`.

    max_leaf_nodes : int, default=None
        Grow a tree with ``max_leaf_nodes`` in best-first fashion.
        Best nodes are defined as relative reduction in impurity.
        If None then unlimited number of leaf nodes.

    random_state : int, RandomState instance or None, default=None
        Controls the randomness of the estimator. The features are always
        randomly permuted at each split, even if ``splitter`` is set to
        ``"best"``. When ``max_features < n_features``, the algorithm will
        select ``max_features`` at random at each split before finding the best
        split among them. But the best found split may vary across different
        runs, even if ``max_features=n_features``. That is the case, if the
        improvement of the criterion is identical for several splits and one
        split has to be selected at random. To obtain a deterministic behaviour
        during fitting, ``random_state`` has to be fixed to an integer.
        See how scikit-learn defines ``random_state`` for details.

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

    clustering_func : callable
        Scikit-learn compatible clustering function to take the affinity matrix
        and return cluster labels. By default, :class:`sklearn.cluster.AgglomerativeClustering`.

    clustering_func_args : dict
        Clustering function class keyword arguments. Passed to `clustering_func`.
    """

    def __init__(
        self,
        *,
        criterion="twomeans",
        splitter="best",
        max_depth=None,
        min_samples_split=5,
        min_samples_leaf=1,
        min_weight_fraction_leaf=0.0,
        max_features=None,
        max_leaf_nodes=None,
        random_state=None,
        min_impurity_decrease=0.0,
        clustering_func=None,
        clustering_func_args=None,
    ):
        super().__init__(
            criterion=criterion,
            splitter=splitter,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            min_weight_fraction_leaf=min_weight_fraction_leaf,
            max_features=max_features,
            max_leaf_nodes=max_leaf_nodes,
            random_state=random_state,
            min_impurity_decrease=min_impurity_decrease,
        )

        self.clustering_func = clustering_func
        self.clustering_func_args = clustering_func_args

    def fit(self, X, y=None, sample_weight=None, check_input=True):
        if check_input:
            # TODO: allow X to be sparse
            check_X_params = dict(dtype=DTYPE)  # , accept_sparse="csc"
            X = self._validate_data(X, validate_separately=(check_X_params))
            if issparse(X):
                X.sort_indices()

                if X.indices.dtype != np.intc or X.indptr.dtype != np.intc:
                    raise ValueError("No support for np.int64 index based sparse matrices")

        super()._fit(X=X, y=None, sample_weight=sample_weight, check_input=check_input)

        # apply to the leaves
        n_samples = X.shape[0]

        sim_mat = self.compute_similarity_matrix(X)

        # compute the labels and set it
        if n_samples >= 2:
            self.labels_ = self._assign_labels(sim_mat)

        return self

    def _build_tree(
        self,
        X,
        y,
        sample_weight,
        missing_values_in_feature_mask,
        categorical,
        min_samples_leaf,
        min_weight_leaf,
        max_leaf_nodes,
        min_samples_split,
        max_depth,
        random_state,
    ):
        monotonic_cst = None
        self.n_categories_ = np.array(
            [
                np.int32(X[:, i].max()) + 1 if i in categorical else -1
                for i in range(self.n_features_in_)
            ],
            dtype=np.int32,
        )

        criterion = self.criterion
        if not isinstance(criterion, UnsupervisedCriterion):
            criterion = UNSUPERVISED_CRITERIA[self.criterion]()
        else:
            # Make a deepcopy in case the criterion has mutable attributes that
            # might be shared and modified concurrently during parallel fitting
            criterion = copy.deepcopy(criterion)

        splitter = self.splitter
        if not isinstance(self.splitter, UnsupervisedSplitter):
            splitter = UNSUPERVISED_SPLITTERS[self.splitter](
                criterion,
                self.max_features_,
                min_samples_leaf,
                min_weight_leaf,
                random_state,
                monotonic_cst,
            )

        self.tree_ = UnsupervisedTree(self.n_features_in_, self.n_categories_)

        # Use BestFirst if max_leaf_nodes given; use DepthFirst otherwise
        if max_leaf_nodes < 0:
            builder = UnsupervisedDepthFirstTreeBuilder(
                splitter,
                min_samples_split,
                min_samples_leaf,
                min_weight_leaf,
                max_depth,
                self.min_impurity_decrease,
            )
        else:
            builder = UnsupervisedBestFirstTreeBuilder(
                splitter,
                min_samples_split,
                min_samples_leaf,
                min_weight_leaf,
                max_depth,
                max_leaf_nodes,
                self.min_impurity_decrease,
            )

        builder.build(self.tree_, X, sample_weight)

    def predict(self, X, check_input=True):
        """Assign labels based on clustering the affinity matrix.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Array to cluster.

        check_input : bool, optional
            Whether to validate input, by default True.

        Returns
        -------
        labels : array-like of shape (n_samples,)
            The assigned labels for each sample.
        """

        X = self._validate_X_predict(X, check_input=check_input)
        affinity_matrix = self.transform(X)

        # compute the labels and set it
        return self._assign_labels(affinity_matrix)

    def transform(self, X):
        """Transform X to a cluster-distance space.

        In the new space, each dimension is the distance to the cluster
        centers. Note that even if X is sparse, the array returned by
        `transform` will typically be dense.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            New data to transform.

        Returns
        -------
        X_new : ndarray of shape (n_samples, n_samples)
            X transformed in the new space.
        """
        check_is_fitted(self)

        # now compute the affinity matrix and set it
        affinity_matrix = self.compute_similarity_matrix(X)
        return affinity_matrix

    def _assign_labels(self, affinity_matrix):
        """Assign cluster labels given X.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_samples)
            The affinity matrix.

        Returns
        -------
        predict_labels : ndarray of shape (n_samples,)
            The predicted cluster labels
        """
        if self.clustering_func is None:
            self.clustering_func_ = AgglomerativeClustering
        else:
            self.clustering_func_ = self.clustering_func
        if self.clustering_func_args is None:
            self.clustering_func_args_ = dict()
        else:
            self.clustering_func_args_ = self.clustering_func_args
        cluster = self.clustering_func_(**self.clustering_func_args_)

        # apply agglomerative clustering to obtain cluster labels
        predict_labels = cluster.fit_predict(affinity_matrix)
        return predict_labels


class UnsupervisedObliqueDecisionTree(UnsupervisedDecisionTree):
    """Unsupervised oblique decision tree.

    Parameters
    ----------
    criterion : {"twomeans", "fastbic"}, default="twomeans"
        The function to measure the quality of a split. Supported criteria are
        "twomeans" for the variance impurity and "fastbic" for the
        BIC criterion. If ``UnsupervisedCriterion`` instance is passed in, then
        the user must abide by the Cython internal API. See source code.

    splitter : {"best", "random"}, default="best"
        The strategy used to choose the split at each node. Supported
        strategies are "best" to choose the best split and "random" to choose
        the best random split. If ``UnsupervisedSplitter`` instance is passed in, then
        the user must abide by the Cython internal API. See source code.

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

    max_features : int, float or {"auto", "sqrt", "log2"}, default=None
        The number of features to consider when looking for the best split:

            - If int, then consider `max_features` features at each split.
            - If float, then `max_features` is a fraction and
              `max(1, int(max_features * n_features_in_))` features are considered at
              each split.
            - If "auto", then `max_features=sqrt(n_features)`.
            - If "sqrt", then `max_features=sqrt(n_features)`.
            - If "log2", then `max_features=log2(n_features)`.
            - If None, then `max_features=n_features`.

    max_leaf_nodes : int, default=None
        Grow a tree with ``max_leaf_nodes`` in best-first fashion.
        Best nodes are defined as relative reduction in impurity.
        If None then unlimited number of leaf nodes.

    random_state : int, RandomState instance or None, default=None
        Controls the randomness of the estimator. The features are always
        randomly permuted at each split, even if ``splitter`` is set to
        ``"best"``. When ``max_features < n_features``, the algorithm will
        select ``max_features`` at random at each split before finding the best
        split among them. But the best found split may vary across different
        runs, even if ``max_features=n_features``. That is the case, if the
        improvement of the criterion is identical for several splits and one
        split has to be selected at random. To obtain a deterministic behaviour
        during fitting, ``random_state`` has to be fixed to an integer.
        See how scikit-learn defines ``random_state`` for details.

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

    feature_combinations : float, default=1.5
        The number of features to combine on average at each split
        of the decision trees.

    clustering_func : callable
        Scikit-learn compatible clustering function to take the affinity matrix
        and return cluster labels. By default, :class:`sklearn.cluster.AgglomerativeClustering`.

    clustering_func_args : dict
        Clustering function class keyword arguments. Passed to `clustering_func`.
    """

    tree_type = "oblique"

    def __init__(
        self,
        *,
        criterion="twomeans",
        splitter="best",
        max_depth=None,
        min_samples_split=5,
        min_samples_leaf=1,
        min_weight_fraction_leaf=0,
        max_features=None,
        max_leaf_nodes=None,
        random_state=None,
        min_impurity_decrease=0,
        feature_combinations=1.5,
        clustering_func=None,
        clustering_func_args=None,
    ):
        super().__init__(
            criterion=criterion,
            splitter=splitter,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            min_weight_fraction_leaf=min_weight_fraction_leaf,
            max_features=max_features,
            max_leaf_nodes=max_leaf_nodes,
            random_state=random_state,
            min_impurity_decrease=min_impurity_decrease,
            clustering_func=clustering_func,
            clustering_func_args=clustering_func_args,
        )
        self.feature_combinations = feature_combinations

    def _build_tree(
        self,
        X,
        y,
        sample_weight,
        missing_values_in_feature_mask,
        categorical,
        min_samples_leaf,
        min_weight_leaf,
        max_leaf_nodes,
        min_samples_split,
        max_depth,
        random_state,
    ):
        self.n_categories_ = np.array(
            [
                np.int32(X[:, i].max()) + 1 if i in categorical else -1
                for i in range(self.n_features_in_)
            ],
            dtype=np.int32,
        )

        # TODO: add feature_combinations fix that was used in obliquedecisiontreeclassifier
        criterion = self.criterion
        if not isinstance(criterion, UnsupervisedCriterion):
            criterion = UNSUPERVISED_CRITERIA[self.criterion]()
        else:
            # Make a deepcopy in case the criterion has mutable attributes that
            # might be shared and modified concurrently during parallel fitting
            criterion = copy.deepcopy(criterion)

        splitter = self.splitter
        if not isinstance(self.splitter, UnsupervisedObliqueSplitter):
            splitter = UNSUPERVISED_OBLIQUE_SPLITTERS[self.splitter](
                criterion,
                self.max_features_,
                min_samples_leaf,
                min_weight_leaf,
                random_state,
                self.feature_combinations,
            )

        self.tree_ = UnsupervisedObliqueTree(self.n_features_in_, self.n_categories_)

        # Use BestFirst if max_leaf_nodes given; use DepthFirst otherwise
        if max_leaf_nodes < 0:
            builder = UnsupervisedDepthFirstTreeBuilder(
                splitter,
                min_samples_split,
                min_samples_leaf,
                min_weight_leaf,
                max_depth,
                self.min_impurity_decrease,
            )
        else:
            builder = UnsupervisedBestFirstTreeBuilder(
                splitter,
                min_samples_split,
                min_samples_leaf,
                min_weight_leaf,
                max_depth,
                max_leaf_nodes,
                self.min_impurity_decrease,
            )

        builder.build(self.tree_, X, sample_weight)


class ObliqueDecisionTreeClassifier(SimMatrixMixin, DecisionTreeClassifier):
    """A decision tree classifier.

    Read more in the :ref:`User Guide <sklearn:tree>`. The implementation follows
    that of :footcite:`breiman2001random` and :footcite:`TomitaSPORF2020`.

    Parameters
    ----------
    criterion : {"gini", "entropy"}, default="gini"
        The function to measure the quality of a split. Supported criteria are
        "gini" for the Gini impurity and "entropy" for the information gain.

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

    max_features : int, float or {"auto", "sqrt", "log2"}, default=None
        The number of features to consider when looking for the best split:

            - If int, then consider `max_features` features at each split.
            - If float, then `max_features` is a fraction and
              `int(max_features * n_features)` features are considered at each
              split.
            - If "auto", then `max_features=sqrt(n_features)`.
            - If "sqrt", then `max_features=sqrt(n_features)`.
            - If "log2", then `max_features=log2(n_features)`.
            - If None, then `max_features=n_features`.

        Note: the search for a split does not stop until at least one
        valid partition of the node samples is found, even if it requires to
        effectively inspect more than ``max_features`` features.

        Note: Compared to axis-aligned Random Forests, one can set
        max_features to a number greater then ``n_features``.

    random_state : int, RandomState instance or None, default=None
        Controls the randomness of the estimator. The features are always
        randomly permuted at each split, even if ``splitter`` is set to
        ``"best"``. When ``max_features < n_features``, the algorithm will
        select ``max_features`` at random at each split before finding the best
        split among them. But the best found split may vary across different
        runs, even if ``max_features=n_features``. That is the case, if the
        improvement of the criterion is identical for several splits and one
        split has to be selected at random. To obtain a deterministic behaviour
        during fitting, ``random_state`` has to be fixed to an integer.
        See :term:`Glossary <random_state>` for details.

    max_leaf_nodes : int, default=None
        Grow a tree with ``max_leaf_nodes`` in best-first fashion.
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

    class_weight : dict, list of dict or "balanced", default=None
        Weights associated with classes in the form ``{class_label: weight}``.
        If None, all classes are supposed to have weight one. For
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

        For multi-output, the weights of each column of y will be multiplied.

        Note that these weights will be multiplied with sample_weight (passed
        through the fit method) if sample_weight is specified.

    feature_combinations : float, default=None
        The number of features to combine on average at each split
        of the decision trees. If ``None``, then will default to the minimum of
        ``(1.5, n_features)``. This controls the number of non-zeros is the
        projection matrix. Setting the value to 1.0 is equivalent to a
        traditional decision-tree. ``feature_combinations * max_features``
        gives the number of expected non-zeros in the projection matrix of shape
        ``(max_features, n_features)``. Thus this value must always be less than
        ``n_features`` in order to be valid.

    Attributes
    ----------
    classes_ : ndarray of shape (n_classes,) or list of ndarray
        The classes labels (single output problem),
        or a list of arrays of class labels (multi-output problem).

    feature_importances_ : ndarray of shape (n_features,)
        The impurity-based feature importances.
        The higher, the more important the feature.
        The importance of a feature is computed as the (normalized)
        total reduction of the criterion brought by that feature.  It is also
        known as the Gini importance [4]_.

        Warning: impurity-based feature importances can be misleading for
        high cardinality features (many unique values). See
        :func:`sklearn.inspection.permutation_importance` as an alternative.

    max_features_ : int
        The inferred value of max_features.

    n_classes_ : int or list of int
        The number of classes (for single output problems),
        or a list containing the number of classes for each
        output (for multi-output problems).

    n_features_in_ : int
        Number of features seen during :term:`fit`.

    feature_names_in_ : ndarray of shape (`n_features_in_`,)
        Names of features seen during :term:`fit`. Defined only when `X`
        has feature names that are all strings.

    n_outputs_ : int
        The number of outputs when ``fit`` is performed.

    tree_ : Tree instance
        The underlying Tree object. Please refer to
        ``help(sklearn.tree._tree.Tree)`` for
        attributes of Tree object.

    feature_combinations_ : float
        The number of feature combinations on average taken to fit the tree.

    See Also
    --------
    sklearn.tree.DecisionTreeClassifier : An axis-aligned decision tree classifier.

    Notes
    -----
    Compared to ``DecisionTreeClassifier``, oblique trees can sample
    more features then ``n_features``, where ``n_features`` is the number
    of columns in ``X``. This is controlled via the ``max_features``
    parameter. In fact, sampling more times results in better
    trees with the caveat that there is an increased computation. It is
    always recommended to sample more if one is willing to spend the
    computational resources.

    The default values for the parameters controlling the size of the trees
    (e.g. ``max_depth``, ``min_samples_leaf``, etc.) lead to fully grown and
    unpruned trees which can potentially be very large on some data sets. To
    reduce memory consumption, the complexity and size of the trees should be
    controlled by setting those parameter values.

    The :meth:`predict` method operates using the :func:`numpy.argmax`
    function on the outputs of :meth:`predict_proba`. This means that in
    case the highest predicted probabilities are tied, the classifier will
    predict the tied class with the lowest index in :term:`classes_`.

    References
    ----------

    .. [1] https://en.wikipedia.org/wiki/Decision_tree_learning

    .. [2] L. Breiman, J. Friedman, R. Olshen, and C. Stone, "Classification
        and Regression Trees", Wadsworth, Belmont, CA, 1984.

    .. [3] T. Hastie, R. Tibshirani and J. Friedman. "Elements of Statistical
        Learning", Springer, 2009.

    .. [4] L. Breiman, and A. Cutler, "Random Forests",
        https://www.stat.berkeley.edu/~breiman/RandomForests/cc_home.htm

    Examples
    --------
    >>> from sklearn.datasets import load_iris
    >>> from sklearn.model_selection import cross_val_score
    >>> from sktree.tree import ObliqueDecisionTreeClassifier
    >>> clf = ObliqueDecisionTreeClassifier(random_state=0)
    >>> iris = load_iris()
    >>> cross_val_score(clf, iris.data, iris.target, cv=10)
    ...                             # doctest: +SKIP
    ...
    array([ 1.     ,  0.93...,  0.86...,  0.93...,  0.93...,
            0.93...,  0.93...,  1.     ,  0.93...,  1.      ])
    """

    tree_type = "oblique"

    _parameter_constraints = {
        **DecisionTreeClassifier._parameter_constraints,
        "feature_combinations": [
            Interval(Real, 1.0, None, closed="left"),
            None,
        ],
    }

    def __init__(
        self,
        *,
        criterion="gini",
        splitter="best",
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        min_weight_fraction_leaf=0.0,
        max_features=None,
        random_state=None,
        max_leaf_nodes=None,
        min_impurity_decrease=0.0,
        class_weight=None,
        feature_combinations=None,
    ):
        super().__init__(
            criterion=criterion,
            splitter=splitter,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            min_weight_fraction_leaf=min_weight_fraction_leaf,
            max_features=max_features,
            max_leaf_nodes=max_leaf_nodes,
            class_weight=class_weight,
            random_state=random_state,
            min_impurity_decrease=min_impurity_decrease,
        )

        self.feature_combinations = feature_combinations

    def _build_tree(
        self,
        X,
        y,
        sample_weight,
        missing_values_in_feature_mask,
        categorical,
        min_samples_leaf,
        min_weight_leaf,
        max_leaf_nodes,
        min_samples_split,
        max_depth,
        random_state,
    ):
        """Build the actual tree.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            The training input samples. Internally, it will be converted to
            ``dtype=np.float32`` and if a sparse matrix is provided
            to a sparse ``csc_matrix``.

        y : array-like of shape (n_samples,) or (n_samples, n_outputs)
            The target values (class labels) as integers or strings.

        sample_weight : array-like of shape (n_samples,), default=None
            Sample weights. If None, then samples are equally weighted. Splits
            that would create child nodes with net zero or negative weight are
            ignored while searching for a split in each node. Splits are also
            ignored if they would result in any single class carrying a
            negative weight in either child node.

        min_samples_leaf : int or float
            The minimum number of samples required to be at a leaf node.

        min_weight_leaf : float, default=0.0
           The minimum weighted fraction of the sum total of weights.

        max_leaf_nodes : int, default=None
            Grow a tree with ``max_leaf_nodes`` in best-first fashion.

        min_samples_split : int or float, default=2
            The minimum number of samples required to split an internal node.

        max_depth : int, default=None
            The maximum depth of the tree. If None, then nodes are expanded until
            all leaves are pure or until all leaves contain less than
            min_samples_split samples.

        random_state : int, RandomState instance or None, default=None
            Controls the randomness of the estimator.
        """
        monotonic_cst = None
        self.n_categories_ = np.array(
            [
                np.int32(X[:, i].max()) + 1 if i in categorical else -1
                for i in range(self.n_features_in_)
            ],
            dtype=np.int32,
        )
        breiman_shortcut = False
        _, n_features = X.shape

        if self.feature_combinations is None:
            self.feature_combinations_ = min(n_features, 1.5)
        elif self.feature_combinations > n_features:
            raise RuntimeError(
                f"Feature combinations {self.feature_combinations} should not be "
                f"greater than the possible number of features {n_features}"
            )
        else:
            self.feature_combinations_ = self.feature_combinations

        # Build tree
        criterion = self.criterion
        if not isinstance(criterion, BaseCriterion):
            criterion = CRITERIA_CLF[self.criterion](self.n_outputs_, self.n_classes_)
        else:
            # Make a deepcopy in case the criterion has mutable attributes that
            # might be shared and modified concurrently during parallel fitting
            criterion = copy.deepcopy(criterion)

        splitter = self.splitter
        if issparse(X):
            raise ValueError(
                "Sparse input is not supported for oblique trees. "
                "Please convert your data to a dense array."
            )
        else:
            OBLIQUE_SPLITTERS = OBLIQUE_DENSE_SPLITTERS

        if not isinstance(self.splitter, ObliqueSplitter):
            splitter = OBLIQUE_SPLITTERS[self.splitter](
                criterion,
                self.max_features_,
                min_samples_leaf,
                min_weight_leaf,
                random_state,
                monotonic_cst,
                breiman_shortcut,
                self.feature_combinations_,
            )

        self.tree_ = ObliqueTree(
            self.n_features_in_, self.n_classes_, self.n_outputs_, self.n_categories_
        )

        # Use BestFirst if max_leaf_nodes given; use DepthFirst otherwise
        if max_leaf_nodes < 0:
            builder = DepthFirstTreeBuilder(
                splitter,
                min_samples_split,
                min_samples_leaf,
                min_weight_leaf,
                max_depth,
                self.min_impurity_decrease,
            )
        else:
            builder = BestFirstTreeBuilder(
                splitter,
                min_samples_split,
                min_samples_leaf,
                min_weight_leaf,
                max_depth,
                max_leaf_nodes,
                self.min_impurity_decrease,
            )

        builder.build(self.tree_, X, y, sample_weight, None, None)

        if self.n_outputs_ == 1:
            self.n_classes_ = self.n_classes_[0]
            self.classes_ = self.classes_[0]


class ObliqueDecisionTreeRegressor(SimMatrixMixin, DecisionTreeRegressor):
    """An oblique decision tree Regressor.

    Read more in the :ref:`User Guide <sklearn:tree>`. The implementation follows
    that of :footcite:`breiman2001random` and :footcite:`TomitaSPORF2020`.

    Parameters
    ----------
    criterion : {"squared_error", "friedman_mse", "absolute_error", \
            "poisson"}, default="squared_error"
        The function to measure the quality of a split. Supported criteria
        are "squared_error" for the mean squared error, which is equal to
        variance reduction as feature selection criterion and minimizes the L2
        loss using the mean of each terminal node, "friedman_mse", which uses
        mean squared error with Friedman's improvement score for potential
        splits, "absolute_error" for the mean absolute error, which minimizes
        the L1 loss using the median of each terminal node, and "poisson" which
        uses reduction in Poisson deviance to find splits.

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

    max_features : int, float or {"auto", "sqrt", "log2"}, default=None
        The number of features to consider when looking for the best split:

            - If int, then consider `max_features` features at each split.
            - If float, then `max_features` is a fraction and
              `int(max_features * n_features)` features are considered at each
              split.
            - If "auto", then `max_features=sqrt(n_features)`.
            - If "sqrt", then `max_features=sqrt(n_features)`.
            - If "log2", then `max_features=log2(n_features)`.
            - If None, then `max_features=n_features`.

        Note: the search for a split does not stop until at least one
        valid partition of the node samples is found, even if it requires to
        effectively inspect more than ``max_features`` features.
        Note: Compared to axis-aligned Random Forests, one can set
        max_features to a number greater then ``n_features``.

    random_state : int, RandomState instance or None, default=None
        Controls the randomness of the estimator. The features are always
        randomly permuted at each split, even if ``splitter`` is set to
        ``"best"``. When ``max_features < n_features``, the algorithm will
        select ``max_features`` at random at each split before finding the best
        split among them. But the best found split may vary across different
        runs, even if ``max_features=n_features``. That is the case, if the
        improvement of the criterion is identical for several splits and one
        split has to be selected at random. To obtain a deterministic behaviour
        during fitting, ``random_state`` has to be fixed to an integer.
        See :term:`Glossary <random_state>` for details.

    max_leaf_nodes : int, default=None
        Grow a tree with ``max_leaf_nodes`` in best-first fashion.
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

    feature_combinations : float, default=None
        The number of features to combine on average at each split
        of the decision trees. If ``None``, then will default to the minimum of
        ``(1.5, n_features)``. This controls the number of non-zeros is the
        projection matrix. Setting the value to 1.0 is equivalent to a
        traditional decision-tree. ``feature_combinations * max_features``
        gives the number of expected non-zeros in the projection matrix of shape
        ``(max_features, n_features)``. Thus this value must always be less than
        ``n_features`` in order to be valid.

    Attributes
    ----------
    feature_importances_ : ndarray of shape (n_features,)
        The impurity-based feature importances.
        The higher, the more important the feature.
        The importance of a feature is computed as the (normalized)
        total reduction of the criterion brought by that feature.  It is also
        known as the Gini importance [4]_.

        Warning: impurity-based feature importances can be misleading for
        high cardinality features (many unique values). See
        :func:`sklearn.inspection.permutation_importance` as an alternative.

    max_features_ : int
        The inferred value of max_features.

    n_features_in_ : int
        Number of features seen during :term:`fit`.

    feature_names_in_ : ndarray of shape (`n_features_in_`,)
        Names of features seen during :term:`fit`. Defined only when `X`
        has feature names that are all strings.

    n_outputs_ : int
        The number of outputs when ``fit`` is performed.

    tree_ : Tree instance
        The underlying Tree object. Please refer to
        ``help(sklearn.tree._tree.Tree)`` for
        attributes of Tree object.

    feature_combinations_ : float
        The number of feature combinations on average taken to fit the tree.

    See Also
    --------
    sklearn.tree.DecisionTreeRegressor : An axis-aligned decision tree regressor.
    ObliqueDecisionTreeClassifier : An oblique decision tree classifier.

    Notes
    -----
    Compared to ``DecisionTreeClassifier``, oblique trees can sample
    more features then ``n_features``, where ``n_features`` is the number
    of columns in ``X``. This is controlled via the ``max_features``
    parameter. In fact, sampling more times results in better
    trees with the caveat that there is an increased computation. It is
    always recommended to sample more if one is willing to spend the
    computational resources.

    The default values for the parameters controlling the size of the trees
    (e.g. ``max_depth``, ``min_samples_leaf``, etc.) lead to fully grown and
    unpruned trees which can potentially be very large on some data sets. To
    reduce memory consumption, the complexity and size of the trees should be
    controlled by setting those parameter values.

    References
    ----------
    .. [1] https://en.wikipedia.org/wiki/Decision_tree_learning

    .. [2] L. Breiman, J. Friedman, R. Olshen, and C. Stone, "Classification
        and Regression Trees", Wadsworth, Belmont, CA, 1984.

    .. [3] T. Hastie, R. Tibshirani and J. Friedman. "Elements of Statistical
        Learning", Springer, 2009.

    .. [4] L. Breiman, and A. Cutler, "Random Forests",
        https://www.stat.berkeley.edu/~breiman/RandomForests/cc_home.htm

    Examples
    --------
    >>> from sklearn.datasets import load_diabetes
    >>> from sklearn.model_selection import cross_val_score
    >>> from sklearn.tree import ObliqueDecisionTreeRegressor
    >>> X, y = load_diabetes(return_X_y=True)
    >>> regressor = ObliqueDecisionTreeRegressor(random_state=0)
    >>> cross_val_score(regressor, X, y, cv=10)
    ...                    # doctest: +SKIP
    ...
    array([-0.68908909, -0.35854406,  0.35223873, -0.03616902, -0.56008907,
            0.32235221,  0.06945264, -1.1465216 ,  0.34597007, -0.15308512])
    """

    tree_type = "oblique"

    _parameter_constraints = {
        **DecisionTreeRegressor._parameter_constraints,
        "feature_combinations": [
            Interval(Real, 1.0, None, closed="left"),
            None,
        ],
    }

    def __init__(
        self,
        *,
        criterion="squared_error",
        splitter="best",
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        min_weight_fraction_leaf=0.0,
        max_features=None,
        random_state=None,
        max_leaf_nodes=None,
        min_impurity_decrease=0.0,
        feature_combinations=None,
    ):
        super().__init__(
            criterion=criterion,
            splitter=splitter,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            min_weight_fraction_leaf=min_weight_fraction_leaf,
            max_features=max_features,
            max_leaf_nodes=max_leaf_nodes,
            random_state=random_state,
            min_impurity_decrease=min_impurity_decrease,
        )

        self.feature_combinations = feature_combinations

    def _build_tree(
        self,
        X,
        y,
        sample_weight,
        missing_values_in_feature_mask,
        categorical,
        min_samples_leaf,
        min_weight_leaf,
        max_leaf_nodes,
        min_samples_split,
        max_depth,
        random_state,
    ):
        """Build the actual tree.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            The training input samples. Internally, it will be converted to
            ``dtype=np.float32`` and if a sparse matrix is provided
            to a sparse ``csc_matrix``.

        y : array-like of shape (n_samples,) or (n_samples, n_outputs)
            The target values (real numbers). Use ``dtype=np.float64`` and
            ``order='C'`` for maximum efficiency.

        sample_weight : array-like of shape (n_samples,), default=None
            Sample weights. If None, then samples are equally weighted. Splits
            that would create child nodes with net zero or negative weight are
            ignored while searching for a split in each node.

        min_samples_leaf : int or float
            The minimum number of samples required to be at a leaf node.

        min_weight_leaf : float, default=0.0
           The minimum weighted fraction of the sum total of weights.

        max_leaf_nodes : int, default=None
            Grow a tree with ``max_leaf_nodes`` in best-first fashion.

        min_samples_split : int or float, default=2
            The minimum number of samples required to split an internal node.

        max_depth : int, default=None
            The maximum depth of the tree. If None, then nodes are expanded until
            all leaves are pure or until all leaves contain less than
            min_samples_split samples.

        random_state : int, RandomState instance or None, default=None
            Controls the randomness of the estimator.
        """
        monotonic_cst = None
        self.n_categories_ = np.array(
            [
                np.int32(X[:, i].max()) + 1 if i in categorical else -1
                for i in range(self.n_features_in_)
            ],
            dtype=np.int32,
        )
        breiman_shortcut = False

        n_samples, n_features = X.shape

        if self.feature_combinations is None:
            self.feature_combinations_ = min(n_features, 1.5)
        elif self.feature_combinations > n_features:
            raise RuntimeError(
                f"Feature combinations {self.feature_combinations} should not be "
                f"greater than the possible number of features {n_features}"
            )
        else:
            self.feature_combinations_ = self.feature_combinations

        # Build tree
        criterion = self.criterion
        if not isinstance(criterion, BaseCriterion):
            criterion = CRITERIA_REG[self.criterion](self.n_outputs_, n_samples)
        else:
            # Make a deepcopy in case the criterion has mutable attributes that
            # might be shared and modified concurrently during parallel fitting
            criterion = copy.deepcopy(criterion)

        splitter = self.splitter
        if issparse(X):
            raise ValueError(
                "Sparse input is not supported for oblique trees. "
                "Please convert your data to a dense array."
            )
        else:
            OBLIQUE_SPLITTERS = OBLIQUE_DENSE_SPLITTERS

        if not isinstance(self.splitter, ObliqueSplitter):
            splitter = OBLIQUE_SPLITTERS[self.splitter](
                criterion,
                self.max_features_,
                min_samples_leaf,
                min_weight_leaf,
                random_state,
                monotonic_cst,
                breiman_shortcut,
                self.feature_combinations_,
            )

        self.tree_ = ObliqueTree(
            self.n_features_in_,
            np.array([1] * self.n_outputs_, dtype=np.intp),
            self.n_outputs_,
            self.n_categories_,
        )

        # Use BestFirst if max_leaf_nodes given; use DepthFirst otherwise
        if max_leaf_nodes < 0:
            builder = DepthFirstTreeBuilder(
                splitter,
                min_samples_split,
                min_samples_leaf,
                min_weight_leaf,
                max_depth,
                self.min_impurity_decrease,
            )
        else:
            builder = BestFirstTreeBuilder(
                splitter,
                min_samples_split,
                min_samples_leaf,
                min_weight_leaf,
                max_depth,
                max_leaf_nodes,
                self.min_impurity_decrease,
            )

        builder.build(self.tree_, X, y, sample_weight)


class PatchObliqueDecisionTreeClassifier(SimMatrixMixin, DecisionTreeClassifier):
    """A oblique decision tree classifier that operates over patches of data.

    A patch oblique decision tree is also known as a manifold oblique decision tree
    (called MORF in :footcite:`Li2019manifold`), where the splitter is aware of
    the structure in the data. For example, in an image, a patch would be contiguous
    along the rows and columns of the image. In a multivariate time-series, a patch
    would be contiguous over time, but possibly discontiguous over the sensors.

    Parameters
    ----------
    criterion : {"gini", "entropy"}, default="gini"
        The function to measure the quality of a split. Supported criteria are
        "gini" for the Gini impurity and "entropy" for the information gain.

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

    max_features : int, float or {"auto", "sqrt", "log2"}, default=None
        The number of features to consider when looking for the best split:

            - If int, then consider `max_features` features at each split.
            - If float, then `max_features` is a fraction and
              `int(max_features * n_features)` features are considered at each
              split.
            - If "auto", then `max_features=sqrt(n_features)`.
            - If "sqrt", then `max_features=sqrt(n_features)`.
            - If "log2", then `max_features=log2(n_features)`.
            - If None, then `max_features=n_features`.

        Note: the search for a split does not stop until at least one
        valid partition of the node samples is found, even if it requires to
        effectively inspect more than ``max_features`` features.

        Note: Compared to axis-aligned Random Forests, one can set
        max_features to a number greater then ``n_features``.

    random_state : int, RandomState instance or None, default=None
        Controls the randomness of the estimator. The features are always
        randomly permuted at each split, even if ``splitter`` is set to
        ``"best"``. When ``max_features < n_features``, the algorithm will
        select ``max_features`` at random at each split before finding the best
        split among them. But the best found split may vary across different
        runs, even if ``max_features=n_features``. That is the case, if the
        improvement of the criterion is identical for several splits and one
        split has to be selected at random. To obtain a deterministic behaviour
        during fitting, ``random_state`` has to be fixed to an integer.
        See :term:`Glossary <random_state>` for details.

    max_leaf_nodes : int, default=None
        Grow a tree with ``max_leaf_nodes`` in best-first fashion.
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

    class_weight : dict, list of dict or "balanced", default=None
        Weights associated with classes in the form ``{class_label: weight}``.
        If None, all classes are supposed to have weight one. For
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

        For multi-output, the weights of each column of y will be multiplied.

        Note that these weights will be multiplied with sample_weight (passed
        through the fit method) if sample_weight is specified.

    min_patch_dims : array-like, optional
        The minimum dimensions of a patch, by default 1 along all dimensions.

    max_patch_dims : array-like, optional
        The maximum dimensions of a patch, by default 1 along all dimensions.

    dim_contiguous : array-like of bool, optional
        Whether or not each patch is sampled contiguously along this dimension.

    data_dims : array-like, optional
        The presumed dimensions of the un-vectorized feature vector, by default
        will be a 1D vector with (1, n_features) shape.

    boundary : optional, str {'wrap'}
        The boundary condition to use when sampling patches, by default None.
        'wrap' corresponds to the boundary condition as is in numpy and scipy.

    feature_weight : array-like of shape (n_samples,n_features,), default=None
        Feature weights. If None, then features are equally weighted as is.
        If provided, then the feature weights are used to weight the
        patches that are generated. The feature weights are used
        as follows: for every patch that is sampled, the feature weights over
        the entire patch is summed and normalizes the patch.

    Attributes
    ----------
    classes_ : ndarray of shape (n_classes,) or list of ndarray
        The classes labels (single output problem),
        or a list of arrays of class labels (multi-output problem).

    feature_importances_ : ndarray of shape (n_features,)
        The impurity-based feature importances.
        The higher, the more important the feature.
        The importance of a feature is computed as the (normalized)
        total reduction of the criterion brought by that feature.  It is also
        known as the Gini importance [4]_.

        Warning: impurity-based feature importances can be misleading for
        high cardinality features (many unique values). See
        :func:`sklearn.inspection.permutation_importance` as an alternative.

    max_features_ : int
        The inferred value of max_features.

    n_classes_ : int or list of int
        The number of classes (for single output problems),
        or a list containing the number of classes for each
        output (for multi-output problems).

    n_features_in_ : int
        Number of features seen during :term:`fit`.

    feature_names_in_ : ndarray of shape (`n_features_in_`,)
        Names of features seen during :term:`fit`. Defined only when `X`
        has feature names that are all strings.

    n_outputs_ : int
        The number of outputs when ``fit`` is performed.

    tree_ : Tree instance
        The underlying Tree object. Please refer to
        ``help(sklearn.tree._tree.Tree)`` for
        attributes of Tree object.

    min_patch_dims_ : array-like
        The minimum dimensions of a patch.

    max_patch_dims_ : array-like
        The maximum dimensions of a patch.

    data_dims_ : array-like
        The presumed dimensions of the un-vectorized feature vector.

    Notes
    -----
    Patches can be 2D masks that are applied onto the data matrix. Following sklearn
    API standards, ``X`` is always a ``(n_samples, n_features)`` array even if
    X is comprised of images, or multivariate-time series. The ``data_width`` and
    ``data_height`` parameters are used to inform the ``PatchObliqueDecisionTreeClassifier``
    of the original structure of the data. It is required that
    ``data_width * data_height = n_features``.

    When users pass in ``X`` to :meth:`fit`, tt is presumed that all vectorization operations
    are done C-contiguously (i.e. the last axis is contiguous).

    Note that for a patch height and width of size 1, the tree is exactly the same as the
    decision tree, albeit with less efficienc optimizations. Therefore, it is always
    recommended to set the range of patch heights and widths based on the structure of your
    expected input data.

    References
    ----------
    .. footbibliography::
    """

    tree_type = "oblique"
    _parameter_constraints = {
        **DecisionTreeClassifier._parameter_constraints,
        "min_patch_dims": ["array-like", None],
        "max_patch_dims": ["array-like", None],
        "data_dims": ["array-like", None],
        "dim_contiguous": ["array-like", None],
        "boundary": [str, None],
        "feature_weight": ["array-like", None],
    }

    def __init__(
        self,
        *,
        criterion="gini",
        splitter="best",
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        min_weight_fraction_leaf=0.0,
        max_features=None,
        random_state=None,
        max_leaf_nodes=None,
        min_impurity_decrease=0.0,
        class_weight=None,
        min_patch_dims=None,
        max_patch_dims=None,
        dim_contiguous=None,
        data_dims=None,
        boundary=None,
        feature_weight=None,
    ):
        super().__init__(
            criterion=criterion,
            splitter=splitter,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            min_weight_fraction_leaf=min_weight_fraction_leaf,
            max_features=max_features,
            max_leaf_nodes=max_leaf_nodes,
            class_weight=class_weight,
            random_state=random_state,
            min_impurity_decrease=min_impurity_decrease,
        )

        self.min_patch_dims = min_patch_dims
        self.max_patch_dims = max_patch_dims
        self.dim_contiguous = dim_contiguous
        self.data_dims = data_dims
        self.boundary = boundary
        self.feature_weight = feature_weight

    def fit(self, X, y, sample_weight=None, check_input=True):
        """Fit tree.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            The training input samples. Internally, it will be converted to
            ``dtype=np.float32`` and if a sparse matrix is provided
            to a sparse ``csc_matrix``.

        y : array-like of shape (n_samples,) or (n_samples, n_outputs)
            The target values (class labels) as integers or strings.

        sample_weight : array-like of shape (n_samples,), default=None
            Sample weights. If None, then samples are equally weighted. Splits
            that would create child nodes with net zero or negative weight are
            ignored while searching for a split in each node. Splits are also
            ignored if they would result in any single class carrying a
            negative weight in either child node.

        check_input : bool, optional
            Whether or not to check input, by default True.
        """
        if check_input:
            # Need to validate separately here.
            # We can't pass multi_output=True because that would allow y to be
            # csr.
            check_X_params = dict(dtype=DTYPE, accept_sparse="csc")
            check_y_params = dict(ensure_2d=False, dtype=None)
            X, y = self._validate_data(X, y, validate_separately=(check_X_params, check_y_params))
            if self.feature_weight is not None:
                self.feature_weight = self._validate_data(
                    self.feature_weight, ensure_2d=True, dtype=DTYPE
                )
                if self.feature_weight.shape != X.shape:
                    raise ValueError(
                        f"feature_weight has shape {self.feature_weight.shape} but X has "
                        f"shape {X.shape}"
                    )
            if issparse(X):
                X.sort_indices()

                if X.indices.dtype != np.intc or X.indptr.dtype != np.intc:
                    raise ValueError("No support for np.int64 index based sparse matrices")

        if self.data_dims is None:
            self.data_dims_ = np.array((1, X.shape[1]))
        else:
            if np.prod(self.data_dims) != X.shape[1]:
                raise RuntimeError(f"Data dimensions {self.data_dims} do not match {X.shape[1]}.")
            self.data_dims_ = np.array(self.data_dims)
        ndim = len(self.data_dims_)

        # validate contiguous parameter
        if self.dim_contiguous is None:
            self.dim_contiguous_ = np.ones((ndim,), dtype=np.bool_)
        else:
            if len(self.dim_contiguous) != ndim:
                raise ValueError(f"Contiguous dimensions should equal {ndim} dimensions.")
            self.dim_contiguous_ = np.array(self.dim_contiguous).astype(np.bool_)

        # validate data height/width
        if self.min_patch_dims is None:
            self.min_patch_dims_ = np.ones((ndim,), dtype=np.intp)
        else:
            self.min_patch_dims_ = np.array(self.min_patch_dims)

        if self.max_patch_dims is None:
            self.max_patch_dims_ = np.ones((ndim,), dtype=np.intp)
            self.max_patch_dims_[-1] = X.shape[1]
        else:
            self.max_patch_dims_ = np.array(self.max_patch_dims)

        if len(self.min_patch_dims_) != ndim:
            raise ValueError(f"Minimum patch dimensions should equal {ndim} dimensions.")
        if len(self.max_patch_dims_) != ndim:
            raise ValueError(f"Maximum patch dimensions should equal {ndim} dimensions.")

        # validate patch parameters
        for idx in range(ndim):
            if self.min_patch_dims_[idx] > self.max_patch_dims_[idx]:
                raise RuntimeError(
                    f"The minimum patch width {self.min_patch_dims_[idx]} is "
                    f"greater than the maximum patch width {self.max_patch_dims_[idx]}"
                )
            if self.min_patch_dims_[idx] > self.data_dims_[idx]:
                raise RuntimeError(
                    f"The minimum patch width {self.min_patch_dims_[idx]} is "
                    f"greater than the data width {self.data_dims_[idx]}"
                )
            if self.max_patch_dims_[idx] > self.data_dims_[idx]:
                raise RuntimeError(
                    f"The maximum patch width {self.max_patch_dims_[idx]} is "
                    f"greater than the data width {self.data_dims_[idx]}"
                )

        return super().fit(X, y, sample_weight, check_input=False)

    def _build_tree(
        self,
        X,
        y,
        sample_weight,
        missing_values_in_feature_mask,
        categorical,
        min_samples_leaf,
        min_weight_leaf,
        max_leaf_nodes,
        min_samples_split,
        max_depth,
        random_state,
    ):
        """Build the actual tree.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            The training input samples. Internally, it will be converted to
            ``dtype=np.float32`` and if a sparse matrix is provided
            to a sparse ``csc_matrix``.

        y : array-like of shape (n_samples,) or (n_samples, n_outputs)
            The target values (class labels) as integers or strings.

        sample_weight : array-like of shape (n_samples,), default=None
            Sample weights. If None, then samples are equally weighted. Splits
            that would create child nodes with net zero or negative weight are
            ignored while searching for a split in each node. Splits are also
            ignored if they would result in any single class carrying a
            negative weight in either child node.

        min_samples_leaf : int or float
            The minimum number of samples required to be at a leaf node.

        min_weight_leaf : float, default=0.0
           The minimum weighted fraction of the sum total of weights.

        max_leaf_nodes : int, default=None
            Grow a tree with ``max_leaf_nodes`` in best-first fashion.

        min_samples_split : int or float, default=2
            The minimum number of samples required to split an internal node.

        max_depth : int, default=None
            The maximum depth of the tree. If None, then nodes are expanded until
            all leaves are pure or until all leaves contain less than
            min_samples_split samples.

        random_state : int, RandomState instance or None, default=None
            Controls the randomness of the estimator.
        """
        monotonic_cst = None
        self.n_categories_ = np.array(
            [
                np.int32(X[:, i].max()) + 1 if i in categorical else -1
                for i in range(self.n_features_in_)
            ],
            dtype=np.int32,
        )
        breiman_shortcut = False

        # Build tree
        criterion = self.criterion
        if not isinstance(criterion, BaseCriterion):
            criterion = CRITERIA_CLF[self.criterion](self.n_outputs_, self.n_classes_)
        else:
            # Make a deepcopy in case the criterion has mutable attributes that
            # might be shared and modified concurrently during parallel fitting
            criterion = copy.deepcopy(criterion)

        splitter = self.splitter
        if issparse(X):
            raise ValueError(
                "Sparse input is not supported for oblique trees. "
                "Please convert your data to a dense array."
            )
        else:
            PATCH_SPLITTERS = PATCH_DENSE_SPLITTERS

        if not isinstance(self.splitter, PatchSplitter):
            splitter = PATCH_SPLITTERS[self.splitter](
                criterion,
                self.max_features_,
                min_samples_leaf,
                min_weight_leaf,
                random_state,
                monotonic_cst,
                breiman_shortcut,
                self.min_patch_dims_,
                self.max_patch_dims_,
                self.dim_contiguous_,
                self.data_dims_,
                self.boundary,
                self.feature_weight,
            )

        self.tree_ = ObliqueTree(
            self.n_features_in_, self.n_classes_, self.n_outputs_, self.n_categories_
        )

        # Use BestFirst if max_leaf_nodes given; use DepthFirst otherwise
        if max_leaf_nodes < 0:
            builder = DepthFirstTreeBuilder(
                splitter,
                min_samples_split,
                min_samples_leaf,
                min_weight_leaf,
                max_depth,
                self.min_impurity_decrease,
            )
        else:
            builder = BestFirstTreeBuilder(
                splitter,
                min_samples_split,
                min_samples_leaf,
                min_weight_leaf,
                max_depth,
                max_leaf_nodes,
                self.min_impurity_decrease,
            )

        builder.build(self.tree_, X, y, sample_weight)

        if self.n_outputs_ == 1:
            self.n_classes_ = self.n_classes_[0]
            self.classes_ = self.classes_[0]

    def _more_tags(self):
        # XXX: nans should be supportable in SPORF by just using RF-like splits on missing values
        # However, for MORF it is not supported
        allow_nan = False
        return {"multilabel": True, "allow_nan": allow_nan}


class PatchObliqueDecisionTreeRegressor(SimMatrixMixin, DecisionTreeRegressor):
    """A oblique decision tree regressor that operates over patches of data.

    A patch oblique decision tree is also known as a manifold oblique decision tree
    (called MORF in :footcite:`Li2019manifold`), where the splitter is aware of
    the structure in the data. For example, in an image, a patch would be contiguous
    along the rows and columns of the image. In a multivariate time-series, a patch
    would be contiguous over time, but possibly discontiguous over the sensors.

    Parameters
    ----------
    criterion : {"squared_error", "friedman_mse", "absolute_error", \
            "poisson"}, default="squared_error"
        The function to measure the quality of a split. Supported criteria
        are "squared_error" for the mean squared error, which is equal to
        variance reduction as feature selection criterion and minimizes the L2
        loss using the mean of each terminal node, "friedman_mse", which uses
        mean squared error with Friedman's improvement score for potential
        splits, "absolute_error" for the mean absolute error, which minimizes
        the L1 loss using the median of each terminal node, and "poisson" which
        uses reduction in Poisson deviance to find splits.

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

    max_features : int, float or {"auto", "sqrt", "log2"}, default=None
        The number of features to consider when looking for the best split:

            - If int, then consider `max_features` features at each split.
            - If float, then `max_features` is a fraction and
              `int(max_features * n_features)` features are considered at each
              split.
            - If "auto", then `max_features=sqrt(n_features)`.
            - If "sqrt", then `max_features=sqrt(n_features)`.
            - If "log2", then `max_features=log2(n_features)`.
            - If None, then `max_features=n_features`.

        Note: the search for a split does not stop until at least one
        valid partition of the node samples is found, even if it requires to
        effectively inspect more than ``max_features`` features.
        Note: Compared to axis-aligned Random Forests, one can set
        max_features to a number greater then ``n_features``.

    random_state : int, RandomState instance or None, default=None
        Controls the randomness of the estimator. The features are always
        randomly permuted at each split, even if ``splitter`` is set to
        ``"best"``. When ``max_features < n_features``, the algorithm will
        select ``max_features`` at random at each split before finding the best
        split among them. But the best found split may vary across different
        runs, even if ``max_features=n_features``. That is the case, if the
        improvement of the criterion is identical for several splits and one
        split has to be selected at random. To obtain a deterministic behaviour
        during fitting, ``random_state`` has to be fixed to an integer.
        See :term:`Glossary <random_state>` for details.

    max_leaf_nodes : int, default=None
        Grow a tree with ``max_leaf_nodes`` in best-first fashion.
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

    min_patch_dims : array-like, optional
        The minimum dimensions of a patch, by default 1 along all dimensions.

    max_patch_dims : array-like, optional
        The maximum dimensions of a patch, by default 1 along all dimensions.

    dim_contiguous : array-like of bool, optional
        Whether or not each patch is sampled contiguously along this dimension.

    data_dims : array-like, optional
        The presumed dimensions of the un-vectorized feature vector, by default
        will be a 1D vector with (1, n_features) shape.

    boundary : optional, str {'wrap'}
        The boundary condition to use when sampling patches, by default None.
        'wrap' corresponds to the boundary condition as is in numpy and scipy.

    feature_weight : array-like of shape (n_samples,n_features,), default=None
        Feature weights. If None, then features are equally weighted as is.
        If provided, then the feature weights are used to weight the
        patches that are generated. The feature weights are used
        as follows: for every patch that is sampled, the feature weights over
        the entire patch is summed and normalizes the patch.

    Attributes
    ----------
    feature_importances_ : ndarray of shape (n_features,)
        The impurity-based feature importances.
        The higher, the more important the feature.
        The importance of a feature is computed as the (normalized)
        total reduction of the criterion brought by that feature.  It is also
        known as the Gini importance [4]_.

        Warning: impurity-based feature importances can be misleading for
        high cardinality features (many unique values). See
        :func:`sklearn.inspection.permutation_importance` as an alternative.

    max_features_ : int
        The inferred value of max_features.

    n_features_in_ : int
        Number of features seen during :term:`fit`.

    feature_names_in_ : ndarray of shape (`n_features_in_`,)
        Names of features seen during :term:`fit`. Defined only when `X`
        has feature names that are all strings.

    n_outputs_ : int
        The number of outputs when ``fit`` is performed.

    tree_ : Tree instance
        The underlying Tree object. Please refer to
        ``help(sklearn.tree._tree.Tree)`` for
        attributes of Tree object.

    min_patch_dims_ : array-like
        The minimum dimensions of a patch.

    max_patch_dims_ : array-like
        The maximum dimensions of a patch.

    data_dims_ : array-like
        The presumed dimensions of the un-vectorized feature vector.

    Notes
    -----
    Patches are 2D masks that are applied onto the data matrix. Following sklearn
    API standards, ``X`` is always a ``(n_samples, n_features)`` array even if
    X is comprised of images, or multivariate-time series. The ``data_width`` and
    ``data_height`` parameters are used to inform the ``PatchObliqueDecisionTreeRegressor``
    of the original structure of the data. It is required that
    ``data_width * data_height = n_features``.

    When users pass in ``X`` to :meth:`fit`, tt is presumed that all vectorization operations
    are done C-contiguously (i.e. the last axis is contiguous).

    Note that for a patch height and width of size 1, the tree is exactly the same as the
    decision tree, albeit with less efficienc optimizations. Therefore, it is always
    recommended to set the range of patch heights and widths based on the structure of your
    expected input data.

    References
    ----------
    .. footbibliography::

    Examples
    --------
    >>> from sklearn.datasets import load_diabetes
    >>> from sklearn.model_selection import cross_val_score
    >>> X, y = load_diabetes(return_X_y=True)
    >>> from sktree.tree import PatchObliqueDecisionTreeRegressor as RGS
    >>> regressor = RGS(random_state=0)
    >>> cross_val_score(regressor, X, y, cv=10)
    ...                    # doctest: +SKIP
    ...
    array([-0.10163671, -0.78786738,  0.01490768,  0.32737289, -0.24816698,
            0.41881754,  0.0588273 , -1.48722913, -0.07927208, -0.15600762])
    """

    tree_type = "oblique"
    _parameter_constraints = {
        **DecisionTreeRegressor._parameter_constraints,
        "min_patch_dims": ["array-like", None],
        "max_patch_dims": ["array-like", None],
        "data_dims": ["array-like", None],
        "dim_contiguous": ["array-like", None],
        "boundary": [str, None],
        "feature_weight": ["array-like", None],
    }

    def __init__(
        self,
        *,
        criterion="squared_error",
        splitter="best",
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        min_weight_fraction_leaf=0.0,
        max_features=None,
        random_state=None,
        max_leaf_nodes=None,
        min_impurity_decrease=0.0,
        min_patch_dims=None,
        max_patch_dims=None,
        dim_contiguous=None,
        data_dims=None,
        boundary=None,
        feature_weight=None,
    ):
        super().__init__(
            criterion=criterion,
            splitter=splitter,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            min_weight_fraction_leaf=min_weight_fraction_leaf,
            max_features=max_features,
            max_leaf_nodes=max_leaf_nodes,
            random_state=random_state,
            min_impurity_decrease=min_impurity_decrease,
        )

        self.min_patch_dims = min_patch_dims
        self.max_patch_dims = max_patch_dims
        self.dim_contiguous = dim_contiguous
        self.data_dims = data_dims
        self.boundary = boundary
        self.feature_weight = feature_weight

    def fit(self, X, y, sample_weight=None, check_input=True):
        """Fit tree.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            The training input samples. Internally, it will be converted to
            ``dtype=np.float32`` and if a sparse matrix is provided
            to a sparse ``csc_matrix``.

        y : array-like of shape (n_samples,) or (n_samples, n_outputs)
            The target values (real numbers). Use ``dtype=np.float64`` and
            ``order='C'`` for maximum efficiency.

        sample_weight : array-like of shape (n_samples,), default=None
            Sample weights. If None, then samples are equally weighted. Splits
            that would create child nodes with net zero or negative weight are
            ignored while searching for a split in each node.

        check_input : bool, optional
            Whether or not to check input, by default True.
        """
        if check_input:
            # Need to validate separately here.
            # We can't pass multi_output=True because that would allow y to be
            # csr.
            check_X_params = dict(dtype=DTYPE, accept_sparse="csc")
            check_y_params = dict(ensure_2d=False, dtype=None)
            X, y = self._validate_data(X, y, validate_separately=(check_X_params, check_y_params))
            if self.feature_weight is not None:
                self.feature_weight = self._validate_data(
                    self.feature_weight, ensure_2d=True, dtype=DTYPE
                )
                if self.feature_weight.shape != X.shape:
                    raise ValueError(
                        f"feature_weight has shape {self.feature_weight.shape} but X has "
                        f"shape {X.shape}"
                    )
            if issparse(X):
                X.sort_indices()

                if X.indices.dtype != np.intc or X.indptr.dtype != np.intc:
                    raise ValueError("No support for np.int64 index based sparse matrices")

        if self.data_dims is None:
            self.data_dims_ = np.array((1, X.shape[1]))
        else:
            if np.prod(self.data_dims) != X.shape[1]:
                raise RuntimeError(f"Data dimensions {self.data_dims} do not match {X.shape[1]}.")
            self.data_dims_ = np.array(self.data_dims)
        ndim = len(self.data_dims_)

        # validate contiguous parameter
        if self.dim_contiguous is None:
            self.dim_contiguous_ = np.ones((ndim,), dtype=np.bool_)
        else:
            if len(self.dim_contiguous) != ndim:
                raise ValueError(f"Contiguous dimensions should equal {ndim} dimensions.")
            self.dim_contiguous_ = np.array(self.dim_contiguous).astype(np.bool_)

        # validate data height/width
        if self.min_patch_dims is None:
            self.min_patch_dims_ = np.ones((ndim,), dtype=np.intp)
        else:
            self.min_patch_dims_ = np.array(self.min_patch_dims)

        if self.max_patch_dims is None:
            self.max_patch_dims_ = np.ones((ndim,), dtype=np.intp)
            self.max_patch_dims_[-1] = X.shape[1]
        else:
            self.max_patch_dims_ = np.array(self.max_patch_dims)

        if len(self.min_patch_dims_) != ndim:
            raise ValueError(f"Minimum patch dimensions should equal {ndim} dimensions.")
        if len(self.max_patch_dims_) != ndim:
            raise ValueError(f"Maximum patch dimensions should equal {ndim} dimensions.")

        # validate patch parameters
        for idx in range(ndim):
            if self.min_patch_dims_[idx] > self.max_patch_dims_[idx]:
                raise RuntimeError(
                    f"The minimum patch width {self.min_patch_dims_[idx]} is "
                    f"greater than the maximum patch width {self.max_patch_dims_[idx]}"
                )
            if self.min_patch_dims_[idx] > self.data_dims_[idx]:
                raise RuntimeError(
                    f"The minimum patch width {self.min_patch_dims_[idx]} is "
                    f"greater than the data width {self.data_dims_[idx]}"
                )
            if self.max_patch_dims_[idx] > self.data_dims_[idx]:
                raise RuntimeError(
                    f"The maximum patch width {self.max_patch_dims_[idx]} is "
                    f"greater than the data width {self.data_dims_[idx]}"
                )

        return super().fit(X, y, sample_weight, check_input=False)

    def _build_tree(
        self,
        X,
        y,
        sample_weight,
        missing_values_in_feature_mask,
        categorical,
        min_samples_leaf,
        min_weight_leaf,
        max_leaf_nodes,
        min_samples_split,
        max_depth,
        random_state,
    ):
        """Build the actual tree.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            The training input samples. Internally, it will be converted to
            ``dtype=np.float32`` and if a sparse matrix is provided
            to a sparse ``csc_matrix``.

        y : array-like of shape (n_samples,) or (n_samples, n_outputs)
            The target values (real numbers). Use ``dtype=np.float64`` and
            ``order='C'`` for maximum efficiency.

        sample_weight : array-like of shape (n_samples,), default=None
            Sample weights. If None, then samples are equally weighted. Splits
            that would create child nodes with net zero or negative weight are
            ignored while searching for a split in each node.

        min_samples_leaf : int or float
            The minimum number of samples required to be at a leaf node.

        min_weight_leaf : float, default=0.0
           The minimum weighted fraction of the sum total of weights.

        max_leaf_nodes : int, default=None
            Grow a tree with ``max_leaf_nodes`` in best-first fashion.

        min_samples_split : int or float, default=2
            The minimum number of samples required to split an internal node.

        max_depth : int, default=None
            The maximum depth of the tree. If None, then nodes are expanded until
            all leaves are pure or until all leaves contain less than
            min_samples_split samples.

        random_state : int, RandomState instance or None, default=None
            Controls the randomness of the estimator.
        """
        monotonic_cst = None
        self.n_categories_ = np.array(
            [
                np.int32(X[:, i].max()) + 1 if i in categorical else -1
                for i in range(self.n_features_in_)
            ],
            dtype=np.int32,
        )
        breiman_shortcut = False
        n_samples = X.shape[0]

        # Build tree
        criterion = self.criterion
        if not isinstance(criterion, BaseCriterion):
            criterion = CRITERIA_REG[self.criterion](self.n_outputs_, n_samples)
        else:
            # Make a deepcopy in case the criterion has mutable attributes that
            # might be shared and modified concurrently during parallel fitting
            criterion = copy.deepcopy(criterion)

        splitter = self.splitter
        if issparse(X):
            raise ValueError(
                "Sparse input is not supported for oblique trees. "
                "Please convert your data to a dense array."
            )
        else:
            PATCH_SPLITTERS = PATCH_DENSE_SPLITTERS

        if not isinstance(self.splitter, PatchSplitter):
            splitter = PATCH_SPLITTERS[self.splitter](
                criterion,
                self.max_features_,
                min_samples_leaf,
                min_weight_leaf,
                random_state,
                monotonic_cst,
                breiman_shortcut,
                self.min_patch_dims_,
                self.max_patch_dims_,
                self.dim_contiguous_,
                self.data_dims_,
                self.boundary,
                self.feature_weight,
            )

        self.tree_ = ObliqueTree(
            self.n_features_in_,
            np.array([1] * self.n_outputs_, dtype=np.intp),
            self.n_outputs_,
            self.n_categories_,
        )

        # Use BestFirst if max_leaf_nodes given; use DepthFirst otherwise
        if max_leaf_nodes < 0:
            builder = DepthFirstTreeBuilder(
                splitter,
                min_samples_split,
                min_samples_leaf,
                min_weight_leaf,
                max_depth,
                self.min_impurity_decrease,
            )
        else:
            builder = BestFirstTreeBuilder(
                splitter,
                min_samples_split,
                min_samples_leaf,
                min_weight_leaf,
                max_depth,
                max_leaf_nodes,
                self.min_impurity_decrease,
            )

        builder.build(self.tree_, X, y, sample_weight)

    def _more_tags(self):
        # XXX: nans should be supportable in SPORF by just using RF-like splits on missing values
        # However, for MORF it is not supported
        allow_nan = False
        return {"multilabel": True, "allow_nan": allow_nan}
