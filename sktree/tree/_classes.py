import copy

import numpy as np
from scipy.sparse import issparse
from sklearn.base import ClusterMixin, TransformerMixin
from sklearn.cluster import AgglomerativeClustering
from sklearn.tree import BaseDecisionTree
from sklearn.tree import _tree as _sklearn_tree
from sklearn.utils.validation import check_is_fitted

from . import _unsup_criterion, _unsup_oblique_splitter, _unsup_splitter  # type: ignore
from ._unsup_criterion import UnsupervisedCriterion
from ._unsup_oblique_splitter import UnsupervisedObliqueSplitter
from ._unsup_oblique_tree import UnsupervisedObliqueTree
from ._unsup_splitter import UnsupervisedSplitter
from ._unsup_tree import (  # type: ignore
    UnsupervisedBestFirstTreeBuilder,
    UnsupervisedDepthFirstTreeBuilder,
    UnsupervisedTree,
)

DTYPE = _sklearn_tree.DTYPE
DOUBLE = _sklearn_tree.DOUBLE

CRITERIA = {"twomeans": _unsup_criterion.TwoMeans}

SPLITTERS = {
    "best": _unsup_splitter.BestUnsupervisedSplitter,
}

OBLIQUE_SPLITTERS = {"best": _unsup_oblique_splitter.BestObliqueUnsupervisedSplitter}


class UnsupervisedDecisionTree(TransformerMixin, ClusterMixin, BaseDecisionTree):
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
        min_samples_split=2,
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

        super().fit(X, None, sample_weight, check_input)

        # apply to the leaves
        n_samples = X.shape[0]
        X_leaves = self.apply(X)

        # now compute the affinity matrix and set it
        self.affinity_matrix_ = self._compute_affinity_matrix(X_leaves)

        # compute the labels and set it
        if n_samples >= 2:
            self.labels_ = self._assign_labels(self.affinity_matrix_)

        return self

    def _build_tree(
        self,
        X,
        y,
        sample_weight,
        is_classification,
        min_samples_leaf,
        min_weight_leaf,
        max_leaf_nodes,
        min_samples_split,
        max_depth,
        random_state,
    ):
        criterion = self.criterion
        if not isinstance(criterion, UnsupervisedCriterion):
            criterion = CRITERIA[self.criterion]()
        else:
            # Make a deepcopy in case the criterion has mutable attributes that
            # might be shared and modified concurrently during parallel fitting
            criterion = copy.deepcopy(criterion)

        splitter = self.splitter
        if not isinstance(self.splitter, UnsupervisedSplitter):
            splitter = SPLITTERS[self.splitter](
                criterion,
                self.max_features_,
                min_samples_leaf,
                min_weight_leaf,
                random_state,
            )

        self.tree_ = UnsupervisedTree(self.n_features_in_)

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
        # apply to the leaves
        X_leaves = self.apply(X)

        # now compute the affinity matrix and set it
        affinity_matrix = self._compute_affinity_matrix(X_leaves)
        return affinity_matrix

    def _compute_affinity_matrix(self, X_leaves):
        """Compute the proximity matrix of samples in X.

        Parameters
        ----------
        X_leaves : ndarray of shape (n_samples,)
            For each datapoint x in X and for each tree in the forest,
            is the index of the leaf x ends up in.

        Returns
        -------
        prox_matrix : array-like of shape (n_samples, n_samples)
        """
        n_samples = X_leaves.shape[0]
        aff_matrix = np.zeros((n_samples, n_samples), dtype=np.int32)

        # for every unique leaf in this dataset, count all co-occurrences of samples
        # in the same leaf
        for leaf in np.unique(X_leaves):
            # find out which samples occur with this leaf
            samples_in_leaf = np.atleast_1d(np.argwhere(X_leaves == leaf).squeeze())
            aff_matrix[np.ix_(samples_in_leaf, samples_in_leaf)] += 1

        return aff_matrix

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

    def __init__(
        self,
        *,
        criterion="twomeans",
        splitter="best",
        max_depth=None,
        min_samples_split=2,
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
        is_classification,
        min_samples_leaf,
        min_weight_leaf,
        max_leaf_nodes,
        min_samples_split,
        max_depth,
        random_state,
    ):
        criterion = self.criterion
        if not isinstance(criterion, UnsupervisedCriterion):
            criterion = CRITERIA[self.criterion]()
        else:
            # Make a deepcopy in case the criterion has mutable attributes that
            # might be shared and modified concurrently during parallel fitting
            criterion = copy.deepcopy(criterion)

        splitter = self.splitter
        if not isinstance(self.splitter, UnsupervisedObliqueSplitter):
            splitter = OBLIQUE_SPLITTERS[self.splitter](
                criterion,
                self.max_features_,
                min_samples_leaf,
                min_weight_leaf,
                self.feature_combinations,
                random_state,
            )

        self.tree_ = UnsupervisedObliqueTree(self.n_features_in_)

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
