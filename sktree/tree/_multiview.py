import copy
import math
import numbers
from numbers import Integral, Real

import numpy as np
from scipy.sparse import issparse
from sklearn.utils import check_random_state
from sklearn.utils._param_validation import Interval, RealNotInt, StrOptions

from .._lib.sklearn.tree import DecisionTreeClassifier, _criterion
from .._lib.sklearn.tree import _tree as _sklearn_tree
from .._lib.sklearn.tree._criterion import BaseCriterion
from .._lib.sklearn.tree._tree import BestFirstTreeBuilder, DepthFirstTreeBuilder
from . import _oblique_splitter
from ._neighbors import SimMatrixMixin
from ._oblique_splitter import ObliqueSplitter
from ._oblique_tree import ObliqueTree

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

DENSE_SPLITTERS = {
    "best": _oblique_splitter.MultiViewSplitter,
}

OBLIQUE_DENSE_SPLITTERS = {
    "best": _oblique_splitter.MultiViewObliqueSplitter,
}


class MultiViewDecisionTreeClassifier(SimMatrixMixin, DecisionTreeClassifier):
    """A multi-view axis-aligned decision tree classifier.

    This is an experimental feature that applies an oblique decision tree to
    multiple feature-sets concatenated across columns in ``X``.

    Parameters
    ----------
    criterion : {"gini", "entropy"}, default="gini"
        The function to measure the quality of a split. Supported criteria are
        "gini" for the Gini impurity and "entropy" for the information gain.

    splitter : {"best"}, default="best"
        The strategy used to choose the split at each node.

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

    max_features : array-like, int, float or {"auto", "sqrt", "log2"}, default=None
        The number of features to consider when looking for the best split:

            - If int, then consider `max_features` features at each split.
            - If float, then `max_features` is a fraction and
              `int(max_features * n_features)` features are considered at each
              split.
            - If "auto", then `max_features=sqrt(n_features)`.
            - If "sqrt", then `max_features=sqrt(n_features)`.
            - If "log2", then `max_features=log2(n_features)`.
            - If None, then `max_features=n_features`.

        If array-like, then `max_features` is the number of features to consider
        for each feature set following the same logic as above, where
        ``n_features`` is the number of features in the respective feature set.

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

    ccp_alpha : non-negative float, default=0.0
        Not used.

    store_leaf_values : bool, default=False
        Whether to store the leaf values.

    monotonic_cst : array-like of int of shape (n_features), default=None
        Indicates the monotonicity constraint to enforce on each feature.
          - 1: monotonic increase
          - 0: no constraint
          - -1: monotonic decrease

        Not used.

    feature_set_ends : array-like of int of shape (n_feature_sets,), default=None
        The indices of the end of each feature set. For example, if the first
        feature set is the first 10 features, and the second feature set is the
        next 20 features, then ``feature_set_ends = [10, 30]``. If ``None``,
        then this will assume that there is only one feature set.

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

    feature_set_ends_ : array-like of int of shape (n_feature_sets,)
        The indices of the end of each feature set.

    n_feature_sets_ : int
        The number of feature sets.

    max_features_per_set_ : array-like of int of shape (n_feature_sets,)
        The number of features to sample per feature set. If ``None``, then
        ``max_features`` is applied to the entire feature space.

    See Also
    --------
    sklearn.tree.DecisionTreeClassifier : An axis-aligned decision tree classifier.
    """

    tree_type = "oblique"

    _parameter_constraints = {
        **DecisionTreeClassifier._parameter_constraints,
        "feature_set_ends": ["array-like", None],
    }
    _parameter_constraints.pop("max_features")
    _parameter_constraints["max_features"] = [
        Interval(Integral, 1, None, closed="left"),
        Interval(RealNotInt, 0.0, 1.0, closed="right"),
        StrOptions({"sqrt", "log2"}),
        "array-like",
        None,
    ]

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
        ccp_alpha=0.0,
        store_leaf_values=False,
        monotonic_cst=None,
        feature_set_ends=None,
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
            ccp_alpha=ccp_alpha,
            store_leaf_values=store_leaf_values,
            monotonic_cst=monotonic_cst,
        )

        self.feature_set_ends = feature_set_ends
        self._max_features_arr = None

    def _build_tree(
        self,
        X,
        y,
        sample_weight,
        missing_values_in_feature_mask,
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
        self.monotonic_cst_ = monotonic_cst
        _, n_features = X.shape

        self._feature_combinations_ = 1

        # Build tree
        criterion = self.criterion
        if not isinstance(criterion, BaseCriterion):
            criterion = CRITERIA_CLF[self.criterion](self.n_outputs_, self.n_classes_)
        else:
            # Make a deepcopy in case the criterion has mutable attributes that
            # might be shared and modified concurrently during parallel fitting
            criterion = copy.deepcopy(criterion)

        if self.feature_set_ends is None:
            self.feature_set_ends_ = np.asarray([n_features], dtype=np.intp)
        else:
            self.feature_set_ends_ = np.atleast_1d(self.feature_set_ends).astype(np.intp)
        self.n_feature_sets_ = len(self.feature_set_ends_)
        if self.feature_set_ends_[-1] != n_features:
            raise ValueError(
                f"The last feature set end must be equal to the number of features, "
                f"{n_features}, but got {self.feature_set_ends_[-1]}."
            )

        splitter = self.splitter
        if issparse(X):
            raise ValueError(
                "Sparse input is not supported for oblique trees. "
                "Please convert your data to a dense array."
            )
        else:
            SPLITTERS = DENSE_SPLITTERS

        if isinstance(self._max_features_arr, (Integral, Real, str, type(None))):
            max_features_arr_ = [self._max_features_arr] * self.n_feature_sets_
        else:
            if not isinstance(self._max_features_arr, (list, np.ndarray)):
                raise ValueError(
                    f"max_features must be an array-like, int, float, str, or None; "
                    f"got {type(self._max_features_arr)}"
                )
            if len(self._max_features_arr) != self.n_feature_sets_:
                raise ValueError(
                    f"max_features must be an array-like of length {self.n_feature_sets_}; "
                    f"got {len(self.max_features)}"
                )
            max_features_arr_ = self._max_features_arr

        self.n_features_in_set_ = []
        # XXX: experimental
        # we can replace max_features_ here based on whether or not uniform logic over
        # feature sets
        max_features_per_set = []
        n_features_in_prev = 0
        for idx in range(self.n_feature_sets_):
            max_features = max_features_arr_[idx]

            n_features_in_ = self.feature_set_ends_[idx] - n_features_in_prev
            n_features_in_prev += n_features_in_
            self.n_features_in_set_.append(n_features_in_)
            if isinstance(max_features, str):
                if max_features == "sqrt":
                    max_features = max(1, math.ceil(np.sqrt(n_features_in_)))
                elif max_features == "log2":
                    max_features = max(1, math.ceil(np.log2(n_features_in_)))
            elif max_features is None:
                max_features = n_features_in_
            elif isinstance(max_features, numbers.Integral):
                max_features = max_features
            else:  # float
                if max_features > 0.0:
                    max_features = max(1, math.ceil(max_features * n_features_in_))
                else:
                    max_features = 0

            if max_features > n_features_in_:
                raise ValueError(
                    f"max_features must be less than or equal to "
                    f"the number of features in feature set {idx}: {n_features_in_}, but "
                    f"max_features = {max_features} when applying sampling"
                    f"per feature set."
                )

            max_features_per_set.append(max_features)
        self.max_features_ = np.sum(max_features_per_set)
        if self.max_features_ > n_features:
            raise ValueError(
                "max_features is greater than the number of features: "
                f"{max_features} > {n_features}."
                "This should not be possible. Please submit a bug report."
            )
        self.max_features_per_set_ = np.asarray(max_features_per_set, dtype=np.intp)
        # the total number of features to sample per split
        self.max_features_ = np.sum(self.max_features_per_set_)

        if not isinstance(self.splitter, ObliqueSplitter):
            splitter = SPLITTERS[self.splitter](
                criterion,
                self.max_features_,
                min_samples_leaf,
                min_weight_leaf,
                random_state,
                monotonic_cst,
                self._feature_combinations_,
                self.feature_set_ends_,
                self.n_feature_sets_,
                self.max_features_per_set_,
            )

        self.tree_ = ObliqueTree(self.n_features_in_, self.n_classes_, self.n_outputs_)

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

        builder.build(self.tree_, X, y, sample_weight, None)

        if self.n_outputs_ == 1:
            self.n_classes_ = self.n_classes_[0]
            self.classes_ = self.classes_[0]

    def _update_tree(self, X, y, sample_weight):
        # Update tree
        max_leaf_nodes = -1 if self.max_leaf_nodes is None else self.max_leaf_nodes
        min_samples_split = self.min_samples_split_
        min_samples_leaf = self.min_samples_leaf_
        min_weight_leaf = self.min_weight_leaf_
        # set decision-tree model parameters
        max_depth = np.iinfo(np.int32).max if self.max_depth is None else self.max_depth

        monotonic_cst = self.monotonic_cst_

        # Build tree
        # Note: this reconstructs the builder with the same state it had during the
        # initial fit. This is necessary because the builder is not saved as part
        # of the class, and thus the state may be lost if pickled/unpickled.
        criterion = self.criterion
        if not isinstance(criterion, BaseCriterion):
            criterion = CRITERIA_CLF[self.criterion](self.n_outputs_, self._n_classes_)
        else:
            # Make a deepcopy in case the criterion has mutable attributes that
            # might be shared and modified concurrently during parallel fitting
            criterion = copy.deepcopy(criterion)

        random_state = check_random_state(self.random_state)

        splitter = self.splitter
        if issparse(X):
            raise ValueError(
                "Sparse input is not supported for oblique trees. "
                "Please convert your data to a dense array."
            )
        else:
            SPLITTERS = DENSE_SPLITTERS
        if not isinstance(self.splitter, ObliqueSplitter):
            splitter = SPLITTERS[self.splitter](
                criterion,
                self.max_features_,
                min_samples_leaf,
                min_weight_leaf,
                random_state,
                monotonic_cst,
                self._feature_combinations_,
                self.feature_set_ends_,
                self.n_feature_sets_,
                self.max_features_per_set_,
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
                self.store_leaf_values,
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
                self.store_leaf_values,
            )
        builder.initialize_node_queue(self.tree_, X, y, sample_weight)
        builder.build(self.tree_, X, y, sample_weight)

        self._prune_tree()
        return self

    def fit(self, X, y, sample_weight=None, check_input=True, classes=None):
        """Build a decision tree classifier from the training set (X, y).

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

        check_input : bool, default=True
            Allow to bypass several input checking.
            Don't use this parameter unless you know what you're doing.

        classes : array-like of shape (n_classes,), default=None
            List of all the classes that can possibly appear in the y vector.

        Returns
        -------
        self : MultiViewDecisionTreeClassifier
            Fitted estimator.
        """
        return self._fit(
            X, y, sample_weight=sample_weight, check_input=check_input, classes=classes
        )

    def _fit(
        self,
        X,
        y,
        sample_weight=None,
        check_input=True,
        missing_values_in_feature_mask=None,
        classes=None,
    ):
        # XXX: BaseDecisionTree does a check that requires max_features to not be a list/array-like
        # so we need to temporarily set it to an acceptable value
        # in the meantime, we will reset:
        #  - self.max_features_ to the original value
        #  - self.max_features_arr contains a possible array-like setting of max_features
        self._max_features_arr = self.max_features
        self.max_features = None
        super()._fit(X, y, sample_weight, check_input, missing_values_in_feature_mask, classes)
        self.max_features = self._max_features_arr
        return self

    @property
    def _inheritable_fitted_attribute(self):
        """Define additional attributes to pass onto a parent meta tree-estimator.

        Used for passing parameters to HonestTreeClassifier.
        """
        return [
            "max_features_",
            "feature_set_ends_",
            "n_feature_sets_",
            "n_features_in_set_",
            "max_features_per_set_",
        ]


class MultiViewObliqueDecisionTreeClassifier(SimMatrixMixin, DecisionTreeClassifier):
    """A multi-view OBLIQUE decision tree classifier.

    This is an experimental feature that applies an oblique decision tree to
    multiple feature-sets concatenated across columns in ``X``.

    Parameters
    ----------
    criterion : {"gini", "entropy"}, default="gini"
        The function to measure the quality of a split. Supported criteria are
        "gini" for the Gini impurity and "entropy" for the information gain.

    splitter : {"best"}, default="best"
        The strategy used to choose the split at each node.

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

    max_features : array-like, int, float or {"auto", "sqrt", "log2"}, default=None
        The number of features to consider when looking for the best split:

            - If int, then consider `max_features` features at each split.
            - If float, then `max_features` is a fraction and
              `int(max_features * n_features)` features are considered at each
              split.
            - If "auto", then `max_features=sqrt(n_features)`.
            - If "sqrt", then `max_features=sqrt(n_features)`.
            - If "log2", then `max_features=log2(n_features)`.
            - If None, then `max_features=n_features`.

        If array-like, then `max_features` is the number of features to consider
        for each feature set following the same logic as above, where
        ``n_features`` is the number of features in the respective feature set.

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

    ccp_alpha : non-negative float, default=0.0
        Not used.

    store_leaf_values : bool, default=False
        Whether to store the leaf values.

    monotonic_cst : array-like of int of shape (n_features), default=None
        Indicates the monotonicity constraint to enforce on each feature.
          - 1: monotonic increase
          - 0: no constraint
          - -1: monotonic decrease

        Not used.

    feature_set_ends : array-like of int of shape (n_feature_sets,), default=None
        The indices of the end of each feature set. For example, if the first
        feature set is the first 10 features, and the second feature set is the
        next 20 features, then ``feature_set_ends = [10, 30]``. If ``None``,
        then this will assume that there is only one feature set.

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

    feature_set_ends_ : array-like of int of shape (n_feature_sets,)
        The indices of the end of each feature set.

    n_feature_sets_ : int
        The number of feature sets.

    max_features_per_set_ : array-like of int of shape (n_feature_sets,)
        The number of features to sample per feature set. If ``None``, then
        ``max_features`` is applied to the entire feature space.

    See Also
    --------
    sklearn.tree.DecisionTreeClassifier : An axis-aligned decision tree classifier.
    """

    tree_type = "oblique"

    _parameter_constraints = {
        **DecisionTreeClassifier._parameter_constraints,
        "feature_set_ends": ["array-like", None],
        "feature_combinations": [
            Interval(Real, 1.0, None, closed="left"),
            None,
        ],
    }
    _parameter_constraints.pop("max_features")
    _parameter_constraints["max_features"] = [
        Interval(Integral, 1, None, closed="left"),
        Interval(RealNotInt, 0.0, 1.0, closed="right"),
        StrOptions({"sqrt", "log2"}),
        "array-like",
        None,
    ]

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
        ccp_alpha=0.0,
        store_leaf_values=False,
        monotonic_cst=None,
        feature_set_ends=None,
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
            ccp_alpha=ccp_alpha,
            store_leaf_values=store_leaf_values,
            monotonic_cst=monotonic_cst,
        )

        self.feature_set_ends = feature_set_ends
        self.feature_combinations = feature_combinations
        self._max_features_arr = None

    def _build_tree(
        self,
        X,
        y,
        sample_weight,
        missing_values_in_feature_mask,
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
        _, n_features = X.shape

        self.feature_combinations_ = (
            self.feature_combinations if self.feature_combinations is not None else 1.5
        )

        # Build tree
        criterion = self.criterion
        if not isinstance(criterion, BaseCriterion):
            criterion = CRITERIA_CLF[self.criterion](self.n_outputs_, self.n_classes_)
        else:
            # Make a deepcopy in case the criterion has mutable attributes that
            # might be shared and modified concurrently during parallel fitting
            criterion = copy.deepcopy(criterion)

        if self.feature_set_ends is None:
            self.feature_set_ends_ = np.asarray([n_features], dtype=np.intp)
        else:
            self.feature_set_ends_ = np.atleast_1d(self.feature_set_ends).astype(np.intp)
        self.n_feature_sets_ = len(self.feature_set_ends_)
        if self.feature_set_ends_[-1] != n_features:
            raise ValueError(
                f"The last feature set end must be equal to the number of features, "
                f"{n_features}, but got {self.feature_set_ends_[-1]}."
            )

        splitter = self.splitter
        if issparse(X):
            raise ValueError(
                "Sparse input is not supported for oblique trees. "
                "Please convert your data to a dense array."
            )

        if isinstance(self._max_features_arr, (Integral, Real, str, type(None))):
            max_features_arr_ = [self._max_features_arr] * self.n_feature_sets_
        else:
            if not isinstance(self._max_features_arr, (list, np.ndarray)):
                raise ValueError(
                    f"max_features must be an array-like, int, float, str, or None; "
                    f"got {type(self._max_features_arr)}"
                )
            if len(self._max_features_arr) != self.n_feature_sets_:
                raise ValueError(
                    f"max_features must be an array-like of length {self.n_feature_sets_}; "
                    f"got {len(self.max_features)}"
                )
            max_features_arr_ = self._max_features_arr

        self.n_features_in_set_ = []
        # XXX: experimental
        # we can replace max_features_ here based on whether or not uniform logic over
        # feature sets
        max_features_per_set = []
        n_features_in_prev = 0
        for idx in range(self.n_feature_sets_):
            max_features = max_features_arr_[idx]

            n_features_in_ = self.feature_set_ends_[idx] - n_features_in_prev
            n_features_in_prev += n_features_in_
            self.n_features_in_set_.append(n_features_in_)
            if isinstance(max_features, str):
                if max_features == "sqrt":
                    max_features = max(1, math.ceil(np.sqrt(n_features_in_)))
                elif max_features == "log2":
                    max_features = max(1, math.ceil(np.log2(n_features_in_)))
            elif max_features is None:
                max_features = n_features_in_
            elif isinstance(max_features, numbers.Integral):
                max_features = max_features
            else:  # float
                if max_features > 0.0:
                    max_features = max(1, math.ceil(max_features * n_features_in_))
                else:
                    max_features = 0

            if max_features > n_features_in_:
                raise ValueError(
                    f"max_features must be less than or equal to "
                    f"the number of features in feature set {idx}: {n_features_in_}, but "
                    f"max_features = {max_features} when applying sampling"
                    f"per feature set."
                )

            max_features_per_set.append(max_features)
        self.max_features_ = np.sum(max_features_per_set)
        if self.max_features_ > n_features:
            raise ValueError(
                "max_features is greater than the number of features: "
                f"{max_features} > {n_features}."
                "This should not be possible. Please submit a bug report."
            )
        self.max_features_per_set_ = np.asarray(max_features_per_set, dtype=np.intp)
        # the total number of features to sample per split
        self.max_features_ = np.sum(self.max_features_per_set_)

        if not isinstance(self.splitter, ObliqueSplitter):
            splitter = OBLIQUE_DENSE_SPLITTERS[self.splitter](
                criterion,
                self.max_features_,
                min_samples_leaf,
                min_weight_leaf,
                random_state,
                monotonic_cst,
                self.feature_combinations_,
                self.feature_set_ends_,
                self.n_feature_sets_,
                self.max_features_per_set_,
            )

        self.tree_ = ObliqueTree(self.n_features_in_, self.n_classes_, self.n_outputs_)

        # Use BestFirst if max_leaf_nodes given; use DepthFirst otherwise
        if max_leaf_nodes < 0:
            self.builder_ = DepthFirstTreeBuilder(
                splitter,
                min_samples_split,
                min_samples_leaf,
                min_weight_leaf,
                max_depth,
                self.min_impurity_decrease,
            )
        else:
            self.builder_ = BestFirstTreeBuilder(
                splitter,
                min_samples_split,
                min_samples_leaf,
                min_weight_leaf,
                max_depth,
                max_leaf_nodes,
                self.min_impurity_decrease,
            )

        self.builder_.build(self.tree_, X, y, sample_weight, None)

        if self.n_outputs_ == 1:
            self.n_classes_ = self.n_classes_[0]
            self.classes_ = self.classes_[0]

    @property
    def _inheritable_fitted_attribute(self):
        """Define additional attributes to pass onto a parent meta tree-estimator.

        Used for passing parameters to HonestTreeClassifier.
        """
        return [
            "max_features_",
            "feature_set_ends_",
            "n_feature_sets_",
            "n_features_in_set_",
            "max_features_per_set_",
            "feature_combinations_",
        ]
