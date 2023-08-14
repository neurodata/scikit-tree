"""A set of mixin methods for marginalizing a random forest."""
import numpy as np
from sklearn.ensemble._forest import BaseForest
from sklearn.utils.parallel import Parallel, delayed
from sklearn.utils.validation import check_is_fitted, check_random_state

from sktree._lib.sklearn.tree import BaseDecisionTree
from sktree._lib.sklearn.tree._tree import DTYPE
from sktree.utils import check_is_forest

from ._marginal import apply_marginal_tree

TRAVERSAL_METHOD_MAP = {
    "random": 0,
    "weighted": 1,
}


def apply_marginal(
    est, X, S, traversal_method="weighted", use_sample_weight: bool = False, check_input=True
):
    """Apply a forest to X, while marginalizing over features.

    XXX: this should not work for ObliqueTrees. But we can add a traversal method
    called 'resample', which applies a random conditional resampling of the data
    to preserve X[S] | X[~S] conditional distribution.

    Parameters
    ----------
    est : BaseForest or BaseDecisionTree
        The tree/forest based estimator that was already fit on (X, y) data.
    X : array-like of shape (n_samples, n_features)
        Data that should match the data used to fit the forest.
    S : array-like of shape (n_features), optional
        An index vector of 1's and 0's indicating which features to
        marginalize over. 1's indicate features to keep, 0's indicate
        features to marginalize over.
    traversal_method : str, {'random', 'weighted'}
        The method to use for traversing the tree. If 'random', then
        each time a feature is encountered that is specified by S to
        be marginalized over, a fair coin is flipped and the sample
        is sent to the left child if heads and the right child if tails.
        If 'weighted', then each time a feature is encountered that is
        specified by S to be marginalized over, the sample is sent to
        the left child with probability equal to the fraction of samples
        that went to the left child during training. By default 'weighted'.
    use_sample_weight : bool, optional
        Whether to weight the samples that are sent to the left and
        right child nodes using ``weighted_node_samples``, by default False.
        See :ref:`sklearn.plot_unveil_tree_structure` for more details.
    check_input : bool, optional
        Whether to check the input data, by default True.

    Returns
    -------
    X_leaves : array-like of shape (n_samples, n_estimators)
        For each datapoint x in X and for each tree in the forest, return the
        index of the leaf x ends up in. If it is a tree ``n_estimators=1``.
    """
    check_is_forest(est)
    if hasattr(est, "tree_type") and est.tree_type == "oblique":
        raise RuntimeError("This method only supports axis-aligned trees.")

    random_state = check_random_state(est.random_state)
    # check if this is a forest, or tree
    if hasattr(est, "estimators_"):
        _apply_marginal_func = _apply_marginal_forest
    else:
        _apply_marginal_func = _apply_marginal_tree

    # make sure S is an array
    S = np.asarray(S).astype(np.intp)

    X_leaves = _apply_marginal_func(
        est,
        X,
        S,
        traversal_method=traversal_method,
        use_sample_weight=use_sample_weight,
        check_input=check_input,
        random_state=random_state,
    )
    return X_leaves


def _apply_marginal_forest(
    est,
    X,
    S,
    traversal_method: str,
    use_sample_weight: bool = False,
    check_input=True,
    random_state: np.random.RandomState = None,
):
    """Apply forest to marginalized set of features."""
    check_is_fitted(est)
    if check_input:
        X = est._validate_X_predict(X)

    # if we trained a binning tree, then we should re-bin the data
    # XXX: this is inefficient and should be improved to be in line with what
    # the Histogram Gradient Boosting Tree does, where the binning thresholds
    # are passed into the tree itself, thus allowing us to set the node feature
    # value thresholds within the tree itself.
    if est.max_bins is not None:
        X = est._bin_data(X, is_training_data=False).astype(DTYPE)

    results = Parallel(n_jobs=est.n_jobs, verbose=est.verbose, prefer="threads",)(
        delayed(_apply_marginal_tree)(
            tree,
            X,
            S,
            traversal_method,
            use_sample_weight,
            check_input=False,
            random_state=random_state,
        )
        for tree in est.estimators_
    )
    return np.array(results).T


def _apply_marginal_tree(
    est: BaseDecisionTree,
    X,
    S,
    traversal_method: str,
    use_sample_weight: bool = False,
    check_input=True,
    random_state: np.random.RandomState = None,
):
    """Apply a tree to X, while marginalizing over features.

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        Data that should match the data used to fit the forest.
    S : array-like of shape (n_marginal_features)
        The feature indices to marginalize over (i.e. columns of X).
    traversal_method : str, {'random', 'weighted'}
        The method to use for traversing the tree. Maps to an integer
        to allow easy comparison in Cython/C++. 'random' maps to 0,
        'weighted' maps to 1.
    use_sample_weight : bool, optional
        Whether to weight the samples that are sent to the left and
        right child nodes, by default False.
    check_input : bool, optional
        Whether to check the input data, by default True.

    Returns
    -------
    X_leaves : array-like of shape (n_samples,)
        Index of the leaf that each sample in X ends up in after marginalizing
        certain features.
    """
    check_is_fitted(est)
    X = est._validate_X_predict(X, check_input=check_input)

    traversal_method_int = TRAVERSAL_METHOD_MAP[traversal_method]

    X_leaves = apply_marginal_tree(
        est.tree_, X, S, traversal_method_int, use_sample_weight, random_state=random_state
    )
    return X_leaves


def compute_marginal(self: BaseForest, X, S, n_repeats=10):
    r"""Compute marginal distribution of P(S = s) for each s in X.

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        Data that should match the data used to fit the forest.
    S : array-like of shape (n_features), optional
        An index vector of 1's and 0's indicating which features to
        marginalize over. 1's indicate features we want to compute on,
        0's indicate features to marginalize over.
    n_repeats : int, optional
        Number of times to repeat the marginalization, by default 10.
        Each time, a feature that is encountered in the forest that
        is specified by S to be marginalized over, a random 50\%
        of samples are sent to the left child and the other 50\%
        are sent to the right child. Since this process is random
        and can affect the leaf nodes assignment, we repeat this
        and take the average of the leaf nodes assignment.
    """
    # get the leaf nodes for each sample in X
    # X_leaves = self.apply_marginal(X, S, n_repeats=n_repeats)

    # compute the marginal distribution of P(S = s) for each s in X
    self.apply(X)

    #


def compute_conditional(self, X, S, y=None, n_repeats=10):
    r"""Compute conditional P(Y | X, Z = z) for each X and Z.

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features_x)
        Data that should match the data used to fit the forest.
    S : array-like of shape (n_features), optional
        An index vector of 1's and 0's indicating which features we
        want to fix values for.
    y : array-like of shape (n_samples, n_outputs), optional
        Y data that used to fit the forest, by default None.
    n_repeats : int, optional
        Number of times to repeat the marginalization, by default 10.
        Each time, a feature that is encountered in the forest that
        is specified by S to be marginalized over, a random 50\%
        of samples are sent to the left child and the other 50\%
        are sent to the right child. Since this process is random
        and can affect the leaf nodes assignment, we repeat this
        and take the average of the leaf nodes assignment.

    Returns
    -------
    proba_y_xz : array-like of shape (n_samples, n_outputs)
        For each datapoint x in X and for each tree in the forest, return the
        probability of y given x and specific instance of z.

    Questions for Mike:
    1. What is |l|? - is this the size of the leaf node? What does that mean? The number
    of samples in l?
    """
    # for every tree, and every leaf compute interval Z of data that reaches
    # that leaf = I_{b,l, Z=z}
    for tree in self.estimators_:
        # tree node value (n_leaves, n_outputs, max_classes)
        # tree_leaves = tree.tree_.value

        for leaf in range(tree.tree_.n_leaves):
            # get the size of the leaf

            # compute interval that reaches this leaf

            # now compute P(Y | x \in l) for every leaf
            # proba_y_x = tree_leaves[leaf, :, :]

            # get sample indices that reach this leaf in Z=z
            sample_indices = []

            # now estimate P(Y | Z=z) for each z in the Z sequence
            for sample_idx in sample_indices:
                #
                pass
            pass
        pass
