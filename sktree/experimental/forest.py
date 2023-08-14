from copy import deepcopy

import numpy as np
from numpy.typing import ArrayLike
from scipy.stats import entropy
from sklearn.base import BaseEstimator, MetaEstimatorMixin
from sklearn.ensemble._forest import BaseForest

from sktree.utils import check_is_forest

from .ksg import entropy_continuous


class SupervisedInfoForest(BaseEstimator, MetaEstimatorMixin):
    def __init__(self, estimator, y_categorical: bool = False, n_jobs=None, random_state=None):
        """Meta estimator for mutual information.

        This supervised forest estimator uses two supervised forests to estimate
        the (conditional) mutual information between X and Y given Z.

        Parameters
        ----------
        estimator : _type_
            _description_
        y_categorical : bool, optional
            _description_, by default False
        n_jobs : _type_, optional
            _description_, by default None
        random_state : _type_, optional
            _description_, by default None
        """
        self.estimator = estimator
        self.y_categorical = y_categorical
        self.n_jobs = n_jobs
        self.random_state = random_state

    def fit(self, X, y, Z=None):
        X, y = self._validate_data(X, y, accept_sparse="csc")
        check_is_forest(self.estimator, allow_tree=False, ensure_fitted=False)

        self.estimator_yxz_ = deepcopy(self.estimator)
        self.estimator_yz_ = deepcopy(self.estimator)
        # self.estimator_yxz_ = CalibratedClassifierCV(
        #     copy(self.estimator), method='isotonic', cv=5,
        # )
        # self.estimator_yz_ = CalibratedClassifierCV(
        #     copy(self.estimator), method='isotonic', cv=5,
        # )

        if Z is not None:
            XZ = np.hstack((X, Z))
        else:
            XZ = X

        # compute the entropy of H(Y | X, Z)
        self.estimator_yxz_.fit(XZ, y)

        # compute the entropy of H(Y | Z)
        if Z is None:
            # compute entropy using empirical distribution
            if self.y_categorical:
                _, self.counts_ = np.unique(y, return_counts=True)
            else:
                self.H_yz_ = entropy_continuous(y, k=5, n_jobs=self.n_jobs)
        else:
            self.estimator_yz_.fit(Z, y)

        return self

    def predict_cmi(self, X, y, Z=None):
        if Z is not None:
            X, y = self._validate_data(X, y, accept_sparse="csc")
            Z = self._validate_data(Z, accept_sparse="csc")
        else:
            X = self._validate_data(X, accept_sparse="csc")

        if Z is not None:
            XZ = np.hstack((X, Z))
        else:
            XZ = X

        # if not fitted yet
        if not hasattr(self, "estimator_yz_"):
            self.fit(X, y, Z)

        sample_indices = np.arange(0, X.shape[0])

        # compute the entropy of H(Y | X, Z)
        H_yxz = cond_entropy(
            self.estimator_yxz_, XZ, y, sample_indices, kappa=3, seed=self.random_state
        )

        # compute the entropy of H(Y | Z)
        if Z is None:
            if self.y_categorical:
                # compute entropy using empirical distribution
                H_yz = entropy(self.counts_, base=np.exp(1))
            else:
                H_yz = self.H_yz_
        else:
            H_yz = cond_entropy(
                self.estimator_yz_, Z, y, sample_indices, kappa=3, seed=self.random_state
            )

        return H_yxz - H_yz


def approx_marginal(est: BaseForest, X: ArrayLike, weighted: bool = False):
    """Compute the approximate marginal distribution.

    A fitted tree/forest can be used to compute the approximate marginal
    by counting the number of training (weighted) training samples
    that fall into each leaf node for samples in X.

    Parameters
    ----------
    est : Forest
        Fitted forest or tree estimator.
    X : ArrayLike of shape (n_samples, n_features_in_)
        Samples to compute the approximate marginal distribution for.
    weighted : bool, default=False
        Whether to use the weighted number of samples in each leaf node.
    """
    check_is_forest(est)
    n_samples = X.shape[0]
    p_leaves = np.zeros(n_samples, est.n_estimators)

    # leaf indices (n_samples, n_estimators)
    X_leaves = est.apply(X)

    for idx, tree_ in enumerate(est.estimators_.tree_):
        # compute the empirical distribution of the leaf nodes
        # (n_nodes,) array
        if weighted:
            n_node_samples = tree_.weighted_n_node_samples
        else:
            n_node_samples = tree_.n_node_samples

        # compute the total number of samples used in this tree
        total_samples_in_tree = n_node_samples.sum()

        # compute the empirical distribution of the leaf nodes for each sample in X
        tree_leaves = X_leaves[:, idx]
        p_leaves[:, idx] = n_node_samples[tree_leaves] / total_samples_in_tree

    # return the average over all trees
    return p_leaves.mean(axis=1)


def cond_entropy(
    est: BaseForest, X: ArrayLike, y: ArrayLike, estimators_samples_: ArrayLike, kappa=3, seed=None
):
    rng = np.random.default_rng(seed)

    check_is_forest(est)
    # if "multioutput" in est._get_tags():
    #     raise ValueError("Multioutput is not supported.")

    # get leaves of each tree for each sample (n_samples, n_estimators)
    X_leaves = est.apply(X)
    n_classes = est.n_classes_
    n_samples = X.shape[0]

    cond_entropy = 0.0
    for tree_idx, tree in enumerate(est.estimators_):
        # (n_nodes,) array of number of samples
        node_counts = tree.tree_.n_node_samples
        n_nodes = len(node_counts)

        # (n_nodes, n_classes_)
        class_counts = np.zeros((n_nodes, n_classes))

        # Find the indices of the training set used for partition.
        sampled_indices = estimators_samples_[tree_idx]
        unsampled_indices = np.delete(np.arange(0, n_samples), sampled_indices)

        # Randomly split the rest into voting and evaluation.
        total_unsampled = len(unsampled_indices)
        rng.shuffle(unsampled_indices)
        vote_indices = unsampled_indices[: total_unsampled // 2]
        eval_indices = unsampled_indices[total_unsampled // 2 :]

        # true classes of evaluation points
        true_classes = y[vote_indices]

        # estimated nodes
        tree_vote_leaves = X_leaves[vote_indices, tree_idx]
        n_vote_samples = len(tree_vote_leaves)
        for i in range(n_vote_samples):
            class_counts[tree_vote_leaves[i], true_classes[i]] += 1

        # compute total number of samples in each leaf node
        # then compute probabilies per class for each node
        # (n_nodes, n_classes_)
        n_node_leaves = class_counts.sum(axis=1)
        class_probs = _finite_sample_correction(class_counts, n_node_leaves, n_classes, kappa=kappa)

        # Place evaluation points in their corresponding leaf node.
        # Store evaluation posterior in a num_eval-by-num_class matrix.
        tree_eval_leaves = tree.apply(X[eval_indices, :])
        eval_class_probs = class_probs[tree_eval_leaves, :]

        # compute the entropy per sample
        eval_entropies = [entropy(posterior) for posterior in eval_class_probs]
        cond_entropy += np.mean(eval_entropies)
    return cond_entropy / est.n_estimators


def _finite_sample_correction(class_counts, n_samples_per_node, n_classes, kappa=3):
    """Perform finite sample correction on class probabilities.

    This applies the following procedure:

    - if a leaf (i.e. node) is pure for a certain class, meaning there is 0 samples
        for that class in the node, then the probability of that class is replaced
        with ``1 / (kappa * n_classes)``, where `kappa` is a hyperparameter and


    Parameters
    ----------
    class_counts : array-like of shape (n_nodes, n_classes)
        Class counts in each node.
    n_samples_per_node : array-like of shape (n_nodes,)
        Total number of samples in each node.
    n_classes : int
        Number of classes.
    kappa : float, default=3
        Corrective factor hyperparameter.
    """
    # Avoid divide by zero.
    n_samples_per_node[n_samples_per_node == 0] = 1

    class_probs = np.divide(class_counts, n_samples_per_node[:, np.newaxis])

    # Make the nodes that have no estimation indices uniform.
    # This includes non-leaf nodes, but that will not affect the estimate.
    class_probs[np.argwhere(class_probs.sum(axis=1) == 0)] = 1.0 / n_classes

    # Apply finite sample correction and renormalize.
    where_0 = np.argwhere(class_probs == 0)
    for row, col in where_0:
        class_probs[row, col] = 1.0 / (kappa * class_counts[row].sum())
    row_sums = class_probs.sum(axis=1)
    class_probs = class_probs / row_sums[:, None]

    return class_probs


def approx_joint(est: BaseForest, X: ArrayLike, P_Y: ArrayLike):
    """Compute the approximate joint distribution.

    A fitted tree/forest can be used to compute the approximate joint
    by counting the number of training (weighted) training samples
    that fall into each leaf node for samples in X. Then the conditional
    distribution is computed as the number of Y samples in each for a
    fitted tree or forest for P(Y | X).

    Parameters
    ----------
    est : Forest
        The fitted supervised forest.
    X : ArrayLike of shape (n_samples, n_features_in_)
        Input data.
    P_Y : ArrayLike of shape (n_samples,)
        The approximate marginal distribution of X.
    """
    check_is_forest(est)
    if est._get_tags().get("multioutput", False):
        raise ValueError("Multioutput is not supported.")

    est.apply(X)
