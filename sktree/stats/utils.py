from typing import Optional, Tuple

import numpy as np
from numpy.typing import ArrayLike
from scipy.stats import entropy
from sklearn.metrics import (
    balanced_accuracy_score,
    mean_absolute_error,
    mean_squared_error,
    roc_auc_score,
    roc_curve,
)
from sklearn.utils.validation import check_X_y

from sktree._lib.sklearn.ensemble._forest import ForestClassifier


def _mutual_information(y_true: ArrayLike, y_pred_proba: ArrayLike) -> float:
    """Compute estimate of mutual information for supervised classification setting.

    Parameters
    ----------
    y_true : ArrayLike of shape (n_samples,)
        The true labels.
    y_pred_proba : ArrayLike of shape (n_samples, n_outputs)
        Posterior probabilities.

    Returns
    -------
    float :
        The estimated MI.
    """
    if y_true.squeeze().ndim != 1:
        raise ValueError(f"y_true must be 1d, not {y_true.shape}")

    # entropy averaged over n_samples
    H_YX = np.mean(entropy(y_pred_proba, base=np.exp(1), axis=1))
    # empirical count of each class (n_classes)
    _, counts = np.unique(y_true, return_counts=True)
    H_Y = entropy(counts, base=np.exp(1))
    return H_Y - H_YX


def _cond_entropy(y_true: ArrayLike, y_pred_proba: ArrayLike) -> float:
    """Compute estimate of entropy for supervised classification setting.

    H(Y | X)

    Parameters
    ----------
    y_true : ArrayLike of shape (n_samples,)
        The true labels. Not used in computation of the entropy.
    y_pred_proba : ArrayLike of shape (n_samples, n_outputs)
        Posterior probabilities.

    Returns
    -------
    float :
        The estimated MI.
    """
    if y_true.squeeze().ndim != 1:
        raise ValueError(f"y_true must be 1d, not {y_true.shape}")

    # entropy averaged over n_samples
    H_YX = np.mean(entropy(y_pred_proba, base=np.exp(1), axis=1))
    return H_YX


# Define function to compute the sensitivity at 98% specificity
def _SAS98(y_true: ArrayLike, y_pred_proba: ArrayLike, max_fpr=0.02) -> float:
    """Compute the sensitivity at 98% specificity.
    Parameters
    ----------
    y_true : ArrayLike of shape (n_samples,)
        The true labels.
    y_pred_proba : ArrayLike of shape (n_samples, n_outputs)
        Posterior probabilities.
    max_fpr : float, optional. Default=0.02
    Returns
    -------
    float :
        The estimated SAS98.
    """
    if y_true.squeeze().ndim != 1:
        raise ValueError(f"y_true must be 1d, not {y_true.shape}")
    if 0 in y_true or -1 in y_true :
        fpr, tpr, thresholds = roc_curve(y_true, y_pred_proba[:,1], pos_label=1,drop_intermediate = False)
    else:
        fpr, tpr, thresholds = roc_curve(y_true, y_pred_proba[:,1], pos_label=2,drop_intermediate = False)
    s98 = max([tpr for (fpr, tpr) in zip(fpr, tpr) if fpr <= max_fpr])
    return s98


METRIC_FUNCTIONS = {
    "mse": mean_squared_error,
    "mae": mean_absolute_error,
    "balanced_accuracy": balanced_accuracy_score,
    "auc": roc_auc_score,
    "mi": _mutual_information,
    "cond_entropy": _cond_entropy,
    "sas98": _SAS98,
}

POSTERIOR_FUNCTIONS = ("mi", "auc", "cond_entropy")

POSITIVE_METRICS = ("mi", "auc", "balanced_accuracy")

REGRESSOR_METRICS = ("mse", "mae")


def _non_nan_samples(posterior_arr: ArrayLike) -> ArrayLike:
    """Determine which samples are not nan in the posterior tree array.

    Parameters
    ----------
    posterior_arr : ArrayLike of shape (n_trees, n_samples, n_outputs)
        The 3D posterior array from the forest.

    Returns
    -------
    nonnan_indices : ArrayLike of shape (n_nonnan_samples,)
        The indices of the samples that are not nan in the posterior array
        along axis=1.
    """
    # Find the row indices with NaN values along the specified axis
    nan_indices = np.isnan(posterior_arr).any(axis=2).all(axis=0)

    # Invert the boolean mask to get indices without NaN values
    nonnan_indices = np.where(~nan_indices)[0]
    return nonnan_indices


def _compute_null_distribution_perm(
    X_train: ArrayLike,
    y_train: ArrayLike,
    X_test: ArrayLike,
    y_test: ArrayLike,
    covariate_index: ArrayLike,
    est: ForestClassifier,
    metric: str = "mse",
    n_repeats: int = 1000,
    seed: Optional[int] = None,
    **metric_kwargs,
) -> ArrayLike:
    """Compute null distribution using permutation method.

    Parameters
    ----------
    X_test : ArrayLike of shape (n_samples, n_features)
        The data matrix.
    y_test : ArrayLike of shape (n_samples, n_outputs)
        The output matrix.
    covariate_index : ArrayLike of shape (n_covariates,)
        The indices of the covariates to permute.
    est : ForestClassifier
        The forest that will be used to recompute the test statistic on the permuted data.
    metric : str, optional
        The metric, which to compute the null distribution of statistics, by default 'mse'.
    n_repeats : int, optional
        The number of times to sample the null, by default 1000.
    seed : int, optional
        Random seed, by default None.
    """
    rng = np.random.default_rng(seed)
    X_test, y_test = check_X_y(X_test, y_test, ensure_2d=True, multi_output=True)
    n_samples_test = len(y_test)
    n_samples_train = len(y_train)
    metric_func = METRIC_FUNCTIONS[metric]

    # pre-allocate memory for the index array
    train_index_arr = np.arange(n_samples_train, dtype=int).reshape(-1, 1)
    test_index_arr = np.arange(n_samples_test, dtype=int).reshape(-1, 1)

    null_metrics = np.zeros((n_repeats,))

    for idx in range(n_repeats):
        # permute the covariates inplace
        rng.shuffle(test_index_arr)
        perm_X_cov = X_test[test_index_arr, covariate_index]
        X_test[:, covariate_index] = perm_X_cov

        rng.shuffle(train_index_arr)
        perm_X_cov = X_train[train_index_arr, covariate_index]
        X_train[:, covariate_index] = perm_X_cov

        # train a new forest on the permuted data
        # XXX: should there be a train/test split here? even w/ honest forests?
        est.fit(X_train, y_train.ravel())
        y_pred = est.predict(X_test)

        # compute two instances of the metric from the sampled trees
        metric_val = metric_func(y_test, y_pred, **metric_kwargs)

        null_metrics[idx] = metric_val
    return null_metrics


def _compute_null_distribution_coleman(
    y_test: ArrayLike,
    y_pred_proba_normal: ArrayLike,
    y_pred_proba_perm: ArrayLike,
    metric: str = "mse",
    n_repeats: int = 1000,
    seed: Optional[int] = None,
    **metric_kwargs,
) -> Tuple[ArrayLike, ArrayLike]:
    """Compute null distribution using Coleman method.

    The null distribution is comprised of two forests.

    Parameters
    ----------
    y_test : ArrayLike of shape (n_samples, n_outputs)
        The output matrix.
    y_pred_proba_normal : ArrayLike of shape (n_estimators_normal, n_samples, n_outputs)
        The predicted posteriors from the normal forest. Some of the trees
        may have nans predicted in them, which means the tree used these samples
        for training and not for prediction.
    y_pred_proba_perm : ArrayLike of shape (n_estimators_perm, n_samples, n_outputs)
        The predicted posteriors from the permuted forest. Some of the trees
        may have nans predicted in them, which means the tree used these samples
        for training and not for prediction.
    metric : str, optional
        The metric, which to compute the null distribution of statistics, by default 'mse'.
    n_repeats : int, optional
        The number of times to sample the null, by default 1000.
    seed : int, optional
        Random seed, by default None.
    metric_kwargs : dict, optional
        Keyword arguments to pass to the metric function.

    Returns
    -------
    metric_star : ArrayLike of shape (n_samples,)
        An array of the metrics computed on half the trees.
    metric_star_pi : ArrayLike of shape (n_samples,)
        An array of the metrics computed on the other half of the trees.
    """
    rng = np.random.default_rng(seed)
    metric_func = METRIC_FUNCTIONS[metric]

    # sample two sets of equal number of trees from the combined forest these are the posteriors
    # (n_estimators * 2, n_samples, n_outputs)
    all_y_pred = np.concatenate((y_pred_proba_normal, y_pred_proba_perm), axis=0)

    n_estimators, _, _ = y_pred_proba_normal.shape
    n_samples_test = len(y_test)
    if all_y_pred.shape[1] != n_samples_test:
        raise RuntimeError(
            f"The number of samples in `all_y_pred` {len(all_y_pred)} "
            f"is not equal to 2 * n_samples_test {2 * n_samples_test}"
        )

    # create index array of [1, ..., 2N] to slice into `all_y_pred` the stacks of trees
    y_pred_ind_arr = np.arange((2 * n_estimators), dtype=int)

    metric_star = np.zeros((n_repeats,))
    metric_star_pi = np.zeros((n_repeats,))
    for idx in range(n_repeats):
        # two sets of random indices from 1 : 2N are sampled using Fisher-Yates
        rng.shuffle(y_pred_ind_arr)

        # get random half of the posteriors from two sets of trees
        first_forest_inds = y_pred_ind_arr[: n_estimators // 2]
        second_forest_inds = y_pred_ind_arr[n_estimators // 2 :]

        # get random half of the posteriors as one forest
        first_forest_pred = all_y_pred[first_forest_inds, ...]
        second_forest_pred = all_y_pred[second_forest_inds, ...]

        # determine if there are any nans in the final posterior array, when
        # averaged over the trees
        first_forest_samples = _non_nan_samples(first_forest_pred)
        second_forest_samples = _non_nan_samples(second_forest_pred)

        # todo: is this step necessary?
        # non_nan_samples = np.intersect1d(
        #     first_forest_samples, second_forest_samples, assume_unique=True
        # )
        # now average the posteriors over the trees for the non-nan samples
        # y_pred_first_half = np.nanmean(first_forest_pred[:, non_nan_samples, :], axis=0)
        # y_pred_second_half = np.nanmean(second_forest_pred[:, non_nan_samples, :], axis=0)
        # # compute two instances of the metric from the sampled trees
        # first_half_metric = metric_func(y_test[non_nan_samples, :], y_pred_first_half)
        # second_half_metric = metric_func(y_test[non_nan_samples, :], y_pred_second_half)

        y_pred_first_half = np.nanmean(first_forest_pred[:, first_forest_samples, :], axis=0)
        y_pred_second_half = np.nanmean(second_forest_pred[:, second_forest_samples, :], axis=0)

        # compute two instances of the metric from the sampled trees
        first_half_metric = metric_func(
            y_test[first_forest_samples, :], y_pred_first_half, **metric_kwargs
        )
        second_half_metric = metric_func(
            y_test[second_forest_samples, :], y_pred_second_half, **metric_kwargs
        )

        metric_star[idx] = first_half_metric
        metric_star_pi[idx] = second_half_metric

    return metric_star, metric_star_pi


def compute_posterior_metrics(posteriors: ArrayLike, y_true, test_idx, metric, max_fpr=0.1):
    """Compute the metric from the posteriors.

    Parameters
    ----------
    posteriors : ArrayLike of shape (n_estimators, n_samples, n_outputs)
        The posteriors from the forest.
    y_true : ArrayLike of shape (n_estimators,n_samples,)
        The true labels.
    test_idx : ArrayLike of shape (n_samples_test,)
        The indices of the test samples.
    metric : str
        The metric to compute. The supported metrics are `mi`, `pauc`,`sas98`, `pvalues`.
    max_fpr : float, optional. For `sas98` and `pauc`  default=0.1 )

    Returns
    -------
    metric_val : float
        The metric value.
    """
    # test the inputs
    if posteriors.ndim != 3:
        raise ValueError(f"posteriors must be 3d, not {posteriors.shape}")

    n_estimators, _, _ = posteriors.shape
    n_samples_test = test_idx.shape[1]
    # check the test index and n_sample_test are the same
    if len(test_idx) != n_samples_test:
        raise RuntimeError(
            f"The number of samples in `test_idx` {len(test_idx)} "
            f"is not equal to the length of `y_true` {n_samples_test}"
        )
    # check that y_true is 1d
    if y_true.squeeze().ndim != 1:
        raise ValueError(f"y_true must be 1d, not {y_true.shape}")

    # check the metric to be calculated is implemented
    if metric not in METRIC_FUNCTIONS.keys():
        raise ValueError(f"metric {metric} is not implemented")

    metric_func = METRIC_FUNCTIONS[metric]
    if metric in ["auc", "sas98"]:
        metric_func = lambda y_true, y_pred_proba: roc_auc_score(
            y_true, y_pred_proba, max_fpr=max_fpr
        )
    scores = np.zeros(n_estimators)
    for idx in range(n_estimators):
        scores[idx] = metric_func(y_true[test_idx], posteriors[idx, test_idx, :])
    return np.mean(scores)
