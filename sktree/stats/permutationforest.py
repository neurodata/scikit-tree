from typing import Optional

import numpy as np
from numpy.typing import ArrayLike
from sklearn.base import MetaEstimatorMixin, clone, is_classifier
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.ensemble._forest import ForestClassifier as sklearnForestClassifier
from sklearn.ensemble._forest import ForestRegressor as sklearnForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.utils.validation import check_X_y

from sktree._lib.sklearn.ensemble._forest import BaseForest, ForestClassifier, ForestRegressor

from .utils import (
    METRIC_FUNCTIONS,
    POSITIVE_METRICS,
    POSTERIOR_FUNCTIONS,
    REGRESSOR_METRICS,
    _compute_null_distribution_perm,
)


class BasePermutationForest(MetaEstimatorMixin):
    def __init__(
        self,
        estimator=None,
        test_size=0.2,
        random_state=None,
        verbose=0,
    ):
        self.estimator = estimator
        self.test_size = test_size
        self.random_state = random_state
        self.verbose = verbose

    def reset(self):
        class_attributes = dir(type(self))
        instance_attributes = dir(self)

        for attr_name in instance_attributes:
            if attr_name.endswith("_") and attr_name not in class_attributes:
                delattr(self, attr_name)

    def _get_estimator(self):
        pass

    @property
    def train_test_samples_(self):
        """
        The subset of drawn samples for each base estimator.

        Returns a dynamically generated list of indices identifying
        the samples used for fitting each member of the ensemble, i.e.,
        the in-bag samples.

        Note: the list is re-created at each call to the property in order
        to reduce the object memory footprint by not storing the sampling
        data. Thus fetching the property may be slower than expected.
        """
        indices = np.arange(self._n_samples_, dtype=int)

        # Get drawn indices along both sample and feature axes
        indices_train, indices_test = train_test_split(
            indices, test_size=self.test_size, shuffle=True, random_state=self.random_state
        )
        return indices_train, indices_test

    def _statistic(
        self,
        estimator: BaseForest,
        X: ArrayLike,
        y: ArrayLike,
        covariate_index: Optional[ArrayLike] = None,
        metric="mse",
        return_posteriors: bool = False,
        seed=None,
        **metric_kwargs,
    ):
        """Helper function to compute the test statistic."""
        metric_func = METRIC_FUNCTIONS[metric]
        if seed is None:
            rng = np.random.default_rng(self.random_state)
        else:
            rng = np.random.default_rng(seed)
        indices_train, indices_test = self.train_test_samples_
        if covariate_index is not None:
            n_samples = X.shape[0]
            indices = np.arange(n_samples, dtype=int)
            # perform permutation of covariates
            index_arr = rng.choice(indices, size=(n_samples, 1), replace=False, shuffle=False)
            X = X.copy()
            X[:, covariate_index] = X[index_arr, covariate_index]

        X_train, X_test = X[indices_train, :], X[indices_test, :]
        y_train, y_test = y[indices_train, :], y[indices_test, :]
        if y_train.shape[1] == 1:
            y_train = y_train.ravel()
            y_test = y_test.ravel()
        estimator.fit(X_train, y_train)

        # Either get the predicted value, or the posterior probabilities
        if metric in POSTERIOR_FUNCTIONS:
            y_pred = estimator.predict_proba(X_test)
        else:
            y_pred = estimator.predict(X_test)

        # set variables to compute metric
        samples = indices_test
        y_true_final = y_test
        posterior_final = y_pred

        stat = metric_func(y_true_final, posterior_final, **metric_kwargs)

        if covariate_index is None:
            # Ignore all NaN values (samples not tested) -> (n_samples_final, n_outputs)
            # arrays of y and predicted posterior
            self.samples_ = samples
            self.y_true_ = y_true_final
            self.posterior_final_ = posterior_final
            self.stat_ = stat

        if return_posteriors:
            return stat, posterior_final, samples

        return stat

    def statistic(
        self,
        X: ArrayLike,
        y: ArrayLike,
        covariate_index: Optional[ArrayLike] = None,
        metric="mse",
        return_posteriors: bool = False,
        check_input: bool = True,
        seed=None,
        **metric_kwargs,
    ):
        """Compute the test statistic.

        Parameters
        ----------
        X : ArrayLike of shape (n_samples, n_features)
            The data matrix.
        y : ArrayLike of shape (n_samples, n_outputs)
            The target matrix.
        covariate_index : ArrayLike, optional of shape (n_covariates,)
            The index array of covariates to shuffle, by default None.
        metric : str, optional
            The metric to compute, by default "mse".
        return_posteriors : bool, optional
            Whether or not to return the posteriors, by default False.
        check_input : bool, optional
            Whether or not to check the input, by default True.
        seed : int, optional
            The random seed to use, by default None.
        **metric_kwargs : dict, optional
            Keyword arguments to pass to the metric function.

        Returns
        -------
        stat : float
            The test statistic.
        posterior_final : ArrayLike of shape (n_samples_final, n_outputs), optional
            If ``return_posteriors`` is True, then the posterior probabilities of the
            samples used in the final test. ``n_samples_final`` is equal to ``n_samples``
            if all samples are encountered in the test set of at least one tree in the
            posterior computation.
        samples : ArrayLike of shape (n_samples_final,), optional
            The indices of the samples used in the final test. ``n_samples_final`` is
            equal to ``n_samples`` if all samples are encountered in the test set of at
            least one tree in the posterior computation.
        """
        if check_input:
            X, y = check_X_y(X, y, ensure_2d=True, multi_output=True)
            if y.ndim != 2:
                y = y.reshape(-1, 1)

        self._n_samples_ = X.shape[0]
        self.estimator_ = self._get_estimator()

        if is_classifier(self.estimator_):
            if metric not in METRIC_FUNCTIONS:
                raise RuntimeError(
                    f"Metric must be one of {list(METRIC_FUNCTIONS.keys())}, got {metric}"
                )
        else:
            if metric not in REGRESSOR_METRICS:
                raise RuntimeError(f'Metric must be either "mse" or "mae", got {metric}')

        if covariate_index is None:
            estimator = self.estimator_
        else:
            self.permuted_estimator_ = clone(self.estimator_)
            estimator = self.permuted_estimator_

        return self._statistic(
            estimator,
            X,
            y,
            covariate_index=covariate_index,
            metric=metric,
            return_posteriors=return_posteriors,
            seed=seed,
            **metric_kwargs,
        )

    def test(
        self,
        X: ArrayLike,
        y: ArrayLike,
        covariate_index: ArrayLike,
        metric: str = "mse",
        n_repeats: int = 1000,
        return_posteriors: bool = False,
        **metric_kwargs,
    ):
        """Perform hypothesis test using permutation testing.

        Parameters
        ----------
        X : ArrayLike of shape (n_samples, n_features)
            The data matrix.
        y : ArrayLike of shape (n_samples, n_outputs)
            The target matrix.
        covariate_index : ArrayLike of shape (n_covariates,)
            The covariate indices of ``X`` to shuffle.
        metric : str, optional
            Metric to compute, by default "mse".
        n_repeats : int, optional
            Number of times to sample the null distribution, by default 1000.
        return_posteriors : bool, optional
            Whether or not to return the posteriors, by default False.
        **metric_kwargs : dict, optional
            Keyword arguments to pass to the metric function.

        Returns
        -------
        observe_stat : float
            Observed test statistic.
        pvalue : float
            Pvalue of the test.
        """
        X, y = check_X_y(X, y, ensure_2d=True, copy=True, multi_output=True)
        if y.ndim != 2:
            y = y.reshape(-1, 1)
        self._n_samples_ = X.shape[0]

        # train/test split
        # XXX: could add stratifying by y when y is classification
        indices_train, indices_test = self.train_test_samples_

        if not hasattr(self, "samples_"):
            # first compute the test statistic on the un-permuted data
            observe_stat, _, _ = self.statistic(
                X,
                y,
                covariate_index=None,
                metric=metric,
                return_posteriors=True,
                check_input=False,
                **metric_kwargs,
            )
        else:
            # observe_samples = self.samples_
            # observe_posteriors = self.posterior_final_
            observe_stat = self.stat_

        # compute null distribution of the test statistic
        # WARNING: this could take a long time, since it fits a new forest
        null_dist = _compute_null_distribution_perm(
            X_train=X[indices_train, :],
            y_train=y[indices_train, :],
            X_test=X[indices_test, :],
            y_test=y[indices_test, :],
            covariate_index=covariate_index,
            est=self.estimator_,
            metric=metric,
            n_repeats=n_repeats,
            seed=self.random_state,
        )

        if not return_posteriors:
            self.null_dist_ = np.array(null_dist)
        else:
            self.null_dist_ = np.array([x[0] for x in null_dist])
            self.posterior_null_ = np.array([x[1] for x in null_dist])

        n_repeats = len(self.null_dist_)
        if metric in POSITIVE_METRICS:
            pvalue = (1 + (self.null_dist_ >= observe_stat).sum()) / (1 + n_repeats)
        else:
            pvalue = (1 + (self.null_dist_ <= observe_stat).sum()) / (1 + n_repeats)
        return observe_stat, pvalue


class PermutationForestRegressor(BasePermutationForest):
    """Hypothesis testing of covariates with a permutation forest regressor.

    This implements permutation testing of a null hypothesis using a random forest.
    The null hypothesis is generated by permuting ``n_repeats`` times the covariate
    indices and then a random forest is trained for each permuted instance. This
    is compared to the original random forest that was computed on the regular
    non-permuted data.

    .. warning:: Permutation testing with forests is computationally expensive.
        As a result, if you are testing for the importance of feature sets, consider
        using `sktree.FeatureImportanceForestRegressor` or
        `sktree.FeatureImportanceForestClassifier` instead, which is
        much more computationally efficient.

    .. note:: This does not allow testing on the posteriors.

    Parameters
    ----------
    estimator : object, default=None
        Type of forest estimator to use. By default `None`, which defaults to
        :class:`sklearn.ensemble.RandomForestRegressor` with default parameters.

    test_size : float, default=0.2
        The proportion of samples to leave out for each tree to compute metric on.

    random_state : int, RandomState instance or None, default=None
        Controls both the randomness of the bootstrapping of the samples used
        when building trees (if ``bootstrap=True``) and the sampling of the
        features to consider when looking for the best split at each node
        (if ``max_features < n_features``).
        See :term:`Glossary <random_state>` for details.

    verbose : int, default=0
        Controls the verbosity when fitting and predicting.

    Attributes
    ----------
    samples_ : ArrayLike of shape (n_samples,)
        The indices of the samples used in the final test.

    y_true_ : ArrayLike of shape (n_samples_final,)
        The true labels of the samples used in the final test.

    posterior_ : ArrayLike of shape (n_samples_final, n_outputs)
        The predicted posterior probabilities of the samples used in the final test.

    null_dist_ : ArrayLike of shape (n_repeats,)
        The null distribution of the test statistic.

    posterior_null_ : ArrayLike of shape (n_samples_final, n_outputs, n_repeats)
        The posterior probabilities of the samples used in the final test for each
        permutation for the null distribution.
    """

    def __init__(
        self,
        estimator=None,
        test_size=0.2,
        random_state=None,
        verbose=0,
    ):
        super().__init__(
            estimator=estimator,
            test_size=test_size,
            random_state=random_state,
            verbose=verbose,
        )

    def _get_estimator(self):
        if not hasattr(self, "estimator_") and self.estimator is None:
            estimator_ = RandomForestRegressor()
        elif not isinstance(self.estimator, (ForestRegressor, sklearnForestRegressor)):
            raise RuntimeError(f"Estimator must be a ForestRegressor, got {type(self.estimator)}")
        else:
            estimator_ = self.estimator
        return estimator_


class PermutationForestClassifier(BasePermutationForest):
    """Hypothesis testing of covariates with a permutation forest classifier.

    This implements permutation testing of a null hypothesis using a random forest.
    The null hypothesis is generated by permuting ``n_repeats`` times the covariate
    indices and then a random forest is trained for each permuted instance. This
    is compared to the original random forest that was computed on the regular
    non-permuted data.

    .. warning:: Permutation testing with forests is computationally expensive.
        As a result, if you are testing for the importance of feature sets, consider
        using `sktree.FeatureImportanceForestRegressor` or
        `sktree.FeatureImportanceForestClassifier` instead, which is
        much more computationally efficient.

    .. note:: This does not allow testing on the posteriors.

    Parameters
    ----------
    estimator : object, default=None
        Type of forest estimator to use. By default `None`, which defaults to
        :class:`sklearn.ensemble.RandomForestClassifier`.

    test_size : float, default=0.2
        The proportion of samples to leave out for each tree to compute metric on.

    random_state : int, RandomState instance or None, default=None
        Controls both the randomness of the bootstrapping of the samples used
        when building trees (if ``bootstrap=True``) and the sampling of the
        features to consider when looking for the best split at each node
        (if ``max_features < n_features``).
        See :term:`Glossary <random_state>` for details.

    verbose : int, default=0
        Controls the verbosity when fitting and predicting.

    Attributes
    ----------
    samples_ : ArrayLike of shape (n_samples,)
        The indices of the samples used in the final test.

    y_true_ : ArrayLike of shape (n_samples_final,)
        The true labels of the samples used in the final test.

    posterior_ : ArrayLike of shape (n_samples_final, n_outputs)
        The predicted posterior probabilities of the samples used in the final test.

    null_dist_ : ArrayLike of shape (n_repeats,)
        The null distribution of the test statistic.

    posterior_null_ : ArrayLike of shape (n_samples_final, n_outputs, n_repeats)
        The posterior probabilities of the samples used in the final test for each
        permutation for the null distribution.
    """

    def __init__(
        self,
        estimator=None,
        test_size=0.2,
        random_state=None,
        verbose=0,
    ):
        super().__init__(
            estimator=estimator,
            test_size=test_size,
            random_state=random_state,
            verbose=verbose,
        )

    def _get_estimator(self):
        if not hasattr(self, "estimator_") and self.estimator is None:
            estimator_ = RandomForestClassifier()
        elif not isinstance(self.estimator, (ForestClassifier, sklearnForestClassifier)):
            raise RuntimeError(f"Estimator must be a ForestClassifier, got {type(self.estimator)}")
        else:
            estimator_ = self.estimator
        return estimator_
