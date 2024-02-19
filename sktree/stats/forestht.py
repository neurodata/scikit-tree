import threading
from typing import Callable, Optional, Tuple, Union

import numpy as np
from joblib import Parallel, delayed
from numpy.typing import ArrayLike
from sklearn.base import MetaEstimatorMixin, clone, is_classifier
from sklearn.ensemble._base import _partition_estimators
from sklearn.ensemble._forest import ForestClassifier as sklearnForestClassifier
from sklearn.ensemble._forest import ForestRegressor as sklearnForestRegressor
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.utils.multiclass import type_of_target
from sklearn.utils.validation import _is_fitted, check_X_y

from .._lib.sklearn.ensemble._forest import (
    ForestClassifier,
    ForestRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
    _get_n_samples_bootstrap,
    _parallel_build_trees,
)
from ..ensemble._honest_forest import HonestForestClassifier
from ..experimental import conditional_resample
from ..tree import DecisionTreeClassifier, DecisionTreeRegressor
from ..tree._classes import DTYPE
from .utils import (
    METRIC_FUNCTIONS,
    POSITIVE_METRICS,
    POSTERIOR_FUNCTIONS,
    REGRESSOR_METRICS,
    _compute_null_distribution_coleman,
    _non_nan_samples,
)


def _parallel_predict_proba(predict_proba, X, indices_test):
    """
    This is a utility function for joblib's Parallel.

    It can't go locally in ForestClassifier or ForestRegressor, because joblib
    complains that it cannot pickle it when placed there.
    """
    # each tree predicts proba with a list of output (n_samples, n_classes[i])
    prediction = predict_proba(X[indices_test, :], check_input=False)
    return prediction


def _parallel_build_trees_with_sepdata(
    tree: Union[DecisionTreeClassifier, DecisionTreeRegressor],
    n_trees: int,
    idx: int,
    indices_train: ArrayLike,
    X: ArrayLike,
    y: ArrayLike,
    covariate_index,
    bootstrap: bool,
    max_samples,
    sample_weight: Optional[ArrayLike] = None,
    class_weight=None,
    missing_values_in_feature_mask=None,
    classes=None,
    random_state=None,
):
    """Parallel function to build trees and compute posteriors.

    This inherently assumes that the caller function defines the indices
    for the training and testing data for each tree.
    """
    rng = np.random.default_rng(random_state)
    X_train = X[indices_train, :]
    y_train = y[indices_train, ...]

    if bootstrap:
        n_samples_bootstrap = _get_n_samples_bootstrap(
            n_samples=X_train.shape[0], max_samples=max_samples
        )
    else:
        n_samples_bootstrap = None

    # XXX: this currently creates a copy of X_train on RAM, which is not ideal
    # individual tree permutation of y labels
    if covariate_index is not None:
        indices = np.arange(X_train.shape[0], dtype=int)
        # perform permutation of covariates
        index_arr = rng.choice(indices, size=(X_train.shape[0], 1), replace=False, shuffle=True)
        perm_X_cov = X_train[index_arr, covariate_index]
        X_train[:, covariate_index] = perm_X_cov

    tree = _parallel_build_trees(
        tree,
        bootstrap,
        X_train,
        y_train,
        sample_weight,
        idx,
        n_trees,
        verbose=0,
        class_weight=class_weight,
        n_samples_bootstrap=n_samples_bootstrap,
        missing_values_in_feature_mask=missing_values_in_feature_mask,
        classes=classes,
    )
    return tree


class BaseForestHT(MetaEstimatorMixin):
    observe_samples_: ArrayLike
    observe_posteriors_: ArrayLike
    observe_stat_: float
    permute_samples_: ArrayLike
    permute_posteriors_: ArrayLike
    permute_stat_: float

    def __init__(
        self,
        estimator=None,
        random_state=None,
        verbose=0,
        test_size=0.2,
        stratify=False,
        conditional_perm=False,
        sample_dataset_per_tree=False,
        permute_forest_fraction=None,
        train_test_split=True,
    ):
        self.estimator = estimator
        self.random_state = random_state
        self.verbose = verbose
        self.test_size = test_size
        self.stratify = stratify
        self.conditional_perm = conditional_perm

        self.train_test_split = train_test_split
        # XXX: possibly removing these parameters
        self.sample_dataset_per_tree = sample_dataset_per_tree
        self.permute_forest_fraction = permute_forest_fraction

        self.n_samples_test_ = None
        self._n_samples_ = None
        self._metric = None
        self._covariate_index_cache_ = None
        self._type_of_target_ = None
        self.n_features_in_ = None
        self._is_fitted = False
        self._seeds = None
        self._perm_seeds = None

    @property
    def n_estimators(self):
        try:
            return self.estimator_.n_estimators
        except Exception:
            return self.permuted_estimator_.n_estimators
        finally:
            return self._get_estimator().n_estimators

    def reset(self):
        class_attributes = dir(type(self))
        instance_attributes = dir(self)

        for attr_name in instance_attributes:
            if attr_name.endswith("_") and attr_name not in class_attributes:
                delattr(self, attr_name)

        self.n_samples_test_ = None
        self._n_samples_ = None
        self._covariate_index_cache_ = None
        self._type_of_target_ = None
        self._metric = None
        self.n_features_in_ = None
        self._is_fitted = False
        self._seeds = None
        self._y = None

    def _get_estimators_indices(self, stratifier=None, sample_separate=False):
        indices = np.arange(self._n_samples_, dtype=int)

        # Get drawn indices along both sample and feature axes
        rng = np.random.default_rng(self.estimator_.random_state)

        if self.permute_forest_fraction is None:
            permute_forest_fraction = 0.0
        else:
            permute_forest_fraction = self.permute_forest_fraction

        # TODO: consolidate how we "sample/permute" per subset of the forest
        if self.sample_dataset_per_tree or permute_forest_fraction > 0.0:
            # sample random seeds
            if self._seeds is None:
                self._seeds = []
                self._n_permutations = 0

                for itree in range(self.n_estimators):
                    # For every N-trees that are defined by permute forest fraction
                    # we will sample a new seed appropriately
                    if itree % max(int(permute_forest_fraction * self.n_estimators), 1) == 0:
                        tree = self.estimator_.estimators_[itree]
                        if tree.random_state is None:
                            seed = rng.integers(low=0, high=np.iinfo(np.int32).max)
                        else:
                            seed = tree.random_state

                    self._seeds.append(seed)
            seeds = self._seeds

            for idx, tree in enumerate(self.estimator_.estimators_):
                seed = seeds[idx]

                # Operations accessing random_state must be performed identically
                # to those in `_parallel_build_trees()`
                indices_train, indices_test = train_test_split(
                    indices,
                    test_size=self.test_size,
                    shuffle=True,
                    stratify=stratifier,
                    random_state=seed,
                )

                yield indices_train, indices_test
        else:
            if self._seeds is None:
                if self.estimator_.random_state is None:
                    self._seeds = rng.integers(low=0, high=np.iinfo(np.int32).max)
                else:
                    self._seeds = self.estimator_.random_state

            indices_train, indices_test = train_test_split(
                indices,
                shuffle=True,
                test_size=self.test_size,
                stratify=stratifier,
                random_state=self._seeds,
            )

            for _ in range(self.estimator_.n_estimators):
                yield indices_train, indices_test

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
        if self._n_samples_ is None:
            raise RuntimeError("The estimator must be fitted before accessing this attribute.")

        # we are not train/test splitting, then
        if not self.train_test_split:
            return [
                (np.arange(self._n_samples_, dtype=int), np.array([], dtype=int))
                for _ in range(len(self.estimator_.estimators_))
            ]

        # Stratifier uses a cached _y attribute if available
        stratifier = self._y if is_classifier(self.estimator_) and self.stratify else None

        return [
            (indices_train, indices_test)
            for indices_train, indices_test in self._get_estimators_indices(stratifier=stratifier)
        ]

    def _statistic(
        self,
        estimator: ForestClassifier,
        X: ArrayLike,
        y: ArrayLike,
        covariate_index: ArrayLike,
        metric: str,
        return_posteriors: bool,
        **metric_kwargs,
    ):
        raise NotImplementedError("Subclasses should implement this!")

    def _check_input(self, X: ArrayLike, y: ArrayLike, covariate_index: Optional[ArrayLike] = None):
        X, y = check_X_y(X, y, ensure_2d=True, copy=True, multi_output=True, dtype=DTYPE)
        if y.ndim != 2:
            y = y.reshape(-1, 1)

        if covariate_index is not None:
            if not isinstance(covariate_index, (list, tuple, np.ndarray)):
                raise RuntimeError("covariate_index must be an iterable of integer indices")
            else:
                if not all(isinstance(idx, (np.integer, int)) for idx in covariate_index):
                    raise RuntimeError("Not all covariate_index are integer indices")

        if self.test_size * X.shape[0] < 5:
            raise RuntimeError(
                f"There are less than 5 testing samples used with "
                f"test_size={self.test_size} for X ({X.shape})."
            )

        if self._n_samples_ is not None and X.shape[0] != self._n_samples_:
            raise RuntimeError(
                f"X must have {self._n_samples_} samples, got {X.shape[0]}. "
                f"If running on a new dataset, call the 'reset' method."
            )
        if self.n_features_in_ is not None and X.shape[1] != self.n_features_in_:
            raise RuntimeError(
                f"X must have {self.n_features_in_} features, got {X.shape[1]}. "
                f"If running on a new dataset, call the 'reset' method."
            )
        if self._type_of_target_ is not None and type_of_target(y) != self._type_of_target_:
            raise RuntimeError(
                f"y must have type {self._type_of_target_}, got {type_of_target(y)}. "
                f"If running on a new dataset, call the 'reset' method."
            )

        if not self.train_test_split and not isinstance(self.estimator, HonestForestClassifier):
            raise RuntimeError("Train test split must occur if not using honest forest classifier.")

        if self.permute_forest_fraction is not None and self.permute_forest_fraction < 0.0:
            raise RuntimeError("permute_forest_fraction must be non-negative.")

        if (
            self.permute_forest_fraction is not None
            and self.permute_forest_fraction * self.n_estimators < 1.0
        ):
            raise RuntimeError(
                "permute_forest_fraction must be greater than 1./n_estimators, "
                f"got {self.permute_forest_fraction}."
            )

        return X, y, covariate_index

    def statistic(
        self,
        X: ArrayLike,
        y: ArrayLike,
        covariate_index: Optional[ArrayLike] = None,
        metric="sa98",
        return_posteriors: bool = False,
        check_input: bool = True,
        **metric_kwargs,
    ) -> Tuple[float, ArrayLike, ArrayLike]:
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
            The metric to compute, by default "sa98".
        return_posteriors : bool, optional
            Whether or not to return the posteriors, by default False.
        check_input : bool, optional
            Whether or not to check the input, by default True.
        **metric_kwargs : dict, optional
            Additional keyword arguments to pass to the metric function.

        Returns
        -------
        stat : float
            The test statistic.
        posterior_final : ArrayLike of shape (n_estimators, n_samples_final, n_outputs) or
            (n_estimators, n_samples_final), optional
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
            X, y, covariate_index = self._check_input(X, y, covariate_index)

        if self._n_samples_ is None:
            self._n_samples_, self.n_features_in_ = X.shape

        # Infer type of target y
        if self._type_of_target_ is None:
            self._type_of_target_ = type_of_target(y)

        if covariate_index is None:
            self.estimator_ = self._get_estimator()
            estimator = self.estimator_
        else:
            self.permuted_estimator_ = self._get_estimator()
            estimator = self.permuted_estimator_

            if not hasattr(self, "estimator_"):
                self.estimator_ = self._get_estimator()

                # XXX: this can be improved as an extra fit can be avoided, by
                # just doing error-checking
                # and then setting the internal meta data structures
                # first run a dummy fit on the samples to initialize the
                # internal data structure of the forest
                if is_classifier(self.estimator_):
                    _unique_y = []
                    for axis in range(y.shape[1]):
                        _unique_y.append(np.unique(y[:, axis]))
                    unique_y = np.hstack(_unique_y)
                    if unique_y.ndim > 1 and unique_y.shape[1] == 1:
                        unique_y = unique_y.ravel()
                    X_dummy = np.zeros((unique_y.shape[0], X.shape[1]))
                    self.estimator_.fit(X_dummy, unique_y)
                else:
                    if y.ndim > 1 and y.shape[1] == 1:
                        self.estimator_.fit(X[:2], y[:2].ravel())
                    else:
                        self.estimator_.fit(X[:2], y[:2])

        # Store a cache of the y variable
        if is_classifier(estimator):
            self._y = y.copy()

        # # XXX: this can be improved as an extra fit can be avoided, by just doing error-checking
        # # and then setting the internal meta data structures
        # # first run a dummy fit on the samples to initialize the
        # # internal data structure of the forest
        if not _is_fitted(estimator) and is_classifier(estimator):
            _unique_y = []
            for axis in range(y.shape[1]):
                _unique_y.append(np.unique(y[:, axis]))
            unique_y = np.hstack(_unique_y)
            if unique_y.ndim > 1 and unique_y.shape[1] == 1:
                unique_y = unique_y.ravel()
            X_dummy = np.zeros((unique_y.shape[0], X.shape[1]))
            estimator.fit(X_dummy, unique_y)
        elif not _is_fitted(estimator):
            if y.ndim > 1 and y.shape[1] == 1:
                estimator.fit(X[:2], y[:2].ravel())
            else:
                estimator.fit(X[:2], y[:2])

        # sampling a separate train/test per tree
        if self.sample_dataset_per_tree:
            self.n_samples_test_ = self._n_samples_
        else:
            # here we fix a training/testing dataset
            test_size_ = int(self.test_size * self._n_samples_)

            # Fit each tree and compute posteriors with train test splits
            self.n_samples_test_ = test_size_

        if self._metric is not None and self._metric != metric:
            raise RuntimeError(
                f"Metric must be {self._metric}, got {metric}. "
                f"If running on a new dataset, call the 'reset' method."
            )
        self._metric = metric

        if not is_classifier(estimator) and metric not in REGRESSOR_METRICS:
            raise RuntimeError(
                f'Metric must be either "mse" or "mae" if using Regression, got {metric}'
            )

        if estimator.n_outputs_ > 1 and metric == "auc":
            raise ValueError("AUC metric is not supported for multi-output")

        return self._statistic(
            estimator,
            X,
            y,
            covariate_index=covariate_index,
            metric=metric,
            return_posteriors=return_posteriors,
            **metric_kwargs,
        )

    def test(
        self,
        X,
        y,
        covariate_index: Optional[ArrayLike] = None,
        metric: str = "mi",
        n_repeats: int = 1000,
        return_posteriors: bool = True,
        **metric_kwargs,
    ):
        """Perform hypothesis test using Coleman method.

        X is split into a training/testing split. Optionally, the covariate index
        columns are shuffled.

        On the training dataset, two honest forests are trained and then the posterior
        is estimated on the testing dataset. One honest forest is trained on the
        permuted dataset and the other is trained on the original dataset.

        Finally, resample the posteriors of the two forests to compute the null
        distribution of the statistics.

        Parameters
        ----------
        X : ArrayLike of shape (n_samples, n_features)
            The data matrix.
        y : ArrayLike of shape (n_samples, n_outputs)
            The target matrix.
        covariate_index : ArrayLike, optional of shape (n_covariates,)
            The index array of covariates to shuffle, will shuffle all columns by
            default (corresponding to None).
        metric : str, optional
            The metric to compute, by default "mse".
        n_repeats : int, optional
            Number of times to sample the null distribution, by default 1000.
        return_posteriors : bool, optional
            Whether or not to return the posteriors, by default True.
        **metric_kwargs : dict, optional
            Additional keyword arguments to pass to the metric function.

        Returns
        -------
        stat : float
            The test statistic. To compute the test statistic, take
            ``permute_stat_`` and subtract ``observe_stat_``.
        pval : float
            The p-value of the test statistic.
        """
        X, y, covariate_index = self._check_input(X, y, covariate_index)

        if not self._is_fitted:
            # first compute the test statistic on the un-permuted data
            observe_stat, observe_posteriors, observe_samples = self.statistic(
                X,
                y,
                covariate_index=None,
                metric=metric,
                return_posteriors=return_posteriors,
                check_input=False,
                **metric_kwargs,
            )
        else:
            observe_samples = self.observe_samples_
            observe_posteriors = self.observe_posteriors_
            observe_stat = self.observe_stat_

        if covariate_index is None:
            covariate_index = np.arange(X.shape[1], dtype=int)

        # next permute the data
        permute_stat, permute_posteriors, permute_samples = self.statistic(
            X,
            y,
            covariate_index=covariate_index,
            metric=metric,
            return_posteriors=return_posteriors,
            check_input=False,
            **metric_kwargs,
        )
        self.permute_stat_ = permute_stat

        # Note: at this point, both `estimator` and `permuted_estimator_` should
        # have been fitted already, so we can now compute on the null by resampling
        # the posteriors and computing the test statistic on the resampled posteriors
        if self.sample_dataset_per_tree:
            metric_star, metric_star_pi = _compute_null_distribution_coleman(
                y_test=y,
                y_pred_proba_normal=observe_posteriors,
                y_pred_proba_perm=permute_posteriors,
                metric=metric,
                n_repeats=n_repeats,
                seed=self.random_state,
                **metric_kwargs,
            )
        else:
            # If not sampling a new dataset per tree, then we may either be
            # permuting the covariate index per tree or per forest. If not permuting
            # there is only one train and test split, so we can just use that
            _, indices_test = self.train_test_samples_[0]
            indices_test = observe_samples
            y_test = y[indices_test, :]
            y_pred_proba_normal = observe_posteriors[:, indices_test, :]
            y_pred_proba_perm = permute_posteriors[:, indices_test, :]

            metric_star, metric_star_pi = _compute_null_distribution_coleman(
                y_test=y_test,
                y_pred_proba_normal=y_pred_proba_normal,
                y_pred_proba_perm=y_pred_proba_perm,
                metric=metric,
                n_repeats=n_repeats,
                seed=self.random_state,
                **metric_kwargs,
            )
        # metric^\pi - metric = observed test statistic, which under the
        # null is normally distributed around 0
        observe_test_stat = permute_stat - observe_stat

        # metric^\pi_j - metric_j, which is centered at 0
        null_dist = metric_star_pi - metric_star

        # compute pvalue
        if metric in POSITIVE_METRICS:
            pvalue = (1 + (null_dist <= observe_test_stat).sum()) / (1 + n_repeats)
        else:
            pvalue = (1 + (null_dist >= observe_test_stat).sum()) / (1 + n_repeats)

        if return_posteriors:
            self.observe_posteriors_ = observe_posteriors
            self.permute_posteriors_ = permute_posteriors
            self.observe_samples_ = observe_samples
            self.permute_samples_ = permute_samples

        self.null_dist_ = null_dist
        return observe_test_stat, pvalue


class FeatureImportanceForestRegressor(BaseForestHT):
    """Forest hypothesis testing with continuous `y` variable.

    Implements the algorithm described in :footcite:`coleman2022scalable`.

    The dataset is split into a training and testing dataset initially. Then there
    are two forests that are trained: one on the original dataset, and one on the
    permuted dataset. The dataset is either permuted once, or independently for
    each tree in the permuted forest. The original test statistic is computed by
    comparing the metric on both forests ``(metric_forest - metric_perm_forest)``.

    Then the output predictions are randomly sampled to recompute the test statistic
    ``n_repeats`` times. The p-value is computed as the proportion of times the
    null test statistic is greater than the original test statistic.

    Parameters
    ----------
    estimator : object, default=None
        Type of forest estimator to use. By default `None`, which defaults to
        :class:`sklearn.ensemble.RandomForestRegressor`.

    random_state : int, RandomState instance or None, default=None
        Controls both the randomness of the bootstrapping of the samples used
        when building trees (if ``bootstrap=True``) and the sampling of the
        features to consider when looking for the best split at each node
        (if ``max_features < n_features``).
        See :term:`Glossary <random_state>` for details.

    verbose : int, default=0
        Controls the verbosity when fitting and predicting.

    test_size : float, default=0.2
        Proportion of samples per tree to use for the test set.

    sample_dataset_per_tree : bool, default=False
        Whether to sample the dataset per tree or per forest.

    conditional_perm : bool, default=False
        Whether or not to conditionally permute the covariate index. If True,
        then the covariate index is permuted while preserving the joint with respect
        to the rest of the covariates.

    permute_forest_fraction : float, default=None
        The fraction of trees to permute the covariate index for. If None, then
        just one permutation is performed. If sampling a permutation per tree
        is desirable, then the fraction should be set to ``1. / n_estimators``.

    train_test_split : bool, default=True
        Whether to split the dataset before passing to the forest.

    Attributes
    ----------
    estimator_ : BaseForest
        The estimator used to compute the test statistic.

    n_samples_test_ : int
        The number of samples used in the final test set.

    indices_train_ : ArrayLike of shape (n_samples_train,)
        The indices of the samples used in the training set.

    indices_test_ : ArrayLike of shape (n_samples_test,)
        The indices of the samples used in the testing set.

    samples_ : ArrayLike of shape (n_samples_final,)
        The indices of the samples used in the final test set that would slice
        the original ``(X, y)`` input.

    y_true_final_ : ArrayLike of shape (n_samples_final,)
        The true labels of the samples used in the final test.

    observe_posteriors_ : ArrayLike of shape (n_estimators, n_samples, n_outputs) or
        (n_estimators, n_samples, n_classes)
        The predicted posterior probabilities of the samples used in the final test.
        For samples that are NaNs for all estimators, means the sample was not used
        in the test set at all across all trees.

    null_dist_ : ArrayLike of shape (n_repeats,)
        The null distribution of the test statistic.

    Notes
    -----
    This class trains two forests: one on the original dataset, and one on the
    permuted dataset. The forest from the original dataset is cached and re-used to
    compute the test-statistic each time the :meth:`test` method is called. However,
    the forest from the permuted dataset is re-trained each time the :meth:`test` is called
    if the ``covariate_index`` differs from the previous run.

    To fully start from a new dataset, call the ``reset`` method, which will then
    re-train both forests upon calling the :meth:`test` and :meth:`statistic` methods.

    References
    ----------
    .. footbibliography::
    """

    def __init__(
        self,
        estimator=None,
        random_state=None,
        verbose=0,
        test_size=0.2,
        sample_dataset_per_tree=False,
        conditional_perm=False,
        permute_forest_fraction=None,
        train_test_split=True,
    ):
        super().__init__(
            estimator=estimator,
            random_state=random_state,
            verbose=verbose,
            test_size=test_size,
            sample_dataset_per_tree=sample_dataset_per_tree,
            conditional_perm=conditional_perm,
            permute_forest_fraction=permute_forest_fraction,
            train_test_split=train_test_split,
        )

    def _get_estimator(self):
        if self.estimator is None:
            estimator_ = RandomForestRegressor()
        elif not isinstance(self.estimator, (ForestRegressor, sklearnForestRegressor)):
            raise RuntimeError(f"Estimator must be a ForestRegressor, got {type(self.estimator)}")
        else:
            estimator_ = self.estimator
        return clone(estimator_)

    def statistic(
        self,
        X: ArrayLike,
        y: ArrayLike,
        covariate_index: Optional[ArrayLike] = None,
        metric="mse",
        return_posteriors: bool = False,
        check_input: bool = True,
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
        **metric_kwargs : dict, optional
            Additional keyword arguments to pass to the metric function.

        Returns
        -------
        stat : float
            The test statistic.
        posterior_final : ArrayLike of shape (n_estimators, n_samples_final, n_outputs) or
            (n_estimators, n_samples_final), optional
            If ``return_posteriors`` is True, then the posterior probabilities of the
            samples used in the final test. ``n_samples_final`` is equal to ``n_samples``
            if all samples are encountered in the test set of at least one tree in the
            posterior computation.
        samples : ArrayLike of shape (n_samples_final,), optional
            The indices of the samples used in the final test. ``n_samples_final`` is
            equal to ``n_samples`` if all samples are encountered in the test set of at
            least one tree in the posterior computation.
        """
        return super().statistic(
            X, y, covariate_index, metric, return_posteriors, check_input, **metric_kwargs
        )

    def _statistic(
        self,
        estimator: ForestClassifier,
        X: ArrayLike,
        y: ArrayLike,
        covariate_index: ArrayLike,
        metric: str,
        return_posteriors: bool,
        **metric_kwargs,
    ):
        """Helper function to compute the test statistic."""
        metric_func: Callable[[ArrayLike, ArrayLike], float] = METRIC_FUNCTIONS[metric]
        rng = np.random.default_rng(self.random_state)

        posterior_arr = np.full((self.n_estimators, self._n_samples_, estimator.n_outputs_), np.nan)

        # both sampling dataset per tree or permuting per tree requires us to bypass the
        # sklearn API to fit each tree individually
        if self.sample_dataset_per_tree or self.permute_forest_fraction:
            if self.permute_forest_fraction and covariate_index is not None:
                random_states = [tree.random_state for tree in estimator.estimators_]
            else:
                random_states = [estimator.random_state] * len(estimator.estimators_)

            trees = Parallel(n_jobs=estimator.n_jobs, verbose=self.verbose, prefer="threads")(
                delayed(_parallel_build_trees_with_sepdata)(
                    estimator.estimators_[idx],
                    len(estimator.estimators_),
                    idx,
                    indices_train,
                    X,
                    y,
                    covariate_index,
                    bootstrap=estimator.bootstrap,
                    max_samples=estimator.max_samples,
                    random_state=random_states[idx],
                )
                for idx, (indices_train, _) in enumerate(self.train_test_samples_)
            )
            estimator.estimators_ = trees
        else:
            # fitting a forest will only get one unique train/test split
            indices_train, indices_test = self.train_test_samples_[0]

            X_train, _ = X[indices_train, :], X[indices_test, :]
            y_train, _ = y[indices_train, :], y[indices_test, :]

            if covariate_index is not None:
                # perform permutation of covariates
                if self.conditional_perm:
                    X_perm_cov_ind = conditional_resample(
                        X_train, X_train[:, covariate_index], replace=False, random_state=rng
                    )
                    X_train[:, covariate_index] = X_perm_cov_ind
                else:
                    n_samples_train = X_train.shape[0]
                    index_arr = rng.choice(
                        np.arange(n_samples_train, dtype=int),
                        size=(n_samples_train, 1),
                        replace=False,
                        shuffle=True,
                    )
                    X_train[:, covariate_index] = X_train[index_arr, covariate_index]

            if self._type_of_target_ == "binary":
                y_train = y_train.ravel()
            estimator.fit(X_train, y_train)

        # TODO: probably a more elegant way of doing this
        if self.train_test_split:
            # accumulate the predictions across all trees
            all_proba = Parallel(n_jobs=estimator.n_jobs, verbose=self.verbose)(
                delayed(_parallel_predict_proba)(
                    estimator.estimators_[idx].predict, X, indices_test
                )
                for idx, (_, indices_test) in enumerate(self.train_test_samples_)
            )
            for itree, (proba, est_indices) in enumerate(zip(all_proba, self.train_test_samples_)):
                _, indices_test = est_indices
                posterior_arr[itree, indices_test, ...] = proba.reshape(-1, estimator.n_outputs_)
        else:
            all_indices = np.arange(self._n_samples_, dtype=int)

            # accumulate the predictions across all trees
            all_proba = Parallel(n_jobs=estimator.n_jobs, verbose=self.verbose)(
                delayed(_parallel_predict_proba)(estimator.estimators_[idx].predict, X, all_indices)
                for idx in range(len(estimator.estimators_))
            )
            for itree, proba in enumerate(all_proba):
                posterior_arr[itree, ...] = proba.reshape(-1, estimator.n_outputs_)

        # determine if there are any nans in the final posterior array, when
        # averaged over the trees
        samples = _non_nan_samples(posterior_arr)

        # Ignore all NaN values (samples not tested)
        y_true_final = y[(samples), :]

        # Average all posteriors (n_samples_test, n_outputs) to compute the statistic
        posterior_forest = np.nanmean(posterior_arr[:, (samples), :], axis=0)
        stat = metric_func(y_true_final, posterior_forest, **metric_kwargs)
        if covariate_index is None:
            # Ignore all NaN values (samples not tested) -> (n_samples_final, n_outputs)
            # arrays of y and predicted posterior
            self.observe_samples_ = samples
            self.y_true_final_ = y_true_final
            self.observe_posteriors_ = posterior_arr
            self.observe_stat_ = stat
            self._is_fitted = True

        if return_posteriors:
            return stat, posterior_arr, samples

        return stat


class FeatureImportanceForestClassifier(BaseForestHT):
    """Forest hypothesis testing with categorical `y` variable.

    Implements the algorithm described in :footcite:`coleman2022scalable`.

    The dataset is split into a training and testing dataset initially. Then there
    are two forests that are trained: one on the original dataset, and one on the
    permuted dataset. The dataset is either permuted once, or independently for
    each tree in the permuted forest. The original test statistic is computed by
    comparing the metric on both forests ``(metric_forest - metric_perm_forest)``.

    Then the output predictions are randomly sampled to recompute the test statistic
    ``n_repeats`` times. The p-value is computed as the proportion of times the
    null test statistic is greater than the original test statistic.

    Parameters
    ----------
    estimator : object, default=None
        Type of forest estimator to use. By default `None`, which defaults to
        :class:`sklearn.ensemble.RandomForestRegressor`.

    random_state : int, RandomState instance or None, default=None
        Controls both the randomness of the bootstrapping of the samples used
        when building trees (if ``bootstrap=True``) and the sampling of the
        features to consider when looking for the best split at each node
        (if ``max_features < n_features``).
        See :term:`Glossary <random_state>` for details.

    verbose : int, default=0
        Controls the verbosity when fitting and predicting.

    test_size : float, default=0.2
        Proportion of samples per tree to use for the test set.

    stratify : bool, default=True
        Whether to stratify the samples by class labels.

    conditional_perm : bool, default=False
        Whether or not to conditionally permute the covariate index. If True,
        then the covariate index is permuted while preserving the joint with respect
        to the rest of the covariates.

    sample_dataset_per_tree : bool, default=False
        Whether to sample the dataset per tree or per forest.

    permute_forest_fraction : float, default=None
        The fraction of trees to permute the covariate index for. If None, then
        just one permutation is performed.

    train_test_split : bool, default=True
        Whether to split the data into train/test before passing to the forest.

    Attributes
    ----------
    estimator_ : BaseForest
        The estimator used to compute the test statistic.

    n_samples_test_ : int
        The number of samples used in the final test set.

    indices_train_ : ArrayLike of shape (n_samples_train,)
        The indices of the samples used in the training set.

    indices_test_ : ArrayLike of shape (n_samples_test,)
        The indices of the samples used in the testing set.

    samples_ : ArrayLike of shape (n_samples_final,)
        The indices of the samples used in the final test set that would slice
        the original ``(X, y)`` input along the rows.

    y_true_final_ : ArrayLike of shape (n_samples_final,)
        The true labels of the samples used in the final test.

    observe_posteriors_ : ArrayLike of shape (n_estimators, n_samples_final, n_outputs) or
        (n_estimators, n_samples_final, n_classes)
        The predicted posterior probabilities of the samples used in the final test.

    null_dist_ : ArrayLike of shape (n_repeats,)
        The null distribution of the test statistic.

    Notes
    -----
    This class trains two forests: one on the original dataset, and one on the
    permuted dataset. The forest from the original dataset is cached and re-used to
    compute the test-statistic each time the :meth:`test` method is called. However,
    the forest from the permuted dataset is re-trained each time the :meth:`test` is called
    if the ``covariate_index`` differs from the previous run.

    To fully start from a new dataset, call the ``reset`` method, which will then
    re-train both forests upon calling the :meth:`test` and :meth:`statistic` methods.

    References
    ----------
    .. footbibliography::
    """

    def __init__(
        self,
        estimator=None,
        random_state=None,
        verbose=0,
        test_size=0.2,
        stratify=True,
        conditional_perm=False,
        sample_dataset_per_tree=False,
        permute_forest_fraction=None,
        train_test_split=True,
    ):
        super().__init__(
            estimator=estimator,
            random_state=random_state,
            verbose=verbose,
            test_size=test_size,
            sample_dataset_per_tree=sample_dataset_per_tree,
            stratify=stratify,
            conditional_perm=conditional_perm,
            train_test_split=train_test_split,
            permute_forest_fraction=permute_forest_fraction,
        )

    def _get_estimator(self):
        if self.estimator is None:
            estimator_ = RandomForestClassifier()
        elif not isinstance(self.estimator, (ForestClassifier, sklearnForestClassifier)):
            raise RuntimeError(f"Estimator must be a ForestClassifier, got {type(self.estimator)}")
        else:
            # self.estimator is an instance of a ForestEstimator
            estimator_ = self.estimator
        return clone(estimator_)

    def _statistic(
        self,
        estimator: ForestClassifier,
        X: ArrayLike,
        y: ArrayLike,
        covariate_index: ArrayLike,
        metric: str,
        return_posteriors: bool,
        **metric_kwargs,
    ):
        """Helper function to compute the test statistic."""
        metric_func: Callable[[ArrayLike, ArrayLike], float] = METRIC_FUNCTIONS[metric]
        rng = np.random.default_rng(estimator.random_state)

        if metric in POSTERIOR_FUNCTIONS:
            predict_posteriors = True
        else:
            predict_posteriors = False

        if predict_posteriors:
            # now initialize posterior array as (n_trees, n_samples_test, n_classes)
            # XXX: currently assumes n_outputs_ == 1
            posterior_arr = np.full(
                (self.n_estimators, self._n_samples_, estimator.n_classes_), np.nan
            )
        else:
            # now initialize posterior array as (n_trees, n_samples_test, n_outputs)
            posterior_arr = np.full(
                (self.n_estimators, self._n_samples_, estimator.n_outputs_), np.nan
            )

        # both sampling dataset per tree or permuting per tree requires us to bypass the
        # sklearn API to fit each tree individually
        if self.sample_dataset_per_tree or self.permute_forest_fraction:
            if self.permute_forest_fraction and covariate_index is not None:
                random_states = [tree.random_state for tree in estimator.estimators_]
            else:
                random_states = [estimator.random_state] * len(estimator.estimators_)

            trees = Parallel(n_jobs=estimator.n_jobs, verbose=self.verbose, prefer="threads")(
                delayed(_parallel_build_trees_with_sepdata)(
                    estimator.estimators_[idx],
                    len(estimator.estimators_),
                    idx,
                    indices_train,
                    X,
                    y,
                    covariate_index,
                    bootstrap=estimator.bootstrap,
                    max_samples=estimator.max_samples,
                    random_state=random_states[idx],
                )
                for idx, (indices_train, _) in enumerate(self.train_test_samples_)
            )
            estimator.estimators_ = trees
        else:
            # fitting a forest will only get one unique train/test split
            indices_train, indices_test = self.train_test_samples_[0]

            X_train, _ = X[indices_train, :], X[indices_test, :]
            y_train = y[indices_train, :]

            if covariate_index is not None:
                # perform permutation of covariates
                if self.conditional_perm:
                    X_perm_cov_ind = conditional_resample(
                        X_train, X_train[:, covariate_index], replace=False, random_state=rng
                    )
                    X_train[:, covariate_index] = X_perm_cov_ind
                else:
                    n_samples_train = X_train.shape[0]
                    index_arr = rng.choice(
                        np.arange(n_samples_train, dtype=int),
                        size=(n_samples_train, 1),
                        replace=False,
                        shuffle=True,
                    )
                    X_train[:, covariate_index] = X_train[index_arr, covariate_index]

            if self._type_of_target_ == "binary" or (y.ndim > 1 and y.shape[1] == 1):
                y_train = y_train.ravel()
            estimator.fit(X_train, y_train)

        # list of tree outputs. Each tree output is (n_samples, n_outputs), or (n_samples,)
        if predict_posteriors:
            # all_proba = Parallel(n_jobs=estimator.n_jobs, verbose=self.verbose)(
            #     delayed(_parallel_predict_proba)(
            #         estimator.estimators_[idx].predict_proba, X, indices_test
            #     )
            #     for idx, (_, indices_test) in enumerate(self.train_test_samples_)
            # )

            # TODO: probably a more elegant way of doing this
            if self.train_test_split:
                # accumulate the predictions across all trees
                all_proba = Parallel(n_jobs=estimator.n_jobs, verbose=self.verbose)(
                    delayed(_parallel_predict_proba)(
                        estimator.estimators_[idx].predict_proba, X, indices_test
                    )
                    for idx, (_, indices_test) in enumerate(self.train_test_samples_)
                )
            else:
                all_indices = np.arange(self._n_samples_, dtype=int)

                # accumulate the predictions across all trees
                all_proba = Parallel(n_jobs=estimator.n_jobs, verbose=self.verbose)(
                    delayed(_parallel_predict_proba)(
                        estimator.estimators_[idx].predict_proba, X, all_indices
                    )
                    for idx in range(len(estimator.estimators_))
                )
        else:
            all_proba = Parallel(n_jobs=estimator.n_jobs, verbose=self.verbose)(
                delayed(_parallel_predict_proba)(
                    estimator.estimators_[idx].predict, X, indices_test
                )
                for idx, (_, indices_test) in enumerate(self.train_test_samples_)
            )
        for itree, (proba, est_indices) in enumerate(zip(all_proba, self.train_test_samples_)):
            _, indices_test = est_indices

            if predict_posteriors:
                if self.train_test_split:
                    posterior_arr[itree, indices_test, ...] = proba.reshape(
                        -1, estimator.n_classes_
                    )
                else:
                    posterior_arr[itree, ...] = proba.reshape(-1, estimator.n_classes_)
            else:
                posterior_arr[itree, indices_test, ...] = proba.reshape(-1, estimator.n_outputs_)

        if metric == "auc":
            # at this point, posterior_final is the predicted posterior for only the positive class
            # as more than one output is not supported.
            if self._type_of_target_ == "binary":
                posterior_arr = posterior_arr[..., (1,)]
            else:
                raise RuntimeError(
                    f"AUC metric is not supported for {self._type_of_target_} targets."
                )

        # determine if there are any nans in the final posterior array, when
        # averaged over the trees
        samples = _non_nan_samples(posterior_arr)

        # Ignore all NaN values (samples not tested)
        y_true_final = y[(samples), :]

        # Average all posteriors (n_samples_test, n_outputs) to compute the statistic
        posterior_forest = np.nanmean(posterior_arr[:, (samples), :], axis=0)
        stat = metric_func(y_true_final, posterior_forest, **metric_kwargs)

        if covariate_index is None:
            # Ignore all NaN values (samples not tested) -> (n_samples_final, n_outputs)
            # arrays of y and predicted posterior
            self.observe_samples_ = samples
            self.y_true_final_ = y_true_final
            self.observe_posteriors_ = posterior_arr
            self.observe_stat_ = stat
            self._is_fitted = True

        if return_posteriors:
            return stat, posterior_arr, samples

        return stat


def _parallel_predict_proba_oob(predict_proba, X, out, idx, test_idx, lock):
    """
    This is a utility function for joblib's Parallel.
    It can't go locally in ForestClassifier or ForestRegressor, because joblib
    complains that it cannot pickle it when placed there.
    """
    # each tree predicts proba with a list of output (n_samples, n_classes[i])
    prediction = predict_proba(X, check_input=False)

    indices = np.zeros(X.shape[0], dtype=bool)
    indices[test_idx] = True
    with lock:
        out[idx, test_idx, :] = prediction[test_idx, :]
    return prediction


def build_coleman_forest(
    est,
    perm_est,
    X,
    y,
    covariate_index=None,
    metric="s@98",
    n_repeats=1000,
    verbose=False,
    seed=None,
    return_posteriors=True,
    **metric_kwargs,
):
    """Build a hypothesis testing forest using oob samples.

    Parameters
    ----------
    est : Forest
        The type of forest to use. Must be enabled with ``bootstrap=True``.
    perm_est : Forest
        The forest to use for the permuted dataset.
    X : ArrayLike of shape (n_samples, n_features)
        Data.
    y : ArrayLike of shape (n_samples, n_outputs)
        Binary target, so ``n_outputs`` should be at most 1.
    covariate_index : ArrayLike, optional of shape (n_covariates,)
        The index array of covariates to shuffle, by default None.
    metric : str, optional
        The metric to compute, by default "s@98", for sensitivity at
        98% specificity.
    n_repeats : int, optional
        Number of times to bootstrap sample the two forests to construct
        the null distribution, by default 1000.
    verbose : bool, optional
        Verbosity, by default False.
    seed : int, optional
        Random seed, by default None.
    return_posteriors : bool, optional
        Whether or not to return the posteriors, by default False.
    **metric_kwargs : dict, optional
        Additional keyword arguments to pass to the metric function.

    Returns
    -------
    observe_stat : float
        The test statistic. To compute the test statistic, take
        ``permute_stat_`` and subtract ``observe_stat_``.
    pvalue : float
        The p-value of the test statistic.
    orig_forest_proba : ArrayLike of shape (n_estimators, n_samples, n_outputs)
        The predicted posterior probabilities for each estimator on their
        out of bag samples.
    perm_forest_proba : ArrayLike of shape (n_estimators, n_samples, n_outputs)
        The predicted posterior probabilities for each of the permuted estimators
        on their out of bag samples.
    """
    rng = np.random.default_rng(seed)
    metric_func: Callable[[ArrayLike, ArrayLike], float] = METRIC_FUNCTIONS[metric]

    if covariate_index is None:
        covariate_index = np.arange(X.shape[1], dtype=int)

    # perform permutation of covariates
    # TODO: refactor permutations into the HonestForest(?)
    n_samples_train = X.shape[0]
    index_arr = rng.choice(
        np.arange(n_samples_train, dtype=int),
        size=(n_samples_train, 1),
        replace=False,
        shuffle=True,
    )
    X_permute = X.copy()
    X_permute[:, covariate_index] = X_permute[index_arr, covariate_index]

    # build two sets of forests
    est, orig_forest_proba = build_hyppo_oob_forest(est, X, y, verbose=verbose)
    perm_est, perm_forest_proba = build_hyppo_oob_forest(
        perm_est, X_permute, y, verbose=verbose, covariate_index=covariate_index
    )

    metric_star, metric_star_pi = _compute_null_distribution_coleman(
        y,
        orig_forest_proba,
        perm_forest_proba,
        metric,
        n_repeats=n_repeats,
        seed=seed,
        **metric_kwargs,
    )

    y_pred_proba_orig = np.nanmean(orig_forest_proba, axis=0)
    y_pred_proba_perm = np.nanmean(perm_forest_proba, axis=0)
    observe_stat = metric_func(y, y_pred_proba_orig, **metric_kwargs)
    permute_stat = metric_func(y, y_pred_proba_perm, **metric_kwargs)

    # metric^\pi - metric = observed test statistic, which under the
    # null is normally distributed around 0
    observe_test_stat = permute_stat - observe_stat

    # metric^\pi_j - metric_j, which is centered at 0
    null_dist = metric_star_pi - metric_star

    # compute pvalue
    if metric in POSITIVE_METRICS:
        pvalue = (1 + (null_dist <= observe_test_stat).sum()) / (1 + n_repeats)
    else:
        pvalue = (1 + (null_dist >= observe_test_stat).sum()) / (1 + n_repeats)

    if return_posteriors:
        return observe_test_stat, pvalue, orig_forest_proba, perm_forest_proba
    else:
        return observe_test_stat, pvalue


def build_hyppo_oob_forest(est, X, y, verbose=False, **est_kwargs):
    """Build a hypothesis testing forest using oob samples.

    Parameters
    ----------
    est : Forest
        The type of forest to use. Must be enabled with ``bootstrap=True``.
    X : ArrayLike of shape (n_samples, n_features)
        Data.
    y : ArrayLike of shape (n_samples, n_outputs)
        Binary target, so ``n_outputs`` should be at most 1.
    verbose : bool, optional
        Verbosity, by default False.
    **est_kwargs : dict, optional
        Additional keyword arguments to pass to the forest estimator.

    Returns
    -------
    est : Forest
        Fitted forest.
    all_proba : ArrayLike of shape (n_estimators, n_samples, n_outputs)
        The predicted posterior probabilities for each estimator on their
        out of bag samples.
    """
    assert est.bootstrap
    assert type_of_target(y) in ("binary")
    est = clone(est)

    # build forest
    est.fit(X, y.ravel(), **est_kwargs)

    # now evaluate
    X = est._validate_X_predict(X)

    # if we trained a binning tree, then we should re-bin the data
    # XXX: this is inefficient and should be improved to be in line with what
    # the Histogram Gradient Boosting Tree does, where the binning thresholds
    # are passed into the tree itself, thus allowing us to set the node feature
    # value thresholds within the tree itself.
    if est.max_bins is not None:
        X = est._bin_data(X, is_training_data=False).astype(DTYPE)

    # Assign chunk of trees to jobs
    n_jobs, _, _ = _partition_estimators(est.n_estimators, est.n_jobs)

    # avoid storing the output of every estimator by summing them here
    lock = threading.Lock()
    # accumulate the predictions across all trees
    all_proba = np.full(
        (len(est.estimators_), X.shape[0], est.n_classes_), np.nan, dtype=np.float64
    )
    Parallel(n_jobs=n_jobs, verbose=verbose, require="sharedmem")(
        delayed(_parallel_predict_proba_oob)(e.predict_proba, X, all_proba, idx, test_idx, lock)
        for idx, (e, test_idx) in enumerate(zip(est.estimators_, est.oob_samples_))
    )
    return est, all_proba


def build_hyppo_cv_forest(
    est,
    X,
    y,
    cv=5,
    test_size=0.2,
    verbose=False,
    seed=None,
):
    """Build a hypothesis testing forest using oob samples.

    Parameters
    ----------
    est : Forest
        The type of forest to use. Must be enabled with ``bootstrap=True``.
    X : ArrayLike of shape (n_samples, n_features)
        Data.
    y : ArrayLike of shape (n_samples, n_outputs)
        Binary target, so ``n_outputs`` should be at most 1.
    cv : int, optional
        Number of folds to use for cross-validation, by default 5.
    test_size : float, optional
        Proportion of samples per tree to use for the test set, by default 0.2.
    verbose : bool, optional
        Verbosity, by default False.
    seed : int, optional
        Random seed, by default None.

    Returns
    -------
    est : Forest
        Fitted forest.
    all_proba_list : list of ArrayLike of shape (n_estimators, n_samples, n_outputs)
        The predicted posterior probabilities for each estimator on their
        out of bag samples. Length of list is equal to the number of splits.
    """
    X = X.astype(np.float32)
    if cv is not None:
        cv = StratifiedKFold(n_splits=cv, shuffle=True, random_state=seed)
        n_splits = cv.get_n_splits()
        train_idx_list, test_idx_list = [], []
        for train_idx, test_idx in cv.split(X, y):
            train_idx_list.append(train_idx)
            test_idx_list.append(test_idx)
    else:
        n_samples_idx = np.arange(len(y))
        train_idx, test_idx = train_test_split(
            n_samples_idx, test_size=test_size, random_state=seed, shuffle=True, stratify=y
        )

        train_idx_list = [train_idx]
        test_idx_list = [test_idx]
        n_splits = 1

    est_list = []
    all_proba_list = []
    for isplit in range(n_splits):
        X_train, y_train = X[train_idx_list[isplit], :], y[train_idx_list[isplit]]
        # X_test = X[test_idx_list[isplit], :]

        # build forest
        est.fit(X_train, y_train)

        # now evaluate
        X = est._validate_X_predict(X)

        # if we trained a binning tree, then we should re-bin the data
        # XXX: this is inefficient and should be improved to be in line with what
        # the Histogram Gradient Boosting Tree does, where the binning thresholds
        # are passed into the tree itself, thus allowing us to set the node feature
        # value thresholds within the tree itself.
        if est.max_bins is not None:
            X = est._bin_data(X, is_training_data=False).astype(DTYPE)

        # Assign chunk of trees to jobs
        n_jobs, _, _ = _partition_estimators(est.n_estimators, est.n_jobs)

        # avoid storing the output of every estimator by summing them here
        all_proba = Parallel(n_jobs=n_jobs, verbose=verbose)(
            delayed(_parallel_predict_proba)(e.predict_proba, X, test_idx_list[isplit])
            for e in est.estimators_
        )
        posterior_arr = np.full((est.n_estimators, X.shape[0], est.n_classes_), np.nan)
        for itree, (proba) in enumerate(all_proba):
            posterior_arr[itree, test_idx_list[isplit], ...] = proba.reshape(-1, est.n_classes_)

        all_proba_list.append(posterior_arr)
        est_list.append(est)

    return est_list, all_proba_list


def predict_oob_proba(est, X, verbose=False):
    """Predict out of bag posterior probabilities."""
    from sklearn.utils.validation import check_is_fitted

    check_is_fitted(est)

    # now evaluate
    X = est._validate_X_predict(X)

    # if we trained a binning tree, then we should re-bin the data
    # XXX: this is inefficient and should be improved to be in line with what
    # the Histogram Gradient Boosting Tree does, where the binning thresholds
    # are passed into the tree itself, thus allowing us to set the node feature
    # value thresholds within the tree itself.
    if est.max_bins is not None:
        X = est._bin_data(X, is_training_data=False).astype(DTYPE)

    # Assign chunk of trees to jobs
    n_jobs, _, _ = _partition_estimators(est.n_estimators, est.n_jobs)

    # avoid storing the output of every estimator by summing them here
    lock = threading.Lock()
    # accumulate the predictions across all trees
    all_proba = np.full(
        (len(est.estimators_), X.shape[0], est.n_classes_), np.nan, dtype=np.float64
    )
    Parallel(n_jobs=n_jobs, verbose=verbose, require="sharedmem")(
        delayed(_parallel_predict_proba_oob)(e.predict_proba, X, all_proba, idx, test_idx, lock)
        for idx, (e, test_idx) in enumerate(zip(est.estimators_, est.oob_samples_))
    )
    return all_proba
