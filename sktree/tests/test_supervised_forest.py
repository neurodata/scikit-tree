from typing import Any, Dict

import numpy as np
import pytest
from sklearn.datasets import load_diabetes, make_classification, make_regression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.utils._testing import assert_array_almost_equal
from sklearn.utils.estimator_checks import parametrize_with_checks
from sklearn.utils.validation import check_random_state

from sktree import (
    ExtraObliqueRandomForestClassifier,
    ExtraObliqueRandomForestRegressor,
    ObliqueRandomForestClassifier,
    ObliqueRandomForestRegressor,
    PatchObliqueRandomForestClassifier,
    PatchObliqueRandomForestRegressor,
)

# Larger classification sample used for testing feature importances
X_large, y_large = make_classification(
    n_samples=500,
    n_features=10,
    n_informative=3,
    n_redundant=0,
    n_repeated=0,
    shuffle=False,
    random_state=0,
)

# Larger regression sample used for testing feature importances
X_large_reg, y_large_reg = make_regression(
    n_samples=500,
    n_features=10,
    n_informative=3,
    shuffle=False,
    random_state=0,
)

# load the diabetes dataset
# and randomly permute it
rng = np.random.RandomState(1)
diabetes = load_diabetes()
perm = rng.permutation(diabetes.target.size)
diabetes.data = diabetes.data[perm]
diabetes.target = diabetes.target[perm]


FOREST_CLASSIFIERS = {
    "ExtraObliqueRandomForestClassifier": ExtraObliqueRandomForestClassifier,
    "ObliqueRandomForestClassifier": ObliqueRandomForestClassifier,
    "PatchObliqueRandomForestClassifier": PatchObliqueRandomForestClassifier,
}

FOREST_REGRESSORS = {
    "ExtraObliqueDecisionTreeRegressor": ExtraObliqueRandomForestRegressor,
    "ObliqueRandomForestRegressor": ObliqueRandomForestRegressor,
    "PatchObliqueRandomForestRegressor": PatchObliqueRandomForestRegressor,
}

FOREST_ESTIMATORS: Dict[str, Any] = dict()
FOREST_ESTIMATORS.update(FOREST_CLASSIFIERS)
FOREST_ESTIMATORS.update(FOREST_REGRESSORS)

REG_CRITERIONS = ("squared_error", "absolute_error", "friedman_mse")


def _sparse_parity(n, p=20, p_star=3, random_state=None):
    """Generate sparse parity dataset.

    Sparse parity is a multivariate generalization of the
    XOR problem.

    Parameters
    ----------
    n : int
        Number of sample to generate.
    p : int, optional
        The dimensionality of the dataset, by default 20
    p_star : int, optional
        The number of informative dimensions, by default 3.
    random_state : Random State, optional
        Random state, by default None.

    Returns
    -------
    X : np.ndarray of shape (n, p)
        Sparse parity dataset as a dense array.
    y : np.ndarray of shape (n,)
        Labels of the dataset
    """
    rng = np.random.RandomState(seed=random_state)
    X = rng.uniform(-1, 1, (n, p))
    y = np.zeros(n)

    for i in range(0, n):
        y[i] = sum(X[i, :p_star] > 0) % 2

    return X, y


def _orthant(n, p=6, random_state=None):
    """Generate orthant dataset.

    Parameters
    ----------
    n : int
        Number of sample to generate.
    p : int, optional
        The dimensionality of the dataset and the number of
        unique labels, by default 6.
    rec : int, optional
        _description_, by default 1
    random_state : Random State, optional
        Random state, by default None.

    Returns
    -------
    X : np.ndarray of shape (n, p)
        Orthant dataset as a dense array.
    y : np.ndarray of shape (n,)
        Labels of the dataset
    """
    rng = np.random.RandomState(seed=random_state)
    orth_labels = np.asarray([2**i for i in range(0, p)][::-1])

    X = rng.uniform(-1, 1, (n, p))
    y = np.zeros(n)

    for i in range(0, n):
        idx = np.where(X[i, :] > 0)[0]
        y[i] = sum(orth_labels[idx])

    if len(np.unique(y)) < 2**p:
        raise RuntimeError("Increase sample size to get a label in each orthant.")

    return X, y


def _trunk(n, p=10, random_state=None):
    """Generate trunk dataset.

    Parameters
    ----------
    n : int
        Number of sample to generate.
    p : int, optional
        The dimensionality of the dataset and the number of
        unique labels, by default 10.
    random_state : Random State, optional
        Random state, by default None.

    Returns
    -------
    X : np.ndarray of shape (n, p)
        Trunk dataset as a dense array.
    y : np.ndarray of shape (n,)
        Labels of the dataset

    References
    ----------
    [1] Gerard V. Trunk. A problem of dimensionality: A
    simple example. IEEE Transactions on Pattern Analysis
    and Machine Intelligence, 1(3):306–307, 1979.
    """
    rng = np.random.RandomState(seed=random_state)

    mu_1 = np.array([1 / i for i in range(1, p + 1)])
    mu_0 = -1 * mu_1
    cov = np.identity(p)

    X = np.vstack(
        (
            rng.multivariate_normal(mu_0, cov, int(n / 2)),
            rng.multivariate_normal(mu_1, cov, int(n / 2)),
        )
    )
    y = np.concatenate((np.zeros(int(n / 2)), np.ones(int(n / 2))))
    return X, y


@parametrize_with_checks(
    [
        ExtraObliqueRandomForestClassifier(random_state=12345, n_estimators=10),
        ObliqueRandomForestClassifier(random_state=12345, n_estimators=10),
        PatchObliqueRandomForestClassifier(random_state=12345, n_estimators=10),
        ExtraObliqueRandomForestRegressor(random_state=12345, n_estimators=10),
        ObliqueRandomForestRegressor(random_state=12345, n_estimators=10),
        PatchObliqueRandomForestRegressor(random_state=12345, n_estimators=10),
    ]
)
def test_sklearn_compatible_estimator(estimator, check):
    # TODO: remove when we can replicate the CI error...
    # this seems to be due to a compiler issue since it is not replicable on MacOSx
    if isinstance(
        estimator,
        (
            ExtraObliqueRandomForestClassifier,
            ObliqueRandomForestClassifier,
            PatchObliqueRandomForestClassifier,
        ),
    ) and check.func.__name__ in ["check_fit_score_takes_y"]:
        pytest.skip()
    check(estimator)


def test_oblique_forest_sparse_parity():
    # Sparse parity dataset
    n = 500
    X, y = _sparse_parity(n, random_state=0)
    n_test = 0.2
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=n_test,
        random_state=0,
    )

    rc_clf = ObliqueRandomForestClassifier(random_state=0)
    rc_clf.fit(X_train, y_train)
    y_hat = rc_clf.predict(X_test)
    rc_accuracy = accuracy_score(y_test, y_hat)

    ri_clf = RandomForestClassifier(random_state=0)
    ri_clf.fit(X_train, y_train)
    y_hat = ri_clf.predict(X_test)
    ri_accuracy = accuracy_score(y_test, y_hat)

    assert (
        ri_accuracy < rc_accuracy
    ), f"Oblique forest: {rc_accuracy} < Axis-aligned forest: {ri_accuracy}"
    assert ri_accuracy > 0.45, f"Axis-aligned forest: {ri_accuracy} < 0.45"
    assert rc_accuracy > 0.5, f"Oblique forest: {rc_accuracy} < 0.5"


def test_oblique_forest_orthant():
    """Test oblique forests on orthant problem.

    It is expected that axis-aligned and oblique-aligned
    forests will perform similarly.
    """
    n = 500
    X, y = _orthant(n, p=6, random_state=0)
    n_test = 0.3
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=n_test,
        random_state=0,
    )

    rc_clf = ObliqueRandomForestClassifier(max_features=None, random_state=0)
    rc_clf.fit(X_train, y_train)
    y_hat = rc_clf.predict(X_test)
    rc_accuracy = accuracy_score(y_test, y_hat)

    ri_clf = RandomForestClassifier(max_features="sqrt", random_state=0)
    ri_clf.fit(X_train, y_train)
    y_hat = ri_clf.predict(X_test)
    ri_accuracy = accuracy_score(y_test, y_hat)

    assert (
        rc_accuracy >= ri_accuracy
    ), f"Oblique forest: {rc_accuracy} < Axis-aligned forest: {ri_accuracy}"
    assert ri_accuracy > 0.84
    assert rc_accuracy > 0.85


def test_oblique_forest_trunk():
    """Test oblique vs axis-aligned forests on Trunk."""
    n = 1000
    X, y = _trunk(n, p=100, random_state=0)
    n_test = 0.2
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=n_test,
        random_state=0,
    )

    rc_clf = ObliqueRandomForestClassifier(max_features=X.shape[1], random_state=0)
    rc_clf.fit(X_train, y_train)
    y_hat = rc_clf.predict(X_test)
    rc_accuracy = accuracy_score(y_test, y_hat)

    ri_clf = RandomForestClassifier(max_features="sqrt", random_state=0)
    ri_clf.fit(X_train, y_train)
    y_hat = ri_clf.predict(X_test)
    ri_accuracy = accuracy_score(y_test, y_hat)

    assert rc_accuracy > ri_accuracy
    assert ri_accuracy > 0.83
    assert rc_accuracy > 0.86


@pytest.mark.parametrize("dtype", (np.float64, np.float32))
@pytest.mark.parametrize(
    "estimator, criterion",
    (
        [ExtraObliqueRandomForestClassifier, "gini"],
        [ExtraObliqueRandomForestClassifier, "log_loss"],
        [ObliqueRandomForestClassifier, "gini"],
        [ObliqueRandomForestClassifier, "log_loss"],
        [ExtraObliqueRandomForestRegressor, "squared_error"],
        [ExtraObliqueRandomForestRegressor, "friedman_mse"],
        [ExtraObliqueRandomForestRegressor, "poisson"],
        [ObliqueRandomForestRegressor, "squared_error"],
        [ObliqueRandomForestRegressor, "friedman_mse"],
        [ObliqueRandomForestRegressor, "poisson"],
    ),
)
@pytest.mark.parametrize("feature_combinations", (2, 5))
def test_check_importances_oblique(estimator, criterion, dtype, feature_combinations):
    """Test checking feature importances for oblique trees."""
    tolerance = 0.01

    # cast as dype
    X = X_large.astype(dtype, copy=False)
    y = y_large.astype(dtype, copy=False)

    est = estimator(
        n_estimators=50,
        criterion=criterion,
        random_state=123,
        feature_combinations=feature_combinations,
    )
    est.fit(X, y)
    importances = est.feature_importances_

    # The forest estimator can detect that only the first 3 features of the
    # dataset are informative:
    n_important = np.sum(importances > 0.1)
    assert importances.shape[0] == 10
    if feature_combinations == 2:
        assert n_important == 3
    else:
        assert n_important >= 3
    assert np.all(importances[:3] > 0.1)

    # Check with parallel
    importances = est.feature_importances_
    est.set_params(n_jobs=2)
    importances_parallel = est.feature_importances_
    assert_array_almost_equal(importances, importances_parallel)

    # Check with sample weights
    sample_weight = check_random_state(0).randint(1, 10, len(X))
    est = estimator(
        n_estimators=10,
        random_state=0,
        criterion=criterion,
        feature_combinations=feature_combinations,
    )
    est.fit(X, y, sample_weight=sample_weight)
    importances = est.feature_importances_
    assert np.all(importances >= 0.0)

    for scale in [0.5, 100]:
        est = estimator(
            n_estimators=10,
            random_state=0,
            criterion=criterion,
            feature_combinations=feature_combinations,
        )
        est.fit(X, y, sample_weight=scale * sample_weight)
        importances_bis = est.feature_importances_
        assert np.abs(importances - importances_bis).mean() < tolerance


@pytest.mark.parametrize("dtype", (np.float64, np.float32))
@pytest.mark.parametrize(
    "estimator, criterion",
    (
        [PatchObliqueRandomForestClassifier, "gini"],
        [PatchObliqueRandomForestClassifier, "log_loss"],
        [PatchObliqueRandomForestRegressor, "squared_error"],
        [PatchObliqueRandomForestRegressor, "friedman_mse"],
        [PatchObliqueRandomForestRegressor, "poisson"],
    ),
)
def test_check_importances_patch(estimator, criterion, dtype):
    """Test checking feature importances for oblique trees."""
    tolerance = 0.01

    # cast as dype
    X = X_large.astype(dtype, copy=False)
    y = y_large.astype(dtype, copy=False)

    est = estimator(
        n_estimators=50,
        criterion=criterion,
        random_state=0,
        max_patch_dims=(2, 2),
        data_dims=(2, 5),
    )
    est.fit(X, y)
    importances = est.feature_importances_

    # The forest estimator can detect that only the first 3 features of the
    # dataset are informative:
    n_important = np.sum(importances > 0.1)
    assert importances.shape[0] == 10
    assert n_important >= 3

    # Check with parallel
    importances = est.feature_importances_
    est.set_params(n_jobs=2)
    importances_parallel = est.feature_importances_
    assert_array_almost_equal(importances, importances_parallel)

    # Check with sample weights
    sample_weight = check_random_state(0).randint(1, 10, len(X))
    est = estimator(
        n_estimators=10,
        random_state=0,
        criterion=criterion,
        max_patch_dims=(1, 5),
        data_dims=(2, 5),
    )
    est.fit(X, y, sample_weight=sample_weight)
    importances = est.feature_importances_
    assert np.all(importances >= 0.0)

    for scale in [0.5, 100]:
        est = estimator(
            n_estimators=10,
            random_state=0,
            criterion=criterion,
            max_patch_dims=(1, 5),
            data_dims=(2, 5),
        )
        est.fit(X, y, sample_weight=scale * sample_weight)
        importances_bis = est.feature_importances_
        assert np.abs(importances - importances_bis).mean() < tolerance


# Unit tests for ObliqueRandomForestRegressor
@pytest.mark.parametrize("forest", FOREST_REGRESSORS.values())
@pytest.mark.parametrize("criterion", REG_CRITERIONS)
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_regression(forest, criterion, dtype):
    estimator = forest(n_estimators=10, criterion=criterion, random_state=123)
    n_test = 0.1
    X = X_large_reg.astype(dtype, copy=False)
    y = y_large_reg.astype(dtype, copy=False)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=n_test, random_state=123)
    estimator.fit(X_train, y_train)
    assert estimator.score(X_test, y_test) > 0.88, f"Failed for {estimator} and {criterion}"
