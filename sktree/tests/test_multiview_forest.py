import numpy as np
import pytest
from numpy.testing import assert_array_equal
from sklearn.datasets import make_blobs
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.utils.estimator_checks import parametrize_with_checks

from sktree import MultiViewRandomForestClassifier, RandomForestClassifier
from sktree.datasets.multiview import make_joint_factor_model

seed = 12345


@parametrize_with_checks(
    [
        MultiViewRandomForestClassifier(random_state=12345, n_estimators=10),
    ]
)
def test_sklearn_compatible_estimator(estimator, check):
    check(estimator)


@pytest.mark.parametrize("baseline_est", [RandomForestClassifier])
def test_multiview_classification(baseline_est):
    """Test that explicit knowledge of multi-view structure improves classification accuracy.

    In very high-dimensional noise setting across two views, when the max_depth and max_features
    are constrained, the multi-view random forests will still obtain perfect accuracy, while
    the single-view random forest will not.
    """
    rng = np.random.default_rng(seed=seed)

    n_samples = 100
    n_estimators = 10
    n_features_1 = 5
    n_features_2 = 1000
    cluster_std = 2.0

    # Create a high-dimensional multiview dataset with a low-dimensional informative
    # subspace in one view of the dataset.
    X0_first, y0 = make_blobs(
        n_samples=n_samples,
        cluster_std=cluster_std,
        n_features=n_features_1,
        random_state=rng.integers(1, 10000),
        centers=1,
    )

    X1_first, y1 = make_blobs(
        n_samples=n_samples,
        cluster_std=cluster_std,
        n_features=n_features_1,
        random_state=rng.integers(1, 10000),
        centers=1,
    )
    y1[:] = 1
    X0 = np.concatenate([X0_first, rng.standard_normal(size=(n_samples, n_features_2))], axis=1)
    X1 = np.concatenate([X1_first, rng.standard_normal(size=(n_samples, n_features_2))], axis=1)
    X = np.vstack((X0, X1))
    y = np.hstack((y0, y1)).T

    # Compare multiview decision tree vs single-view decision tree
    clf = MultiViewRandomForestClassifier(
        random_state=seed,
        feature_set_ends=[n_features_1, X.shape[1]],
        max_features="sqrt",
        max_depth=4,
        n_estimators=n_estimators,
    )
    clf.fit(X, y)
    assert (
        accuracy_score(y, clf.predict(X)) == 1.0
    ), f"Accuracy score: {accuracy_score(y, clf.predict(X))}"
    assert (
        cross_val_score(clf, X, y, cv=5).mean() == 1.0
    ), f"CV score: {cross_val_score(clf, X, y, cv=5).mean()}"

    clf = baseline_est(
        random_state=seed,
        max_depth=4,
        max_features="sqrt",
        n_estimators=n_estimators,
    )
    clf.fit(X, y)
    assert (
        accuracy_score(y, clf.predict(X)) > 0.99
    ), f"Accuracy score: {accuracy_score(y, clf.predict(X))}"
    assert (
        cross_val_score(clf, X, y, cv=3).mean() <= 0.95
    ), f"CV score for {clf}: {cross_val_score(clf, X, y, cv=3).mean()}"


@pytest.mark.parametrize(
    "n_views, max_features",
    [
        (2, 0.3),
        (3, 0.3),
        (4, 0.3),
        (5, 0.3),
    ],
)
def test_three_view_dataset(n_views, max_features):
    rng = np.random.default_rng(seed=seed)

    n_estimators = 50

    n_samples = 100
    n_views = 5
    n_features = 200

    # number of dimensions that are informative
    n_signals = 5

    # Create a high-dimensional multiview dataset
    Xs = make_joint_factor_model(
        n_features=n_signals,
        m=2.0,
        n_views=n_views,
        n_samples=n_samples,
        joint_rank=5,
        random_state=seed + 1,
    )
    X = np.empty((n_samples, 0))
    feature_set_ends = []
    for _X in Xs:
        X = np.hstack((X, _X, rng.standard_normal(size=(n_samples, n_features - n_signals))))
        feature_set_ends.append(X.shape[1])

    Xs = make_joint_factor_model(
        n_features=n_signals,
        m=2.0,
        n_views=n_views,
        n_samples=n_samples,
        joint_rank=5,
        random_state=seed + 2,
    )
    second_X = np.empty((n_samples, 0))
    for _X in Xs:
        second_X = np.hstack(
            (second_X, _X, rng.standard_normal(size=(n_samples, n_features - n_signals)))
        )
    X = np.vstack((X, second_X))
    y = np.concatenate([np.zeros(n_samples), np.ones(n_samples)]).T

    # perform train/test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=seed)

    # Compare multiview decision tree vs single-view decision tree
    clf = MultiViewRandomForestClassifier(
        random_state=seed,
        feature_set_ends=feature_set_ends,
        max_features=max_features,
        n_estimators=n_estimators,
    )
    clf.fit(X_train, y_train)

    rf_clf = RandomForestClassifier(
        random_state=seed,
        max_features=max_features,
        n_estimators=n_estimators,
    )
    rf_clf.fit(X_train, y_train)
    print(n_features, max_features, n_features * max_features)
    assert_array_equal(
        clf.estimators_[0].max_features_per_set_, [n_features * max_features] * n_views
    )

    rf_acc = accuracy_score(y_test, rf_clf.predict(X_test))
    acc = accuracy_score(y_test, clf.predict(X_test))
    assert acc > rf_acc + 0.01, f"Accuracy score: {acc} < {rf_acc}"
