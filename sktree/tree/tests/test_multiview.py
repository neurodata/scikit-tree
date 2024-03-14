import math

import numpy as np
import pytest
from numpy.testing import assert_array_equal
from sklearn.datasets import make_blobs
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score
from sklearn.utils.estimator_checks import parametrize_with_checks

from sktree.tree import DecisionTreeClassifier, MultiViewDecisionTreeClassifier

seed = 12345


@parametrize_with_checks(
    [
        MultiViewDecisionTreeClassifier(random_state=12),
    ]
)
def test_sklearn_compatible_estimator(estimator, check):
    check(estimator)


@pytest.mark.parametrize("baseline_est", [MultiViewDecisionTreeClassifier, DecisionTreeClassifier])
def test_multiview_classification(baseline_est):
    """Test that explicit knowledge of multi-view structure improves classification accuracy.

    In very high-dimensional noise setting across two views, when the max_depth and max_features
    are constrained, the multi-view decision tree will still obtain perfect accuracy, while
    the single-view decision tree will not.
    """
    rng = np.random.default_rng(seed=seed)

    n_samples = 20
    n_features_1 = 5
    n_features_2 = 1000
    cluster_std = 5.0

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
    clf = MultiViewDecisionTreeClassifier(
        random_state=seed,
        feature_set_ends=[n_features_1, X.shape[1]],
        max_features=0.3,
    )
    clf.fit(X, y)
    assert (
        accuracy_score(y, clf.predict(X)) == 1.0
    ), f"Accuracy score: {accuracy_score(y, clf.predict(X))}"
    assert (
        cross_val_score(clf, X, y, cv=5).mean() > 0.9
    ), f"CV score: {cross_val_score(clf, X, y, cv=5).mean()}"

    base_clf = baseline_est(
        random_state=seed,
        max_features=0.3,
    )
    assert cross_val_score(base_clf, X, y, cv=5).mean() < cross_val_score(clf, X, y, cv=5).mean(), (
        f"CV score: {cross_val_score(base_clf, X, y, cv=5).mean()} vs "
        f"{cross_val_score(clf, X, y, cv=5).mean()}"
    )


def test_multiview_errors():
    """Test that an error is raised when max_features is greater than the number of features."""
    X = np.random.random((10, 5))
    y = np.random.randint(0, 2, size=10)

    clf = MultiViewDecisionTreeClassifier(
        random_state=seed,
        feature_set_ends=[3, 10],
        max_features=2,
    )
    with pytest.raises(ValueError, match="The last feature set end must be equal"):
        clf.fit(X, y)

    # Test that an error is raised when max_features is greater than the number of features.
    clf = MultiViewDecisionTreeClassifier(
        random_state=seed,
        feature_set_ends=[3, 5],
        max_features=6,
    )
    with pytest.raises(ValueError, match="the number of features in feature set"):
        clf.fit(X, y)


def test_multiview_separate_feature_set_sampling_sets_attributes():
    X = np.random.random((20, 10))
    y = np.random.randint(0, 2, size=20)

    # test with max_features as a float
    clf = MultiViewDecisionTreeClassifier(
        random_state=seed,
        feature_set_ends=[6, 10],
        max_features=0.5,
    )
    clf.fit(X, y)

    assert_array_equal(clf.max_features_per_set_, [3, 2])
    assert clf.max_features_ == 5

    # test with max_features as sqrt
    X = np.random.random((20, 13))
    clf = MultiViewDecisionTreeClassifier(
        random_state=seed,
        feature_set_ends=[9, 13],
        max_features="sqrt",
    )
    clf.fit(X, y)
    assert_array_equal(clf.max_features_per_set_, [3, 2])
    assert clf.max_features_ == 5

    # test with max_features as 'sqrt' but not a perfect square
    X = np.random.random((20, 9))
    clf = MultiViewDecisionTreeClassifier(
        random_state=seed,
        feature_set_ends=[5, 9],
        max_features="sqrt",
    )
    clf.fit(X, y)
    assert_array_equal(clf.max_features_per_set_, [3, 2])
    assert clf.max_features_ == 5


def test_at_least_one_feature_per_view_is_sampled():
    """Test that multiview decision tree always samples at least one feature per view."""

    X = np.random.random((20, 10))
    y = np.random.randint(0, 2, size=20)

    # test with max_features as a float
    clf = MultiViewDecisionTreeClassifier(
        random_state=seed,
        feature_set_ends=[1, 2, 4, 10],
        max_features=0.4,
    )
    clf.fit(X, y)

    assert_array_equal(clf.max_features_per_set_, [1, 1, 1, np.ceil(6 * clf.max_features)])
    assert clf.max_features_ == np.sum(clf.max_features_per_set_), np.sum(clf.max_features_per_set_)


def test_multiview_separate_feature_set_sampling_is_consistent():
    rng = np.random.default_rng(seed)
    X = rng.standard_normal(size=(20, 10))
    y = rng.integers(0, 2, size=20)

    # test with max_features as an array but apply_max_features is off
    clf = MultiViewDecisionTreeClassifier(
        random_state=seed,
        feature_set_ends=[1, 3, 6, 10],
        max_features=[1, 2, 2, 3],
    )
    clf.fit(X, y)

    assert_array_equal(clf.max_features_per_set_, [1, 2, 2, 3])
    assert clf.max_features_ == np.sum(clf.max_features_per_set_), np.sum(clf.max_features_per_set_)
    assert_array_equal(clf.max_features_per_set_, [1, 2, 2, 3])
    assert clf.max_features_ == np.sum(clf.max_features_per_set_), np.sum(clf.max_features_per_set_)

    # test with max_features as an array but apply_max_features is off
    other_clf = MultiViewDecisionTreeClassifier(
        random_state=seed,
        feature_set_ends=[1, 3, 6, 10],
        max_features=[1, 2, 2, 3],
    )
    other_clf.fit(X, y)

    assert_array_equal(other_clf.tree_.value, clf.tree_.value)


def test_separate_mtry_per_feature_set():
    """Test that multiview decision tree can sample different numbers of features per view.

    Sets the ``max_feature`` argument as an array-like.
    """
    X = np.random.random((20, 10))
    y = np.random.randint(0, 2, size=20)

    # test with max_features as an array
    clf = MultiViewDecisionTreeClassifier(
        random_state=seed,
        feature_set_ends=[1, 2, 4, 10],
        max_features=[0.4, 0.5, 0.6, 0.7],
    )
    clf.fit(X, y)

    assert_array_equal(clf.max_features_per_set_, [1, 1, 2, math.ceil(6 * 0.7)])
    assert clf.max_features_ == np.sum(clf.max_features_per_set_), np.sum(clf.max_features_per_set_)

    # test with max_features as an array
    clf = MultiViewDecisionTreeClassifier(
        random_state=seed,
        feature_set_ends=[1, 2, 4, 10],
        max_features=[1, 1, 1, 1.0],
    )
    clf.fit(X, y)
    assert_array_equal(clf.max_features_per_set_, [1, 1, 1, 6])
    assert clf.max_features_ == np.sum(clf.max_features_per_set_), np.sum(clf.max_features_per_set_)

    # test with max_features as 1.0
    clf = MultiViewDecisionTreeClassifier(
        random_state=seed,
        feature_set_ends=[1, 2, 4, 10],
        max_features=1.0,
    )
    clf.fit(X, y)
    assert_array_equal(clf.max_features_per_set_, [1, 1, 2, 6])
    assert clf.max_features_ == 10, np.sum(clf.max_features_per_set_)


def test_multiview_without_feature_view_stratification():
    """Test that multiview decision tree can sample different numbers of features per view.

    Sets the ``max_feature`` argument as an array-like.
    """
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((20, 497))
    X = np.hstack((X, rng.standard_normal((20, 3))))
    y = rng.integers(0, 2, size=20)

    # test with max_features as a float
    clf = MultiViewDecisionTreeClassifier(
        random_state=seed,
        feature_set_ends=[497, 500],
        max_features=0.3,
    )
    clf.fit(X, y)

    assert_array_equal(clf.max_features_per_set_, [150, 1]), clf.max_features_per_set_
    assert clf.max_features_ == math.ceil(497.0 * clf.max_features) + math.ceil(
        3 * clf.max_features
    )
