import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_almost_equal, assert_array_equal
from sklearn import datasets
from sklearn.metrics import accuracy_score, r2_score, roc_auc_score
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeClassifier as skDecisionTreeClassifier
from sklearn.utils import check_random_state
from sklearn.utils.estimator_checks import parametrize_with_checks

from sktree._lib.sklearn.tree import DecisionTreeClassifier
from sktree.datasets import make_quadratic_classification
from sktree.ensemble import HonestForestClassifier
from sktree.stats.utils import _mutual_information
from sktree.tree import (
    MultiViewDecisionTreeClassifier,
    ObliqueDecisionTreeClassifier,
    PatchObliqueDecisionTreeClassifier,
)

CLF_CRITERIONS = ("gini", "entropy")

# also load the iris dataset
# and randomly permute it
iris = datasets.load_iris()
rng = np.random.RandomState(1)
perm = rng.permutation(iris.target.size)
iris.data = iris.data[perm]
iris.target = iris.target[perm]

# Larger classification sample used for testing feature importances
X_large, y_large = datasets.make_classification(
    n_samples=500,
    n_features=10,
    n_informative=3,
    n_redundant=0,
    n_repeated=0,
    shuffle=False,
    random_state=0,
)


def test_toy_accuracy():
    clf = HonestForestClassifier(n_estimators=10)
    X = np.ones((20, 4))
    X[10:] *= -1
    y = [0] * 10 + [1] * 10
    clf = clf.fit(X, y)
    np.testing.assert_array_equal(clf.predict(X), y)


@pytest.mark.parametrize("criterion", ["gini", "entropy"])
@pytest.mark.parametrize("max_features", [None, 2])
@pytest.mark.parametrize("honest_prior", ["empirical", "uniform", "ignore", "error"])
@pytest.mark.parametrize(
    "estimator",
    [
        None,
        DecisionTreeClassifier(),
        ObliqueDecisionTreeClassifier(),
        PatchObliqueDecisionTreeClassifier(),
    ],
)
def test_iris(criterion, max_features, honest_prior, estimator):
    # Check consistency on dataset iris.
    clf = HonestForestClassifier(
        criterion=criterion,
        random_state=0,
        max_features=max_features,
        n_estimators=10,
        honest_prior=honest_prior,
        tree_estimator=estimator,
    )
    if honest_prior == "error":
        with pytest.raises(ValueError, match="honest_prior error not a valid input."):
            clf.fit(iris.data, iris.target)
    else:
        clf.fit(iris.data, iris.target)
        score = accuracy_score(clf.predict(iris.data), iris.target)

        assert (
            score > 0.5 and score < 1.0
        ), "Failed with {0}, criterion = {1} and score = {2}".format("HForest", criterion, score)

        score = accuracy_score(clf.predict(iris.data), clf.predict_proba(iris.data).argmax(1))
        assert score == 1.0, "Failed with {0}, criterion = {1} and score = {2}".format(
            "HForest", criterion, score
        )


@pytest.mark.parametrize("criterion", ["gini", "entropy"])
@pytest.mark.parametrize("max_features", [None, 2])
@pytest.mark.parametrize("honest_prior", ["empirical", "uniform", "ignore", "error"])
@pytest.mark.parametrize(
    "estimator",
    [
        DecisionTreeClassifier(),
        ObliqueDecisionTreeClassifier(),
        PatchObliqueDecisionTreeClassifier(),
    ],
)
def test_iris_multi(criterion, max_features, honest_prior, estimator):
    n_estimators = 10

    # Check consistency on dataset iris.
    clf = HonestForestClassifier(
        criterion=criterion,
        random_state=0,
        max_features=max_features,
        n_estimators=n_estimators,
        honest_prior=honest_prior,
        tree_estimator=estimator,
    )

    second_y = np.concatenate([(np.ones(50) * 3), (np.ones(50) * 4), (np.ones(50) * 5)])

    X = iris.data
    y = np.stack((iris.target, second_y[perm])).T
    if honest_prior == "error":
        with pytest.raises(ValueError, match="honest_prior error not a valid input."):
            clf.fit(X, y)
    else:
        clf.fit(X, y)
        score = r2_score(clf.predict(X), y)
        if honest_prior == "ignore":
            assert (
                score > 0.4 and score < 1.0
            ), "Failed with {0}, criterion = {1} and score = {2}".format(
                "HForest", criterion, score
            )
        else:
            assert (
                score > 0.9 and score < 1.0
            ), "Failed with {0}, criterion = {1} and score = {2}".format(
                "HForest", criterion, score
            )


def test_max_samples():
    """Test different max_samples argument for HonestForestClassifier."""
    max_samples_list = [8, 0.5, None]
    depths = []
    X = rng.normal(0, 1, (100, 2))
    X[:50] *= -1
    y = [0, 1] * 50
    for ms in max_samples_list:
        uf = HonestForestClassifier(
            n_estimators=2, random_state=0, max_samples=ms, bootstrap=True, n_jobs=-1
        )
        uf = uf.fit(X, y)
        depths.append(uf.estimators_[0].get_depth())

    assert all(np.diff(depths) > 0), np.diff(depths)

    # Should work for a float greater than 1
    uf = HonestForestClassifier(
        n_estimators=2, random_state=0, max_samples=1.6, bootstrap=True, n_jobs=-1
    )
    uf = uf.fit(X, y)


@pytest.mark.parametrize("max_samples", [0.75, 1.0])
def test_honest_forest_has_deterministic_sampling_for_oob_structure_and_leaves(max_samples):
    """Test that honest forest can produce the oob, structure and leaf-node samples.

    When bootstrap is True, oob should be exclusive from structure and leaf-node samples.
    When bootstrap is False, there is no oob.
    """
    bootstrap = True

    n_estimators = 5
    est = HonestForestClassifier(
        n_estimators=n_estimators,
        random_state=0,
        bootstrap=bootstrap,
        max_samples=max_samples,
    )
    X = rng.normal(0, 1, (100, 2))
    X[:50] *= -1
    y = [0, 1] * 50
    samples = np.arange(len(y))

    est.fit(X, y)

    inbag_samples = est.estimators_samples_
    oob_samples = [
        [idx for idx in samples if idx not in inbag_samples[jdx]] for jdx in range(n_estimators)
    ]
    structure_samples = est.structure_indices_
    leaf_samples = est.honest_indices_
    if not bootstrap and max_samples == 1.0:
        assert all(oob_list_ == [] for oob_list_ in oob_samples)

        with pytest.raises(RuntimeError, match="Cannot extract out-of-bag samples"):
            est.oob_samples_
    else:
        oob_samples_ = est.oob_samples_
        for itree in range(n_estimators):
            assert len(oob_samples[itree]) > 1, oob_samples[itree]
            assert (
                set(structure_samples[itree])
                .union(set(leaf_samples[itree]))
                .intersection(set(oob_samples_[itree]))
                == set()
            )

            assert set(structure_samples[itree]).union(set(leaf_samples[itree])) == set(
                inbag_samples[itree]
            )
            assert set(inbag_samples[itree]).intersection(set(oob_samples_[itree])) == set()
            assert_array_equal(oob_samples_[itree], oob_samples[itree])


@pytest.mark.parametrize(
    "honest_prior, val",
    [
        ("uniform", 0.5),
        ("empirical", 0.75),
        ("ignore", np.nan),
    ],
)
def test_impute_posteriors(honest_prior, val):
    X = rng.normal(0, 1, (100, 2))
    y = [0] * 75 + [1] * 25
    clf = HonestForestClassifier(
        honest_fraction=0.02, random_state=0, honest_prior=honest_prior, n_estimators=2
    )
    clf = clf.fit(X, y)

    y_proba = clf.predict_proba(X)
    if np.isnan(val):
        assert (
            len(np.where(np.isnan(y_proba[:, 0]))[0]) > 50
        ), f"Failed with {honest_prior}, prior {clf.estimators_[0].empirical_prior_}"
    else:
        assert (
            len(np.where(y_proba[:, 0] == val)[0]) > 50
        ), f"Failed with {honest_prior}, prior {clf.estimators_[0].empirical_prior_}"


@pytest.mark.parametrize(
    "honest_fraction, val",
    [
        (0.8, 0.5),
        (0.02, np.nan),
    ],
)
def test_honest_decision_function(honest_fraction, val):
    X = rng.normal(0, 1, (100, 2))
    y = [0] * 75 + [1] * 25
    clf = HonestForestClassifier(honest_fraction=honest_fraction, random_state=0, n_estimators=2)
    clf = clf.fit(X, y)

    y_proba = clf.honest_decision_function_
    if np.isnan(val):
        assert len(np.where(np.isnan(y_proba[:, 0]))[0]) > 50, f"Failed with {honest_fraction}"
    else:
        assert len(np.where(y_proba[:, 1] < val)[0]) > 50, f"Failed with {honest_fraction}"


@parametrize_with_checks(
    [HonestForestClassifier(n_estimators=10, honest_fraction=0.5, random_state=0)]
)
def test_sklearn_compatible_estimator(estimator, check):
    # 1. check_class_weight_classifiers is not supported since it requires sample weight
    # XXX: can include this "generalization" in the future if it's useful
    #  zero sample weight is not "really supported" in honest subsample trees since sample weight
    #  for fitting the tree's splits
    if check.func.__name__ in [
        "check_class_weight_classifiers",
        # TODO: this is an error. Somehow a segfault is raised when fit is called first and
        # then partial_fit
        # "check_fit_score_takes_y",
    ]:
        pytest.skip()
    check(estimator)


@pytest.mark.parametrize("dtype", (np.float64, np.float32))
@pytest.mark.parametrize("criterion", ["gini", "log_loss"])
def test_importances(dtype, criterion):
    """Ported from sklearn unit-test.

    Used to ensure that honest forest feature importances are consistent with sklearn's.
    """
    tolerance = 0.01

    # cast as dtype
    X = X_large.astype(dtype, copy=False)
    y = y_large.astype(dtype, copy=False)

    ForestEstimator = HonestForestClassifier

    est = ForestEstimator(n_estimators=10, criterion=criterion, random_state=0)
    est.fit(X, y)

    importances = est.feature_importances_

    # The forest estimator can detect that only the first 3 features of the
    # dataset are informative:
    n_important = np.sum(importances > 0.1)
    assert importances.shape[0] == 10
    assert n_important == 3
    assert np.all(importances[:3] > 0.1)

    # Check with parallel
    importances = est.feature_importances_
    est.set_params(n_jobs=2)
    importances_parallel = est.feature_importances_
    assert_array_almost_equal(importances, importances_parallel)

    # Check with sample weights
    sample_weight = check_random_state(0).randint(1, 10, len(X))
    est = ForestEstimator(n_estimators=10, random_state=0, criterion=criterion)
    est.fit(X, y, sample_weight=sample_weight)
    importances = est.feature_importances_
    assert np.all(importances >= 0.0)

    for scale in [0.5, 100]:
        est = ForestEstimator(n_estimators=10, random_state=0, criterion=criterion)
        est.fit(X, y, sample_weight=scale * sample_weight)
        importances_bis = est.feature_importances_
        assert np.abs(importances - importances_bis).mean() < tolerance


def test_honest_forest_with_sklearn_trees():
    """Test against regression in power-curves discussed in:
    https://github.com/neurodata/scikit-tree/pull/157."""

    # generate the high-dimensional quadratic data
    X, y = make_quadratic_classification(256, 2048, noise=True, seed=0)
    y = y.squeeze()
    print(X.shape, y.shape)
    print(np.sum(y) / len(y))

    clf = HonestForestClassifier(
        n_estimators=10, tree_estimator=skDecisionTreeClassifier(), random_state=0
    )
    honestsk_scores = cross_val_score(clf, X, y, cv=5)
    print(honestsk_scores)

    clf = HonestForestClassifier(
        n_estimators=10, tree_estimator=DecisionTreeClassifier(), random_state=0
    )
    honest_scores = cross_val_score(clf, X, y, cv=5)
    print(honest_scores)

    # XXX: surprisingly, when we use the default which uses the fork DecisionTree,
    # we get different results
    # clf = HonestForestClassifier(n_estimators=10, random_state=0)
    # honest_scores = cross_val_score(clf, X, y, cv=5)
    # print(honest_scores)

    print(honestsk_scores, honest_scores)
    print(np.mean(honestsk_scores), np.mean(honest_scores))
    assert_allclose(np.mean(honestsk_scores), np.mean(honest_scores))


def test_honest_forest_with_sklearn_trees_with_auc():
    """Test against regression in power-curves discussed in:
    https://github.com/neurodata/scikit-tree/pull/157.

    This unit-test tests the equivalent of the AUC using sklearn's DTC
    vs our forked version of sklearn's DTC as the base tree.
    """
    skForest = HonestForestClassifier(
        n_estimators=10, tree_estimator=skDecisionTreeClassifier(), random_state=0
    )

    Forest = HonestForestClassifier(
        n_estimators=10, tree_estimator=DecisionTreeClassifier(), random_state=0
    )

    max_fpr = 0.1
    scores = []
    sk_scores = []
    for idx in range(10):
        X, y = make_quadratic_classification(256, 2048, noise=True, seed=idx)
        y = y.squeeze()

        skForest.fit(X, y)
        Forest.fit(X, y)

        # compute MI
        y_pred_proba = skForest.predict_proba(X)[:, 1].reshape(-1, 1)
        sk_mi = roc_auc_score(y, y_pred_proba, max_fpr=max_fpr)

        y_pred_proba = Forest.predict_proba(X)[:, 1].reshape(-1, 1)
        mi = roc_auc_score(y, y_pred_proba, max_fpr=max_fpr)

        scores.append(mi)
        sk_scores.append(sk_mi)

    print(scores, sk_scores)
    print(np.mean(scores), np.mean(sk_scores))
    print(np.std(scores), np.std(sk_scores))
    assert_allclose(np.mean(sk_scores), np.mean(scores), atol=0.005)


def test_honest_forest_with_sklearn_trees_with_mi():
    """Test against regression in power-curves discussed in:
    https://github.com/neurodata/scikit-tree/pull/157.

    This unit-test tests the equivalent of the MI using sklearn's DTC
    vs our forked version of sklearn's DTC as the base tree.
    """
    skForest = HonestForestClassifier(
        n_estimators=10, tree_estimator=skDecisionTreeClassifier(), random_state=0
    )

    Forest = HonestForestClassifier(
        n_estimators=10, tree_estimator=DecisionTreeClassifier(), random_state=0
    )

    scores = []
    sk_scores = []
    for idx in range(10):
        X, y = make_quadratic_classification(256, 2048, noise=True, seed=idx)
        y = y.squeeze()

        skForest.fit(X, y)
        Forest.fit(X, y)

        # compute MI
        sk_posterior = skForest.predict_proba(X)
        sk_score = _mutual_information(y, sk_posterior)

        posterior = Forest.predict_proba(X)
        score = _mutual_information(y, posterior)

        scores.append(score)
        sk_scores.append(sk_score)

    print(scores, sk_scores)
    print(np.mean(scores), np.mean(sk_scores))
    print(np.std(scores), np.std(sk_scores))
    assert_allclose(np.mean(sk_scores), np.mean(scores), atol=0.005)


@pytest.mark.parametrize(
    "tree, tree_kwargs",
    [
        (MultiViewDecisionTreeClassifier(), {"feature_set_ends": [10, 20]}),
        (ObliqueDecisionTreeClassifier(), {"feature_combinations": 2}),
        (PatchObliqueDecisionTreeClassifier(), {"max_patch_dims": 5}),
    ],
)
def test_honest_forest_with_tree_estimator_params(tree, tree_kwargs):
    """Test that honest forest inherits all the fitted parameters of the tree estimator."""
    X = np.ones((20, 4))
    X[10:] *= -1
    y = [0] * 10 + [1] * 10

    # test with a parameter that is a repeat of an init parameter
    clf = HonestForestClassifier(
        tree_estimator=DecisionTreeClassifier(), random_state=0, **tree_kwargs
    )
    with pytest.raises(ValueError, match=r"Invalid parameter\(s\)"):
        clf.fit(X, y)

    # test with a parameter that is not in any init signature
    clf = HonestForestClassifier(
        tree_estimator=tree,
        random_state=0,
        blah=0,
    )
    with pytest.raises(ValueError, match=r"Invalid parameter\(s\)"):
        clf.fit(X, y)

    # passing in a valid argument to the tree_estimator should work
    clf = HonestForestClassifier(tree_estimator=tree, random_state=0, **tree_kwargs)
    clf.fit(X, y)
    checked_attrs = [
        "classes_",
        "n_classes_",
        "n_features_in_",
        "n_outputs_",
    ]
    checked_attrs + getattr(tree, "_inheritable_fitted_attribute", [])
    for attr_name in checked_attrs:
        if not attr_name.startswith("_") and attr_name.endswith("_"):
            if isinstance(getattr(clf, attr_name), np.ndarray):
                np.testing.assert_array_equal(
                    getattr(clf, attr_name), getattr(clf.estimators_[0], attr_name)
                )
            else:
                assert getattr(clf, attr_name) == getattr(clf.estimators_[0], attr_name)
