import numpy as np
import pytest
from sklearn import datasets
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeClassifier as skDecisionTreeClassifier
from sklearn.utils.estimator_checks import parametrize_with_checks

from treeple._lib.sklearn.tree import DecisionTreeClassifier
from treeple.tree import (
    HonestTreeClassifier,
    MultiViewDecisionTreeClassifier,
    ObliqueDecisionTreeClassifier,
    PatchObliqueDecisionTreeClassifier,
)

# also load the iris dataset
# and randomly permute it
iris = datasets.load_iris()
rng = np.random.RandomState(1)
perm = rng.permutation(iris.target.size)
iris.data = iris.data[perm]
iris.target = iris.target[perm]


@pytest.mark.parametrize("criterion", ["gini", "entropy"])
@pytest.mark.parametrize("max_features", [None, 2])
@pytest.mark.parametrize(
    "estimator",
    [
        DecisionTreeClassifier(),
        ObliqueDecisionTreeClassifier(),
        PatchObliqueDecisionTreeClassifier(),
    ],
)
def test_iris(criterion, max_features, estimator):
    # Check consistency on dataset iris.
    clf = HonestTreeClassifier(
        criterion=criterion,
        random_state=0,
        max_features=max_features,
        tree_estimator=estimator,
    )
    clf.fit(iris.data, iris.target)
    score = accuracy_score(clf.predict(iris.data), iris.target)
    assert score > 0.5 and score < 1.0, "Failed with {0}, criterion = {1} and score = {2}".format(
        "HonestTree", criterion, score
    )

    score = accuracy_score(clf.predict(iris.data), clf.predict_proba(iris.data).argmax(1))
    assert score == 1.0, "Failed with {0}, criterion = {1} and score = {2}".format(
        "HonestTree", criterion, score
    )

    print(clf.honest_indices_)
    assert len(clf.honest_indices_) < len(iris.target)
    assert len(clf.structure_indices_) < len(iris.target)


@pytest.mark.parametrize("honest_method", ["apply", "prune"])
def test_toy_accuracy(honest_method):
    clf = HonestTreeClassifier(honest_method=honest_method)
    X = np.ones((20, 4))
    X[10:] *= -1
    y = [0] * 10 + [1] * 10
    clf = clf.fit(X, y)
    np.testing.assert_array_equal(clf.predict(X), y)


@pytest.mark.parametrize(
    "tree, tree_kwargs",
    [
        (MultiViewDecisionTreeClassifier(), {"feature_set_ends": [10, 20]}),
        (ObliqueDecisionTreeClassifier(), {"feature_combinations": 2}),
        (PatchObliqueDecisionTreeClassifier(), {"max_patch_dims": 5}),
    ],
)
def test_honest_tree_with_tree_estimator_params(tree, tree_kwargs):
    """Test that honest tree inherits all the fitted parameters of the tree estimator."""
    X = np.ones((20, 4))
    X[10:] *= -1
    y = [0] * 10 + [1] * 10

    # test with a parameter that is a repeat of an init parameter
    clf = HonestTreeClassifier(
        tree_estimator=DecisionTreeClassifier(), random_state=0, **tree_kwargs
    )
    with pytest.raises(ValueError, match=r"Invalid parameter\(s\)"):
        clf.fit(X, y)

    # test with a parameter that is not in any init signature
    clf = HonestTreeClassifier(
        tree_estimator=tree,
        random_state=0,
        blah=0,
    )
    with pytest.raises(ValueError, match=r"Invalid parameter\(s\)"):
        clf.fit(X, y)

    # passing in a valid argument to the tree_estimator should work
    clf = HonestTreeClassifier(tree_estimator=tree, random_state=0, **tree_kwargs)
    clf.fit(X, y)
    for attr_name in dir(clf.estimator_):
        if not attr_name.startswith("_") and attr_name.endswith("_"):
            if isinstance(getattr(clf, attr_name), np.ndarray):
                np.testing.assert_array_equal(
                    getattr(clf, attr_name), getattr(clf.estimator_, attr_name)
                )
            else:
                assert getattr(clf, attr_name) == getattr(clf.estimator_, attr_name)


@pytest.mark.parametrize(
    "honest_prior, val",
    [
        ("uniform", 0.5),
        ("empirical", 0.75),
        ("ignore", np.nan),
    ],
)
def test_impute_posteriors(honest_prior, val):
    # XXX: test could be improved as we don't actually test proper behavior of the imputation
    np.random.seed(0)
    X = np.random.normal(0, 1, (100, 2))
    y = [0] * 75 + [1] * 25
    clf = HonestTreeClassifier(honest_fraction=0.02, random_state=0, honest_prior=honest_prior)
    clf = clf.fit(X, y)

    y_proba = clf.predict_proba(X)
    if np.isnan(val):
        assert (
            len(np.where(np.isnan(y_proba[:, 0]))[0]) > 50
        ), f"Failed with {honest_prior}, {len(np.where(np.isnan(y_proba[:, 0]))[0])}"
        # assert len(np.where(np.isnan(y_proba[:, 0]))[0]) < 60, f"Failed with {honest_prior}, {len(np.where(np.isnan(y_proba[:, 0]))[0])}"
    else:
        assert (
            len(np.where(y_proba[:, 0] == val)[0]) > 50
        ), f"Failed with {honest_prior}, {len(np.where(y_proba[:, 0] == val)[0])}"


def test_increasing_leaves():
    np.random.seed(0)
    X = np.random.normal(0, 1, (200, 2))
    y = [0] * 100 + [1] * 100

    n_leaves = []
    for hf in [0.9, 0.7, 0.4, 0.2]:
        clf = HonestTreeClassifier(honest_fraction=hf, random_state=0)
        clf = clf.fit(X, y)
        n_leaves.append(clf.get_n_leaves())

    assert np.all(np.diff(n_leaves) > 0)


def test_impute_classes():
    np.random.seed(0)
    X = np.random.normal(0, 1, (101, 2))
    y = [2] * 50 + [1] * 50 + [0]
    clf = HonestTreeClassifier(honest_fraction=0.02, random_state=0)
    clf = clf.fit(X, y)

    y_proba = clf.predict_proba(X)

    assert y_proba.shape[1] == 3


@parametrize_with_checks(
    [
        HonestTreeClassifier(random_state=0),
        HonestTreeClassifier(random_state=0, tree_estimator=DecisionTreeClassifier()),
    ]
)
def test_sklearn_compatible_estimator(estimator, check):
    # 1. check_class_weight_classifiers is not supported since it requires sample weight
    # XXX: can include this "generalization" in the future if it's useful
    #  zero sample weight is not "really supported" in honest subsample trees since sample weight
    #  for fitting the tree's splits
    if check.func.__name__ in ["check_class_weight_classifiers", "check_classifier_multioutput"]:
        pytest.skip()
    check(estimator)


def test_with_sklearn_trees():
    X = np.ones((20, 4))
    X[10:] *= -1
    y = [0] * 10 + [1] * 10

    # with pytest.raises(RuntimeError, match="Instead of using sklearn.tree"):
    clf = HonestTreeClassifier(tree_estimator=skDecisionTreeClassifier())
    clf.fit(X, y)


@pytest.mark.skip()
def test_sklearn_tree_regression():
    """Test against regression in power-curves discussed in:"""

    def quadratic(n, p, noise=False, seed=None):
        rng = np.random.default_rng(seed)

        x = rng.standard_normal(size=(n, p))
        coeffs = np.array([np.exp(-0.0325 * (i + 24)) for i in range(p)])
        eps = rng.standard_normal(size=(n, p))

        x_coeffs = x * coeffs
        y = x_coeffs**2 + noise * eps

        n1 = x.shape[0]
        n2 = y.shape[0]
        v = np.vstack([np.zeros((n1, 1)), np.ones((n2, 1))])
        x = np.vstack((x, y))
        return x, v

    # generate the high-dimensional quadratic data
    X, y = quadratic(1024, 4096, noise=False, seed=0)
    print(X.shape, y.shape)
    print(np.sum(y) / len(y))
    assert False
    clf = HonestTreeClassifier(tree_estimator=skDecisionTreeClassifier(), random_state=0)
    honestsk_scores = cross_val_score(clf, X, y, cv=5)
    print(honestsk_scores)

    clf = HonestTreeClassifier(tree_estimator=DecisionTreeClassifier(), random_state=0)
    honest_scores = cross_val_score(clf, X, y, cv=5)
    print(honest_scores)

    clf = HonestTreeClassifier(random_state=0)
    honest_scores = cross_val_score(clf, X, y, cv=5)
    print(honest_scores)

    skest = skDecisionTreeClassifier(random_state=0)
    sk_scores = cross_val_score(skest, X, y, cv=5)

    est = DecisionTreeClassifier(random_state=0)
    scores = cross_val_score(est, X, y, cv=5)

    print(sk_scores, scores)
    print(np.mean(sk_scores), np.mean(scores))
    assert np.mean(sk_scores) == np.mean(scores)
    assert False
