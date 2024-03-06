import pickle
import sys

import numpy as np
import pytest
from numpy.testing import assert_allclose
from sklearn import datasets
from sklearn.base import is_classifier
from sklearn.metrics import accuracy_score, mean_poisson_deviance, mean_squared_error
from sklearn.model_selection import cross_val_score
from sklearn.utils._testing import skip_if_32bit
from sklearn.utils.estimator_checks import parametrize_with_checks

from sktree._lib.sklearn.tree import DecisionTreeClassifier
from sktree.tree import (
    ExtraObliqueDecisionTreeClassifier,
    ExtraObliqueDecisionTreeRegressor,
    ObliqueDecisionTreeClassifier,
    ObliqueDecisionTreeRegressor,
    PatchObliqueDecisionTreeClassifier,
    PatchObliqueDecisionTreeRegressor,
    UnsupervisedDecisionTree,
    UnsupervisedObliqueDecisionTree,
)

from .test_all_trees import assert_tree_equal

CLUSTER_CRITERIONS = ("twomeans", "fastbic")
REG_CRITERIONS = ("squared_error", "absolute_error", "friedman_mse", "poisson")
CLF_CRITERIONS = ("gini", "entropy")

TREE_CLUSTERS = {
    "UnsupervisedDecisionTree": UnsupervisedDecisionTree,
    "UnsupervisedObliqueDecisionTree": UnsupervisedObliqueDecisionTree,
}

REG_TREES = {
    "ExtraObliqueDecisionTreeRegressor": ExtraObliqueDecisionTreeRegressor,
    "ObliqueDecisionTreeRegressor": ObliqueDecisionTreeRegressor,
    "PatchObliqueDecisionTreeRegressor": PatchObliqueDecisionTreeRegressor,
}

CLF_TREES = {
    "ExtraObliqueDecisionTreeClassifier": ExtraObliqueDecisionTreeClassifier,
    "ObliqueDecisionTreeClassifier": ObliqueDecisionTreeClassifier,
    "PatchObliqueTreeClassifier": PatchObliqueDecisionTreeClassifier,
}

OBLIQUE_TREES = {
    "ExtraObliqueDecisionTreeClassifier": ExtraObliqueDecisionTreeClassifier,
    "ExtraObliqueDecisionTreeRegressor": ExtraObliqueDecisionTreeRegressor,
    "ObliqueDecisionTreeClassifier": ObliqueDecisionTreeClassifier,
    "ObliqueDecisionTreeRegressor": ObliqueDecisionTreeRegressor,
}

PATCH_OBLIQUE_TREES = {
    "PatchObliqueDecisionTreeClassifier": PatchObliqueDecisionTreeClassifier,
    "PatchObliqueDecisionTreeRegressor": PatchObliqueDecisionTreeRegressor,
}

ALL_TREES = {
    "ExtraObliqueDecisionTreeClassifier": ExtraObliqueDecisionTreeClassifier,
    "ExtraObliqueDecisionTreeRegressor": ExtraObliqueDecisionTreeRegressor,
    "ObliqueDecisionTreeClassifier": ObliqueDecisionTreeClassifier,
    "ObliqueDecisionTreeRegressor": ObliqueDecisionTreeRegressor,
    "PatchObliqueDecisionTreeClassifier": PatchObliqueDecisionTreeClassifier,
    "PatchObliqueDecisionTreeRegressor": PatchObliqueDecisionTreeRegressor,
    "UnsupervisedDecisionTree": UnsupervisedDecisionTree,
    "UnsupervisedObliqueDecisionTree": UnsupervisedObliqueDecisionTree,
}
X_small = np.array(
    [
        [0, 0, 4, 0, 0, 0, 1, -14, 0, -4, 0, 0, 0, 0],
        [0, 0, 5, 3, 0, -4, 0, 0, 1, -5, 0.2, 0, 4, 1],
        [-1, -1, 0, 0, -4.5, 0, 0, 2.1, 1, 0, 0, -4.5, 0, 1],
        [-1, -1, 0, -1.2, 0, 0, 0, 0, 0, 0, 0.2, 0, 0, 1],
        [-1, -1, 0, 0, 0, 0, 0, 3, 0, 0, 0, 0, 0, 1],
        [-1, -2, 0, 4, -3, 10, 4, 0, -3.2, 0, 4, 3, -4, 1],
        [2.11, 0, -6, -0.5, 0, 11, 0, 0, -3.2, 6, 0.5, 0, -3, 1],
        [2.11, 0, -6, -0.5, 0, 11, 0, 0, -3.2, 6, 0, 0, -2, 1],
        [2.11, 8, -6, -0.5, 0, 11, 0, 0, -3.2, 6, 0, 0, -2, 1],
        [2.11, 8, -6, -0.5, 0, 11, 0, 0, -3.2, 6, 0.5, 0, -1, 0],
        [2, 8, 5, 1, 0.5, -4, 10, 0, 1, -5, 3, 0, 2, 0],
        [2, 0, 1, 1, 1, -1, 1, 0, 0, -2, 3, 0, 1, 0],
        [2, 0, 1, 2, 3, -1, 10, 2, 0, -1, 1, 2, 2, 0],
        [1, 1, 0, 2, 2, -1, 1, 2, 0, -5, 1, 2, 3, 0],
        [3, 1, 0, 3, 0, -4, 10, 0, 1, -5, 3, 0, 3, 1],
        [2.11, 8, -6, -0.5, 0, 1, 0, 0, -3.2, 6, 0.5, 0, -3, 1],
        [2.11, 8, -6, -0.5, 0, 1, 0, 0, -3.2, 6, 1.5, 1, -1, -1],
        [2.11, 8, -6, -0.5, 0, 10, 0, 0, -3.2, 6, 0.5, 0, -1, -1],
        [2, 0, 5, 1, 0.5, -2, 10, 0, 1, -5, 3, 1, 0, -1],
        [2, 0, 1, 1, 1, -2, 1, 0, 0, -2, 0, 0, 0, 1],
        [2, 1, 1, 1, 2, -1, 10, 2, 0, -1, 0, 2, 1, 1],
        [1, 1, 0, 0, 1, -3, 1, 2, 0, -5, 1, 2, 1, 1],
        [3, 1, 0, 1, 0, -4, 1, 0, 1, -2, 0, 0, 1, 0],
    ]
)

y_small = [1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0]
y_small_reg = [
    1.0,
    2.1,
    1.2,
    0.05,
    10,
    2.4,
    3.1,
    1.01,
    0.01,
    2.98,
    3.1,
    1.1,
    0.0,
    1.2,
    2,
    11,
    0,
    0,
    4.5,
    0.201,
    1.06,
    0.9,
    0,
]


# also load the iris dataset
# and randomly permute it
iris = datasets.load_iris()
rng = np.random.RandomState(1)
perm = rng.permutation(iris.target.size)
iris.data = iris.data[perm]
iris.target = iris.target[perm]

# also load the diabetes dataset
# and randomly permute it
diabetes = datasets.load_diabetes()
perm = rng.permutation(diabetes.target.size)
diabetes.data = diabetes.data[perm]
diabetes.target = diabetes.target[perm]

# load digits dataset and randomly permute it
digits = datasets.load_digits()
perm = rng.permutation(digits.target.size)
digits.data = digits.data[perm]
digits.target = digits.target[perm]


def test_pickle_splitters():
    """Test that splitters are picklable."""
    import tempfile

    import joblib

    from sktree._lib.sklearn.tree._criterion import Gini
    from sktree.tree._oblique_splitter import BestObliqueSplitter, RandomObliqueSplitter
    from sktree.tree.manifold._morf_splitter import BestPatchSplitter

    criterion = Gini(1, np.array((0, 1)))
    max_features = 6
    min_samples_leaf = 1
    min_weight_leaf = 0.0
    monotonic_cst = np.array([1, 1, 1, 1, 1, 1], dtype=np.int8)
    random_state = np.random.RandomState(100)
    boundary = None
    feature_weight = None
    min_patch_dims = np.array((1, 1))
    max_patch_dims = np.array((3, 1))
    dim_contiguous = np.array((True, True))
    data_dims = np.array((5, 5))

    feature_combinations = 1.5
    splitter = BestObliqueSplitter(
        criterion,
        max_features,
        min_samples_leaf,
        min_weight_leaf,
        random_state,
        monotonic_cst,
        feature_combinations,
    )
    with tempfile.TemporaryFile() as f:
        joblib.dump(splitter, f)

    splitter = RandomObliqueSplitter(
        criterion,
        max_features,
        min_samples_leaf,
        min_weight_leaf,
        random_state,
        monotonic_cst,
        feature_combinations,
    )
    with tempfile.TemporaryFile() as f:
        joblib.dump(splitter, f)

    splitter = BestPatchSplitter(
        criterion,
        max_features,
        min_samples_leaf,
        min_weight_leaf,
        random_state,
        monotonic_cst,
        feature_combinations,
        min_patch_dims,
        max_patch_dims,
        dim_contiguous,
        data_dims,
        boundary,
        feature_weight,
    )
    with tempfile.TemporaryFile() as f:
        joblib.dump(splitter, f)


@parametrize_with_checks(
    [
        ExtraObliqueDecisionTreeClassifier(random_state=12),
        ExtraObliqueDecisionTreeRegressor(random_state=12),
        ObliqueDecisionTreeClassifier(random_state=12),
        ObliqueDecisionTreeRegressor(random_state=12),
        PatchObliqueDecisionTreeClassifier(random_state=12),
        PatchObliqueDecisionTreeRegressor(random_state=12),
    ]
)
def test_sklearn_compatible_estimator(estimator, check):
    # TODO: remove when we can replicate the CI error...
    if isinstance(
        estimator, (PatchObliqueDecisionTreeClassifier, ExtraObliqueDecisionTreeClassifier)
    ) and check.func.__name__ in ["check_fit_score_takes_y"]:
        pytest.skip()
    check(estimator)


@pytest.mark.parametrize("Tree", CLF_TREES.values())
def test_oblique_tree_sampling(Tree, random_state=0):
    """Test Oblique Decision Trees.

    Oblique trees can sample more candidate splits than
    a normal axis-aligned tree.
    """
    X, y = iris.data, iris.target
    n_samples, n_features = X.shape

    # add additional noise dimensions
    rng = np.random.RandomState(random_state)
    X_noise = rng.random((n_samples, n_features))
    X = np.concatenate((X, X_noise), axis=1)

    # oblique decision trees can sample significantly more
    # diverse sets of splits and will do better if allowed
    # to sample more
    tree_ri = DecisionTreeClassifier(random_state=random_state, max_features=n_features)
    tree_rc = Tree(random_state=random_state, max_features=n_features * 2)
    ri_cv_scores = cross_val_score(tree_ri, X, y, scoring="accuracy", cv=10, error_score="raise")
    rc_cv_scores = cross_val_score(tree_rc, X, y, scoring="accuracy", cv=10, error_score="raise")
    assert rc_cv_scores.mean() > ri_cv_scores.mean()
    assert rc_cv_scores.std() < ri_cv_scores.std()
    assert rc_cv_scores.mean() > 0.91


@pytest.mark.parametrize("splitter", ["best", "random"])
@pytest.mark.parametrize("Tree", OBLIQUE_TREES.values())
def test_oblique_trees_feature_combinations_less_than_n_features(Tree, splitter):
    """Test the hyperparameter ``feature_combinations`` behaves properly."""

    X, y = iris.data[:5, :], iris.target[:5, ...]
    _, n_features = X.shape

    # asset that the feature combinations is less than the number of features
    estimator = ObliqueDecisionTreeClassifier(
        splitter=splitter, random_state=0, feature_combinations=3
    )
    estimator.fit(X, y)
    assert estimator.feature_combinations_ < n_features

    X, y = diabetes.data[:5, :], diabetes.target[:5, ...]
    _, n_features = X.shape

    # asset that the feature combinations is less than the number of features
    estimator = Tree(random_state=0, feature_combinations=3)
    estimator.fit(X, y)
    assert estimator.feature_combinations_ < n_features


@pytest.mark.parametrize("splitter", ["best", "random"])
@pytest.mark.parametrize("Tree", OBLIQUE_TREES.values())
def test_oblique_trees_feature_combinations(Tree, splitter):
    """Test the hyperparameter ``feature_combinations`` behaves properly."""

    if is_classifier(Tree):
        X, y = iris.data, iris.target
    else:
        X, y = diabetes.data, diabetes.target
    _, n_features = X.shape

    X = X[:5, :]
    y = y[:5, ...]

    with pytest.raises(
        RuntimeError, match=f"Feature combinations {n_features + 1} should not be greater"
    ):
        estimator = Tree(splitter=splitter, random_state=0, feature_combinations=n_features + 1)
        estimator.fit(X, y)

    # asset that the feature combinations is less than the number of features
    estimator = Tree(splitter=splitter, random_state=0, feature_combinations=3)
    estimator.fit(X, y)
    assert estimator.feature_combinations_ < n_features

    # default option should make it 1.5 if n_features > 1.5
    estimator = Tree(splitter=splitter, random_state=0)
    estimator.fit(X, y)
    assert estimator.feature_combinations_ == 1.5

    # setting the feature combinations explicitly is fine as long as it is < n_features
    estimator = Tree(splitter=splitter, random_state=0, feature_combinations=3)
    estimator.fit(X, y)
    assert estimator.feature_combinations_ == 3

    # edge-case of only a single feature should set feature_combinations properly
    X = X[:, 0:1]
    estimator = Tree(splitter=splitter, random_state=0)
    estimator.fit(X, y)
    assert estimator.feature_combinations_ == 1


@pytest.mark.parametrize("Tree", PATCH_OBLIQUE_TREES.values())
def test_patch_tree_errors(Tree):
    """Test errors that are specifically raised by manifold trees."""
    X, y = digits.data, digits.target

    # passed in data should match expected data shape
    with pytest.raises(RuntimeError, match="Data dimensions"):
        clf = Tree(
            data_dims=(8, 9),
        )
        clf.fit(X, y)

    # minimum patch height/width should be always less than or equal to
    # the maximum patch height/width
    with pytest.raises(RuntimeError, match="The minimum patch"):
        clf = Tree(
            min_patch_dims=(2, 1),
            max_patch_dims=(1, 1),
            data_dims=(8, 8),
        )
        clf.fit(X, y)

    # the maximum patch height/width should not exceed the data height/width
    with pytest.raises(RuntimeError, match="The maximum patch width"):
        clf = Tree(
            max_patch_dims=(9, 1),
            data_dims=(8, 8),
        )
        clf.fit(X, y)


def test_patch_tree_overfits():
    """Test of performance of patch tree on image-like data."""
    X, y = digits.data, digits.target

    clf = PatchObliqueDecisionTreeClassifier(
        min_patch_dims=(2, 2),
        max_patch_dims=(6, 6),
        data_dims=(8, 8),
        random_state=1,
    )

    # the single tree without depth limitations should have almost 0 bias
    clf.fit(X, y)
    assert accuracy_score(y, clf.predict(X)) > 0.99


def test_patch_tree_compared():
    """Test patch tree against other tree models."""
    X, y = digits.data, digits.target
    _, n_features = X.shape

    clf = PatchObliqueDecisionTreeClassifier(
        min_patch_dims=(1, 1),
        max_patch_dims=(8, 8),
        data_dims=(8, 8),
        random_state=1,
        max_features=n_features,
    )

    clf.fit(X, y)

    # a well-parametrized patch tree should be relatively accurate
    patch_tree_score = np.mean(cross_val_score(clf, X, y, scoring="accuracy", cv=5))
    assert patch_tree_score > 0.6

    # similar to oblique trees, we can sample more and we should improve
    clf = PatchObliqueDecisionTreeClassifier(
        min_patch_dims=(1, 1),
        max_patch_dims=(8, 8),
        data_dims=(8, 8),
        random_state=1,
        max_features=n_features * 2,
    )
    new_patch_tree_score = np.mean(cross_val_score(clf, X, y, scoring="accuracy", cv=5))
    assert new_patch_tree_score > patch_tree_score

    clf = ObliqueDecisionTreeClassifier(
        random_state=1,
        max_features=n_features,
    )
    assert np.mean(cross_val_score(clf, X, y, scoring="accuracy", cv=2)) < new_patch_tree_score

    clf = DecisionTreeClassifier(max_features=n_features, random_state=1)
    assert np.mean(cross_val_score(clf, X, y, scoring="accuracy", cv=2)) < new_patch_tree_score


def test_patch_oblique_tree_feature_weights():
    """Test patch oblique tree when feature weights are passed in."""
    X, y = digits.data, digits.target

    with pytest.raises(ValueError, match="feature_weight has shape"):
        clf = PatchObliqueDecisionTreeClassifier(
            min_patch_dims=(2, 2),
            max_patch_dims=(6, 6),
            data_dims=(8, 8),
            random_state=1,
            feature_weight=np.ones((X.shape[0], 1), dtype=np.float32),
        )
        clf.fit(X, y)


@pytest.mark.parametrize("Tree", REG_TREES.values())
@pytest.mark.parametrize("criterion", REG_CRITERIONS)
def test_regression_toy(Tree, criterion):
    # Check regression on a toy dataset.

    # toy sample
    X = [[-2, -1], [-1, -1], [-1, -2], [1, 1], [1, 2], [2, 1]]
    y = [-1, -1, -1, 1, 1, 1]
    T = [[-1, -1], [2, 2], [3, 2]]
    true_result = [-1, 1, 1]

    if criterion == "poisson":
        # make target positive while not touching the original y and
        # true_result
        a = np.abs(np.min(y)) + 1
        y_train = np.array(y) + a
        y_test = np.array(true_result) + a
    else:
        y_train = y
        y_test = true_result

    regressor = Tree(criterion=criterion, random_state=1)
    regressor.fit(X, y_train)
    assert_allclose(regressor.predict(T), y_test)

    regressor = Tree(criterion=criterion, max_features=1, random_state=1)
    regressor.fit(X, y_train)
    assert_allclose(regressor.predict(T), y_test)


@pytest.mark.parametrize("name, Tree", REG_TREES.items())
@pytest.mark.parametrize("criterion", REG_CRITERIONS)
def test_diabetes_overfit(name, Tree, criterion):
    # check consistency of overfitted trees on the diabetes dataset
    # since the trees will overfit, we expect an MSE of 0
    reg = Tree(criterion=criterion, random_state=12)
    reg.fit(diabetes.data, diabetes.target)
    score = mean_squared_error(diabetes.target, reg.predict(diabetes.data))
    assert score == pytest.approx(
        0
    ), f"Failed with {name}, criterion = {criterion} and score = {score}"


@skip_if_32bit
@pytest.mark.parametrize("name, Tree", REG_TREES.items())
@pytest.mark.parametrize(
    "criterion, max_depth, metric, max_loss",
    [
        ("squared_error", 15, mean_squared_error, 65),
        ("absolute_error", 25, mean_squared_error, 60),
        ("friedman_mse", 15, mean_squared_error, 65),
        ("poisson", 15, mean_poisson_deviance, 30),
    ],
)
def test_diabetes_underfit(name, Tree, criterion, max_depth, metric, max_loss):
    # check consistency of trees when the depth and the number of features are
    # limited

    reg = Tree(criterion=criterion, max_depth=max_depth, max_features=10, random_state=1234)
    reg.fit(diabetes.data, diabetes.target)
    loss = metric(diabetes.target, reg.predict(diabetes.data))
    assert (
        0.0 <= loss < max_loss
    ), f"Failed with {name}, criterion = {criterion} and loss = {loss}, and max_loss: {max_loss}"


def test_numerical_stability():
    # Check numerical stability.
    X = np.array(
        [
            [152.08097839, 140.40744019, 129.75102234, 159.90493774],
            [142.50700378, 135.81935120, 117.82884979, 162.75781250],
            [127.28772736, 140.40744019, 129.75102234, 159.90493774],
            [132.37025452, 143.71923828, 138.35694885, 157.84558105],
            [103.10237122, 143.71928406, 138.35696411, 157.84559631],
            [127.71276855, 143.71923828, 138.35694885, 157.84558105],
            [120.91514587, 140.40744019, 129.75102234, 159.90493774],
        ]
    )

    y = np.array([1.0, 0.70209277, 0.53896582, 0.0, 0.90914464, 0.48026916, 0.49622521])

    with np.errstate(all="raise"):
        for name, Tree in REG_TREES.items():
            reg = Tree(random_state=0)
            reg.fit(X, y)
            reg.fit(X, -y)
            reg.fit(-X, y)
            reg.fit(-X, -y)


@pytest.mark.parametrize("criterion", ["squared_error", "friedman_mse", "poisson"])
@pytest.mark.parametrize("Tree", REG_TREES.values())
def test_balance_property(criterion, Tree):
    # Test that sum(y_pred)=sum(y_true) on training set.
    # This works if the mean is predicted (should even be true for each leaf).
    # MAE predicts the median and is therefore excluded from this test.
    # Choose a training set with non-negative targets (for poisson)
    X, y = diabetes.data, diabetes.target
    reg = Tree(criterion=criterion)
    reg.fit(X, y)
    assert np.sum(reg.predict(X)) == pytest.approx(
        np.sum(y)
    ), f"Failed with {Tree} and {criterion}: {np.sum(reg.predict(X))} != {np.sum(y)}"


@pytest.mark.parametrize("Tree", ALL_TREES.values())
def test_deterministic_pickle(Tree):
    # Non-regression test for:
    # https://github.com/scikit-learn/scikit-learn/issues/27268
    # Uninitialised memory would lead to the two pickle strings being different.
    tree1 = Tree(random_state=0).fit(iris.data, iris.target)
    tree2 = Tree(random_state=0).fit(iris.data, iris.target)

    pickle1 = pickle.dumps(tree1)
    pickle2 = pickle.dumps(tree2)

    pickle1_tree = pickle.loads(pickle1)
    pickle2_tree = pickle.loads(pickle2)
    assert_tree_equal(
        pickle1_tree.tree_,
        pickle2_tree.tree_,
        "The trees of the original and loaded classifiers are not equal.",
    )
    assert sys.getsizeof(pickle1_tree) == sys.getsizeof(pickle2_tree)

    # TODO: this does not work properly it seems
    if Tree not in (list(CLF_TREES.values()) + list(REG_TREES.values())):
        assert pickle1 == pickle2, f"Failed with {Tree}"
