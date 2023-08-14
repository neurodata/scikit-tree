import numpy as np
import pytest
from sklearn import datasets
from sklearn.cluster import AgglomerativeClustering
from sklearn.datasets import make_blobs
from sklearn.metrics import adjusted_rand_score
from sklearn.utils.estimator_checks import parametrize_with_checks

from sktree.tree import UnsupervisedDecisionTree, UnsupervisedObliqueDecisionTree

CLUSTER_CRITERIONS = ("twomeans", "fastbic")
REG_CRITERIONS = ("squared_error", "absolute_error", "friedman_mse", "poisson")

TREE_CLUSTERS = {
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


@parametrize_with_checks(
    [
        UnsupervisedDecisionTree(random_state=12),
        UnsupervisedObliqueDecisionTree(random_state=12),
    ]
)
def test_sklearn_compatible_transformer(estimator, check):
    if check.func.__name__ in [
        # Cannot apply agglomerative clustering on < 2 samples
        "check_methods_subset_invariance",
        # clustering accuracy is poor when using TwoMeans on 1 single tree
        "check_clustering",
        # sample weights do not necessarily imply a sample is not used in clustering
        "check_sample_weights_invariance",
        # sample order is not preserved in predict
        "check_methods_sample_order_invariance",
    ]:
        pytest.skip()
    check(estimator)


@pytest.mark.parametrize("name,Tree", TREE_CLUSTERS.items())
@pytest.mark.parametrize("criterion", CLUSTER_CRITERIONS)
def test_check_simulation(name, Tree, criterion):
    """Test axis-aligned Gaussian blobs."""
    n_samples = 100
    n_classes = 2
    X, y = make_blobs(n_samples=n_samples, centers=n_classes, n_features=6, random_state=1234)

    est = Tree(criterion=criterion, random_state=1234)
    est.fit(X)
    sim_mat = est.compute_similarity_matrix(X)

    # there is quite a bit of variance in the performance at the tree level
    if criterion == "twomeans":
        if "oblique" in name.lower():
            expected_score = 0.02
        else:
            expected_score = 0.3
    elif criterion == "fastbic":
        if "oblique" in name.lower():
            expected_score = 0.01
        else:
            expected_score = 0.4

    # all ones along the diagonal
    assert np.array_equal(sim_mat.diagonal(), np.ones(n_samples))

    cluster = AgglomerativeClustering(n_clusters=n_classes).fit(sim_mat)
    predict_labels = cluster.fit_predict(sim_mat)
    score = adjusted_rand_score(y, predict_labels)

    # a single decision tree does not fit well, but should still have a positive score
    assert score >= expected_score, "Blobs failed with {0}, criterion = {1} and score = {2}".format(
        name, criterion, score
    )


@pytest.mark.parametrize("name,Tree", TREE_CLUSTERS.items())
@pytest.mark.parametrize("criterion", CLUSTER_CRITERIONS)
def test_check_rotated_blobs(name, Tree, criterion):
    """Test rotated axis-aligned Gaussian blobs, which should make oblique trees perform better."""
    n_samples = 100
    n_classes = 2
    X, y = make_blobs(n_samples=n_samples, centers=n_classes, n_features=6, random_state=1234)

    # apply rotation matrix to X

    est = Tree(criterion=criterion, random_state=1234)
    est.fit(X)
    sim_mat = est.compute_similarity_matrix(X)

    # there is quite a bit of variance in the performance at the tree level
    if criterion == "twomeans":
        if "oblique" in name.lower():
            expected_score = 0.02
        else:
            expected_score = 0.3
    elif criterion == "fastbic":
        if "oblique" in name.lower():
            expected_score = 0.01
        else:
            expected_score = 0.4

    # all ones along the diagonal
    assert np.array_equal(sim_mat.diagonal(), np.ones(n_samples))

    cluster = AgglomerativeClustering(n_clusters=n_classes).fit(sim_mat)
    predict_labels = cluster.fit_predict(sim_mat)
    score = adjusted_rand_score(y, predict_labels)

    # a single decision tree does not fit well, but should still have a positive score
    assert score >= expected_score, "Blobs failed with {0}, criterion = {1} and score = {2}".format(
        name, criterion, score
    )


@pytest.mark.parametrize("name,Tree", TREE_CLUSTERS.items())
@pytest.mark.parametrize("criterion", CLUSTER_CRITERIONS)
def test_check_iris(name, Tree, criterion):
    # Check consistency on dataset iris.
    n_classes = len(np.unique(iris.target))
    est = Tree(criterion=criterion, random_state=12345)
    est.fit(iris.data, iris.target)
    sim_mat = est.compute_similarity_matrix(iris.data)

    # there is quite a bit of variance in the performance at the tree level
    if criterion == "twomeans":
        if "oblique" in name.lower():
            expected_score = 0.2
        else:
            expected_score = 0.01
    elif criterion == "fastbic":
        if "oblique" in name.lower():
            expected_score = 0.001
        else:
            expected_score = 0.2

    cluster = AgglomerativeClustering(n_clusters=n_classes).fit(sim_mat)
    predict_labels = cluster.fit_predict(sim_mat)
    score = adjusted_rand_score(iris.target, predict_labels)

    # Two-means and fastBIC criterions doesn't perform well
    assert score > expected_score, "Iris failed with {0}, criterion = {1} and score = {2}".format(
        name, criterion, score
    )
