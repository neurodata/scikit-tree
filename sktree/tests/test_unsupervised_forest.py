import numpy as np
import pytest
from sklearn import datasets
from sklearn.cluster import AgglomerativeClustering
from sklearn.datasets import make_blobs
from sklearn.metrics import adjusted_rand_score
from sklearn.utils.estimator_checks import parametrize_with_checks

from sktree.ensemble import UnsupervisedObliqueRandomForest, UnsupervisedRandomForest

CLUSTER_CRITERIONS = ("twomeans", "fastbic")

FOREST_CLUSTERS = {
    "UnsupervisedRandomForest": UnsupervisedRandomForest,
    "UnsupervisedObliqueRandomForest": UnsupervisedObliqueRandomForest,
}

# load iris dataset
iris = datasets.load_iris()
rng = np.random.RandomState(1)
perm = rng.permutation(iris.target.size)
iris.data = iris.data[perm]
iris.target = iris.target[perm]


@parametrize_with_checks(
    [
        UnsupervisedRandomForest(random_state=12345, n_estimators=50),
        UnsupervisedObliqueRandomForest(random_state=12345, n_estimators=50),
    ]
)
def test_sklearn_compatible_estimator(estimator, check):
    if check.func.__name__ in [
        # Cannot apply agglomerative clustering on < 2 samples
        "check_methods_subset_invariance",
        # # sample weights do not necessarily imply a sample is not used in clustering
        "check_sample_weights_invariance",
        # # sample order is not preserved in predict
        "check_methods_sample_order_invariance",
    ]:
        pytest.skip()
    check(estimator)


@pytest.mark.parametrize("name, forest", FOREST_CLUSTERS.items())
@pytest.mark.parametrize("criterion", CLUSTER_CRITERIONS)
def test_check_simulation(name, forest, criterion):
    n_samples = 200
    n_classes = 2

    #
    if name == "UnsupervisedRandomForest":
        n_features = 5
        if criterion == "twomeans":
            expected_score = 0.05
        elif criterion in ("fastbic", "fasterbic"):
            expected_score = 0.35
    else:
        n_features = 20

        # in the forest setting, we can overfit the training dataset perfectly
        expected_score = 1.0
    X, y = make_blobs(
        n_samples=n_samples, centers=n_classes, n_features=n_features, random_state=12345
    )

    clf = forest(criterion=criterion, random_state=12345)
    clf.fit(X)
    sim_mat = clf.compute_similarity_matrix(X)

    # all ones along the diagonal
    assert np.array_equal(sim_mat.diagonal(), np.ones(n_samples))

    cluster = AgglomerativeClustering(n_clusters=n_classes).fit(sim_mat)
    predict_labels = cluster.fit_predict(sim_mat)
    score = adjusted_rand_score(y, predict_labels)

    # XXX: This should be > 0.9 according to the UReRF. However, that could be because they used
    # the oblique projections by default
    assert (
        score >= expected_score
    ), f"{name}-blobs failed with criterion {criterion} and score = {score}"


@pytest.mark.parametrize("name, forest", FOREST_CLUSTERS.items())
@pytest.mark.parametrize("criterion", CLUSTER_CRITERIONS)
def test_check_iris(name, forest, criterion):
    # Check consistency on dataset iris.
    n_classes = 3
    est = forest(criterion=criterion, random_state=12345)
    est.fit(iris.data, iris.target)
    sim_mat = est.compute_similarity_matrix(iris.data)

    if criterion == "twomeans":
        if "oblique" in name.lower():
            expected_score = 0.21
        else:
            expected_score = 0.2
    elif criterion in ("fastbic", "fasterbic"):
        if "oblique" in name.lower():
            expected_score = 0.55
        else:
            expected_score = 0.3

    cluster = AgglomerativeClustering(n_clusters=n_classes).fit(sim_mat)
    predict_labels = cluster.fit_predict(sim_mat)
    score = adjusted_rand_score(iris.target, predict_labels)

    # Two-means and fastBIC criterions perform similarly here
    assert (
        score > expected_score
    ), f"{name}-iris failed with criterion {criterion} and score = {score}"
