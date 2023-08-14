import joblib
import numpy as np
import pytest
from numpy.testing import assert_almost_equal, assert_array_almost_equal, assert_array_equal
from sklearn.base import is_classifier
from sklearn.datasets import make_blobs
from sklearn.tree._tree import TREE_LEAF

from sktree.tree import (
    ObliqueDecisionTreeClassifier,
    ObliqueDecisionTreeRegressor,
    PatchObliqueDecisionTreeClassifier,
    PatchObliqueDecisionTreeRegressor,
    UnsupervisedDecisionTree,
    UnsupervisedObliqueDecisionTree,
)

ALL_TREES = [
    ObliqueDecisionTreeRegressor,
    PatchObliqueDecisionTreeRegressor,
    ObliqueDecisionTreeClassifier,
    PatchObliqueDecisionTreeClassifier,
    UnsupervisedDecisionTree,
    UnsupervisedObliqueDecisionTree,
]


def assert_tree_equal(d, s, message):
    assert s.node_count == d.node_count, "{0}: inequal number of node ({1} != {2})".format(
        message, s.node_count, d.node_count
    )

    assert_array_equal(d.children_right, s.children_right, message + ": inequal children_right")
    assert_array_equal(d.children_left, s.children_left, message + ": inequal children_left")

    external = d.children_right == TREE_LEAF
    internal = np.logical_not(external)

    assert_array_equal(d.feature[internal], s.feature[internal], message + ": inequal features")
    assert_array_equal(
        d.threshold[internal], s.threshold[internal], message + ": inequal threshold"
    )
    assert_array_equal(
        d.n_node_samples.sum(),
        s.n_node_samples.sum(),
        message + ": inequal sum(n_node_samples)",
    )
    assert_array_equal(d.n_node_samples, s.n_node_samples, message + ": inequal n_node_samples")

    assert_almost_equal(d.impurity, s.impurity, err_msg=message + ": inequal impurity")

    assert_array_almost_equal(
        d.value[external], s.value[external], err_msg=message + ": inequal value"
    )


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


@pytest.mark.parametrize(
    "TREE",
    [ObliqueDecisionTreeClassifier, UnsupervisedDecisionTree, UnsupervisedObliqueDecisionTree],
)
def test_tree_deserialization_from_read_only_buffer(tmpdir, TREE):
    """Check that Trees can be deserialized with read only buffers.

    Non-regression test for gh-25584.
    """
    pickle_path = str(tmpdir.join("clf.joblib"))
    clf = TREE(random_state=0)

    if is_classifier(TREE):
        clf.fit(X_small, y_small)
    else:
        clf.fit(X_small)

    joblib.dump(clf, pickle_path)
    loaded_clf = joblib.load(pickle_path, mmap_mode="r")

    assert_tree_equal(
        loaded_clf.tree_,
        clf.tree_,
        "The trees of the original and loaded classifiers are not equal.",
    )


@pytest.mark.parametrize("tree", ALL_TREES)
def test_similarity_matrix(tree):
    n_samples = 200
    n_classes = 2
    n_features = 5

    X, y = make_blobs(
        n_samples=n_samples, centers=n_classes, n_features=n_features, random_state=12345
    )

    clf = tree(random_state=12345)
    clf.fit(X, y)
    sim_mat = clf.compute_similarity_matrix(X)

    assert np.allclose(sim_mat, sim_mat.T)
    assert np.all((sim_mat.diagonal() == 1))
