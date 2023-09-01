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
    ObliqueDecisionTreeClassifier,
    ObliqueDecisionTreeRegressor,
    PatchObliqueDecisionTreeClassifier,
    PatchObliqueDecisionTreeRegressor,
    UnsupervisedDecisionTree,
    UnsupervisedObliqueDecisionTree,
)

CLUSTER_CRITERIONS = ("twomeans", "fastbic")
REG_CRITERIONS = ("squared_error", "absolute_error", "friedman_mse", "poisson")

TREE_CLUSTERS = {
    "UnsupervisedDecisionTree": UnsupervisedDecisionTree,
    "UnsupervisedObliqueDecisionTree": UnsupervisedObliqueDecisionTree,
}

REG_TREES = {
    "ObliqueDecisionTreeRegressor": ObliqueDecisionTreeRegressor,
    "PatchObliqueDecisionTreeRegressor": PatchObliqueDecisionTreeRegressor,
}

CLF_TREES = {
    "ObliqueDecisionTreeClassifier": ObliqueDecisionTreeClassifier,
    "PatchObliqueTreeClassifier": PatchObliqueDecisionTreeClassifier,
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


X, y = iris.data, iris.target
n_samples, n_features = X.shape

# add additional noise dimensions
rng = np.random.RandomState(0)
X_noise = rng.random((n_samples, n_features))
X = np.concatenate((X, X_noise), axis=1)

# oblique decision trees can sample significantly more
# diverse sets of splits and will do better if allowed
# to sample more
tree_ri = DecisionTreeClassifier(random_state=0, max_features=n_features)
tree_rc = ObliqueDecisionTreeClassifier(random_state=0, max_features=n_features * 2)
ri_cv_scores = cross_val_score(tree_ri, X, y, scoring="accuracy", cv=10, error_score="raise")
rc_cv_scores = cross_val_score(tree_rc, X, y, scoring="accuracy", cv=10, error_score="raise")
assert rc_cv_scores.mean() > ri_cv_scores.mean(), f"{rc_cv_scores.mean()} <= {ri_cv_scores.mean()}"
assert rc_cv_scores.std() < ri_cv_scores.std(), f"{rc_cv_scores.std()} >= {ri_cv_scores.std()}"
assert rc_cv_scores.mean() > 0.91, f"{rc_cv_scores.mean()} <= 0.91"