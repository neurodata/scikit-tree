import joblib
import numpy as np
import pytest
from numpy.testing import assert_almost_equal, assert_array_almost_equal, assert_array_equal
from sklearn import datasets
from sklearn.utils.estimator_checks import parametrize_with_checks

from sktree.causal import CausalTree


@parametrize_with_checks(
    [
        CausalTree(random_state=12),
    ]
)
def test_sklearn_compatible_estimator(estimator, check):
    # TODO: remove when we implement Regressor classes
    if check.func.__name__ in ["check_requires_y_none"]:
        pytest.skip()
    check(estimator)
