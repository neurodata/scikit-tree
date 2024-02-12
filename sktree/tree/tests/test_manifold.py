import numpy as np

from sktree import PatchObliqueRandomForestClassifier
from sktree.tree import PatchObliqueDecisionTreeClassifier


def test_contiguous_and_discontiguous_patch():
    """Test regression reported in https://github.com/neurodata/scikit-tree/issues/215."""
    n, a, b = 100, 10, 10
    x = np.random.normal(size=(n, a, b))
    y = np.random.binomial(1, 0.5, size=(n))

    for seed in range(1000):
        est = PatchObliqueDecisionTreeClassifier(
            min_patch_dims=[1, 1],
            max_patch_dims=[4, 4],
            dim_contiguous=(True, False),
            data_dims=(a, b),
            random_state=seed,
        )

        est.fit(x.reshape(100, -1), y)


def test_contiguous_and_discontiguous_patch_forest():
    """Test regression reported in https://github.com/neurodata/scikit-tree/issues/215."""
    n, a, b = 100, 10, 10
    x = np.random.normal(size=(n, a, b))
    y = np.random.binomial(1, 0.5, size=(n))

    est = PatchObliqueRandomForestClassifier(
        n_estimators=20,
        min_patch_dims=[1, 1],
        max_patch_dims=[4, 4],
        dim_contiguous=(False, True),
        data_dims=(a, b),
        n_jobs=-1,
    )

    est.fit(x.reshape(100, -1), y)
