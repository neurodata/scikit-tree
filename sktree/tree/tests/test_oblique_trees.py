import numpy as np

from sktree import HonestForestClassifier
from sktree.tree import MultiViewDecisionTreeClassifier


def test_oblique_tree():
    """Test regression reported in https://github.com/neurodata/scikit-tree/issues/215."""
    n, a = (
        10,
        20,
    )
    x = np.random.normal(size=(n, a))
    y = np.random.binomial(1, 0.5, size=(n))

    for seed in range(100):
        # est = MultiViewDecisionTreeClassifier(
        #     max_features=0.3,
        #     feature_set_ends=[15, 20],
        #     random_state=seed,
        # )

        est = HonestForestClassifier(
            n_estimators=10,
            max_features=0.3,
            feature_set_ends=[15, 20],
            # bootstrap=True,
            # max_samples=1.6,
            tree_estimator=MultiViewDecisionTreeClassifier(),
            random_state=seed,
            n_jobs=-1,
        )

        est.fit(x, y)
