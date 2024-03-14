import pickle
from copy import deepcopy
from itertools import combinations
from pathlib import Path

import numpy as np
import pytest
from flaky import flaky
from joblib import Parallel, delayed
from numpy.testing import assert_almost_equal, assert_array_equal
from scipy.stats import wilcoxon
from sklearn import datasets

from sktree import HonestForestClassifier, RandomForestClassifier, RandomForestRegressor
from sktree._lib.sklearn.tree import DecisionTreeClassifier
from sktree.stats import (
    FeatureImportanceForestClassifier,
    FeatureImportanceForestRegressor,
    PermutationHonestForestClassifier,
    build_coleman_forest,
    build_permutation_forest,
)
from sktree.stats.utils import _non_nan_samples
from sktree.tree import MultiViewDecisionTreeClassifier, ObliqueDecisionTreeClassifier

# load the iris dataset (n_samples, 4)
# and randomly permute it
iris = datasets.load_iris()
seed = 12345
rng = np.random.default_rng(seed)

# remove third class
iris_X = iris.data[iris.target != 2]
iris_y = iris.target[iris.target != 2]

p = rng.permutation(iris_X.shape[0])
iris_X = iris_X[p]
iris_y = iris_y[p]


@pytest.mark.parametrize("sample_dataset_per_tree", [True, False])
def test_featureimportance_forest_permute_pertree(sample_dataset_per_tree):
    n_samples = 50
    n_estimators = 10
    est = FeatureImportanceForestClassifier(
        estimator=RandomForestClassifier(
            n_estimators=n_estimators,
            random_state=seed,
        ),
        permute_forest_fraction=1.0 / n_estimators * 5,
        test_size=0.7,
        random_state=seed,
        sample_dataset_per_tree=sample_dataset_per_tree,
    )
    est.statistic(iris_X[:n_samples], iris_y[:n_samples], metric="mse")

    assert (
        len(est.train_test_samples_[0][1]) == n_samples * est.test_size
    ), f"{len(est.train_test_samples_[0][1])} {n_samples * est.test_size}"
    assert len(est.train_test_samples_[0][0]) == est._n_samples_ - n_samples * est.test_size

    est.test(iris_X[:n_samples], iris_y[:n_samples], [0, 1], n_repeats=10, metric="mse")
    assert (
        len(est.train_test_samples_[0][1]) == n_samples * est.test_size
    ), f"{len(est.train_test_samples_[0][1])} {n_samples * est.test_size}"
    assert len(est.train_test_samples_[0][0]) == est._n_samples_ - n_samples * est.test_size

    # covariate index should work with mse
    est.reset()
    est.statistic(iris_X[:n_samples], iris_y[:n_samples], covariate_index=[1], metric="mse")
    with pytest.raises(RuntimeError, match="Metric must be"):
        est.statistic(iris_X[:n_samples], iris_y[:n_samples], covariate_index=[1], metric="mi")

    # covariate index should work with mse
    est.reset()
    est.statistic(iris_X[:n_samples], iris_y[:n_samples], covariate_index=[1], metric="mse")
    with pytest.raises(RuntimeError, match="Metric must be"):
        est.statistic(iris_X[:n_samples], iris_y[:n_samples], covariate_index=[1], metric="mi")

    # covariate index must be an iterable
    est.reset()
    with pytest.raises(RuntimeError, match="covariate_index must be an iterable"):
        est.statistic(iris_X[:n_samples], iris_y[:n_samples], 0, metric="mi")

    # covariate index must be an iterable of ints
    est.reset()
    with pytest.raises(RuntimeError, match="Not all covariate_index"):
        est.statistic(iris_X[:n_samples], iris_y[:n_samples], [0, 1.0], metric="mi")

    with pytest.raises(
        RuntimeError, match="permute_forest_fraction must be greater than 1./n_estimators"
    ):
        est = FeatureImportanceForestClassifier(
            estimator=RandomForestClassifier(
                n_estimators=n_estimators,
                random_state=seed,
            ),
            permute_forest_fraction=1.0 / n_samples,
            test_size=0.7,
            random_state=seed,
            sample_dataset_per_tree=sample_dataset_per_tree,
        )
        est.statistic(iris_X[:n_samples], iris_y[:n_samples], metric="mse")

    with pytest.raises(RuntimeError, match="permute_forest_fraction must be non-negative."):
        est = FeatureImportanceForestClassifier(
            estimator=RandomForestClassifier(
                n_estimators=n_estimators,
                random_state=seed,
            ),
            permute_forest_fraction=-1.0 / n_estimators * 5,
            test_size=0.7,
            random_state=seed,
            sample_dataset_per_tree=sample_dataset_per_tree,
        )
        est.statistic(iris_X[:n_samples], iris_y[:n_samples], metric="mse")


@pytest.mark.parametrize("covariate_index", [None, [0, 1]])
def test_featureimportance_forest_statistic_with_covariate_index(covariate_index):
    """Tests that calling `est.statistic` with covariate_index defined works.

    There should be no issue calling `est.statistic` with covariate_index defined.
    """
    n_estimators = 10
    n_samples = 10

    est = FeatureImportanceForestClassifier(
        estimator=RandomForestClassifier(
            n_estimators=n_estimators,
            random_state=seed,
        ),
        permute_forest_fraction=1.0 / n_estimators * 5,
        test_size=0.7,
        random_state=seed,
    )
    est.statistic(
        iris_X[:n_samples], iris_y[:n_samples], covariate_index=covariate_index, metric="mi"
    )


@pytest.mark.parametrize("sample_dataset_per_tree", [True, False])
def test_featureimportance_forest_stratified(sample_dataset_per_tree):
    n_samples = 100
    n_estimators = 10
    est = FeatureImportanceForestClassifier(
        estimator=RandomForestClassifier(
            n_estimators=n_estimators,
            random_state=seed,
        ),
        permute_forest_fraction=1.0 / n_estimators * 5,
        test_size=0.7,
        random_state=seed,
        sample_dataset_per_tree=sample_dataset_per_tree,
    )
    est.statistic(iris_X[:n_samples], iris_y[:n_samples], metric="mi")

    _, indices_test = est.train_test_samples_[0]
    y_test = iris_y[indices_test]

    assert len(y_test[y_test == 0]) == len(y_test[y_test == 1]), (
        f"{len(y_test[y_test==0])} " f"{len(y_test[y_test==1])}"
    )

    est.test(iris_X[:n_samples], iris_y[:n_samples], [0, 1], n_repeats=10, metric="mi")

    _, indices_test = est.train_test_samples_[0]
    y_test = iris_y[indices_test]

    assert len(y_test[y_test == 0]) == len(y_test[y_test == 1]), (
        f"{len(y_test[y_test==0])} " f"{len(y_test[y_test==1])}"
    )


def test_featureimportance_forest_errors():
    sample_dataset_per_tree = True
    n_estimators = 10
    est = FeatureImportanceForestClassifier(
        estimator=RandomForestClassifier(
            n_estimators=n_estimators,
        ),
        test_size=0.5,
        permute_forest_fraction=None,
        sample_dataset_per_tree=sample_dataset_per_tree,
    )
    with pytest.raises(RuntimeError, match="The estimator must be fitted"):
        est.train_test_samples_

    with pytest.raises(RuntimeError, match="There are less than 5 testing samples"):
        est.statistic(iris_X[:5], iris_y[:5])

    est = FeatureImportanceForestClassifier(estimator=RandomForestRegressor, test_size=0.5)
    with pytest.raises(RuntimeError, match="Estimator must be"):
        est.statistic(iris_X[:20], iris_y[:20])

    est = FeatureImportanceForestRegressor(estimator=RandomForestClassifier, test_size=0.5)
    with pytest.raises(RuntimeError, match="Estimator must be"):
        est.statistic(iris_X[:20], iris_y[:20])


@flaky(max_runs=2)
@pytest.mark.parametrize("criterion", ["gini", "entropy"])
@pytest.mark.parametrize("honest_prior", ["empirical", "uniform"])
@pytest.mark.parametrize(
    "estimator",
    [
        None,
        DecisionTreeClassifier(),
        ObliqueDecisionTreeClassifier(),
    ],
)
@pytest.mark.parametrize(
    "permute_forest_fraction",
    [
        None,
        0.5,
    ],
)
@pytest.mark.parametrize("sample_dataset_per_tree", [True, False])
def test_iris_pauc_statistic(
    criterion, honest_prior, estimator, permute_forest_fraction, sample_dataset_per_tree
):
    limit = 0.1
    max_features = "sqrt"
    n_repeats = 200
    n_estimators = 25
    test_size = 0.2

    # Check consistency on dataset iris.
    clf = FeatureImportanceForestClassifier(
        estimator=HonestForestClassifier(
            criterion=criterion,
            n_estimators=n_estimators,
            max_features=max_features,
            tree_estimator=estimator,
            honest_prior=honest_prior,
            random_state=0,
            n_jobs=1,
        ),
        test_size=test_size,
        sample_dataset_per_tree=sample_dataset_per_tree,
        permute_forest_fraction=permute_forest_fraction,
    )
    # now add completely uninformative feature
    X = np.hstack((iris_X, rng.standard_normal(size=(iris_X.shape[0], 4))))

    # test for unimportant feature set
    clf.reset()
    if sample_dataset_per_tree and permute_forest_fraction is None:
        # test in another test
        return

    stat, pvalue = clf.test(
        X,
        iris_y,
        np.arange(iris_X.shape[0], X.shape[1], dtype=int).tolist(),
        n_repeats=n_repeats,
        metric="auc",
    )
    print(pvalue)
    assert pvalue > 0.05, f"pvalue: {pvalue}"

    # test for important features that are permuted
    stat, pvalue = clf.test(X, iris_y, [0, 1, 2, 3], n_repeats=n_repeats, metric="auc")
    print(pvalue)
    assert pvalue < 0.05, f"pvalue: {pvalue}"

    # one must call `reset()` to make sure the test is run on a "new" feature set properly
    with pytest.raises(RuntimeError, match="X must have 8 features"):
        clf.statistic(iris_X, iris_y, metric="auc", max_fpr=limit)

    clf.reset()
    score = clf.statistic(iris_X, iris_y, metric="auc", max_fpr=limit)
    assert score >= 0.8, "Failed with pAUC: {0} for max fpr: {1}".format(score, limit)

    assert isinstance(clf.estimator_, HonestForestClassifier)


@pytest.mark.parametrize(
    "forest_hyppo",
    [
        FeatureImportanceForestRegressor(
            estimator=RandomForestRegressor(
                n_estimators=10,
            ),
            random_state=seed,
        ),
        FeatureImportanceForestClassifier(
            estimator=RandomForestClassifier(
                n_estimators=10,
            ),
            random_state=seed,
            permute_forest_fraction=None,
            sample_dataset_per_tree=False,
        ),
    ],
)
def test_forestht_check_inputs(forest_hyppo):
    n_samples = 100
    n_features = 5
    X = rng.uniform(size=(n_samples, n_features))
    y = rng.integers(0, 2, size=n_samples)  # Binary classification

    # Test case 1: Valid input
    forest_hyppo.statistic(X, y)

    # Test case 2: Invalid input with different number of samples
    X_invalid = np.random.rand(n_samples + 1, X.shape[1])
    y_invalid = rng.integers(0, 2, size=n_samples + 1)
    with pytest.raises(RuntimeError, match="X must have"):
        forest_hyppo.statistic(X_invalid, y_invalid)

    # Test case 3: Invalid input with different number of features
    X_invalid = np.random.rand(X.shape[0], n_features + 1)
    with pytest.raises(RuntimeError, match="X must have"):
        forest_hyppo.statistic(X_invalid, y)

    # Test case 4: Invalid input with incorrect y type target
    y_invalid = np.random.rand(X.shape[0])
    with pytest.raises(RuntimeError, match="y must have type"):
        forest_hyppo.statistic(X, y_invalid)


@pytest.mark.parametrize("backend", ["loky", "threading"])
@pytest.mark.parametrize("n_jobs", [1, 2])
def test_parallelization(backend, n_jobs):
    """Test parallelization of training forests."""
    n_samples = 20
    n_features = 5
    X = rng.uniform(size=(n_samples, n_features))
    y = rng.integers(0, 2, size=n_samples)  # Binary classification

    def run_forest(covariate_index=None):
        clf = FeatureImportanceForestClassifier(
            estimator=HonestForestClassifier(
                n_estimators=10, random_state=seed, n_jobs=n_jobs, honest_fraction=0.2
            ),
            test_size=0.5,
        )
        pvalue = clf.test(X, y, covariate_index=[covariate_index], metric="mi")
        return pvalue

    out = Parallel(n_jobs=-1, backend=backend)(
        delayed(run_forest)(covariate_index) for covariate_index in range(n_features)
    )
    assert len(out) == n_features


def test_pickle(tmpdir):
    """Test that pickling works and preserves fitted attributes."""
    n_samples = 100
    n_features = 5
    X = rng.uniform(size=(n_samples, n_features))
    y = rng.integers(0, 2, size=n_samples)  # Binary classification
    n_repeats = 1000

    clf = FeatureImportanceForestClassifier(
        estimator=HonestForestClassifier(
            n_estimators=10, random_state=seed, n_jobs=1, honest_fraction=0.2
        ),
        test_size=0.5,
    )
    stat, pvalue = clf.test(X, y, covariate_index=[1], metric="mi", n_repeats=n_repeats)

    with open(Path(tmpdir) / "clf.pkl", "wb") as fpath:
        pickle.dump(clf, fpath)

    with open(Path(tmpdir) / "clf.pkl", "rb") as fpath:
        clf_pickle = pickle.load(fpath)

    # recompute pvalue manually and compare
    pickle_pvalue = (
        1.0 + (clf_pickle.null_dist_ <= (clf_pickle.permute_stat_ - clf_pickle.observe_stat_)).sum()
    ) / (1.0 + n_repeats)
    assert pvalue == pickle_pvalue
    assert clf_pickle.observe_stat_ == -(
        stat - clf_pickle.permute_stat_
    ), f"{clf_pickle.observe_stat_} != {-(stat - clf_pickle.permute_stat_)}"

    attr_list = [
        "test_size",
        "observe_samples_",
        "y_true_final_",
        "observe_posteriors_",
        "observe_stat_",
        "_is_fitted",
        "permute_samples_",
        "permute_posteriors_",
        "permute_stat_",
        "n_samples_test_",
        "_n_samples_",
        "_metric",
        "train_test_samples_",
    ]
    for attr in attr_list:
        assert_array_equal(getattr(clf, attr), getattr(clf_pickle, attr))


@pytest.mark.parametrize(
    "permute_forest_fraction",
    [None, 0.5],
    ids=["no_permute", "permute_forest_fraction"],
)
@pytest.mark.parametrize(
    "sample_dataset_per_tree", [True, False], ids=["sample_dataset_per_tree", "no_sample_dataset"]
)
def test_sample_size_consistency_of_estimator_indices_(
    permute_forest_fraction, sample_dataset_per_tree
):
    """Test that the test-sample indices are what is expected."""
    clf = FeatureImportanceForestClassifier(
        estimator=HonestForestClassifier(
            n_estimators=10, random_state=seed, n_jobs=1, honest_fraction=0.2
        ),
        test_size=0.5,
        permute_forest_fraction=permute_forest_fraction,
        sample_dataset_per_tree=sample_dataset_per_tree,
        stratify=False,
    )

    n_samples = 100
    n_features = 5
    X = rng.uniform(size=(n_samples, n_features))
    y = rng.integers(0, 2, size=n_samples)  # Binary classification

    _, posteriors, samples = clf.statistic(
        X, y, covariate_index=None, return_posteriors=True, metric="mi"
    )

    if sample_dataset_per_tree or permute_forest_fraction is not None:
        # check the non-nans
        non_nan_idx = _non_nan_samples(posteriors)
        if sample_dataset_per_tree:
            assert clf.n_samples_test_ == n_samples, f"{clf.n_samples_test_} != {n_samples}"

        sorted_sample_idx = sorted(np.unique(samples))
        sorted_est_samples_idx = sorted(
            np.unique(np.concatenate([x[1] for x in clf.train_test_samples_]).flatten())
        )
        assert_array_equal(sorted_sample_idx, non_nan_idx)

        # the sample indices are equal to the output of the train/test indices_
        # only if there are no nans in the posteriors over all the samples
        if np.sum(non_nan_idx) == n_samples:
            assert_array_equal(
                sorted_sample_idx,
                sorted_est_samples_idx,
                f"{set(sorted_sample_idx) - set(sorted_est_samples_idx)} and "
                f"{set(sorted_est_samples_idx) - set(sorted_sample_idx)}",
            )
    else:
        assert_array_equal(
            samples,
            sorted(clf.train_test_samples_[0][1]),
            err_msg=f"Samples {set(samples) - set(sorted(clf.train_test_samples_[0][1]))}.",
        )
    assert len(_non_nan_samples(posteriors)) == len(samples)


@pytest.mark.parametrize("sample_dataset_per_tree", [True, False])
@pytest.mark.parametrize("seed", [None, 0])
def test_sample_per_tree_samples_consistency_with_sklearnforest(seed, sample_dataset_per_tree):
    n_samples = 100
    n_features = 5
    X = rng.uniform(size=(n_samples, n_features))
    y = rng.integers(0, 2, size=n_samples)  # Binary classification
    n_estimators = 10
    clf = FeatureImportanceForestClassifier(
        estimator=HonestForestClassifier(
            n_estimators=n_estimators, random_state=seed, n_jobs=1, honest_fraction=0.2
        ),
        test_size=0.5,
        permute_forest_fraction=1.0 / n_estimators,
        sample_dataset_per_tree=sample_dataset_per_tree,
    )
    other_clf = FeatureImportanceForestClassifier(
        estimator=HonestForestClassifier(
            n_estimators=10, random_state=seed, n_jobs=1, honest_fraction=0.2
        ),
        test_size=0.5,
        permute_forest_fraction=None,
        sample_dataset_per_tree=sample_dataset_per_tree,
    )

    clf.statistic(X, y, covariate_index=None, metric="mi")
    other_clf.statistic(X, y, covariate_index=None, metric="mi")

    # estimator indices should be preserved over multiple calls
    estimator_train_test_indices = deepcopy(clf.train_test_samples_)
    for idx in range(clf.n_estimators):
        assert_array_equal(clf.train_test_samples_[idx][0], estimator_train_test_indices[idx][0])
        assert_array_equal(clf.train_test_samples_[idx][1], estimator_train_test_indices[idx][1])

    # if the sample_dataset_per_tree, then the indices should be different across all trees
    if sample_dataset_per_tree or clf.permute_forest_fraction > 0.0:
        for indices, other_indices in combinations(clf.train_test_samples_, 2):
            assert not np.array_equal(indices[0], other_indices[0])
            assert not np.array_equal(indices[1], other_indices[1])
    else:
        for indices, other_indices in combinations(clf.train_test_samples_, 2):
            assert_array_equal(indices[0], other_indices[0])
            assert_array_equal(indices[1], other_indices[1])

    # estimator indices should be preserved over multiple calls
    estimator_train_test_indices = deepcopy(other_clf.train_test_samples_)
    for idx in range(clf.n_estimators):
        assert_array_equal(
            other_clf.train_test_samples_[idx][0], estimator_train_test_indices[idx][0]
        )
        assert_array_equal(
            other_clf.train_test_samples_[idx][1], estimator_train_test_indices[idx][1]
        )

        # when seed is passed, the indices should be deterministic
        if seed is not None and sample_dataset_per_tree:
            assert_array_equal(
                clf.train_test_samples_[idx][0], other_clf.train_test_samples_[idx][0]
            )
            assert_array_equal(
                clf.train_test_samples_[idx][1], other_clf.train_test_samples_[idx][1]
            )


@pytest.mark.parametrize("seed", [None, 0])
def test_small_dataset_independent(seed):
    n_samples = 32
    n_features = 5

    X = rng.uniform(size=(n_samples, n_features))
    y = rng.integers(0, 2, size=n_samples)  # Binary classification

    clf = FeatureImportanceForestClassifier(
        estimator=HonestForestClassifier(
            n_estimators=10, random_state=seed, n_jobs=1, honest_fraction=0.5
        ),
        test_size=0.2,
        permute_forest_fraction=None,
        sample_dataset_per_tree=False,
    )
    stat, pvalue = clf.test(X, y, covariate_index=[1, 2], metric="mi")
    assert ~np.isnan(pvalue)
    assert ~np.isnan(stat)
    assert pvalue > 0.05

    stat, pvalue = clf.test(X, y, metric="mi")
    assert_almost_equal(stat, 0.0, decimal=1)
    assert pvalue > 0.05


@flaky(max_runs=3)
@pytest.mark.parametrize("seed", [None, 0])
def test_small_dataset_dependent(seed):
    n_samples = 100
    n_features = 5
    rng = np.random.default_rng(seed)

    X = rng.uniform(size=(n_samples, n_features))
    X = rng.uniform(size=(n_samples // 2, n_features))
    X2 = X + 3
    X = np.vstack([X, X2])
    y = np.vstack(
        [np.zeros((n_samples // 2, 1)), np.ones((n_samples // 2, 1))]
    )  # Binary classification

    clf = FeatureImportanceForestClassifier(
        estimator=HonestForestClassifier(
            n_estimators=50, random_state=seed, n_jobs=1, honest_fraction=0.5
        ),
        test_size=0.2,
        permute_forest_fraction=None,
        sample_dataset_per_tree=False,
    )
    stat, pvalue = clf.test(X, y, covariate_index=[1, 2], metric="mi")
    assert ~np.isnan(pvalue)
    assert ~np.isnan(stat)
    assert pvalue <= 0.05

    stat, pvalue = clf.test(X, y, metric="mi")
    assert pvalue <= 0.05


@flaky(max_runs=3)
def test_no_traintest_split():
    n_samples = 500
    n_features = 5
    rng = np.random.default_rng(seed)

    X = rng.uniform(size=(n_samples, n_features))
    X = rng.uniform(size=(n_samples // 2, n_features))
    X2 = X * 2
    X = np.vstack([X, X2])
    y = np.vstack(
        [np.zeros((n_samples // 2, 1)), np.ones((n_samples // 2, 1))]
    )  # Binary classification

    clf = FeatureImportanceForestClassifier(
        estimator=HonestForestClassifier(
            n_estimators=50,
            max_features=n_features,
            random_state=seed,
            n_jobs=1,
            honest_fraction=0.5,
        ),
        test_size=0.2,
        train_test_split=False,
        permute_forest_fraction=None,
        sample_dataset_per_tree=False,
    )
    stat, pvalue = clf.test(X, y, covariate_index=[1, 2], metric="mi")

    # since no train-test split, the training is all the data and the testing is none of the data
    assert_array_equal(clf.train_test_samples_[0][0], np.arange(n_samples))
    assert_array_equal(clf.train_test_samples_[0][1], np.array([]))

    assert ~np.isnan(pvalue)
    assert ~np.isnan(stat)
    assert pvalue <= 0.05, f"{pvalue}"

    stat, pvalue = clf.test(X, y, metric="mi")
    assert pvalue <= 0.05, f"{pvalue}"

    X = rng.uniform(size=(n_samples, n_features))
    y = rng.integers(0, 2, size=n_samples)  # Binary classification
    clf.reset()

    stat, pvalue = clf.test(X, y, metric="mi")
    assert_almost_equal(stat, 0.0, decimal=1)
    assert pvalue > 0.05, f"{pvalue}"

    stat, pvalue = clf.test(X, y, covariate_index=[1, 2], metric="mi")
    assert ~np.isnan(pvalue)
    assert ~np.isnan(stat)
    assert pvalue > 0.05, f"{pvalue}"


# @pytest.mark.parametrize("permute_forest_fraction", [1.0 / 10, 0.5, 0.75, 1.0])
# @pytest.mark.parametrize("seed", [None, 0])
# def test_permute_forest_fraction(permute_forest_fraction, seed):
#     """Test proper handling of random seeds, shuffled covariates and train/test splits."""
#     n_estimators = 10
#     clf = FeatureImportanceForestClassifier(
#         estimator=HonestForestClassifier(
#             n_estimators=n_estimators, random_state=seed, n_jobs=1, honest_fraction=0.2
#         ),
#         test_size=0.5,
#         permute_forest_fraction=permute_forest_fraction,
#         stratify=False,
#     )

#     n_samples = 100
#     n_features = 5
#     X = rng.uniform(size=(n_samples, n_features))
#     y = rng.integers(0, 2, size=n_samples)  # Binary classification

#     _ = clf.statistic(X, y, covariate_index=None, return_posteriors=True, metric="mi")

#     seed = None
#     train_test_splits = list(clf.train_test_samples_)
#     train_inds = None
#     test_inds = None
#     for idx, tree in enumerate(clf.estimator_.estimators_):
#         # All random seeds of the meta-forest should be as expected, where
#         # the seed only changes depending on permute forest fraction
#         if idx % int(permute_forest_fraction * clf.n_estimators) == 0:
#             prev_seed = seed
#             seed = clf._seeds[idx]

#             assert seed == tree.random_state
#             assert prev_seed != seed
#         else:
#             assert seed == clf._seeds[idx], f"{seed} != {clf._seeds[idx]}"
#             assert seed == clf._seeds[idx - 1]

#         # Next, train/test splits should be consistent for batches of trees
#         if idx % int(permute_forest_fraction * clf.n_estimators) == 0:
#             prev_train_inds = train_inds
#             prev_test_inds = test_inds

#             train_inds, test_inds = train_test_splits[idx]

#             assert (prev_train_inds != train_inds).any(), f"{prev_train_inds} == {train_inds}"
#             assert (prev_test_inds != test_inds).any(), f"{prev_test_inds} == {test_inds}"
#         else:
#             assert_array_equal(train_inds, train_test_splits[idx][0])
#             assert_array_equal(test_inds, train_test_splits[idx][1])


def test_comight_repeated_feature_sets():
    """Test COMIGHT when there are repeated feature sets."""
    n_samples = 100
    n_features = 500
    rng = np.random.default_rng(seed)

    X = rng.uniform(size=(n_samples, 10))
    X2 = 2 * (X + 5) + rng.standard_normal(size=(n_samples, 10))
    X = np.hstack((X, rng.standard_normal(size=(n_samples, n_features - 10))))
    X2 = np.hstack((X2, rng.standard_normal(size=(n_samples, n_features - 10))))
    X = np.vstack([X, X2])
    y = np.vstack([np.zeros((n_samples, 1)), np.ones((n_samples, 1))])  # Binary classification

    X = np.hstack((X, X))
    # feature_set_ends = [n_features, n_features * 2]
    clf = FeatureImportanceForestClassifier(
        estimator=HonestForestClassifier(
            n_estimators=50,
            random_state=seed,
            n_jobs=1,
            honest_fraction=0.5,
            max_features=0.3,
            # tree_estimator=MultiViewDecisionTreeClassifier(
            #     feature_set_ends=feature_set_ends,
            #     max_features=0.3,
            # ),
        ),
        test_size=0.2,
        permute_forest_fraction=None,
        sample_dataset_per_tree=False,
        random_state=seed,
    )

    # first test MIGHT rejects the null, since there is information
    stat, pvalue = clf.test(X, y, metric="mi")
    assert pvalue < 0.05

    # second test CoMIGHT fails to reject the null, since the information
    # is entirely contained in the first feature set
    stat, pvalue = clf.test(X, y, covariate_index=np.arange(n_features), metric="mi")
    assert pvalue > 0.05, f"{pvalue}"


def test_null_with_partial_auc():
    limit = 0.1
    max_features = "sqrt"
    n_repeats = 1000
    n_estimators = 20
    test_size = 0.2

    # Check consistency on dataset iris.
    clf = FeatureImportanceForestClassifier(
        estimator=HonestForestClassifier(
            n_estimators=n_estimators,
            max_features=max_features,
            random_state=0,
            n_jobs=1,
        ),
        test_size=test_size,
        random_state=0,
    )
    # now add completely uninformative feature
    X = np.hstack((iris_X, rng.standard_normal(size=(iris_X.shape[0], 4))))

    stat, pvalue = clf.test(
        X,
        iris_y,
        covariate_index=np.arange(2),
        n_repeats=n_repeats,
        metric="auc",
    )
    first_null_dist = deepcopy(clf.null_dist_)

    # If we re-run it with a different seed, but now specifying max_fpr
    # there should be a difference in the partial-AUC distribution
    clf = FeatureImportanceForestClassifier(
        estimator=HonestForestClassifier(
            n_estimators=n_estimators,
            max_features=max_features,
            random_state=0,
            n_jobs=1,
        ),
        test_size=test_size,
        random_state=seed,
    )
    stat, pvalue = clf.test(
        X, iris_y, covariate_index=np.arange(2), n_repeats=n_repeats, metric="auc", max_fpr=limit
    )
    second_null_dist = clf.null_dist_
    null_dist_pvalue = wilcoxon(first_null_dist, second_null_dist).pvalue
    assert null_dist_pvalue < 0.05, null_dist_pvalue
    assert pvalue > 0.05, f"pvalue: {pvalue}"


def test_build_coleman_forest():
    """Simple test for building a Coleman forest.

    Test the function under alternative and null hypothesis for a very simple dataset.
    """
    n_estimators = 40
    n_samples = 30
    n_features = 5
    rng = np.random.default_rng(seed)

    _X = rng.uniform(size=(n_samples, n_features))
    _X = rng.uniform(size=(n_samples // 2, n_features))
    X2 = _X + 3
    X = np.vstack([_X, X2])
    y = np.vstack(
        [np.zeros((n_samples // 2, 1)), np.ones((n_samples // 2, 1))]
    )  # Binary classification

    clf = HonestForestClassifier(
        n_estimators=n_estimators,
        random_state=seed,
        n_jobs=-1,
        honest_fraction=0.5,
        bootstrap=True,
        max_samples=1.6,
    )
    perm_clf = PermutationHonestForestClassifier(
        n_estimators=n_estimators,
        random_state=seed,
        n_jobs=-1,
        honest_fraction=0.5,
        bootstrap=True,
        max_samples=1.6,
    )
    with pytest.raises(
        RuntimeError, match="Permutation forest must be a PermutationHonestForestClassifier"
    ):
        build_coleman_forest(clf, clf, X, y)

    forest_result, orig_forest_proba, perm_forest_proba, clf_fitted, perm_clf_fitted = (
        build_coleman_forest(clf, perm_clf, X, y, metric="s@98", n_repeats=1000, seed=seed)
    )
    assert clf_fitted._n_samples_bootstrap == round(n_samples * 1.6)
    assert perm_clf_fitted._n_samples_bootstrap == round(n_samples * 1.6)
    assert_array_equal(perm_clf_fitted.permutation_indices_.shape, (n_samples, 1))

    assert forest_result.pvalue <= 0.05, f"{forest_result.pvalue}"
    assert forest_result.observe_stat > 0.1, f"{forest_result.observe_stat}"
    assert_array_equal(orig_forest_proba.shape, perm_forest_proba.shape)

    X = np.vstack([_X, _X])
    forest_result, _, _, clf_fitted, perm_clf_fitted = build_coleman_forest(
        clf, perm_clf, X, y, metric="s@98"
    )
    assert forest_result.pvalue > 0.05, f"{forest_result.pvalue}"
    assert forest_result.observe_stat < 0.05, f"{forest_result.observe_stat}"


def test_build_coleman_forest_multiview():
    """Simple test for building a Coleman forest.

    Test the function under alternative and null hypothesis for a very simple dataset.
    """
    n_estimators = 40
    n_samples = 30
    n_features = 5
    rng = np.random.default_rng(seed)

    _X = rng.uniform(size=(n_samples, n_features))
    _X = rng.uniform(size=(n_samples // 2, n_features))
    X2 = _X + 3
    X = np.vstack([_X, X2])
    y = np.vstack(
        [np.zeros((n_samples // 2, 1)), np.ones((n_samples // 2, 1))]
    )  # Binary classification

    clf = HonestForestClassifier(
        n_estimators=n_estimators,
        random_state=seed,
        n_jobs=-1,
        honest_fraction=0.5,
        bootstrap=True,
        max_samples=1.6,
        max_features=[1, 1],
        tree_estimator=MultiViewDecisionTreeClassifier(),
        feature_set_ends=[2, 5],
    )
    perm_clf = PermutationHonestForestClassifier(
        n_estimators=n_estimators,
        random_state=seed,
        n_jobs=-1,
        honest_fraction=0.5,
        bootstrap=True,
        max_samples=1.6,
        max_features=[1, 1],
        tree_estimator=MultiViewDecisionTreeClassifier(),
        feature_set_ends=[2, 5],
    )
    with pytest.raises(
        RuntimeError, match="Permutation forest must be a PermutationHonestForestClassifier"
    ):
        build_coleman_forest(clf, clf, X, y)

    forest_result, orig_forest_proba, perm_forest_proba, clf_fitted, perm_clf_fitted = (
        build_coleman_forest(clf, perm_clf, X, y, metric="s@98", n_repeats=1000, seed=seed)
    )
    assert clf_fitted._n_samples_bootstrap == round(n_samples * 1.6)
    assert perm_clf_fitted._n_samples_bootstrap == round(n_samples * 1.6)
    assert_array_equal(perm_clf_fitted.permutation_indices_.shape, (n_samples, 1))

    assert forest_result.pvalue <= 0.05, f"{forest_result.pvalue}"
    assert forest_result.observe_stat > 0.1, f"{forest_result.observe_stat}"
    assert_array_equal(orig_forest_proba.shape, perm_forest_proba.shape)

    X = np.vstack([_X, _X])
    forest_result, _, _, clf_fitted, perm_clf_fitted = build_coleman_forest(
        clf, perm_clf, X, y, metric="s@98"
    )
    assert forest_result.pvalue > 0.05, f"{forest_result.pvalue}"


def test_build_permutation_forest():
    """Simple test for building a permutation forest."""
    n_estimators = 30
    n_samples = 100
    n_features = 3
    rng = np.random.default_rng(seed)

    _X = rng.uniform(size=(n_samples, n_features))
    _X = rng.uniform(size=(n_samples // 2, n_features))
    X2 = _X + 10
    X = np.vstack([_X, X2])
    y = np.vstack(
        [np.zeros((n_samples // 2, 1)), np.ones((n_samples // 2, 1))]
    )  # Binary classification

    clf = HonestForestClassifier(
        n_estimators=n_estimators, random_state=seed, n_jobs=-1, honest_fraction=0.5, bootstrap=True
    )
    perm_clf = PermutationHonestForestClassifier(
        n_estimators=n_estimators, random_state=seed, n_jobs=-1, honest_fraction=0.5, bootstrap=True
    )
    with pytest.raises(
        RuntimeError, match="Permutation forest must be a PermutationHonestForestClassifier"
    ):
        build_permutation_forest(clf, clf, X, y, seed=seed)

    forest_result, orig_forest_proba, perm_forest_proba = build_permutation_forest(
        clf, perm_clf, X, y, metric="s@98", n_repeats=20, seed=seed
    )
    assert forest_result.observe_test_stat > 0.1, f"{forest_result.observe_stat}"
    assert forest_result.pvalue <= 0.05, f"{forest_result.pvalue}"
    assert_array_equal(orig_forest_proba.shape, perm_forest_proba.shape)

    X = np.vstack([_X, _X])
    forest_result, _, _ = build_permutation_forest(
        clf, perm_clf, X, y, metric="s@98", n_repeats=10, seed=seed
    )
    assert forest_result.pvalue > 0.05, f"{forest_result.pvalue}"
    assert forest_result.observe_test_stat < 0.05, f"{forest_result.observe_test_stat}"
