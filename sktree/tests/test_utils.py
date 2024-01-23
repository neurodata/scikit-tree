import numpy as np
import pytest
from sklearn import datasets

from sklearn.ensemble import RandomForestClassifier

from ..stats.utils import _SAS98, compute_posterior_metrics

# set the constant seed for reproducibility
SEED = 123
np.random.seed(SEED)
# generate toy samples of posteriors, targets
n_samples = 100
posteriors = np.random.randn(n_samples, 3)
n_samples = 100
posteriors = np.random.uniform(0, 1, size=n_samples)
y_true = np.array([posteriors >= 0.5]).astype(int).ravel()

_SAS98(posteriors, y_true)


def test_sas98():
    # test that the function returns 0
    # when all posteriors are 0.5
    y_true = [1, 1, 0, 0]
    y_pred_proba = np.repeat(0.5, 4)
    t = _SAS98(y_true, y_pred_proba, max_fpr=0.02)
    assert t == 0
    # test that the function returns 1
    # when all posteriors are correct
    y_pred_proba = [0.9, 0.9, 0.1, 0.1]
    t = _SAS98(y_true, y_pred_proba, max_fpr=0.02)
    assert t == 1


def test_compute_posterior_metrics():
    # generate toy samples of posteriors, targets
    n_estimators = 10
    n_samples = 100
    N = n_samples + 100
    n_class = 1
    posteriors = np.random.uniform(0, 1, size=(n_estimators, N, n_class))
    # posteriors = np.random.uniform(0, 1, size=n_samples).reshape((1, n_samples,1))
    test_idx = list(range(n_samples))
    y_true = (
        np.random.binomial(1, posteriors[0, test_idx], size=(n_samples, n_class))
        .astype(int)
        .ravel()
    )
    # test that the function returns 0
    # when all posteriors are 0.5
    y_true = [1, 1, 0, 0]
    y_pred_proba = np.repeat(0.5, 4)
    t = compute_posterior_metrics(y_true, y_pred_proba, test_idx, metric="auc", max_fpr=0.1)
    assert t == 0
    # test that the function returns 1
    # when all posteriors are correct
    y_pred_proba = [0.9, 0.9, 0.1, 0.1]
    t = compute_posterior_metrics(y_true, y_pred_proba, test_idx, metric="auc", max_fpr=0.1)
    assert t == 1
    # test that the function returns 0
    # when all posteriors are 0.5
    y_true = [1, 1, 0, 0]
    y_pred_proba = np.repeat(0.5, 4)
    t = compute_posterior_metrics(y_true, y_pred_proba, test_idx, metric="sas98", max_fpr=0.02)
    assert t == 0
    # test that the function returns 1
    # when all posteriors are correct
    y_pred_proba = [0.9, 0.9, 0.1, 0.1]
    t = compute_posterior_metrics(y_true, y_pred_proba, test_idx, metric="sas98", max_fpr=0.02)
    assert t == 1
