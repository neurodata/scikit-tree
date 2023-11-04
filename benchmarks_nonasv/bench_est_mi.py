# Reimplementation of Figure 4 from Uncertainty Forests

import copy
import pickle

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from joblib import Parallel, delayed
from scipy.integrate import nquad
from scipy.stats import entropy, multivariate_normal
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier

from sktree import HonestForestClassifier
from sktree.experimental.simulate import simulate_separate_gaussians
from sktree.tree import ObliqueDecisionTreeClassifier


def plot_setting(X, y, name, ax):
    colors = ["#c51b7d", "#2166ac", "#d95f02"]
    ax.scatter(X[:, 0], X[:, 1], color = np.array(colors)[y], marker = ".")
    
    ax.set_xlim(left = -5.05)
    ax.set_xlim(right = 5.05)
    ax.set_ylabel(name)


def plot_example_2D_gaussians():
    names = ['Spherical Gaussians', 'Elliptical Gaussians', 'Three Class Gaussians']
    fig, axes = plt.subplots(1, len(names), figsize = (18,4))

    mean = 3 if name == 'Three Class Gaussians' else 1
    X, y = simulate_separate_gaussians(n_samples=n, n_dims=2, **setting['kwargs'], mu1 = mean)
    n_samples = 2000
    n_dims = 2
    X, y, means, sigmas, pi = simulate_separate_gaussians(
        n_dims=n_dims, n_samples=n_samples, n_classes=n_classes, seed=seed
    )

if __name__ == '__main__':
    n_jobs = -1
    n_estimators = 100
    feature_combinations = 2.0
    n_nbrs = 5
    seed = 12345

    # hyperparameters of the simulation
    n_samples = 1000
    n_noise_dims = 20
    alpha = 0.001
    n_classes = 2

    # dimensionality of mvg
    n_dims = 3

    # simulate separated multivariate Gaussians
    X, y, means, sigmas, pi = simulate_separate_gaussians(
        n_dims=n_dims, n_samples=n_samples, n_classes=n_classes, seed=seed
    )

    print(X.shape, y.shape)

    # Plot data.
    fig, axes = plt.subplots(1, len(settings), figsize = (18,4))
    for i, setting in enumerate(settings):
        plot_setting(2000, setting, axes[i])
        
    plt.show()
    plt.clf()


    