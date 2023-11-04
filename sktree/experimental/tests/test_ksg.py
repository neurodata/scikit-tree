import itertools

import numpy as np

from sktree.experimental.ksg import entropy_continuous
from sktree.experimental.mutual_info import entropy_gaussian

seed = 12345

rng = np.random.default_rng(seed)


def get_mvn_data(total_rvs, dimensionality=2, scale_sigma_offdiagonal_by=1.0, total_samples=1000):
    data_space_size = total_rvs * dimensionality

    # initialise distribution
    mu = rng.standard_normal(data_space_size)
    sigma = rng.rand(data_space_size, data_space_size)

    # ensures that sigma is positive semi-definite
    sigma = np.dot(sigma.transpose(), sigma)

    # scale off-diagonal entries -- might want to change that to block diagonal entries
    # diag = np.diag(sigma).copy()
    # sigma *= scale_sigma_offdiagonal_by
    # sigma[np.diag_indices(len(diag))] = diag

    # scale off-block diagonal entries
    d = dimensionality
    for ii, jj in itertools.product(list(range(total_rvs)), repeat=2):
        if ii != jj:
            sigma[d * ii : d * (ii + 1), d * jj : d * (jj + 1)] *= scale_sigma_offdiagonal_by

    # get samples
    samples = rng.multivariate_normal(mu, sigma, size=total_samples)

    return [samples[:, ii * d : (ii + 1) * d] for ii in range(total_rvs)]


def test_get_h_1d(k=5, norm="max"):
    X = rng.standard_normal(size=(1000, 1))
    cov_X = np.atleast_2d(np.cov(X.T))

    analytic = entropy_gaussian(cov_X)
    kozachenko = entropy_continuous(X, k=k, norm=norm)

    print("analytic result: {:.5f}".format(analytic))
    print("K-L estimator:   {:.5f}".format(kozachenko))
    assert np.isclose(
        analytic, kozachenko, rtol=0.1, atol=0.1
    ), "K-L estimate strongly differs from analytic expectation!"
