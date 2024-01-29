import numpy as np
from scipy.integrate import nquad
from scipy.stats import entropy, multivariate_normal


def make_bimodal_classification(
    n_samples: int,
    n_dim: int = 10,
    mix: float = 0.5,
    mu_F: float = 0.0,
    mu_G: float = 1.0,
    sigma_F: float = 1.0,
    sigma_G: float = (2 / 3) ** 2,
    return_params: bool = False,
    seed=None,
):
    """Simulate classification data from a bimodal model.

    Bimodal is defined as two Gaussians with different means and covariances.
    $$\frac{1}{2} \mathcal{N}(\mu_0, \Sigma_0*I) + \frac{1}{2} \mathcal{N}(\mu_1, \Sigma_1*I)$$

    Parameters
    ----------
    n_samples : int
        The number of samples to generate.
    n_dim : int, optional
        The number of dimensions in the dataset, by default 10.
    mix : float, optional
        The mixture parameter, by default 0.5.
    mu_F : float, optional
        The mean of the first Gaussian, by default 0.0.
    mu_G : float, optional
        The mean of the second Gaussian, by default 1.0.
    sigma_F : float, optional
        The variance of the first Gaussian, by default 1.0.
    sigma_G : float, optional
        The variance of the second Gaussian when the class label is 0, by default (2/3)**2.
    return_params : bool, optional
        Whether or not to return the distribution parameters of the classes normal distributions.
    seed : int, optional
        Random seed, by default None.

    Returns
    -------
    X : np.ndarray of shape (n_samples, n_dim), dtype=np.float64
        First half from the Normal and the second half from the Bimodal distribution.
    y : np.ndarray of shape (n_samples,), dtype=np.intp
        Labels of the dataset.
    """
    rng = np.random.default_rng(seed=seed)

    mu_F = np.zeros(n_dim)
    mu_G0 = mu_G * np.array([1 / np.sqrt(i) for i in range(1, n_dim + 1)])
    mu_G1 = -mu_G0

    cov_F = sigma_F * np.identity(n_dim)
    cov_G = sigma_G * np.identity(n_dim)

    if mix < 0 or mix > 1:
        raise ValueError("Mix must be between 0 and 1.")

    if n_dim > 1000:
        method = "cholesky"
    else:
        method = "svd"

    F = rng.multivariate_normal(mu_F, cov_F, n_samples // 2, method=method)

    # generate a binomial random variable to determine which class to sample from
    # of size n_samples // 2
    binom = rng.binomial(1, mix, size=n_samples // 2)
    # mix the Gaussians together to generate the data from the two classes
    # if binom is 0, then sample from G0, otherwise sample from G1
    # create a new array to store the data
    G = np.zeros((n_samples // 2, n_dim))
    for i, z in enumerate(binom):
        if z == 0:
            G[i] = rng.multivariate_normal(mu_G0, cov_G, 1, method="svd")
        else:
            G[i] = rng.multivariate_normal(mu_G1, cov_G, 1, method="svd")
    # concatenate the two arrays together
    X = np.vstack((F, G))
    y = np.concatenate((np.zeros(n_samples // 2), np.ones(n_samples // 2)))

    if return_params:
        return X, y, (mu_F, [mu_G0, mu_G1]), (cov_F, [cov_G, cov_G])
    return X, y


def make_quadratic_classification(n_samples: int, n_features: int, noise=False, seed=None):
    """Simulate classification data from a quadratic model.

    This is a form of the simulation used in :footcite:`panda2018learning`.

    Parameters
    ----------
    n_samples : int
        The number of samples to generate.
    n_features : int
        The number of dimensions in the dataset.
    noise : bool, optional
        Whether or not to add noise, by default False.
    seed : int, optional
        Random seed, by default None.

    Returns
    -------
    x : array-like, shape (2 * n_samples, n_features)
        Data array.
    v : array-like, shape (2 * n_samples,)
        Target array of 1's and 0's.

    References
    ----------
    .. footbibliography::
    """
    rng = np.random.default_rng(seed)

    x = rng.standard_normal(size=(n_samples, n_features))
    coeffs = np.array([np.exp(-0.0325 * (i + 24)) for i in range(n_features)])
    eps = rng.standard_normal(size=(n_samples, n_features))

    x_coeffs = x * coeffs
    y = x_coeffs**2 + noise * eps

    # generate the classification labels
    n1 = x.shape[0]
    n2 = y.shape[0]
    v = np.vstack([np.zeros((n1, 1)), np.ones((n2, 1))])
    x = np.vstack((x, y))
    return x, v


def make_trunk_classification(
    n_samples,
    n_dim=10,
    n_informative=10,
    m_factor: int = -1,
    rho: int = 0,
    band_type: str = "ma",
    return_params: bool = False,
    mix: int = 0,
    seed=None,
):
    """Generate trunk dataset.

    For each dimension in the first distribution, there is a mean of :math:`1 / d`, where
    ``d`` is the dimensionality. The covariance is the identity matrix.
    The second distribution has a mean vector that is the negative of the first.
    As ``d`` increases, the two distributions become closer and closer.

    See full details in :footcite:`trunk1982`.

    Instead of the identity covariance matrix, one can implement a banded covariance matrix
    that follows :footcite:`Bickel_2008`.

    Parameters
    ----------
    n_samples : int
        Number of sample to generate.
    n_dim : int, optional
        The dimensionality of the dataset and the number of
        unique labels, by default 10.
    n_informative : int, optional
        The informative dimensions. All others for ``n_dim - n_informative``
        are uniform noise.
    m_factor : int, optional
        The multiplicative factor to apply to the mean-vector of the first
        distribution to obtain the mean-vector of the second distribution.
        By default -1.
    rho : float, optional
        The covariance value of the bands. By default 0 indicating, an identity matrix is used.
    band_type : str
        The band type to use. For details, see Example 1 and 2 in :footcite:`Bickel_2008`.
        Either 'ma', or 'ar'.
    return_params : bool, optional
        Whether or not to return the distribution parameters of the classes normal distributions.
    mix : float, optional
        Whether or not to mix the Gaussians. Should be a value between 0 and 1.
    seed : int, optional
        Random seed, by default None.

    Returns
    -------
    X : np.ndarray of shape (n_samples, n_dim), dtype=np.float64
        Trunk dataset as a dense array.
    y : np.ndarray of shape (n_samples,), dtype=np.intp
        Labels of the dataset.
    means : list of ArrayLike of shape (n_dim,), dtype=np.float64
        The mean vector for each class starting with class 0.
        Returned if ``return_params`` is True.
    covs : list of ArrayLike of shape (n_dim, n_dim), dtype=np.float64
        The covariance for each class. Returned if ``return_params`` is True.

    References
    ----------
    .. footbibliography::
    """
    if n_dim < n_informative:
        raise ValueError(
            f"Number of informative dimensions {n_informative} must be less than number "
            f"of dimensions, {n_dim}"
        )
    rng = np.random.default_rng(seed=seed)

    mu_1 = np.array([1 / np.sqrt(i) for i in range(1, n_informative + 1)])
    mu_0 = m_factor * mu_1

    if rho != 0:
        if band_type == "ma":
            cov = _moving_avg_cov(n_informative, rho)
        elif band_type == "ar":
            cov = _autoregressive_cov(n_informative, rho)
        else:
            raise ValueError(f'Band type {band_type} must be one of "ma", or "ar".')
    else:
        cov = np.identity(n_informative)

    if mix < 0 or mix > 1:
        raise ValueError("Mix must be between 0 and 1.")

    if n_informative > 1000:
        method = "cholesky"
    else:
        method = "svd"

    if mix == 0:
        X = np.vstack(
            (
                rng.multivariate_normal(mu_0, cov, n_samples // 2, method=method),
                rng.multivariate_normal(mu_1, cov, n_samples // 2, method=method),
            )
        )
    else:
        mixture_idx = rng.choice(
            [0, 1], n_samples // 2, replace=True, shuffle=True, p=[mix, 1 - mix]
        )
        X_mixture = np.zeros((n_samples // 2, len(mu_1)))
        for idx in range(n_samples // 2):
            if mixture_idx[idx] == 1:
                X_sample = rng.multivariate_normal(mu_1, cov, 1, method=method)
            else:
                X_sample = rng.multivariate_normal(mu_0, cov, 1, method=method)
            X_mixture[idx, :] = X_sample

        X = np.vstack(
            (
                rng.multivariate_normal(
                    np.zeros(n_informative), cov, n_samples // 2, method=method
                ),
                X_mixture,
            )
        )

    if n_dim > n_informative:
        X = np.hstack((X, rng.uniform(low=0, high=1, size=(n_samples, n_dim - n_informative))))

    y = np.concatenate((np.zeros(n_samples // 2), np.ones(n_samples // 2)))

    if return_params:
        return X, y, [mu_0, mu_1], [cov, cov]
    return X, y


def _moving_avg_cov(n_dim, rho):
    # Create a meshgrid of indices
    i, j = np.meshgrid(np.arange(1, n_dim + 1), np.arange(1, n_dim + 1), indexing="ij")

    # Calculate the covariance matrix using the corrected formula
    cov_matrix = rho ** np.abs(i - j)

    # Apply the banding condition
    cov_matrix[abs(i - j) > 1] = 0
    return cov_matrix


def _autoregressive_cov(n_dim, rho):
    # Create a meshgrid of indices
    i, j = np.meshgrid(np.arange(1, n_dim + 1), np.arange(1, n_dim + 1), indexing="ij")

    # Calculate the covariance matrix using the corrected formula
    cov_matrix = rho ** np.abs(i - j)

    return cov_matrix


def approximate_clf_mutual_information(
    means, covs, class_probs=[0.5, 0.5], base=np.exp(1), seed=None
):
    """Approximate MI for multivariate Gaussian for a classification setting.

    Parameters
    ----------
    means : list of ArrayLike of shape (n_dim,)
        A list of means to sample from for each class.
    covs : list of ArrayLike of shape (n_dim, n_dim)
        A list of covariances to sample from for each class.
    class_probs : list, optional
        List of class probabilities, by default [0.5, 0.5] for
        balanced binary classification.
    base : float, optional
        The bit base to use, by default np.exp(1) for natural logarithm.
    seed : int, optional
        Random seed for the multivariate normal, by default None.

    Returns
    -------
    I_XY : float
        Estimated mutual information.
    H_X : float
        Estimated entropy of X, the mixture of multivariate Gaussians.
    H_XY : float
        The conditional entropy of X given Y.
    int_err : float
        The integration error for ``H_X``.
    """
    # this implicitly assumes that the signal of interest is between -10 and 10
    scale = 10
    n_dims = [cov.shape[1] for cov in covs]
    lims = [[-scale, scale]] * max(n_dims)

    # Compute entropy and X and Y.
    def func(*args):
        x = np.array(args)
        p = 0
        for k in range(len(means)):
            p += class_probs[k] * multivariate_normal.pdf(x, means[k], covs[k])
        return -p * np.log(p) / np.log(base)

    # numerically integrate H(X)
    # opts = dict(limit=1000)
    H_X, int_err = nquad(func, lims)

    # Compute MI.
    H_XY = 0
    for k in range(len(means)):
        H_XY += (
            class_probs[k]
            * (n_dims[k] * np.log(2 * np.pi) + np.log(np.linalg.det(covs[k])) + n_dims[k])
            / (2 * np.log(base))
        )
    I_XY = H_X - H_XY
    return I_XY, H_X, H_XY, int_err


def approximate_clf_mutual_information_with_monte_carlo(
    means, covs, n_samples=100_000, class_probs=[0.5, 0.5], base=np.exp(1), seed=None
):
    """Approximate MI for multivariate Gaussian for a classification setting.

    Parameters
    ----------
    means : list of ArrayLike of shape (n_dim,)
        A list of means to sample from for each class.
    covs : list of ArrayLike of shape (n_dim, n_dim)
        A list of covariances to sample from for each class.
    n_samples : int
        The total number of simulation samples
    class_probs : list, optional
        List of class probabilities, by default [0.5, 0.5] for
        balanced binary classification.
    base : float, optional
        The bit base to use, by default np.exp(1) for natural logarithm.
    seed : int, optional
        Random seed for the multivariate normal, by default None.

    Returns
    -------
    I_XY : float
        Estimated mutual information.
    H_Y : float
        Estimated entropy of Y, the class labels.
    H_Y_on_X : float
        The conditional entropy of Y given X.
    """
    rng = np.random.default_rng(seed=seed)
    P_Y = class_probs

    # Generate samples
    pdf_class = []
    X = []
    for i in range(len(means)):
        pdf_class.append(multivariate_normal(means[i], covs[i], allow_singular=True))
        X.append(
            rng.multivariate_normal(means[i], covs[i], size=int(n_samples * P_Y[i])).reshape(-1, 1)
        )

    X = np.vstack(X)

    # Calculate P(X) by law of total probability
    P_X_l = []
    P_X_on_Y = []
    for i in range(len(means)):
        P_X_on_Y.append(pdf_class[i].pdf(X))
        P_X_l.append(P_X_on_Y[-1] * P_Y[i])
    P_X = sum(P_X_l)

    # Calculate P(Y|X) by Bayes' theorem
    P_Y_on_X = []
    for i in range(len(means)):
        P_Y_on_X.append((P_X_on_Y[i] * P_Y[i] / P_X).reshape(-1, 1))

    P_Y_on_X = np.hstack(P_Y_on_X)
    P_Y_on_X = P_Y_on_X[~np.isnan(P_Y_on_X)].reshape(-1, 2)

    # Calculate the entropy of Y by class counts
    H_Y = entropy(P_Y, base=base)

    # Calculate the conditional entropy of Y on X
    H_Y_on_X = np.mean(entropy(P_Y_on_X, base=base, axis=1))

    MI = H_Y - H_Y_on_X
    return MI, H_Y, H_Y_on_X


def _compute_mi_bounds(means, covs, class_probs):
    # compute bounds using https://arxiv.org/pdf/2101.11670.pdf
    prior_y = np.array(class_probs)
    H_Y = entropy(prior_y, base=np.exp(1))

    # number of mixtures
    N = len(class_probs)

    cov = covs[0]

    H_YX_lb = 0.0
    H_YX_ub = 0.0
    for idx in range(N):
        num = 0.0
        denom = 0.0
        mean_i = means[idx]

        for jdx in range(N):
            mean_j = means[jdx]
            c_alpha, kl_div = _compute_c_alpha_and_kl(mean_i, mean_j, cov, alpha=0.5)
            num += class_probs[jdx] * np.exp(-c_alpha)

        for kdx in range(N):
            mean_k = means[kdx]
            c_alpha, kl_div = _compute_c_alpha_and_kl(mean_i, mean_k, cov, alpha=0.5)
            denom += class_probs[jdx] * np.exp(-kl_div)
        H_YX_lb += class_probs[idx] * np.log(num / denom)
        H_YX_ub += class_probs[idx] * np.log(denom / num)
    I_lb = H_Y - H_YX_lb
    I_ub = H_Y - H_YX_ub
    return I_lb, I_ub


def _compute_c_alpha_and_kl(mean_i, mean_j, cov, alpha=0.5):
    mean_ij = mean_i - mean_j
    lambda_ij = mean_ij.T @ np.linalg.inv(cov) @ mean_ij
    return alpha * (1.0 - alpha) * lambda_ij / 2.0, lambda_ij / 2
