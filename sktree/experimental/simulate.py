import numpy as np
import scipy.linalg
import scipy.special
import scipy.stats
from scipy.integrate import nquad
from scipy.stats import multivariate_normal


def simulate_helix(
    radius_a=0,
    radius_b=1,
    obs_noise_func=None,
    nature_noise_func=None,
    alpha=0.005,
    n_samples=1000,
    return_mi_lb=False,
    random_seed=None,
):
    """Simulate data from a helix.

    Parameters
    ----------
    radius_a : int, optional
        The value of the smallest radius, by default 0.0.
    radius_b : int, optional
        The value of the largest radius, by default 1.0
    obs_noise_func : Callable, optional
        By default None, which defaults to a Uniform distribution from
        (-0.005, 0.005). If passed in, then must be a callable that when
        called returns a random number denoting the noise.
    nature_noise_func : callable, optional
        By defauult None, which will add no noise. The nature noise func
        is just an independent noise term added to ``P`` before it is
        passed to the generation of the X, Y, and Z terms.
    alpha : float, optional
        The value of the noise, by default 0.005.
    n_samples : int, optional
        Number of samples to generate, by default 1000.
    return_mi_lb : bool, optional
        Whether to return the mutual information lower bound, by default False.
    random_seed : int, optional
        The random seed.

    Returns
    -------
    P : array-like of shape (n_samples,)
        The sampled P.
    X : array-like of shape (n_samples,)
        The X dimension.
    Y : array-like of shape (n_samples,)
        The X dimension.
    Z : array-like of shape (n_samples,)
        The X dimension.
    lb : float
        The mutual information lower bound.

    Notes
    -----
    Data is generated as follows: We first sample a radius that
    defines the helix, :math:`R \\approx Unif(radius_a, radius_b)`.
    Afterwards, we generate one sample as follows::

        P = 5\\pi + 3\\pi R
        X = (P + \\epsilon_1) cos(P + \\epsilon_1) / 8\\pi + N_1
        Y = (P + \\epsilon_2) sin(P + \\epsilon_2) / 8\\pi + N_2
        Z = (P + \\epsilon_3) / 8\\pi + N_3

    where :math:`N_1,N_2,N_3` are noise variables that are independently
    sampled for each sample point. And
    :math:`\\epsilon_1, \\epsilon_2, \\epsilon_3` are "nature noise" terms
    which are off by default. This process is repeated ``n_samples`` times.

    Note, that this forms the graphical model::

        R \\rightarrow P

        P \\rightarrow X
        P \\rightarrow Y
        P \\rightarrow Z

    such that P is a confounder among X, Y and Z. This implies that X, Y and Z
    are conditionally dependent on P, whereas
    """
    rng = np.random.default_rng(random_seed)

    Radii = np.zeros((n_samples,))
    P_arr = np.zeros((n_samples,))
    X_arr = np.zeros((n_samples,))
    Y_arr = np.zeros((n_samples,))
    Z_arr = np.zeros((n_samples,))

    if obs_noise_func is None:
        obs_noise_func = lambda: rng.uniform(-alpha, alpha)
    if nature_noise_func is None:
        nature_noise_func = lambda: 0.0

    for idx in range(n_samples):
        Radii[idx] = rng.uniform(radius_a, radius_b)
        P_arr[idx] = 5 * np.pi + 3 * np.pi * Radii[idx]
        X_arr[idx] = (P_arr[idx] + nature_noise_func()) * np.cos(
            P_arr[idx] + nature_noise_func()
        ) / (8 * np.pi) + obs_noise_func()
        Y_arr[idx] = (P_arr[idx] + nature_noise_func()) * np.sin(
            P_arr[idx] + nature_noise_func()
        ) / (8 * np.pi) + obs_noise_func()
        Z_arr[idx] = (P_arr[idx] + nature_noise_func()) / (8 * np.pi) + obs_noise_func()

    if return_mi_lb:
        lb = alpha / 2 - np.log(alpha)
        return P_arr, X_arr, Y_arr, Z_arr, lb

    return P_arr, X_arr, Y_arr, Z_arr


def simulate_sphere(
    radius=1, noise_func=None, alpha=0.005, n_samples=1000, return_mi_lb=False, random_seed=None
):
    """Simulate samples generated on a sphere.

    Parameters
    ----------
    radius : int, optional
        The radius of the sphere, by default 1.
    noise_func : callable, optional
        The noise function to call to add to samples, by default None,
        which defaults to sampling from the uniform distribution [-alpha, alpha].
    alpha : float, optional
        The value of the noise, by default 0.005.
    n_samples : int, optional
        Number of samples to generate, by default 1000.
    return_mi_lb : bool, optional
        Whether to return the mutual information lower bound, by default False.
    random_seed : int, optional
        Random seed, by default None.

    Returns
    -------
    latitude : float
        Latitude.
    longitude : float
        Longitude.
    Y1 : array-like of shape (n_samples,)
        The X coordinate.
    Y2 : array-like of shape (n_samples,)
        The Y coordinate.
    Y3 : array-like of shape (n_samples,)
        The Z coordinate.
    lb : float
        The mutual information lower bound.
    """
    rng = np.random.default_rng(random_seed)
    if noise_func is None:
        noise_func = lambda: rng.uniform(-alpha, alpha)

    latitude = np.zeros((n_samples,))
    longitude = np.zeros((n_samples,))
    Y1 = np.zeros((n_samples,))
    Y2 = np.zeros((n_samples,))
    Y3 = np.zeros((n_samples,))

    # sample latitude and longitude
    for idx in range(n_samples):
        # sample latitude and longitude
        latitude[idx] = rng.uniform(0, 2 * np.pi)
        longitude[idx] = np.arccos(1 - 2 * rng.uniform(0, 1))

        Y1[idx] = np.sin(longitude[idx]) * np.cos(latitude[idx]) * radius + noise_func()
        Y2[idx] = np.sin(longitude[idx]) * np.sin(latitude[idx]) * radius + noise_func()
        Y3[idx] = np.cos(longitude[idx]) * radius + noise_func()

    if return_mi_lb:
        lb = alpha / 2 - np.log(alpha)
        return latitude, longitude, Y1, Y2, Y3, lb

    return latitude, longitude, Y1, Y2, Y3


def simulate_multivariate_gaussian(mean=None, cov=None, d=2, n_samples=1000, seed=1234):
    """Multivariate gaussian simulation for testing entropy and MI estimators.

    Simulates samples from a "known" multivariate gaussian distribution
    and then passes those samples, along with the true analytical MI/CMI.

    Parameters
    ----------
    mean : array-like of shape (n_features,)
        The optional mean array. If None (default), a random standard normal vector is drawn.
    cov : array-like of shape (n_features, n_features)
        The covariance array. If None (default), a random standard normal 2D array is drawn.
        It is then converted to a PD matrix.
    d : int
        The dimensionality of the multivariate gaussian. By default 2.
    n_samples : int
        The number of samples to generate. By default 1000.
    seed : int
        The random seed to feed to :func:`numpy.random.default_rng`.

    Returns
    -------
    data : array-like of shape (n_samples, n_features)
        The generated data from the distribution.
    mean : array-like of shape (n_features,)
        The mean vector of the distribution.
    cov : array-like of shape (n_features, n_features)
        The covariance matrix of the distribution.
    """
    rng = np.random.default_rng(seed)

    if mean is None:
        mean = rng.normal(size=(d,))
    if cov is None:
        # generate random covariance matrix and enure it is symmetric and positive-definite
        cov = rng.normal(size=(d, d))
        cov = 0.5 * (cov.dot(cov.T))
    if not np.all(np.linalg.eigvals(cov) > 0):
        raise RuntimeError("Passed in covariance matrix should be positive definite")
    if not scipy.linalg.issymmetric(cov):
        raise RuntimeError(f"Passed in covariance matrix {cov} should be symmetric")

    if len(mean) != d or len(cov) != d:
        raise RuntimeError(f"Dimensionality of mean and covariance matrix should match {d}")

    data = rng.multivariate_normal(mean=mean, cov=cov, size=(n_samples))

    return data, mean, cov


def embed_high_dims(data, n_dims=50, random_state=None):
    rng = np.random.default_rng(random_state)

    new_data = np.zeros((data.shape[0], n_dims + data.shape[1]))
    new_data[:, : data.shape[1]] = data

    for idim in range(n_dims):
        new_col = rng.standard_normal(size=(data.shape[0],))
        new_data[:, data.shape[1] + idim] = new_col

    return new_data


def simulate_separate_gaussians(
    n_dims=2,
    n_samples=1000,
    n_classes=2,
    mean1: float = None,
    var1: float = None,
    pi=None,
    seed=None,
):
    """Simulate data from separate multivariate Gaussians.

    Parameters
    ----------
    n_dims : int
        The dimensionality of the data. The default is 2.
    n_samples : int
        The number of samples to generate. The default is 1000.
    n_classes : int
        The number of classes to generate. The default is 2.
    mean1 : float
        The mean of the first dimension of the first class. If None (default), then a random
        standard normal vector is drawn.
    var1 : float
        The covariance matrix of the first class. If None (default), then a random standard
        normal 2D array is drawn. It is then converted to a PD matrix.
    pi : array-like of shape (n_classes,)
        The class probabilities. If None (default), then uniform class probabilities are used.
    seed : int
        The random seed to feed to :func:`numpy.random.default_rng`. The default is None.

    Returns
    -------
    data : array-like of shape (n_samples, n_dims)
        The generated data.
    y : array-like of shape (n_samples,)
        The class labels.
    means : list of array-like of shape (n_dims,)
        The means of the Gaussians from each class.
    sigmas : list of array-like of shape (n_dims, n_dims)
        The covariance matrices of the Gaussians from each class.
    pi : array-like of shape (n_classes,)
        The class probabilities.

    Notes
    -----
    This simulates data from separate multivariate Gaussians, where each class has its own
    multivariate Gaussian distribution. The class labels are sampled from a multinomial distribution
    with probabilities `pi`.

    The ground-truth computation of the MI depends on
    """
    rng = np.random.default_rng(seed)

    if pi is None:
        pi = np.ones((n_classes,)) / n_classes
    else:
        if len(pi) != n_classes:
            raise RuntimeError(f"pi should be of length {n_classes}")

    # first sample the class labels according to class probabilities
    counts = rng.multinomial(n_samples, pi, size=1)[0]

    # now sample the multivariate Gaussian for each class
    means = [np.zeros((n_dims,))]
    sigmas = [np.eye(n_dims)]

    if mean1 is not None:
        means[0][0] = mean1
    if var1 is not None:
        sigmas[0][0, 0] = var1

    # sample additional classes if not binary
    for _ in range(1, n_classes):
        mean = rng.standard_normal(size=(n_dims,))
        sigma = np.eye(n_dims)

        means.append(mean)
        sigmas.append(sigma)

    # now sample the data
    X_data = []
    y_data = []
    for k in range(n_classes):
        X_data.append(rng.multivariate_normal(means[k], sigmas[k], counts[k]))
        y_data.append(np.repeat(k, counts[k]))
    X = np.concatenate(tuple(X_data))
    y = np.concatenate(tuple(y_data))

    return X, y, means, sigmas, pi


def mi_separated_gaussians(means, sigmas, pi, seed=None):
    """Compute the ground-truth mutual information between the class labels and the data.

    Parameters
    ----------
    means : list of array-like of shape (n_dims,)
        The means of the Gaussians from each class. The list has length ``n_classes``.
    sigmas : list of array-like of shape (n_dims, n_dims)
        The covariance matrices of the Gaussians from each class.
        The list has length ``n_classes``.
    pi : array-like of shape (n_classes,)
        The class probabilities.
    seed : int
        The random seed to feed to :func:`numpy.random.default_rng`. The default is None.

    Returns
    -------
    I_XY : float
        The ground-truth mutual information between the class labels and the data.
    """
    n_dims = means[0].shape[0]
    n_classes = len(sigmas)

    # compute ground-truth MI
    base = np.exp(1)
    # H_Y = entropy(pi, base=base)

    def func(*args):
        # points at which to evaluate the multivariate-Gaussian
        x_points = np.array(args)

        # compute the probability of the multivariate-Gaussian at various points in the
        # d-dimensional space
        p = 0.0
        for k in range(n_classes):
            p += pi[k] * multivariate_normal.pdf(x_points, mean=means[k], cov=sigmas[k])

        # compute the log-probability
        return -p * np.log(p) / np.log(base)

    # limits over each dimension of the multivariate-Gaussian
    lims = [[-10, 10]] * n_dims
    H_X, _ = nquad(func, ranges=lims)

    # now compute H(X|Y)
    H_XY = _conditional_entropy_separated_gaussians(sigmas, pi, base)

    I_XY = H_X - H_XY
    return I_XY


def _conditional_entropy_separated_gaussians(sigmas, pi, base):
    """Conditional entropy.

    Computes H(X | Y), where X can be multivariate and is assumed to be multivariate-Gaussian.
    The determinant of the covariance matrix of X is used to compute the entropy.

    Y is assumed to be discrete.
    X is some multivariate-Gaussian, where the covariance matrix is given by `sigmas`
    and provides the entropy of X for each class.
    """
    n_classes = len(sigmas)
    n_dims = sigmas[0].shape[0]

    # now compute H(Y|X) = H(X, Y) - H(X)
    H_XY = 0.0
    for k in range(n_classes):
        # [d * log(2 * pi) + log(det(sigma)) + d] / (2 * log(base))
        H_XY += (
            pi[k]
            * (n_dims * np.log(2 * np.pi) + np.log(np.linalg.det(sigmas[k])) + n_dims)
            / (2.0 * np.log(base))
        )
    return H_XY


def cmi_separated_gaussians(means, sigmas, pi, condition_idx, seed=None):
    """Compute the ground-truth conditional mutual information.

    This computes the CMI between the class labels and the data.
    """
    n_classes = len(means)

    x_idx = np.ones((means[0].shape[0],), dtype=np.bool_)
    x_idx[condition_idx] = False
    # Z_sigmas = [sigma[condition_idx, condition_idx] for sigma in sigmas]
    # X_sigmas = [sigma[x_idx, x_idx] for sigma in sigmas]
    base = np.exp(1)

    def func(*args):
        # points at which to evaluate the multivariate-Gaussian
        x_points = np.array(args)

        # compute the probability of the multivariate-Gaussian at various points in the
        # d-dimensional space
        p = 0.0
        for k in range(n_classes):
            p += pi[k] * multivariate_normal.pdf(x_points, mean=means[k], cov=sigmas[k])

        # compute the log-probability
        return -p * np.log(p) / np.log(base)

    # integrate to get approximate H(X, Z)
    n_dims = means[0].shape[0]
    lims = [[-10, 10]] * n_dims
    H_XZ, _ = nquad(func, ranges=lims)

    def func(*args):
        # points at which to evaluate the multivariate-Gaussian
        x_points = np.array(args)

        # compute the probability of the multivariate-Gaussian at various points in the
        # d-dimensional space
        p = 0.0
        for k in range(n_classes):
            p += pi[k] * multivariate_normal.pdf(x_points, mean=z_means[k], cov=z_sigmas[k])

        # compute the log-probability
        return -p * np.log(p) / np.log(base)

    # get approximate H(Z)
    z_means = [mean[condition_idx] for mean in means]
    z_sigmas = [sigma[np.ix_(condition_idx, condition_idx)] for sigma in sigmas]
    n_dims = z_means[0].shape[0]
    lims = [[-10, 10]] * n_dims
    H_Z, _ = nquad(func, ranges=lims)

    # now compute H(X, Z|Y)
    H_XZY = _conditional_entropy_separated_gaussians(sigmas, pi, base)

    # lastly compute H(Z |Y)
    H_ZY = _conditional_entropy_separated_gaussians(z_sigmas, pi, base)

    # now compute H(X|Y,Z)
    print(H_XZ, H_Z, H_XZY, H_ZY)
    I_XYZ = H_XZ - H_Z - H_XZY + H_ZY
    return I_XYZ
