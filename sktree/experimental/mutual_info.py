import numpy as np
import scipy.special


def entropy_gaussian(cov):
    """Compute entropy of a multivariate Gaussian.

    Computes the analytical solution due to :footcite:`Darbellay1999Entropy`.

    Parameters
    ----------
    cov : array-like of shape (d,d)
        The covariance matrix of the distribution.

    Returns
    -------
    true_entropy : float
        The true entropy of the generated multivariate Gaussian distribution.

    References
    ----------
    .. footbibliography::
    """
    d = cov.shape[0]

    true_entropy = 0.5 * d * (1 + np.log(2 * np.pi)) + 0.5 * np.log(np.linalg.det(cov))
    return true_entropy


def mi_gaussian(cov, x_index, y_index):
    """Compute mutual information of a multivariate Gaussian.

    Parameters
    ----------
    cov : array-like of shape (d,d)
        The covariance matrix of the distribution.
    x_index : list or int
        List of indices in ``cov`` that are for the X variable.
    y_index : list or int
        List of indices in ``cov`` that are for the Y variable.

    Returns
    -------
    true_mi : float
        The true analytical entropy of the generated multivariate Gaussian distribution.

    Notes
    -----
    Analytical solution for entropy, :math:`H(X)` of a multivariate Gaussian is given by::

        H(X) = \\frac{d}{2} (1 + log(2\\pi)) + \\frac{1}{2} log(det(C))
    """
    x_index = np.atleast_1d(x_index)
    y_index = np.atleast_1d(y_index)

    # computes the true MI
    det_x = np.linalg.det(cov[np.ix_(x_index, x_index)])
    det_y = np.linalg.det(cov[np.ix_(y_index, y_index)])
    det_xy = np.linalg.det(
        cov[np.ix_(np.concatenate((x_index, y_index)), np.concatenate((x_index, y_index)))]
    )
    true_mi = 0.5 * np.log((det_x * det_y) / det_xy)
    return true_mi


def cmi_gaussian(cov, x_index, y_index, z_index=None):
    """Compute the analytical CMI for a multivariate Gaussian distribution.

    Parameters
    ----------
    cov : array-like of shape (d,d)
        The covariance matrix of the distribution.
    x_index : list or int
        List of indices in ``cov`` that are for the X variable.
    y_index : list or int
        List of indices in ``cov`` that are for the Y variable.
    z_index : list or int
        List of indices in ``cov`` that are for the Z variable.

    Returns
    -------
    true_mi : float
        The true analytical mutual information of the generated multivariate Gaussian distribution.

    Notes
    -----
    Analytical solution for conditional mutual information, :math:`I(X;Y|Z)` of a
    multivariate Gaussian is given by::

        I(X;Y | Z) = H(X, Z) + H(Y, Z) - H(Z) - H(X, Y, Z)

    where we plug in the analytical solutions for entropy as shown in :func:`entropy_gaussian`.
    """
    x_index = np.atleast_1d(x_index).astype(int)
    y_index = np.atleast_1d(y_index).astype(int)
    if z_index is None:
        z_index = []
    z_index = np.atleast_1d(z_index).astype(int)

    xz_index = np.concatenate((x_index, z_index))
    yz_index = np.concatenate((y_index, z_index))

    cov_xz = cov[np.ix_(xz_index, xz_index)]
    cov_yz = cov[np.ix_(yz_index, yz_index)]
    cov_z = cov[np.ix_(z_index, z_index)]
    cov_xyz = cov[
        np.ix_(
            np.concatenate((x_index, y_index, z_index)), np.concatenate((x_index, y_index, z_index))
        )
    ]
    cmi = (
        entropy_gaussian(cov_xz)
        + entropy_gaussian(cov_yz)
        - entropy_gaussian(cov_z)
        - entropy_gaussian(cov_xyz)
    )
    return cmi


def entropy_weibull(alpha, k):
    """Analytical solution for entropy of Weibull distribution.

    https://en.wikipedia.org/wiki/Weibull_distribution
    """
    return np.euler_gamma * (1.0 - 1.0 / alpha) - np.log(alpha * np.power(k, 1.0 / alpha)) + 1


def mi_gamma(theta):
    """Analytical solution for"""
    return scipy.special.digamma(theta + 1) - np.log(theta)


def mi_from_entropy(hx, hy, hxy):
    """Analytic formula for MI given plug-in estimates of entropies."""
    return hx + hy - hxy


def cmi_from_entropy(hxz, hyz, hz, hxyz):
    """Analytic formula for CMI given plug-in estimates of entropies."""
    return hxz + hyz - hz - hxyz
