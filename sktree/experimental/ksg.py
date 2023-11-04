from typing import Optional

import numpy as np
import scipy.linalg
import scipy.special
import scipy.stats
from numpy.typing import ArrayLike
from sklearn.base import BaseEstimator
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler


def entropy_continuous(X, k=1, norm="max", min_dist=0.0, n_jobs=-1):
    """Estimates the entropy H of a continuous random variable.

    The approach uses the kth-nearest neighbour distances between sample points
    from :footcite:`kozachenko1987sample`. See:
    https://github.com/paulbrodersen/entropy_estimators/blob/master/

    Parameters
    ----------
    X : ArrayLike of shape (n_samples, n_features)
        N samples from a d-dimensional multivariate distribution

    k : int (default 1)
        kth nearest neighbour to use in density estimate;
        imposes smoothness on the underlying probability distribution

    norm : 'euclidean' or 'max'
        p-norm used when computing k-nearest neighbour distances

    min_dist : float (default 0.)
        minimum distance between data points;
        smaller distances will be capped using this value

        n_jobs : int (default 1)
                number of workers to use for parallel processing in query;
                -1 uses all CPU threads

    Returns
    -------
    h: float
        entropy H(X)

    References
    ----------
    .. footbibliography::
    """
    if X.ndim == 1:
        X = X[:, np.newaxis]
    n, d = X.shape

    if norm == "max":  # max norm:
        p = np.inf
        log_c_d = 0  # volume of the d-dimensional unit ball
    elif norm == "euclidean":  # euclidean norm
        p = 2
        log_c_d = (d / 2.0) * np.log(np.pi) - np.log(scipy.special.gamma(d / 2.0 + 1))
    else:
        raise NotImplementedError("Variable 'norm' either 'max' or 'euclidean'")

    kdtree = scipy.spatial.KDTree(X)

    # query all points -- k+1 as query point also in initial set
    # distances, _ = kdtree.query(x, k + 1, eps=0, p=norm)
    distances, _ = kdtree.query(X, k + 1, eps=0, p=p, workers=n_jobs)
    distances = distances[:, -1]

    # enforce non-zero distances
    distances[distances < min_dist] = min_dist

    sum_log_dist = np.sum(np.log(2.0 * distances))  # where did the 2 come from? radius -> diameter
    h = (
        -scipy.special.digamma(k)
        + scipy.special.digamma(n)
        + log_c_d
        + (d / float(n)) * sum_log_dist
    )

    return h


def mutual_info_ksg_nn(
    X,
    Y,
    Z=None,
    k: float = 0.2,
    norm="max",
    n_jobs: int = -1,
    transform: str = "rank",
    random_seed: int = None,
    verbose: bool = False,
):
    rng = np.random.default_rng(random_seed)

    data = np.hstack((X, Y))
    x_idx = np.arange(X.shape[1])
    y_idx = np.arange(Y.shape[1]) + X.shape[1]
    z_idx = np.array([])
    if Z is not None:
        z_idx = np.arange(Z.shape[1]) + data.shape[1]
        data = np.hstack((data, Z))

    data = _preprocess_data(data, transform, rng)
    if verbose:
        print(f"Data shape: {data.shape}")
        print(f"X shape: {X.shape}, Y shape: {Y.shape}")
        if Z is not None:
            print(f"Z shape: {Z.shape}")
        print("Preprocessing complete.")
    n_samples = data.shape[0]

    # get the number of neighbors to use in estimating the CMI
    if k < 1:
        knn_here = max(1, int(k * n_samples))
    else:
        knn_here = max(1, int(k))

    if verbose:
        print(f"Using {knn_here} neighbors to define D-dimensional volume.")

    if Z is not None:
        val = _cmi_ksg_scipy(data, x_idx, y_idx, z_idx, knn_here, n_jobs=n_jobs)
    else:
        val = _mi_ksg_scipy(data, x_idx, y_idx, knn_here, n_jobs=n_jobs)

    if norm == "max":
        norm_constant = np.log(1)
    else:
        norm_constant = np.log(
            np.pi ** (data.shape[1] / 2)
            / scipy.special.gamma(data.shape[1] / 2 + 1)
            / 2 ** data.shape[1]
        )

    return val + norm_constant


def mutual_info_ksg(
    X,
    Y,
    Z=None,
    k: float = 0.2,
    norm="max",
    nn_estimator=None,
    n_jobs: int = -1,
    transform: str = "rank",
    random_seed: int = None,
    verbose: bool = False,
):
    """Compute the generalized (conditional) mutual information KSG estimate.

    Parameters
    ----------
    X : ArrayLike of shape (n_samples, n_features_x)
        The X covariate space.
    Y : ArrayLike of shape (n_samples, n_features_y)
        The Y covariate space.
    Z : ArrayLike of shape (n_samples, n_features_z), optional
        The Z covariate space, by default None. If None, then the MI is computed.
        If Z is defined, then the CMI is computed.
    k : float, optional
        The number of neighbors to use in defining the radius, by default 0.2.
        If a number less than 1, then the number of neighbors is computed as
        ``k * n_samples``.
    norm : str, optional, {'max', 'euclidean'}
        The norm to use in computing the distance, by default 'max'. This is the norm
        used for computing the distance between points in the covariate space and
        affects the constant term in the KSG estimator. For 'max' norm, the constant
        term is :math:`\\log(1) = 0`, and for 'euclidean' norm, the constant term is
        :math:`\\pi^{d/2} / \\Gamma(d/2 + 1) / 2^d)`, where :math:`d` is the dimension.
    nn_estimator : str, optional
        The nearest neighbor estimator to use, by default None. If None will default
        to using :class:`sklearn.neighbors.NearestNeighbors` with default parameters.
    n_jobs : int, optional
        Number of parallel jobs, by default -1.
    transform : one of {'rank', 'standardize', 'uniform'}
        Preprocessing, by default "rank".
    random_seed : int, optional
        Random seed, by default None.
    verbose : bool, optional
        Whether to print verbose output, by default False.

    Returns
    -------
    val : float
        The estimated MI, or CMI value.

    Notes
    -----
    Given a dataset with ``n`` samples, the KSG estimator proceeds by:

    1. For fixed k, get the distance to the kth nearest-nbr in XYZ subspace, call it 'r'
    2. Get the number of NN in XZ subspace within radius 'r'
    3. Get the number of NN in YZ subspace within radius 'r'
    4. Get the number of NN in Z subspace within radius 'r'
    5. Apply analytic solution for KSG estimate

    For MI, the analytical solution is:

    .. math::

        \\psi(k) - E[(\\psi(n_x) + \\psi(n_y))] + \\psi(n)

    For CMI, the analytical solution is:

    .. math::

        \\psi(k) - E[(\\psi(n_{xz}) + \\psi(n_{yz}) - \\psi(n_{z}))]

    where :math:`\\psi` is the DiGamma function, and each expectation term
    is estimated by taking the sample average.

    Note that the :math:`n_i` terms denote the number of neighbors within
    radius 'r' in the subspace of 'i', where 'i' could be for example the
    X, Y, XZ, etc. subspaces. This term does not include the sample itself.

    The hyperparamter ``k`` defines the number of points in the D-dimensional
    ball with a specified radius. The larger the k, the higher the bias in
    the estimate, but the lower the variance. The smaller the k, the lower
    the bias, but the higher the variance. The default value of 0.2 is set
    to allow scaling with the number of samples.
    """
    rng = np.random.default_rng(random_seed)

    if nn_estimator is None:
        nn_estimator = NearestNeighbors(n_jobs=n_jobs)

    data = np.hstack((X, Y))
    x_idx = np.arange(X.shape[1])
    y_idx = np.arange(Y.shape[1]) + X.shape[1]
    z_idx = np.array([])
    if Z is not None:
        z_idx = np.arange(Z.shape[1]) + data.shape[1]
        data = np.hstack((data, Z))

    data = _preprocess_data(data, transform, rng)
    if verbose:
        print(f"Data shape: {data.shape}")
        print(f"X shape: {X.shape}, Y shape: {Y.shape}")
        if Z is not None:
            print(f"Z shape: {Z.shape}")
        print("Preprocessing complete.")
    n_samples = data.shape[0]

    # get the number of neighbors to use in estimating the CMI
    if k < 1:
        knn_here = max(1, int(k * n_samples))
    else:
        knn_here = max(1, int(k))

    if verbose:
        print(f"Using {knn_here} neighbors to define D-dimensional volume.")
        print(data.shape, x_idx, y_idx, z_idx)

    if Z is not None:
        val = _cmi_ksg(data, x_idx, y_idx, z_idx, nn_estimator, knn_here)
    else:
        val = _mi_ksg(data, x_idx, y_idx, nn_estimator, knn_here)

    if norm == "max":
        norm_constant = np.log(1)
    else:
        norm_constant = np.log(
            np.pi ** (data.shape[1] / 2)
            / scipy.special.gamma(data.shape[1] / 2 + 1)
            / 2 ** data.shape[1]
        )

    return val + norm_constant


def _preprocess_data(data, transform, rng):
    if transform not in ("rank", "standardize", "uniform", None):
        raise ValueError(
            f"Unknown transform {transform}. Must "
            f"be one of 'rank', 'standardize', 'uniform', or None."
        )

    n_samples, n_features = data.shape

    # add minor noise to make sure there are no ties
    random_noise = rng.random((n_samples, n_features))
    data += 1e-5 * random_noise @ np.std(data, axis=0).reshape(n_features, 1)

    # optionally transform the data
    if transform == "standardize":
        # standardize with standard scaling
        data = data.astype(np.float64)
        scaler = StandardScaler()
        data = scaler.fit_transform(data)
    elif transform == "uniform":
        data = _trafo2uniform(data)
    elif transform == "rank":
        # rank transform each column
        data = scipy.stats.rankdata(data, axis=0)
    return data


def _cmi_ksg_scipy(data, x_idx, y_idx, z_idx, knn_here: int, n_jobs: int = -1) -> float:
    tree_xyz = scipy.spatial.KDTree(data)
    radius_per_sample = tree_xyz.query(data, k=[knn_here + 1], p=np.inf, eps=0.0, workers=n_jobs)[
        0
    ][:, 0].astype(np.float64)

    # To search neighbors < eps
    radius_per_sample = np.multiply(radius_per_sample, 0.99999)

    # compute on the subspace of X, Z
    xz_idx = np.concatenate((x_idx, z_idx)).squeeze()
    tree_xz = scipy.spatial.KDTree(data[:, xz_idx])
    num_nn_xz = tree_xz.query_ball_point(
        data[:, xz_idx], r=radius_per_sample, eps=0.0, p=np.inf, workers=n_jobs, return_length=True
    )

    # compute on the subspace of Y, Z
    yz_idx = np.concatenate((y_idx, z_idx)).squeeze()
    tree_yz = scipy.spatial.KDTree(data[:, yz_idx])
    num_nn_yz = tree_yz.query_ball_point(
        data[:, yz_idx], r=radius_per_sample, eps=0.0, p=np.inf, workers=n_jobs, return_length=True
    )

    tree_z = scipy.spatial.KDTree(data[:, z_idx])
    num_nn_z = tree_z.query_ball_point(
        data[:, z_idx], r=radius_per_sample, eps=0.0, p=np.inf, workers=n_jobs, return_length=True
    )

    # compute the final CMI value
    hxyz = scipy.special.digamma(knn_here)
    hxz = scipy.special.digamma(num_nn_xz)
    hyz = scipy.special.digamma(num_nn_yz)
    hz = scipy.special.digamma(num_nn_z)
    val = hxyz - (hxz + hyz - hz).mean()
    return val


def _mi_ksg_scipy(data, x_idx, y_idx, knn_here: int, n_jobs: int = -1) -> float:
    n_samples = data.shape[0]

    tree_xyz = scipy.spatial.KDTree(data)
    radius_per_sample = tree_xyz.query(data, k=[knn_here + 1], p=np.inf, eps=0.0, workers=n_jobs)[
        0
    ][:, 0].astype(np.float64)

    # To search neighbors < eps
    radius_per_sample = np.multiply(radius_per_sample, 0.99999)

    # compute on the subspace of X
    tree_x = scipy.spatial.KDTree(data[:, x_idx])
    num_nn_x = tree_x.query_ball_point(
        data[:, x_idx], r=radius_per_sample, eps=0.0, p=np.inf, workers=n_jobs, return_length=True
    )

    # compute on the subspace of Y
    tree_y = scipy.spatial.KDTree(data[:, y_idx])
    num_nn_y = tree_y.query_ball_point(
        data[:, x_idx], r=radius_per_sample, eps=0.0, p=np.inf, workers=n_jobs, return_length=True
    )

    # compute the final MI value
    # \\psi(k) - E[(\\psi(n_x) + \\psi(n_y))] + \\psi(n)
    hxy = scipy.special.digamma(knn_here)
    hx = scipy.special.digamma(num_nn_x)
    hy = scipy.special.digamma(num_nn_y)
    hn = scipy.special.digamma(n_samples)
    val = hxy - (hx + hy).mean() + hn
    return val


def _mi_ksg(
    data, x_idx, y_idx, nn_estimator: BaseEstimator, knn_here: int, verbose: bool = False
) -> float:
    """Compute KSG estimate of MI.

    Parameters
    ----------
    data : ArrayLike
        Stacked data of X and Y.
    x_idx : ArrayLike
        Indices for the X data stored as a 1D array.
    y_idx : ArrayLike
        Indices for the Y data stored as a 1D array.
    nn_estimator : BaseEstimator
        Nearest neighbor estimator.
    knn_here : int
        Number of nearest neighbors used in nn_estimator to estimate the volume
        of the joint distribution.

    Returns
    -------
    val : float
        Estimated MI value.
    """
    n_samples = data.shape[0]

    if verbose:
        print(f"Fitting nearest neighbors estimator with {data.shape} data.")

    # estimate distance to the kth NN in XYZ subspace for each sample
    neigh = nn_estimator.fit(data)
    dists, _ = neigh.kneighbors(n_neighbors=knn_here)

    if verbose:
        print(f"Computing radii for {knn_here} nn and got dists {dists.shape}.")

    # - get the radius we want to use per sample as the distance to the kth neighbor
    #   in the joint distribution space
    radius_per_sample = dists[:, -1]

    # compute on the subspace of X
    if verbose:
        print(f"Computing radius neighbors for X with {x_idx.shape} indices.")
    num_nn_x = _compute_radius_nbrs(data, radius_per_sample, nn_estimator, col_idx=x_idx)

    # compute on the subspace of Y
    if verbose:
        print(f"Computing radius neighbors for Y with {y_idx.shape} indices.")
    num_nn_y = _compute_radius_nbrs(data, radius_per_sample, nn_estimator, col_idx=y_idx)

    # compute the final MI value
    # \\psi(k) - E[(\\psi(n_x) + \\psi(n_y))] + \\psi(n)
    hxy = scipy.special.digamma(knn_here)
    hx = scipy.special.digamma(num_nn_x)
    hy = scipy.special.digamma(num_nn_y)
    hn = scipy.special.digamma(n_samples)
    val = hxy - (hx + hy).mean() + hn
    return val


def _cmi_ksg(
    data: ArrayLike, x_idx, y_idx, z_idx, nn_estimator: BaseEstimator, knn_here: int
) -> float:
    """Compute KSG estimate of CMI.

    Parameters
    ----------
    data : ArrayLike
        Stacked data of X, Y and Z.
    x_idx : ArrayLike
        Indices for the X data stored as a 1D array.
    y_idx : ArrayLike
        Indices for the Y data stored as a 1D array.
    z_idx : ArrayLike
        Indices for the Z data stored as a 1D array.
    nn_estimator : BaseEstimator
        Nearest neighbor estimator.
    knn_here : int
        Number of nearest neighbors used in nn_estimator to estimate the volume
        of the joint distribution.

    Returns
    -------
    val : float
        Estimated CMI value.
    """
    # estimate distance to the kth NN in XYZ subspace for each sample
    nn_estimator = nn_estimator.fit(data)
    dists, _ = nn_estimator.kneighbors(n_neighbors=knn_here)

    # get the radius we want to use per sample as the distance to the kth neighbor
    # in the joint distribution space
    radius_per_sample = dists[:, -1]

    # compute on the subspace of XZ
    xz_idx = np.concatenate((x_idx, z_idx)).squeeze()
    num_nn_xz = _compute_radius_nbrs(
        data,
        radius_per_sample,
        nn_estimator,
        col_idx=xz_idx,
    )

    # compute on the subspace of YZ
    yz_idx = np.concatenate((y_idx, z_idx)).squeeze()
    num_nn_yz = _compute_radius_nbrs(
        data,
        radius_per_sample,
        nn_estimator,
        col_idx=yz_idx,
    )

    # compute on the subspace of XZ
    num_nn_z = _compute_radius_nbrs(
        data,
        radius_per_sample,
        nn_estimator,
        col_idx=z_idx,
    )

    # compute the final CMI value
    hxyz = scipy.special.digamma(knn_here)
    hxz = scipy.special.digamma(num_nn_xz)
    hyz = scipy.special.digamma(num_nn_yz)
    hz = scipy.special.digamma(num_nn_z)
    val = hxyz - (hxz + hyz - hz).mean()
    return val


def _compute_radius_nbrs(
    data: ArrayLike,
    radius_per_sample: ArrayLike,
    nn_estimator: BaseEstimator,
    col_idx: Optional[ArrayLike] = None,
):
    # compute distances in the subspace defined by data
    nn_estimator.fit(data[:, col_idx])

    # compute the radius neighbors for each sample
    if getattr(nn_estimator, "_supports_multi_radii", False):
        nn = nn_estimator.radius_neighbors(
            X=data[:, col_idx], radius=radius_per_sample, return_distance=False
        )
    else:
        nn = []
        for idx, radius in enumerate(radius_per_sample):
            nn_ = nn_estimator.radius_neighbors(
                X=np.atleast_2d(data[:, col_idx]), radius=radius, return_distance=False
            )
            nn.append(nn_[idx])

    num_nn_data = np.array([len(nn_) for nn_ in nn])
    return num_nn_data


def _trafo2uniform(X):
    """Transforms input array to uniform marginals.

    Assumes x.shape = (dim, T)

    Parameters
    ----------
    X : arraylike
        The input data with (n_samples,) rows and (n_features,) columns.

    Returns
    -------
    u : array-like
        array with uniform marginals.
    """

    def trafo(xi):
        xisorted = np.sort(xi)
        yi = np.linspace(1.0 / len(xi), 1, len(xi))
        return np.interp(xi, xisorted, yi)

    _, n_features = X.shape

    # apply a uniform transformation for each feature
    for idx in range(n_features):
        marginalized_feature = trafo(X[:, idx].to_numpy().squeeze())
        X[:, idx] = marginalized_feature
    return X
