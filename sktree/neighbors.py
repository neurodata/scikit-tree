import numbers
from copy import copy

import numpy as np
from joblib import Parallel, delayed
from sklearn.base import BaseEstimator, MetaEstimatorMixin
from sklearn.exceptions import NotFittedError
from sklearn.neighbors import NearestNeighbors
from sklearn.utils.validation import check_is_fitted

from sktree.tree._neighbors import _compute_distance_matrix, compute_forest_similarity_matrix


def forest_distance(clf, X, Y) -> float:
    """Compute a valid distance metric between two samples using a decision-tree or forest model."""
    X = np.atleast_2d(X)
    Y = np.atleast_2d(Y)
    XY = np.concatenate((X, Y), axis=0)
    aff_matrix = compute_forest_similarity_matrix(clf, XY)

    # dists should be (2, 2)
    dists = _compute_distance_matrix(aff_matrix)
    if dists.shape != (2, 2):
        raise RuntimeError("This shouldn't happen")

    return dists[0, 1]


class NearestNeighborsMetaEstimator(BaseEstimator, MetaEstimatorMixin):
    """Meta-estimator for nearest neighbors.

    Uses a decision-tree, or forest model to compute distances between samples
    and then uses the sklearn's nearest-neighbors API to compute neighbors. Thus,
    this meta-estimator is a two-stage process:

    1. Fit a forest on X (n_samples, n_features) to compute a distance matrix
         (n_samples, n_samples).
    2. Fit an instance of `sklearn.neighbors.NearestNeighbors` on the distance matrix to compute
         nearest neighbors.

    Parameters
    ----------
    estimator : BaseDecisionTree, BaseForest
        The estimator to use for computing distances.
    n_neighbors : int, optional
        Number of neighbors to use by default for kneighbors queries, by default 5.
    radius : float, optional
        Range of parameter space to use by default for radius_neighbors queries, by default 1.0.
    algorithm : str, optional
        Algorithm used to compute the nearest-neighbors, by default 'auto'.
        See :class:`sklearn.neighbors.NearestNeighbors` for details.
    n_jobs : int, optional
        The number of parallel jobs to run for neighbors, by default None.
    force_fit : bool, optional
        If True, the estimator will be fit even if it is already fitted, by default False.
    verbose : bool, optional
        If True, print out additional information, by default False.
    """

    _supports_multi_radii: bool = True

    def __init__(
        self,
        estimator,
        n_neighbors=5,
        radius=1.0,
        algorithm="auto",
        n_jobs=None,
        force_fit=False,
        verbose: bool = False,
    ):
        self.estimator = estimator
        self.n_neighbors = n_neighbors
        self.algorithm = algorithm
        self.radius = radius
        self.n_jobs = n_jobs
        self.force_fit = force_fit
        self.verbose = verbose

    def fit(self, X, y=None):
        """Fit the nearest neighbors estimator from the training dataset.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            The training input samples. Internally, it will be converted to
            ``dtype=np.float32`` and if a sparse matrix is provided
            to a sparse ``csc_matrix``.

        y : array-like of shape (n_samples,) or (n_samples, n_outputs)
            The target values, by default None.

        Returns
        -------
        self : object
            Fitted estimator.
        """
        if y is not None:
            X, y = self._validate_data(X, y, accept_sparse="csc")
        else:
            X = self._validate_data(X, accept_sparse="csc")

        self.estimator_ = copy(self.estimator)
        if self.force_fit:
            self.estimator_.fit(X, y)
        else:
            try:
                check_is_fitted(self.estimator_)
            except NotFittedError:
                self.estimator_.fit(X, y)

        if self.verbose:
            print(f"Finished fitting estimator: {self.estimator_}")

        # get the number of neighbors to use in estimating the CMI
        n_samples = X.shape[0]
        if self.n_neighbors < 1:
            knn_here = max(1, int(self.n_neighbors * n_samples))
        else:
            knn_here = max(1, int(self.n_neighbors))

        self._fit(X, knn_here)
        return self

    def _fit(self, X, n_neighbors):
        self.neigh_est_ = NearestNeighbors(
            n_neighbors=n_neighbors,
            algorithm=self.algorithm,
            metric="precomputed",
            n_jobs=self.n_jobs,
        )

        # compute the distance matrix
        dists = self._compute_distance_matrix(X)

        if self.verbose:
            print(f"Finished computing distance matrix: {dists.shape}")

        # fit the nearest-neighbors estimator
        self.neigh_est_.fit(dists)

    def _compute_distance_matrix(self, X):
        # compute the distance matrix
        aff_matrix = compute_forest_similarity_matrix(self.estimator_, X)
        dists = _compute_distance_matrix(aff_matrix)
        return dists

    def kneighbors(self, X=None, n_neighbors=None, return_distance=True):
        """Find the K-neighbors of a point.

        Returns indices of and distances to the neighbors of each point.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape (n_queries, n_features), \
            or (n_queries, n_indexed) if metric == 'precomputed', default=None
            Not used, present for API consistency by convention.

        n_neighbors : int, default=None
            Number of neighbors required for each sample. The default is the
            value passed to the constructor.

        return_distance : bool, default=True
            Whether or not to return the distances.

        Returns
        -------
        neigh_dist : ndarray of shape (n_queries, n_neighbors)
            Array representing the lengths to points, only present if
            return_distance=True.

        neigh_ind : ndarray of shape (n_queries, n_neighbors)
            Indices of the nearest points in the population matrix.
        """
        check_is_fitted(self)

        if n_neighbors is None:
            n_neighbors = self.n_neighbors
        elif n_neighbors <= 0:
            raise ValueError("Expected n_neighbors > 0. Got %d" % n_neighbors)
        elif not isinstance(n_neighbors, numbers.Integral):
            raise TypeError(
                "n_neighbors does not take %s value, enter integer value" % type(n_neighbors)
            )

        if X is not None:
            self._fit(X, n_neighbors)

        return self.neigh_est_.kneighbors(n_neighbors=n_neighbors, return_distance=return_distance)

    def radius_neighbors(self, X=None, radius=None, return_distance=True, sort_results=False):
        """Find the neighbors within a given radius of a point or points.

        Return the indices and distances of each point from the dataset
        lying in a ball with size ``radius`` around the points of the query
        array. Points lying on the boundary are included in the results.

        The result points are *not* necessarily sorted by distance to their
        query point.

        Parameters
        ----------
        X : {array-like, sparse matrix} of (n_samples, n_features), default=None
            The query point or points.
            If not provided, neighbors of each indexed point are returned.
            In this case, the query point is not considered its own neighbor.

        radius : float, or array-like of shape (n_samples,) default=None
            Limiting distance of neighbors to return. The default is the value
            passed to the constructor. If an array-like of shape (n_samples),
            then will query for each sample point with a different radius.

        return_distance : bool, default=True
            Whether or not to return the distances.

        sort_results : bool, default=False
            If True, the distances and indices will be sorted by increasing
            distances before being returned. If False, the results may not
            be sorted. If `return_distance=False`, setting `sort_results=True`
            will result in an error.

            .. versionadded:: 0.22

        Returns
        -------
        neigh_dist : ndarray of shape (n_samples,) of arrays
            Array representing the distances to each point, only present if
            `return_distance=True`. The distance values are computed according
            to the ``metric`` constructor parameter.

        neigh_ind : ndarray of shape (n_samples,) of arrays
            An array of arrays of indices of the approximate nearest points
            from the population matrix that lie within a ball of size
            ``radius`` around the query points.

        Notes
        -----
        Because the number of neighbors of each point is not necessarily
        equal, the results for multiple query points cannot be fit in a
        standard data array.
        For efficiency, `radius_neighbors` returns arrays of objects, where
        each object is a 1D array of indices or distances.
        """
        check_is_fitted(self)

        if X is not None:
            n_samples = X.shape[0]
            X = self._compute_distance_matrix(X)

            if self.verbose:
                print(f"Finished computing distance matrix: {X.shape}")
        else:
            n_samples = self.neigh_est_.n_samples_fit_

        # now construct nearest neighbor indices and distances within radius
        if self.verbose:
            print(f"Computing radius neighbors for {n_samples} samples")

        if isinstance(radius, numbers.Number):
            # return only neighbors within one fixed radius across all samples
            return self.neigh_est_.radius_neighbors(
                X=X, radius=radius, return_distance=return_distance, sort_results=sort_results
            )
        else:
            # forest-based nearest neighbors needs to support all samples to get pairwise
            # distances before querying the radius neighbors API
            if X is None:
                raise RuntimeError("Must provide X if radius is an array of numbers")

            if len(radius) != n_samples:
                raise RuntimeError(f"Expected {n_samples} radius values, got {len(radius)}")

            nn_inds_arr = np.zeros((n_samples,), dtype=object)
            nn_dist_arr = np.zeros((n_samples,), dtype=object)

            # compute radius neighbors for each sample in parallel
            if self.verbose:
                print("Computing radius neighbors in parallel...")

            result = Parallel(n_jobs=self.n_jobs)(
                delayed(_parallel_radius_nbrs)(
                    self.neigh_est_.radius_neighbors,
                    np.atleast_2d(X[idx, :]),
                    radius[idx],
                    return_distance,
                )
                for idx in range(n_samples)
            )

            for idx, nn in enumerate(result):
                if return_distance:
                    nn_inds_arr[idx] = nn[0][0]
                    nn_dist_arr[idx] = nn[1][0]
                else:
                    nn_inds_arr[idx] = nn[0]

            if return_distance:
                return nn_dist_arr, nn_inds_arr
            else:
                return nn_inds_arr


def _parallel_radius_nbrs(radius_nbr_func, X, radius, return_distance):
    return radius_nbr_func(X, radius, return_distance)
