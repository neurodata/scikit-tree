from typing import Optional

import numpy as np
from numpy.typing import ArrayLike
from scipy.sparse import issparse
from sklearn.neighbors import NearestNeighbors
from sklearn.utils import _approximate_mode, _safe_indexing, check_array, check_consistent_length


def _conditional_shuffle(nbrs: ArrayLike, replace: bool = False, rng=None) -> ArrayLike:
    """Compute a permutation of neighbors with restrictions.

    Parameters
    ----------
    nbrs : ArrayLike of shape (n_samples, k)
        The k-nearest-neighbors for each sample index. Each row corresponds to the
        original sample. Each element corresponds to another sample index that is deemed
        as the k-nearest neighbors with respect to the original sample.
    replace : bool, optional
        Whether or not to allow replacement of samples, by default False.
    random_seed : int, optional
        Random seed, by default None.

    Returns
    -------
    restricted_perm : ArrayLike of shape (n_samples)
        The final permutation order of the sample indices. There may be
        repeating samples. See Notes for details.

    Notes
    -----
    Restricted permutation goes through random samples and looks at the k-nearest
    neighbors (columns of ``nbrs``) and shuffles the closest neighbor index only
    if it has not been used to permute another sample. If it has been, then the
    algorithm looks at the next nearest-neighbor and so on. If all k-nearest
    neighbors of a sample has been checked, then a random neighbor is chosen. In this
    manner, the algorithm tries to perform permutation without replacement, but
    if necessary, will choose a repeating neighbor sample.
    """
    n_samples, k_dims = nbrs.shape
    rng = np.random.default_rng(seed=rng)

    # initialize the final permutation order
    restricted_perm = np.zeros((n_samples,), dtype=np.intp)

    # generate a random order of samples to go through
    random_order = rng.permutation(n_samples)

    # keep track of values we have already used
    used = set()

    # go through the random order
    for idx in random_order:
        if replace:
            restricted_perm[idx] = rng.choice(nbrs[idx, :], size=1)
        else:
            m = 0
            use_idx = nbrs[idx, m]

            # if the current nbr is already used, continue incrementing
            # until we have either found a new sample to use, or if
            # we have reach the maximum number of shuffles to consider
            while (use_idx in used) and (m < k_dims - 1):
                m += 1
                use_idx = nbrs[idx, m]

            # check whether or not we have exhaustively checked all kNN
            if use_idx in used and m == k_dims:
                # XXX: Note this step is not in the original paper
                # choose a random neighbor to permute
                restricted_perm[idx] = rng.choice(nbrs[idx, :], size=1)
            else:
                # permute with the existing neighbor
                restricted_perm[idx] = use_idx
            used.add(use_idx)
    return restricted_perm


def conditional_resample(
    conditional_array,
    *arrays,
    nn_estimator=None,
    replace=True,
    replace_nbrs=True,
    n_samples=None,
    random_state=None,
    stratify=None,
):
    """Conditionally resample arrays or sparse matrices in a consistent way.

    The default strategy implements one step of the bootstrapping
    procedure. Conditional resampling is a modification of the bootstrap
    technique that preserves the conditional distribution of the data. This
    is done by fitting a nearest neighbors estimator on the conditional array
    and then resampling the nearest neighbors of each sample.

    Parameters
    ----------
    conditional_array : array-like of shape (n_samples, n_features)
        The array, which we preserve the conditional distribution of.

    *arrays : sequence of array-like of shape (n_samples,) or \
            (n_samples, n_outputs)
        Indexable data-structures can be arrays, lists, dataframes or scipy
        sparse matrices with consistent first dimension.

    nn_estimator : estimator object, default=None
        The nearest neighbors estimator to use. If None, then a
        :class:`sklearn.neighbors.NearestNeighbors` instance is used.

    replace : bool, default=True
        Implements resampling with replacement. If False, this will implement
        (sliced) random permutations. The replacement will take place at the level
        of the sample index.

    replace_nbrs : bool, default=True
        Implements resampling with replacement at the level of the nearest neighbors.

    n_samples : int, default=None
        Number of samples to generate. If left to None this is
        automatically set to the first dimension of the arrays.
        If replace is False it should not be larger than the length of
        arrays.

    random_state : int, RandomState instance or None, default=None
        Determines random number generation for shuffling
        the data.
        Pass an int for reproducible results across multiple function calls.
        See :term:`Glossary <random_state>`.

    stratify : array-like of shape (n_samples,) or (n_samples, n_outputs), \
            default=None
        If not None, data is split in a stratified fashion, using this as
        the class labels.

    Returns
    -------
    resampled_arrays : sequence of array-like of shape (n_samples,) or \
            (n_samples, n_outputs)
        Sequence of resampled copies of the collections. The original arrays
        are not impacted.
    """
    max_n_samples = n_samples
    rng = np.random.default_rng(random_state)

    if len(arrays) == 0:
        return None

    first = arrays[0]
    n_samples = first.shape[0] if hasattr(first, "shape") else len(first)

    if max_n_samples is None:
        max_n_samples = n_samples
    elif (max_n_samples > n_samples) and (not replace):
        raise ValueError(
            "Cannot sample %d out of arrays with dim %d when replace is False"
            % (max_n_samples, n_samples)
        )

    check_consistent_length(conditional_array, *arrays)

    # fit nearest neighbors onto the conditional array
    if nn_estimator is None:
        nn_estimator = NearestNeighbors()
    nn_estimator.fit(conditional_array)

    if stratify is None:
        if replace:
            indices = rng.integers(0, n_samples, size=(max_n_samples,))
        else:
            indices = np.arange(n_samples)
            rng.shuffle(indices)
            indices = indices[:max_n_samples]
    else:
        # Code adapted from StratifiedShuffleSplit()
        y = check_array(stratify, ensure_2d=False, dtype=None)
        if y.ndim == 2:
            # for multi-label y, map each distinct row to a string repr
            # using join because str(row) uses an ellipsis if len(row) > 1000
            y = np.array([" ".join(row.astype("str")) for row in y])

        classes, y_indices = np.unique(y, return_inverse=True)
        n_classes = classes.shape[0]

        class_counts = np.bincount(y_indices)

        # Find the sorted list of instances for each class:
        # (np.unique above performs a sort, so code is O(n logn) already)
        class_indices = np.split(
            np.argsort(y_indices, kind="mergesort"), np.cumsum(class_counts)[:-1]
        )

        n_i = _approximate_mode(class_counts, max_n_samples, random_state)

        indices = []

        for i in range(n_classes):
            indices_i = rng.choice(class_indices[i], n_i[i], replace=replace)
            indices.extend(indices_i)

        indices = rng.permutation(indices)

    # now get the kNN indices for each sample (n_samples, n_neighbors)
    sample_nbrs = nn_estimator.kneighbors(X=conditional_array[indices, :], return_distance=False)

    # actually sample the indices using a conditional permutation
    indices = _conditional_shuffle(sample_nbrs, replace=replace_nbrs, rng=rng)

    # convert sparse matrices to CSR for row-based indexing
    arrays = [a.tocsr() if issparse(a) else a for a in arrays]
    resampled_arrays = [_safe_indexing(a, indices) for a in arrays]
    if len(resampled_arrays) == 1:
        # syntactic sugar for the unit argument case
        return resampled_arrays[0]
    else:
        return resampled_arrays


def conditional_shuffle(
    conditional_array, *arrays, random_state: Optional[int] = None, n_samples: Optional[int] = None
):
    """_summary_

    Parameters
    ----------
    conditional_array : ArrayLike of shape (n_samples, k)
        The array, which we preserve the conditional distribution of.
    *arrays : sequence of ArrayLike
        Indexable data-structures can be arrays, lists, dataframes or scipy sparse matrices
        with consistent first dimension.
    random_state : int, optional
        Random seed, by default None.
    n_samples : int, optional
        Number of samples to generate, by default None. If left to None this is
        automatically set to the first dimension of the arrays. It should not be larger
        than the length of arrays.
    """
    pass
