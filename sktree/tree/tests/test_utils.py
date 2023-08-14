from collections import deque
from itertools import product

import numpy as np
from numpy.testing import assert_equal

from sktree._lib.sklearn.tree._criterion import Gini
from sktree._lib.sklearn.tree._utils import _any_isnan_axis0

from .._utils import ravel_multi_index, unravel_index
from ..manifold._morf_splitter import BestPatchSplitterTester


def is_contiguous_patch(patch_arr, patch_indices, data_dims):
    """Check patch is contiguous.

    Parameters
    ----------
    patch_arr : array-like of shape (n_dim_0, n_dim_1, ...)
        The array of the patch, with elements > 0 indicating the weight of the
        element, which is a part of the patch.
    patch_indices : array-like of shape (nnz, len(data_dims))
        The non-zero indices of the patch.
    data_dims : Tuple of int
        The dimensions of the data.
    """
    # create a visited array to mark which indices have been visited
    visited = dict()

    ndim = len(data_dims)

    # choose the starting index
    starting_index = tuple(patch_indices[0])
    # print(f'Starting index is: {starting_index}')

    # initialize the queue with the starting index
    queue = deque([starting_index])

    # perform a BFS
    while queue:
        # pop the next index from the queue
        current_index = queue.popleft()

        # store the raveled index
        # mark this index as visited
        # print(current_index, data_dims)
        ravel_index = np.ravel_multi_index(current_index, data_dims)
        visited[ravel_index] = 1

        # get the neighbors of the current index
        for dim_idx in range(ndim):
            for offset in [-1, 1]:
                neighbor_index = list(current_index)
                neighbor_index[dim_idx] += offset

                # check that this neighbor is valid
                if all(
                    [
                        neighbor_index[i] >= 0 and neighbor_index[i] < data_dims[i]
                        for i in range(ndim)
                    ]
                ):
                    neighbor = tuple(neighbor_index)

                    # add unvisited neighbors to the queue
                    ravel_index = np.ravel_multi_index(neighbor, data_dims)
                    if ravel_index not in visited and patch_arr[neighbor] > 0:
                        queue.append(neighbor)

    # check if all indices have been visited
    visited_set = set(visited.keys())
    return visited_set == set(np.ravel_multi_index(idx, data_dims) for idx in patch_indices)


def test_best_patch_splitter_contiguous():
    """Test that patch splitter generates patches correctly."""
    criterion = Gini(1, np.array((0, 1)))
    max_features = 6
    min_samples_leaf = 1
    min_weight_leaf = 0.0
    random_state = np.random.RandomState(10)

    # initialize some dummy data
    data_dims = np.array((2, 3, 10))
    X = np.repeat(np.arange(data_dims.prod()).astype(np.float32), 5).reshape(5, -1)
    y = np.array([0, 0, 0, 1, 1]).reshape(-1, 1).astype(np.float64)
    sample_weight = np.ones(5)

    print(X.shape, y.shape, sample_weight.shape)

    # We will make the patch 2D, which samples multiple rows contiguously. This is
    # a 2D patch of size 3 in the columns and 2 in the rows.
    min_patch_dims = np.array((1, 2, 1))
    max_patch_dims = np.array((2, 3, 6))
    dim_contiguous = np.array((True, True, True))

    boundary = None
    feature_weight = None

    # monotonic constraints are not supported for oblique splits
    monotonic_cst = None

    splitter = BestPatchSplitterTester(
        criterion,
        max_features,
        min_samples_leaf,
        min_weight_leaf,
        random_state,
        monotonic_cst,
        min_patch_dims,
        max_patch_dims,
        dim_contiguous,
        data_dims,
        boundary,
        feature_weight,
    )
    feature_has_missing = _any_isnan_axis0(X)
    splitter.init_test(X, y, sample_weight, feature_has_missing)

    proj_i = 0
    for _ in range(10):
        top_left_patch_seed, patch_size, patch_dims = splitter.sample_top_left_seed_cpdef()

        # sample a bunch of projection vectors
        proj_vec = splitter.sample_projection_vector(
            proj_i, patch_size, top_left_patch_seed, patch_dims
        )
        # print(proj_vec.reshape(data_dims))

        # Check if the patch is contiguous
        patch_indices = np.array(proj_vec).reshape(data_dims)
        nnz_unravel_ind = np.argwhere(patch_indices != 0)
        assert is_contiguous_patch(patch_indices, nnz_unravel_ind, data_dims)


def test_unravel_index():
    # Test with 1D array
    indices = np.array([0, 1, 2, 3, 4])
    shape = np.asarray((5,))
    expected_output = [(0,), (1,), (2,), (3,), (4,)]
    for idx, index in enumerate(indices):
        assert unravel_index(index, shape) == expected_output[idx]

    # Test with 2D array
    indices = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8])
    shape = np.array((3, 3))
    expected_output = list(product(range(3), repeat=2))
    for idx, index in enumerate(indices):
        assert_equal(unravel_index(index, shape), expected_output[idx])

    # Test with 3D array
    indices = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15])
    shape = np.array((2, 2, 2, 2))
    expected_output = list(product(range(2), repeat=4))
    for idx, index in enumerate(indices):
        assert_equal(unravel_index(index, shape), expected_output[idx])

    # Test with empty array
    indices = np.array([])
    shape = (2, 2)
    expected_output = []
    for idx, index in enumerate(indices):
        assert unravel_index(index, shape) == expected_output[idx]

    # Test with out-of-bounds index
    # indices = np.array([0, 1, 2, 3, 4])
    # shape = np.array((2, 2))
    # try:
    #     for idx, index in enumerate(indices):
    #         ind = unravel_index(index, shape)
    #         print(ind)
    #     assert False
    # except ValueError as e:
    #     assert str(e) == "Invalid index"
    # else:
    #     print(e)
    #     assert False, "Expected IndexError but no exception was raised"


def test_ravel_multi_index():
    # Test 1D array
    index = np.array([3], dtype=np.intp)
    shape = np.array([6], dtype=np.intp)
    assert_equal(ravel_multi_index(index, shape), 3)

    # Test 2D array
    index = np.array([1, 2], dtype=np.intp)
    shape = np.array([2, 3], dtype=np.intp)
    assert_equal(ravel_multi_index(index, shape), 5)

    # Test invalid index
    # index = np.array([2, 4], dtype=np.intp)
    # shape = np.array([2, 3], dtype=np.intp)
    # try:
    #     ind = ravel_multi_index(index, shape)
    #     print(ind)
    #     assert False
    # except ValueError as e:
    #     assert str(e) == "Invalid index"
    # else:
    #     assert False, "Expected ValueError"
