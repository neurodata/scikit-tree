# distutils: language=c++
# cython: cdivision=True
# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: initializedcheck=False

import numpy as np

cimport numpy as cnp

cnp.import_array()

from sktree._lib.sklearn.tree._utils cimport rand_uniform


cdef inline int rand_weighted_binary(double p0, UINT32_t* random_state) noexcept nogil:
    """Sample from integers 0 and 1 with different probabilities.

    Parameters
    ----------
    p0 : double
        The probability of sampling 0.
    random_state : UINT32_t*
        The random state.
    """
    cdef double random_value = rand_uniform(0.0, 1.0, random_state)

    if random_value < p0:
        return 0
    else:
        return 1

cpdef unravel_index(
    SIZE_t index,
    cnp.ndarray[SIZE_t, ndim=1] shape
):
    """Converts a flat index or array of flat indices into a tuple of coordinate arrays.

    Purely used for testing purposes.

    Parameters
    ----------
    index : SIZE_t
        A flat index.
    shape : numpy.ndarray[SIZE_t, ndim=1]
        The shape of the array into which the flat indices should be converted.

    Returns
    -------
    numpy.ndarray[SIZE_t, ndim=1]
        A coordinate array having the same shape as the input `shape`.
    """
    index = np.intp(index)
    shape = np.array(shape)
    coords = np.empty(shape.shape[0], dtype=np.intp)
    unravel_index_cython(index, shape, coords)
    return coords


cpdef ravel_multi_index(SIZE_t[:] coords, const SIZE_t[:] shape):
    """Converts a tuple of coordinate arrays into a flat index.

    Purely used for testing purposes.

    Parameters
    ----------
    coords : numpy.ndarray[SIZE_t, ndim=1]
        An array of coordinate arrays to be converted.
    shape : numpy.ndarray[SIZE_t, ndim=1]
        The shape of the array into which the coordinates should be converted.

    Returns
    -------
    SIZE_t
        The resulting flat index.

    Raises
    ------
    ValueError
        If the input `coords` have invalid indices.
    """
    return ravel_multi_index_cython(coords, shape)


cdef void unravel_index_cython(SIZE_t index, const SIZE_t[:] shape, SIZE_t[:] coords) noexcept nogil:
    """Converts a flat index into a tuple of coordinate arrays.

    Parameters
    ----------
    index : SIZE_t
        The flat index to be converted.
    shape : numpy.ndarray[SIZE_t, ndim=1]
        The shape of the array into which the flat index should be converted.
    coords : numpy.ndarray[SIZE_t, ndim=1]
        A preinitialized memoryview array of coordinate arrays to be converted.

    Returns
    -------
    numpy.ndarray[SIZE_t, ndim=1]
        An array of coordinate arrays, with each coordinate array having the same shape as the input `shape`.
    """
    cdef SIZE_t ndim = shape.shape[0]
    cdef SIZE_t j, size

    for j in range(ndim - 1, -1, -1):
        size = shape[j]
        coords[j] = index % size
        index //= size


cdef SIZE_t ravel_multi_index_cython(SIZE_t[:] coords, const SIZE_t[:] shape) noexcept nogil:
    """Converts a tuple of coordinate arrays into a flat index.

    Parameters
    ----------
    coords : numpy.ndarray[SIZE_t, ndim=1]
        An array of coordinate arrays to be converted.
    shape : numpy.ndarray[SIZE_t, ndim=1]
        The shape of the array into which the coordinates should be converted.

    Returns
    -------
    SIZE_t
        The resulting flat index.

    Raises
    ------
    ValueError
        If the input `coords` have invalid indices.
    """
    cdef SIZE_t i, ndim
    cdef SIZE_t flat_index, index

    ndim = len(shape)

    # Compute flat index
    flat_index = 0
    for i in range(ndim):
        index = coords[i]
        # if index < 0 or index >= shape[i]:
        #     raise ValueError("Invalid index")
        flat_index += index
        if i < ndim - 1:
            flat_index *= shape[i + 1]

    return flat_index

cdef SIZE_t vector_hash(
    const vector[SIZE_t]& v
) noexcept nogil:
    """Hash a vector of size_t."""
    cdef SIZE_t seed = v.size()
    cdef SIZE_t i, hash_val
    for i in range(v.size()):
        seed = seed ^ (v[i] + <SIZE_t>0x9e3779b9 + (seed << 6) + (seed >> 2))
    hash_val = seed
    return hash_val
