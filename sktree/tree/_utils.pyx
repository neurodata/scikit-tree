# cython: cdivision=True
# distutils: language=c++
# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: profile=True

import numpy as np

cimport numpy as cnp

cnp.import_array()

from sklearn.tree._utils cimport rand_int


cdef void fisher_yates_shuffle(
    int size,
    SIZE_t[:] index_buffer,
    UINT32_t* random_state
) noexcept nogil:
    """Performs an in-place Fisher-Yates shuffle of the given index buffer.

    XXX: Not used yet, because not sure if we should just use extern c++

    Parameters
    ----------
    size : int
        The size of the index buffer.
    index_buffer : numpy.ndarray[SIZE_t, ndim=1]
        The index buffer to be shuffled that has been pre-allocated memory.
    random_state : UINT32_t*
        The random state to be used for shuffling.
    """
    cdef int i, j

    # fill with values 0, 1, ..., dimension - 1
    for i in range(0, size):
        index_buffer[i] = i
    # then shuffle indices using Fisher-Yates
    for i in range(size):
        j = rand_int(0, size - i, random_state)
        index_buffer[i], index_buffer[j] = \
            index_buffer[j], index_buffer[i]


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


cpdef ravel_multi_index(SIZE_t[:] coords, SIZE_t[:] shape):
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


cdef SIZE_t ravel_multi_index_cython(SIZE_t[:] coords, SIZE_t[:] shape) noexcept nogil:
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
