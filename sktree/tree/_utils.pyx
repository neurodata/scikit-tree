# distutils: language=c++
# cython: cdivision=True
# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: initializedcheck=False

import numpy as np

cimport numpy as cnp

cnp.import_array()
from libcpp.vector cimport vector

from .._lib.sklearn.tree._utils cimport rand_uniform


cdef inline int rand_weighted_binary(float64_t p0, UINT32_t* random_state) noexcept nogil:
    """Sample from integers 0 and 1 with different probabilities.

    Parameters
    ----------
    p0 : float64_t
        The probability of sampling 0.
    random_state : UINT32_t*
        The random state.
    """
    cdef float64_t random_value = rand_uniform(0.0, 1.0, random_state)

    if random_value < p0:
        return 0
    else:
        return 1

cpdef unravel_index(
    intp_t index,
    cnp.ndarray[intp_t, ndim=1] shape
):
    """Converts a flat index or array of flat indices into a tuple of coordinate arrays.

    Purely used for testing purposes.

    Parameters
    ----------
    index : intp_t
        A flat index.
    shape : numpy.ndarray[intp_t, ndim=1]
        The shape of the array into which the flat indices should be converted.

    Returns
    -------
    numpy.ndarray[intp_t, ndim=1]
        A coordinate array having the same shape as the input `shape`.
    """
    index = np.intp(index)
    shape = np.array(shape)
    coords = unravel_index_cython(index, shape)
    return coords


cpdef ravel_multi_index(vector[intp_t] coords, const intp_t[:] shape):
    """Converts a tuple of coordinate arrays into a flat index.

    Purely used for testing purposes.

    Parameters
    ----------
    coords : numpy.ndarray[intp_t, ndim=1]
        An array of coordinate arrays to be converted.
    shape : numpy.ndarray[intp_t, ndim=1]
        The shape of the array into which the coordinates should be converted.

    Returns
    -------
    intp_t
        The resulting flat index.

    Raises
    ------
    ValueError
        If the input `coords` have invalid indices.
    """
    return ravel_multi_index_cython(coords, shape)


cdef vector[intp_t] unravel_index_cython(intp_t index, const intp_t[:] shape) noexcept nogil:
    """Converts a flat index into a tuple of coordinate arrays.

    Parameters
    ----------
    index : intp_t
        The flat index to be converted.
    shape : numpy.ndarray[intp_t, ndim=1]
        The shape of the array into which the flat index should be converted.
    coords : numpy.ndarray[intp_t, ndim=1]
        A preinitialized memoryview array of coordinate arrays to be converted.

    Returns
    -------
    numpy.ndarray[intp_t, ndim=1]
        An array of coordinate arrays, with each coordinate array having the same shape as the input `shape`.
    """
    cdef intp_t ndim = shape.shape[0]
    cdef intp_t j, size
    cdef vector[intp_t] coords = vector[intp_t](ndim)

    for j in range(ndim - 1, -1, -1):
        size = shape[j]
        coords[j] = index % size
        index //= size

    return coords


cdef intp_t ravel_multi_index_cython(vector[intp_t] coords, const intp_t[:] shape) noexcept nogil:
    """Converts a tuple of coordinate arrays into a flat index.

    Parameters
    ----------
    coords : numpy.ndarray[intp_t, ndim=1]
        An array of coordinate arrays to be converted.
    shape : numpy.ndarray[intp_t, ndim=1]
        The shape of the array into which the coordinates should be converted.

    Returns
    -------
    intp_t
        The resulting flat index.

    Raises
    ------
    ValueError
        If the input `coords` have invalid indices.
    """
    cdef intp_t i, ndim
    cdef intp_t flat_index, index

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


cdef vector[vector[intp_t]] cartesian_cython(vector[vector[intp_t]] sequences) noexcept nogil:
    cdef vector[vector[intp_t]] results = vector[vector[intp_t]](1)
    cdef vector[vector[intp_t]] next_results
    for new_values in sequences:
        for result in results:
            for value in new_values:
                result_copy = result
                result_copy.push_back(value)
                next_results.push_back(result_copy)
        results = next_results
        next_results.clear()
    return results


cpdef cartesian_python(vector[vector[intp_t]]& sequences):
    return cartesian_cython(sequences)
