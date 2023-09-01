# distutils: language = c++
# distutils: sources = _vector_hash.hpp
from libcpp.vector cimport vector

cdef extern from "_vector_hash.cpp":
    pass

cdef extern from "_vector_hash.hpp":
    cdef cppclass VectorHash:
        VectorHash() except +
        size_t operator()(const vector[size_t]& v)
