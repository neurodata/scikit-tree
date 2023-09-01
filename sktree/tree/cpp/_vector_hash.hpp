// vector_hash_class.hpp
#pragma once
#include <vector>
#include <cstddef>

class VectorHash {
public:
    VectorHash() = default;
    size_t operator()(const std::vector<size_t>& v) const;
};
