#include "_vector_hash.hpp"

size_t VectorHash::operator()(const std::vector<size_t>& v) const {
    size_t seed = v.size();
    for (const size_t& i : v) {
        seed ^= i + 0x9e3779b9 + (seed << 6) + (seed >> 2);
    }
    return seed;
}
