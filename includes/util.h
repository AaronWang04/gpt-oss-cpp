#pragma once

#include "checkpoint.h"

#include <iosfwd>
#include <ostream>
#include <string>
#include <vector>

std::string to_string(DType d);
std::string to_string(const std::vector<std::uint64_t>& vec);
std::ostream& operator<<(std::ostream& os, const TensorMeta_& T);

template<class T>
std::ostream& operator<<(std::ostream& os, std::vector<T> const& v) {
    os << "[";
    for (size_t i = 0; i < v.size(); ++i) {
        if (i) os << ", ";
        os << v[i];
    }
    os << "]";
    return os;
}
