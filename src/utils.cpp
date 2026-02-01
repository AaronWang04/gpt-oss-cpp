#include "util.h"

#include <ostream>
#include <sstream>

std::string to_string(DType d) {
    switch (d) {
        case DType::BF16:
            return "DType::BF16";
        case DType::U8:
            return "DType::U8";
    }
    return "unknown";
}

std::string to_string(const std::vector<std::uint64_t>& vec) {
    std::stringstream ss;
    ss << '[';
    for (auto elm : vec) {
        ss << std::to_string(elm);
    }
    ss << ']';
    return ss.str();
}

std::ostream& operator<<(std::ostream& os, const TensorMeta_& T) {
    return os << T.name << ": " << to_string(T.dtype) << ", Shape: " << to_string(T.shape)
              << ", offset: " << to_string(T.offset) << '\n';
}
