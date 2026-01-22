#include <algorithm>
#include <cctype>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <unordered_map>
#include <utility>
#include <vector>
#include <cmath>
#include <memory>

namespace fs = std::filesystem;

// MXFP4 constants
static constexpr int kBytesPerBlock = 16; // 32 fp4 values packed in 16 bytes
static constexpr float kFp4Values[16] = {
    +0.0f, +0.5f, +1.0f, +1.5f, +2.0f, +3.0f, +4.0f, +6.0f,
    -0.0f, -0.5f, -1.0f, -1.5f, -2.0f, -3.0f, -4.0f, -6.0f,
};

enum class DType {
    U8,
    F32,
    BF16,
    Unknown,
};

struct TensorMeta {
    DType dtype{DType::Unknown};
    std::vector<size_t> shape{};
    size_t dataOffset{0}; // absolute file offset where tensor data starts
    size_t nbytes{0};     // byte length of tensor data
};

// Minimal SafeTensors reader sufficient for our needs.
class SafeTensorsFile {
public:
    explicit SafeTensorsFile(std::string path) : path_(std::move(path)) { parse(); }

    bool has(std::string_view name) const { return meta_.find(std::string(name)) != meta_.end(); }

    const TensorMeta& meta(const std::string& name) const {
        auto it = meta_.find(name);
        if (it == meta_.end()) throw std::runtime_error("tensor not found: " + name);
        return it->second;
    }

    // Read raw bytes for a tensor
    std::vector<uint8_t> readBytes(const std::string& name) const {
        const auto& m = meta(name);
        std::ifstream f(path_, std::ios::binary);
        if (!f) throw std::runtime_error("failed to open safetensors file: " + path_);
        f.seekg(static_cast<std::streamoff>(m.dataOffset), std::ios::beg);
        std::vector<uint8_t> bytes(m.nbytes);
        if (!f.read(reinterpret_cast<char*>(bytes.data()), static_cast<std::streamsize>(m.nbytes))) {
            throw std::runtime_error("failed to read tensor bytes: " + name);
        }
        return bytes;
    }

    const std::unordered_map<std::string, TensorMeta>& all() const { return meta_; }

private:
    static uint64_t readU64LE(std::ifstream& f) {
        uint64_t v = 0;
        char buf[8];
        if (!f.read(buf, 8)) throw std::runtime_error("failed to read u64 header");
        // little-endian
        for (int i = 7; i >= 0; --i) {
            v = (v << 8) | static_cast<unsigned char>(buf[i]);
        }
        return v;
    }

    static void skipWhitespace(const std::string& s, size_t& i) {
        while (i < s.size() && std::isspace(static_cast<unsigned char>(s[i]))) ++i;
    }

    static bool matchLiteral(const std::string& s, size_t& i, std::string_view lit) {
        skipWhitespace(s, i);
        if (s.compare(i, lit.size(), lit) == 0) {
            i += lit.size();
            return true;
        }
        return false;
    }

    static std::string parseJSONString(const std::string& s, size_t& i) {
        skipWhitespace(s, i);
        if (i >= s.size() || s[i] != '"') throw std::runtime_error("expected JSON string");
        ++i; // skip opening quote
        std::string out;
        while (i < s.size()) {
            char c = s[i++];
            if (c == '\\') {
                if (i >= s.size()) break;
                char esc = s[i++];
                switch (esc) {
                    case '"': out.push_back('"'); break;
                    case '\\': out.push_back('\\'); break;
                    case '/': out.push_back('/'); break;
                    case 'b': out.push_back('\b'); break;
                    case 'f': out.push_back('\f'); break;
                    case 'n': out.push_back('\n'); break;
                    case 'r': out.push_back('\r'); break;
                    case 't': out.push_back('\t'); break;
                    // We don't handle \uXXXX as it's not expected here
                    default: out.push_back(esc); break;
                }
            } else if (c == '"') {
                break;
            } else {
                out.push_back(c);
            }
        }
        return out;
    }

    static size_t findMatchingBrace(const std::string& s, size_t openPos) {
        // s[openPos] must be '{'
        int depth = 0;
        for (size_t i = openPos; i < s.size(); ++i) {
            char c = s[i];
            if (c == '{') depth++;
            else if (c == '}') {
                depth--;
                if (depth == 0) return i;
            }
        }
        throw std::runtime_error("unmatched brace while parsing safetensors header");
    }

    static DType parseDType(std::string_view s) {
        if (s == "U8") return DType::U8;
        if (s == "F32") return DType::F32;
        if (s == "BF16") return DType::BF16;
        return DType::Unknown;
    }

    static std::vector<size_t> parseShapeArray(const std::string& s, size_t& i) {
        skipWhitespace(s, i);
        if (s[i] != '[') throw std::runtime_error("expected '[' for shape array");
        ++i;
        std::vector<size_t> out;
        while (i < s.size()) {
            skipWhitespace(s, i);
            if (s[i] == ']') { ++i; break; }
            // parse integer
            size_t j = i;
            while (j < s.size() && (std::isdigit(static_cast<unsigned char>(s[j])))) j++;
            if (j == i) throw std::runtime_error("expected integer in shape array");
            size_t val = std::stoull(s.substr(i, j - i));
            out.push_back(val);
            i = j;
            skipWhitespace(s, i);
            if (s[i] == ',') { ++i; continue; }
            if (s[i] == ']') { ++i; break; }
        }
        return out;
    }

    static std::pair<size_t, size_t> parseOffsetsArray(const std::string& s, size_t& i) {
        skipWhitespace(s, i);
        if (s[i] != '[') throw std::runtime_error("expected '[' for data_offsets");
        ++i;
        auto parseOne = [&](size_t& pos) -> size_t {
            skipWhitespace(s, pos);
            size_t j = pos;
            while (j < s.size() && (std::isdigit(static_cast<unsigned char>(s[j])))) j++;
            if (j == pos) throw std::runtime_error("expected integer in data_offsets");
            size_t v = std::stoull(s.substr(pos, j - pos));
            pos = j;
            return v;
        };
        size_t a = parseOne(i);
        skipWhitespace(s, i);
        if (s[i] != ',') throw std::runtime_error("expected comma in data_offsets");
        ++i;
        size_t b = parseOne(i);
        skipWhitespace(s, i);
        if (s[i] != ']') throw std::runtime_error("expected closing ']' in data_offsets");
        ++i;
        return {a, b};
    }

    void parseHeaderJSON(const std::string& json, size_t dataSectionStart) {
        // Top-level object: { "name": {"dtype":..., "shape":[...], "data_offsets":[s,e]}, ... }
        size_t i = 0;
        skipWhitespace(json, i);
        if (i >= json.size() || json[i] != '{') throw std::runtime_error("invalid safetensors header");
        ++i; // skip '{'
        while (i < json.size()) {
            skipWhitespace(json, i);
            if (i < json.size() && json[i] == '}') { ++i; break; }
            if (i >= json.size() || json[i] != '"') break; // done
            std::string name = parseJSONString(json, i);
            skipWhitespace(json, i);
            if (i >= json.size() || json[i] != ':') throw std::runtime_error("expected ':' after key");
            ++i;
            skipWhitespace(json, i);
            if (i >= json.size() || json[i] != '{') throw std::runtime_error("expected '{' for value object");
            size_t objStart = i;
            size_t objEnd = findMatchingBrace(json, objStart);
            // parse within [objStart, objEnd]
            size_t j = objStart + 1;
            DType dtype = DType::Unknown;
            std::vector<size_t> shape;
            std::pair<size_t, size_t> offsets{0, 0};
            bool haveDType = false, haveShape = false, haveOffsets = false;
            while (j < objEnd) {
                skipWhitespace(json, j);
                if (j >= objEnd) break;
                if (json[j] != '"') { ++j; continue; }
                std::string key = parseJSONString(json, j);
                skipWhitespace(json, j);
                if (j >= objEnd || json[j] != ':') break;
                ++j;
                if (key == "dtype") {
                    std::string dtypeStr = parseJSONString(json, j);
                    dtype = parseDType(dtypeStr);
                    haveDType = true;
                } else if (key == "shape") {
                    shape = parseShapeArray(json, j);
                    haveShape = true;
                } else if (key == "data_offsets") {
                    offsets = parseOffsetsArray(json, j);
                    haveOffsets = true;
                } else {
                    // skip value (object/array/string/number)
                    // Simple skip: if starts with '{' or '[' find matching, if '"' parse string, else skip token
                    if (j < objEnd && (json[j] == '{')) {
                        size_t k = findMatchingBrace(json, j);
                        j = k + 1;
                    } else if (j < objEnd && json[j] == '[') {
                        // find matching ']' naive
                        int depth = 0; size_t k = j;
                        do {
                            if (json[k] == '[') depth++; else if (json[k] == ']') depth--; k++;
                        } while (k < objEnd && depth > 0);
                        j = k;
                    } else if (j < objEnd && json[j] == '"') {
                        (void)parseJSONString(json, j);
                    } else {
                        // number, boolean, null
                        while (j < objEnd && json[j] != ',' && json[j] != '}') j++;
                    }
                }
                skipWhitespace(json, j);
                if (j < objEnd && json[j] == ',') ++j;
            }
            if (haveDType && haveShape && haveOffsets && name != "__metadata__") {
                TensorMeta t{};
                t.dtype = dtype;
                t.shape = std::move(shape);
                t.dataOffset = dataSectionStart + offsets.first;
                t.nbytes = offsets.second - offsets.first;
                meta_.emplace(std::move(name), std::move(t));
            }
            i = objEnd + 1;
            skipWhitespace(json, i);
            if (i < json.size() && json[i] == ',') ++i;
        }
    }

    void parse() {
        std::ifstream f(path_, std::ios::binary);
        if (!f) throw std::runtime_error("failed to open safetensors file: " + path_);
        uint64_t headerLen = readU64LE(f);
        std::string header;
        header.resize(static_cast<size_t>(headerLen));
        if (!f.read(header.data(), static_cast<std::streamsize>(headerLen))) {
            throw std::runtime_error("failed to read safetensors header: " + path_);
        }
        size_t dataStart = 8 + static_cast<size_t>(headerLen);
        parseHeaderJSON(header, dataStart);
    }

    std::string path_;
    std::unordered_map<std::string, TensorMeta> meta_;
};

// Logical parameter mapping like in pyt/weights.py (PARAM_NAME_MAP)
struct ParamRef {
    // If isMXFP4 is true, use blocks/scales; otherwise use single
    bool isMXFP4{false};
    std::string single;
    std::string blocks;
    std::string scales;
};

static std::unordered_map<std::string, ParamRef> buildParamNameMap() {
    std::unordered_map<std::string, ParamRef> m;
    for (int n = 0; n < 36; ++n) {
        // Biases map directly
        {
            std::string k = "block." + std::to_string(n) + ".mlp.mlp1_bias";
            m.emplace(k, ParamRef{false, k, {}, {}});
        }
        {
            std::string k = "block." + std::to_string(n) + ".mlp.mlp2_bias";
            m.emplace(k, ParamRef{false, k, {}, {}});
        }
        // Weights are MXFP4 (blocks + scales)
        {
            std::string logical = "block." + std::to_string(n) + ".mlp.mlp1_weight";
            std::string blocks = logical + ".blocks";
            std::string scales = logical + ".scales";
            m.emplace(logical, ParamRef{true, {}, blocks, scales});
        }
        {
            std::string logical = "block." + std::to_string(n) + ".mlp.mlp2_weight";
            std::string blocks = logical + ".blocks";
            std::string scales = logical + ".scales";
            m.emplace(logical, ParamRef{true, {}, blocks, scales});
        }
    }
    return m;
}

class CheckPoint {
public:
    struct TensorF32 {
        std::vector<size_t> shape;
        std::vector<float> data; // row-major
    };

    explicit CheckPoint(const std::string& dirPath) : dirPath_(dirPath), paramMap_(buildParamNameMap()) {
        // Discover .safetensors files and index their tensors
        for (const auto& entry : fs::directory_iterator(dirPath)) {
            if (!entry.is_regular_file()) continue;
            auto path = entry.path();
            if (path.extension() == ".safetensors") {
                files_.emplace_back(std::make_unique<SafeTensorsFile>(path.string()));
                size_t idx = files_.size() - 1;
                for (const auto& kv : files_.back()->all()) {
                    nameToFile_.emplace(kv.first, idx);
                }
            }
        }
        if (files_.empty()) {
            throw std::runtime_error("no .safetensors files found in directory: " + dirPath);
        }
    }

    TensorF32 get(const std::string& name) const {
        auto it = paramMap_.find(name);
        if (it != paramMap_.end()) {
            const ParamRef& ref = it->second;
            if (ref.isMXFP4) {
                return getMXFP4(ref.blocks, ref.scales);
            } else {
                return getTensorAsF32(ref.single);
            }
        }
        // default: try to load tensor directly
        return getTensorAsF32(name);
    }

private:
    static size_t numElements(const std::vector<size_t>& shape) {
        size_t n = 1;
        for (size_t d : shape) n *= d;
        return n;
    }

    static float bf16ToF32(uint16_t bf16) {
        uint32_t u = static_cast<uint32_t>(bf16) << 16;
        float f;
        std::memcpy(&f, &u, sizeof(float));
        return f;
    }

    TensorF32 getTensorAsF32(const std::string& tensorName) const {
        const SafeTensorsFile* file = locate(tensorName);
        const TensorMeta& m = file->meta(tensorName);
        TensorF32 t{};
        t.shape = m.shape;
        const auto bytes = file->readBytes(tensorName);
        if (m.dtype == DType::F32) {
            size_t n = m.nbytes / sizeof(float);
            t.data.resize(n);
            std::memcpy(t.data.data(), bytes.data(), m.nbytes);
        } else if (m.dtype == DType::BF16) {
            size_t n = m.nbytes / sizeof(uint16_t);
            t.data.resize(n);
            const uint16_t* src = reinterpret_cast<const uint16_t*>(bytes.data());
            for (size_t i = 0; i < n; ++i) t.data[i] = bf16ToF32(src[i]);
        } else if (m.dtype == DType::U8) {
            // Promote to float
            size_t n = m.nbytes;
            t.data.resize(n);
            for (size_t i = 0; i < n; ++i) t.data[i] = static_cast<float>(bytes[i]);
        } else {
            throw std::runtime_error("unsupported dtype for tensor: " + tensorName);
        }
        return t;
    }

    TensorF32 getMXFP4(const std::string& blocksName, const std::string& scalesName) const {
        const SafeTensorsFile* fileBlocks = locate(blocksName);
        const SafeTensorsFile* fileScales = locate(scalesName);
        const TensorMeta& mb = fileBlocks->meta(blocksName);
        const TensorMeta& ms = fileScales->meta(scalesName);
        if (mb.dtype != DType::U8 || ms.dtype != DType::U8) {
            throw std::runtime_error("MXFP4 expects U8 blocks and U8 scales");
        }
        if (mb.shape.size() < 2 || ms.shape.size() < 1) {
            throw std::runtime_error("unexpected MXFP4 tensor shapes");
        }
        // blocks shape: *prefix, G, B
        // scales shape: *prefix, G
        std::vector<size_t> prefix(mb.shape.begin(), mb.shape.end() - 2);
        size_t G = mb.shape[mb.shape.size() - 2];
        size_t B = mb.shape[mb.shape.size() - 1];

        // Validate scales prefix matches
        if (ms.shape.size() + 1 != mb.shape.size()) {
            throw std::runtime_error("MXFP4 scales shape rank mismatch vs blocks");
        }
        for (size_t d = 0; d + 1 < ms.shape.size(); ++d) {
            if (ms.shape[d] != prefix[d]) {
                throw std::runtime_error("MXFP4 scales prefix shape mismatch vs blocks");
            }
        }
        if (ms.shape.back() != G) {
            throw std::runtime_error("MXFP4 scales last dim (G) mismatch vs blocks");
        }
        // Flattened rows total
        size_t rowsTotal = 1;
        for (size_t d : prefix) rowsTotal *= d;
        rowsTotal *= G;

        const auto blocksBytes = fileBlocks->readBytes(blocksName);
        const auto scalesBytes = fileScales->readBytes(scalesName);

        if (blocksBytes.size() != numElements(mb.shape)) {
            throw std::runtime_error("blocks byte size mismatch vs shape");
        }
        if (scalesBytes.size() != numElements(ms.shape)) {
            throw std::runtime_error("scales byte size mismatch vs shape");
        }

        // Prepare output: shape *prefix, (G * B * 2)
        std::vector<size_t> outShape = prefix;
        outShape.push_back(G * B * 2);
        std::vector<float> out(rowsTotal * (B * 2));

        // Decode in rows
        const uint8_t* blk = blocksBytes.data();
        const uint8_t* scl = scalesBytes.data();
        for (size_t r = 0; r < rowsTotal; ++r) {
            int exp = static_cast<int>(scl[r]) - 127;
            size_t rowIn  = r * B;
            size_t rowOut = r * (B * 2);
            for (size_t b = 0; b < B; ++b) {
                uint8_t v = blk[rowIn + b];
                uint8_t idxLo = static_cast<uint8_t>(v & 0x0Fu);
                uint8_t idxHi = static_cast<uint8_t>((v >> 4) & 0x0Fu);
                float lo = std::ldexp(kFp4Values[idxLo], exp);
                float hi = std::ldexp(kFp4Values[idxHi], exp);
                out[rowOut + (b * 2 + 0)] = lo;
                out[rowOut + (b * 2 + 1)] = hi;
            }
        }

        return TensorF32{std::move(outShape), std::move(out)};
    }

    const SafeTensorsFile* locate(const std::string& tensorName) const {
        auto it = nameToFile_.find(tensorName);
        if (it == nameToFile_.end()) {
            throw std::runtime_error("tensor not found in any checkpoint file: " + tensorName);
        }
        size_t idx = it->second;
        return files_[idx].get();
    }

    std::string dirPath_;
    std::vector<std::unique_ptr<SafeTensorsFile>> files_;
    std::unordered_map<std::string, size_t> nameToFile_;
    std::unordered_map<std::string, ParamRef> paramMap_;
};
