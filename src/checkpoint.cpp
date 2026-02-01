/*
saftensor file format (predictably in this order)
```
{
  "TENSOR_NAME": {
    "dtype": "F16",
    "shape": [1, 16, 256],
    "data_offsets": [BEGIN, END]
  },
  "__metadata__": {
    "any_key": "any_string_value"
  }
}
```

*/
#include "checkpoint.h"
#include "util.h"

#include <algorithm>
#include <cassert>
#include <cctype>
#include <cstddef>
#include <cstring>
#include <fstream>
#include <iostream>
#include <stdexcept>

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>


Checkpoint::Checkpoint(const std::string& path) {
    path_ = path;
    std::ifstream file_stream(path, std::ios::binary);
    if (!file_stream) throw std::runtime_error("wrong path or file DNE");

    try {
        file_stream.read(reinterpret_cast<char*>(&header_len), 8);
        header.resize(header_len);
        file_stream.read(&header[0], header_len);
    } catch (const std::exception& e) {
        std::cerr << e.what() << std::endl;
    }
    processHeader();
    file_stream.close();
    // debugPrintCheckpoint();

    // memory map the file
    mmap_weights();
    finalizeTensorData();
}

Checkpoint::~Checkpoint() {
    if (map_base_ && map_base_ != MAP_FAILED) {
        munmap(map_base_, map_length_);
    }
    if (fd_ >= 0) {
        close(fd_);
    }
}

const TensorMeta_& Checkpoint::get(const std::string& name) const {
    auto it = meta_.find(name);
    if (it == meta_.end()) {
        throw std::runtime_error("tensor not found in checkpoint: " + name);
    }
    return it->second;
}

const std::uint16_t* Checkpoint::get_bf16_ptr(const std::string& name) const {
    const auto& meta = get(name);
    if (meta.dtype != DType::BF16) {
        throw std::runtime_error("tensor dtype is not BF16: " + name);
    }
    return reinterpret_cast<const std::uint16_t*>(meta.data);
}

std::size_t Checkpoint::get_bf16_count(const std::string& name) const {
    const auto& meta = get(name);
    if (meta.dtype != DType::BF16) {
        throw std::runtime_error("tensor dtype is not BF16: " + name);
    }
    return meta.byte_size / sizeof(std::uint16_t);
}

const std::uint8_t* Checkpoint::get_u8_ptr(const std::string& name) const {
    const auto& meta = get(name);
    if (meta.dtype != DType::U8) {
        throw std::runtime_error("tensor dtype is not U8: " + name);
    }
    return reinterpret_cast<const std::uint8_t*>(meta.data);
}

std::size_t Checkpoint::get_u8_count(const std::string& name) const {
    const auto& meta = get(name);
    if (meta.dtype != DType::U8) {
        throw std::runtime_error("tensor dtype is not U8: " + name);
    }
    return meta.byte_size;
}

Checkpoint::MXFP4Pair Checkpoint::get_mxfp4_pair(
    const std::string& base_name,
    std::initializer_list<std::uint64_t> expected_prefix,
    std::size_t expected_blocks_rank,
    std::size_t expected_scales_rank) const {
    const std::string blocks_name = base_name + ".blocks";
    const std::string scales_name = base_name + ".scales";
    const auto& blocks_meta = get(blocks_name);
    const auto& scales_meta = get(scales_name);
    assert(blocks_meta.shape.size() == expected_blocks_rank);
    assert(scales_meta.shape.size() == expected_scales_rank);
    std::size_t i = 0;
    for (auto expected_dim : expected_prefix) {
        assert(i < blocks_meta.shape.size());
        assert(i < scales_meta.shape.size());
        assert(blocks_meta.shape[i] == expected_dim);
        assert(scales_meta.shape[i] == expected_dim);
        i++;
    }
    return MXFP4Pair{
        get_u8_ptr(blocks_name),
        get_u8_count(blocks_name),
        get_u8_ptr(scales_name),
        get_u8_count(scales_name),
    };
}

void Checkpoint::mmap_weights() {
    fd_ = ::open(path_.c_str(), O_RDONLY);
    if (fd_ < 0) {
        throw std::runtime_error("failed to open safetensor file for mmap");
    }

    struct stat st {};
    if (fstat(fd_, &st) != 0) {
        close(fd_);
        throw std::runtime_error("failed to open safetensor file for mmap");
    }

    std::uint64_t data_offset = 8 + header_len;
    if (st.st_size < static_cast<off_t>(data_offset)) {
        close(fd_);
        fd_ = -1;
        throw std::runtime_error("invalid safetensor file");
    }

    size_t page_size = sysconf(_SC_PAGE_SIZE);
    std::uint64_t page_offset = data_offset % static_cast<std::uint64_t>(page_size);
    const off_t map_offset = static_cast<off_t>(data_offset - page_offset);
    map_length_ = static_cast<std::size_t>(st.st_size - map_offset);

    map_base_ = mmap(nullptr, map_length_, PROT_READ, MAP_PRIVATE, fd_, map_offset);
    if (map_base_ == MAP_FAILED) {
        close(fd_);
        fd_ = -1;
        throw std::runtime_error("mmap failed for safetensor weights");
    }
    weights = reinterpret_cast<std::byte*>(map_base_) + page_offset;
}

void Checkpoint::finalizeTensorData() {
    for (auto& it : meta_) {
        auto& meta = it.second;
        if (meta.offset.size() != 2) {
            throw std::runtime_error("invalid data_offsets for tensor: " + meta.name);
        }
        const std::uint64_t begin = meta.offset[0];
        const std::uint64_t end = meta.offset[1];
        if (end < begin) {
            throw std::runtime_error("invalid data_offsets range for tensor: " + meta.name);
        }
        meta.byte_size = static_cast<std::size_t>(end - begin);
        meta.data = weights + begin;
    }
}

void Checkpoint::debugPrintCheckpoint() {
    for (const auto& it : meta_) {
        std::cout << it.second;
    }
}

void Checkpoint::processHeader() {
    std::uint64_t i = 0;
    skipWhitespace(i);
    if (header[i] == '{')
        i++;
    else
        throw std::runtime_error("safetensor header unable to parse");

    while (i < header.size()) {
        skipWhitespace(i);
        if (header[i] == '}') break;
        TensorMeta_ meta_instance;
        std::string entry_name = parseJSONString(i);
        skipWhitespace(i);
        expectChar(i, ':');
        skipWhitespace(i);

        if (entry_name == "__metadata__") {
            skipJSONValue(i);
        } else {
            meta_instance.name = entry_name;
            parseTensorMeta(i, meta_instance);
            meta_.emplace(meta_instance.name, std::move(meta_instance));
        }

        skipWhitespace(i);
        if (header[i] == ',') {
            i++;
            continue;
        }
        if (header[i] == '}') break;
        throw std::runtime_error("safetensor header unable to parse");
    }
}

void Checkpoint::skipWhitespace(std::uint64_t& i) {
    while (i < header.size() && std::isspace(static_cast<unsigned char>(header[i]))) i++;
}

void Checkpoint::expectChar(std::uint64_t& i, char expected) {
    if (i >= header.size() || header[i] != expected) {
        throw std::runtime_error("safetensor header unable to parse");
    }
    i++;
}

std::string Checkpoint::parseJSONString(std::uint64_t& i) {
    if (header[i] == '"')
        i++;
    else
        throw std::runtime_error("safetensor header unable to parse");

    std::string parsed_string;
    while (i < header.size()) {
        char c = header[i++];
        // assume no escape characters (who would put escape chars in a safetensor nani)
        if (c == '\\') {
            throw std::runtime_error("not implemented");
        } else if (c == '"') {
            break;
        } else {
            parsed_string.push_back(c);
        }
    }
    return parsed_string;
}

std::uint64_t Checkpoint::parseUInt64(std::uint64_t& i) {
    skipWhitespace(i);
    if (i >= header.size() || !std::isdigit(static_cast<unsigned char>(header[i]))) {
        throw std::runtime_error("safetensor header unable to parse");
    }
    std::uint64_t value = 0;
    while (i < header.size() && std::isdigit(static_cast<unsigned char>(header[i]))) {
        value = value * 10 + static_cast<std::uint64_t>(header[i] - '0');
        i++;
    }
    return value;
}

std::vector<std::uint64_t> Checkpoint::parseUInt64Array(std::uint64_t& i) {
    std::vector<std::uint64_t> values;
    expectChar(i, '[');
    skipWhitespace(i);
    if (header[i] == ']') {
        i++;
        return values;
    }
    while (i < header.size()) {
        values.push_back(parseUInt64(i));
        skipWhitespace(i);
        if (header[i] == ',') {
            i++;
            continue;
        }
        if (header[i] == ']') {
            i++;
            break;
        }
        throw std::runtime_error("safetensor header unable to parse");
    }
    return values;
}

DType Checkpoint::parseDTypeString(const std::string& dtype) {
    if (dtype == "BF16") return DType::BF16;
    if (dtype == "U8") return DType::U8;
    throw std::runtime_error("unexpected dtype here");
}

void Checkpoint::parseTensorMeta(std::uint64_t& i, TensorMeta_& meta_instance) {
    expectChar(i, '{');
    while (i < header.size()) {
        skipWhitespace(i);
        if (header[i] == '}') {
            i++;
            break;
        }
        std::string field = parseJSONString(i);
        skipWhitespace(i);
        expectChar(i, ':');
        skipWhitespace(i);

        if (field == "dtype") {
            meta_instance.dtype = parseDTypeString(parseJSONString(i));
        } else if (field == "shape") {
            meta_instance.shape = parseUInt64Array(i);
        } else if (field == "data_offsets") {
            meta_instance.offset = parseUInt64Array(i);
        } else {
            skipJSONValue(i);
        }

        skipWhitespace(i);
        if (header[i] == ',') {
            i++;
            continue;
        }
        if (header[i] == '}') {
            i++;
            break;
        }
        throw std::runtime_error("safetensor header unable to parse");
    }
}

void Checkpoint::skipJSONValue(std::uint64_t& i) {
    skipWhitespace(i);
    if (i >= header.size()) throw std::runtime_error("safetensor header unable to parse");

    char c = header[i];
    if (c == '{') {
        i++;
        while (i < header.size()) {
            skipWhitespace(i);
            if (header[i] == '}') {
                i++;
                break;
            }
            parseJSONString(i);
            skipWhitespace(i);
            expectChar(i, ':');
            skipWhitespace(i);
            skipJSONValue(i);
            skipWhitespace(i);
            if (header[i] == ',') {
                i++;
                continue;
            }
            if (header[i] == '}') {
                i++;
                break;
            }
            throw std::runtime_error("safetensor header unable to parse");
        }
        return;
    }

    if (c == '[') {
        i++;
        while (i < header.size()) {
            skipWhitespace(i);
            if (header[i] == ']') {
                i++;
                break;
            }
            skipJSONValue(i);
            skipWhitespace(i);
            if (header[i] == ',') {
                i++;
                continue;
            }
            if (header[i] == ']') {
                i++;
                break;
            }
            throw std::runtime_error("safetensor header unable to parse");
        }
        return;
    }

    if (c == '"') {
        parseJSONString(i);
        return;
    }

    if (c == '-' || std::isdigit(static_cast<unsigned char>(c))) {
        if (c == '-') i++;
        parseUInt64(i);
        return;
    }

    if (header.compare(i, 4, "true") == 0) {
        i += 4;
        return;
    }
    if (header.compare(i, 5, "false") == 0) {
        i += 5;
        return;
    }
    if (header.compare(i, 4, "null") == 0) {
        i += 4;
        return;
    }

    throw std::runtime_error("safetensor header unable to parse");
}
