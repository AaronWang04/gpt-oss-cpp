#pragma once

#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <iosfwd>
#include <string>
#include <unordered_map>
#include <vector>

enum DType {
    BF16,
    U8
};

struct TensorMeta_ {
    std::string name;
    DType dtype;
    std::vector<std::uint64_t> shape;
    std::vector<std::uint64_t> offset;
    const std::byte* data{nullptr};
    std::size_t byte_size{0};
};

class Checkpoint {
public:
    explicit Checkpoint(const std::string& path);
    ~Checkpoint();

    const TensorMeta_& get(const std::string& name) const;
    const std::uint16_t* get_bf16_ptr(const std::string& name) const;
    std::size_t get_bf16_count(const std::string& name) const;
    const std::uint8_t* get_u8_ptr(const std::string& name) const;
    std::size_t get_u8_count(const std::string& name) const;

    struct MXFP4Pair {
        const std::uint8_t* blocks;
        std::size_t blocks_count;
        const std::uint8_t* scales;
        std::size_t scales_count;
    };

    MXFP4Pair get_mxfp4_pair(
        const std::string& base_name,
        std::initializer_list<std::uint64_t> expected_prefix,
        std::size_t expected_blocks_rank,
        std::size_t expected_scales_rank) const;

private:
    std::uint64_t header_len{};
    std::string header;
    std::unordered_map<std::string, TensorMeta_> meta_;
    std::byte* weights{nullptr};
    std::string path_;
    int fd_{-1};
    void* map_base_{nullptr};
    std::size_t map_length_{0};

    void mmap_weights();
    void finalizeTensorData();
    void debugPrintCheckpoint();
    void processHeader();
    void skipWhitespace(std::uint64_t& i);
    void expectChar(std::uint64_t& i, char expected);
    std::string parseJSONString(std::uint64_t& i);
    std::uint64_t parseUInt64(std::uint64_t& i);
    std::vector<std::uint64_t> parseUInt64Array(std::uint64_t& i);
    DType parseDTypeString(const std::string& dtype);
    void parseTensorMeta(std::uint64_t& i, TensorMeta_& meta_instance);
    void skipJSONValue(std::uint64_t& i);
};
