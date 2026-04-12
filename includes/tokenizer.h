#pragma once

#include <cstdint>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

class Tokenizer {
public:
    explicit Tokenizer(const std::string& path);
    ~Tokenizer();

    std::vector<std::int32_t> encode(std::string text) const;
    std::string decode(std::int32_t token) const;
    std::vector<std::string> regex_split(
        const std::string& text,
        const std::string& pattern) const;

private:
    std::string path_;
    std::vector<std::string> id_to_token_;
    std::unordered_map<std::string, std::int32_t> token_to_id_;

    std::vector<std::int32_t> bpe_encode_piece(const std::string& piece) const;
};
