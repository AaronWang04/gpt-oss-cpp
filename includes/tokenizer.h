#pragma once

#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

class Tokenizer {
public:
    explicit Tokenizer(const std::string& path);
    ~Tokenizer();

    std::vector<int> encode(std::string text) const;
    std::string decode(int token) const;
    std::vector<std::string> regex_split(
        const std::string& text,
        const std::string& pattern) const;

private:
    std::string path_;
    std::vector<std::string> id_to_token_;
    std::unordered_map<std::string, int> token_to_id_;

    std::vector<int> bpe_encode_piece(const std::string& piece) const;
    void load_token_file();
};
