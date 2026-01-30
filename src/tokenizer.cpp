#include <array>
#include <cstdint>
#include <fstream>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>
#include <unordered_map>

// regex engine that supports unicode cuz std::regex is only ECMAScript
#include <unicode/regex.h>
#include <unicode/unistr.h>


constexpr const char* kO200kPatStr =
    "[^\\r\\n\\p{L}\\p{N}]?[\\p{Lu}\\p{Lt}\\p{Lm}\\p{Lo}\\p{M}]*[\\p{Ll}\\p{Lm}\\p{Lo}\\p{M}]+(?i:'s|'t|'re|'ve|'m|'ll|'d)"
    "|[^\\r\\n\\p{L}\\p{N}]?[\\p{Lu}\\p{Lt}\\p{Lm}\\p{Lo}\\p{M}]+[\\p{Ll}\\p{Lm}\\p{Lo}\\p{M}]*(?i:'s|'t|'re|'ve|'m|'ll|'d)"
    "|\\p{N}{1,3}"
    "| ?[^\\s\\p{L}\\p{N}]+[\\r\\n/]*"
    "|\\s*[\\r\\n]+"
    "|\\s+(?!\\S)"
    "|\\s+";

std::string base64_decode(std::string_view input) {
    static const std::array<int8_t, 256> table = [] {
        std::array<int8_t, 256> t{};
        t.fill(-1);
        for (int i = 0; i < 26; ++i) t[static_cast<uint8_t>('A' + i)] = static_cast<int8_t>(i);
        for (int i = 0; i < 26; ++i) t[static_cast<uint8_t>('a' + i)] = static_cast<int8_t>(26 + i);
        for (int i = 0; i < 10; ++i) t[static_cast<uint8_t>('0' + i)] = static_cast<int8_t>(52 + i);
        t[static_cast<uint8_t>('+')] = 62;
        t[static_cast<uint8_t>('/')] = 63;
        return t;
    }();

    std::string output;
    unsigned int buffer = 0;
    int bits_collected = 0;
    output.reserve(input.size()*3/4);
    for (char c : input) {
        if (c == '=') break;
        int8_t val = table[c];
        if (val == -1) {
            throw std::invalid_argument("base64: invalid input character");
        }
        buffer = (buffer << 6) | val;
        bits_collected += 6;
        if (bits_collected >= 8) {
            bits_collected -= 8;
            output.push_back(static_cast<char>((buffer >> bits_collected) & 0xFF));
        }
    }
    return output;
}

class Tokenizer {
public:
    explicit Tokenizer(const std::string& path) {
        path_ = path;
        load_token_file();
    }

    ~Tokenizer() = default;
    
    // byte-pair encoding
    std::vector<int> encode(std::string text) const {
        if (text.empty()) return {};
        auto start = text.find("<|");
        if (start != std::string::npos) {
            auto end = text.find("|>", start + 2);
            if (end != std::string::npos) {
                throw std::runtime_error("special token handling to be implemneted");
            }
        }
        std::vector<int> token_ids;
        auto pieces = regex_split(text, kO200kPatStr);
        for (const auto& piece : pieces) {
            auto piece_tokens = bpe_encode_piece(piece);
            token_ids.insert(token_ids.end(), piece_tokens.begin(), piece_tokens.end());
        }
        return token_ids;
    }

    std::string decode(int token) const {
        return id_to_token_[token];
    }

    std::vector<std::string> regex_split(const std::string& text, const std::string& pattern) const {
        UErrorCode status = U_ZERO_ERROR;
        icu::UnicodeString pattern_u = icu::UnicodeString::fromUTF8(pattern);
        std::unique_ptr<icu::RegexPattern> re(icu::RegexPattern::compile(pattern_u, 0, status));
        if (U_FAILURE(status)) {
            throw std::runtime_error("tokenizer: failed to compile ICU regex");
        }

        icu::UnicodeString text_u = icu::UnicodeString::fromUTF8(text);
        std::unique_ptr<icu::RegexMatcher> matcher(re->matcher(text_u, status));
        if (U_FAILURE(status)) {
            throw std::runtime_error("tokenizer: failed to create ICU matcher");
        }

        std::vector<std::string> pieces;
        while (matcher->find(status)) {
            if (U_FAILURE(status)) {
                throw std::runtime_error("tokenizer: ICU regex match failed");
            }
            icu::UnicodeString match = matcher->group(status);
            if (U_FAILURE(status)) {
                throw std::runtime_error("tokenizer: ICU regex group failed");
            }
            std::string out;
            match.toUTF8String(out);
            pieces.push_back(out);
        }
        return pieces;
    }

private:
    std::string path_;
    std::vector<std::string> id_to_token_;
    std::unordered_map<std::string, int> token_to_id_;

    std::vector<int> bpe_encode_piece(const std::string& piece) const {
        if (piece.empty()) return {};
        std::vector<std::string> symbols;
        symbols.reserve(piece.size());
        for (unsigned char c : piece) {
            symbols.emplace_back(1, static_cast<char>(c));
        }

        while (symbols.size() > 1) {
            int best_rank = std::numeric_limits<int>::max();
            size_t best_idx = 0;
            bool found = false;

            for (size_t i = 0; i + 1 < symbols.size(); ++i) {
                std::string merged = symbols[i] + symbols[i + 1];
                auto it = token_to_id_.find(merged);
                if (it == token_to_id_.end()) continue;
                if (it->second < best_rank) {
                    best_rank = it->second;
                    best_idx = i;
                    found = true;
                }
            }

            if (!found) break;
            symbols[best_idx] += symbols[best_idx + 1];
            symbols.erase(symbols.begin() + best_idx + 1);
        }

        std::vector<int> token_ids;
        token_ids.reserve(symbols.size());
        for (const auto& sym : symbols) {
            auto it = token_to_id_.find(sym);
            if (it == token_to_id_.end()) {
                throw std::runtime_error("tokenizer: missing token for symbol");
            }
            token_ids.push_back(it->second);
        }
        return token_ids;
    }


    void load_token_file() {
        std::ifstream file(path_);
        std::string line;
        while (std::getline(file, line)) {
            if (line.empty()) break;
            int space_idx = line.find(' ');
            std::string_view b64_token(line.data(), space_idx);
            const int token_id = std::stoi(line.substr(space_idx+1));
            std::string token = base64_decode(b64_token);
            token_to_id_[token] = token_id;
            id_to_token_.push_back(std::move(token));
        }
    }

};
