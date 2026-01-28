#include <algorithm>
#include <cstddef>
#include <fstream>
#include <queue>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>


std::string base64_decode(std::string_view input) {
    static constexpr unsigned char kInvalid = 0xFF;
    static constexpr unsigned char table[256] = {
        kInvalid, kInvalid, kInvalid, kInvalid, kInvalid, kInvalid, kInvalid, kInvalid,
        kInvalid, kInvalid, kInvalid, kInvalid, kInvalid, kInvalid, kInvalid, kInvalid,
        kInvalid, kInvalid, kInvalid, kInvalid, kInvalid, kInvalid, kInvalid, kInvalid,
        kInvalid, kInvalid, kInvalid, kInvalid, kInvalid, kInvalid, kInvalid, kInvalid,
        kInvalid, kInvalid, kInvalid, kInvalid, kInvalid, kInvalid, kInvalid, kInvalid,
        kInvalid, kInvalid, kInvalid, 62,       kInvalid, kInvalid, kInvalid, 63,
        52,       53,       54,       55,       56,       57,       58,       59,
        60,       61,       kInvalid, kInvalid, kInvalid, 0,        kInvalid, kInvalid,
        kInvalid, 0,        1,        2,        3,        4,        5,        6,
        7,        8,        9,        10,       11,       12,       13,       14,
        15,       16,       17,       18,       19,       20,       21,       22,
        23,       24,       25,       kInvalid, kInvalid, kInvalid, kInvalid, kInvalid,
        kInvalid, 26,       27,       28,       29,       30,       31,       32,
        33,       34,       35,       36,       37,       38,       39,       40,
        41,       42,       43,       44,       45,       46,       47,       48,
        49,       50,       51,       kInvalid, kInvalid, kInvalid, kInvalid, kInvalid,
        kInvalid, kInvalid, kInvalid, kInvalid, kInvalid, kInvalid, kInvalid, kInvalid,
        kInvalid, kInvalid, kInvalid, kInvalid, kInvalid, kInvalid, kInvalid, kInvalid,
        kInvalid, kInvalid, kInvalid, kInvalid, kInvalid, kInvalid, kInvalid, kInvalid,
        kInvalid, kInvalid, kInvalid, kInvalid, kInvalid, kInvalid, kInvalid, kInvalid,
        kInvalid, kInvalid, kInvalid, kInvalid, kInvalid, kInvalid, kInvalid, kInvalid,
        kInvalid, kInvalid, kInvalid, kInvalid, kInvalid, kInvalid, kInvalid, kInvalid,
        kInvalid, kInvalid, kInvalid, kInvalid, kInvalid, kInvalid, kInvalid, kInvalid,
        kInvalid, kInvalid, kInvalid, kInvalid, kInvalid, kInvalid, kInvalid, kInvalid,
        kInvalid, kInvalid, kInvalid, kInvalid, kInvalid, kInvalid, kInvalid, kInvalid,
        kInvalid, kInvalid, kInvalid, kInvalid, kInvalid, kInvalid, kInvalid, kInvalid,
        kInvalid, kInvalid, kInvalid, kInvalid, kInvalid, kInvalid, kInvalid, kInvalid,
        kInvalid, kInvalid, kInvalid, kInvalid, kInvalid, kInvalid, kInvalid, kInvalid,
        kInvalid, kInvalid, kInvalid, kInvalid, kInvalid, kInvalid, kInvalid, kInvalid,
        kInvalid, kInvalid, kInvalid, kInvalid, kInvalid, kInvalid, kInvalid, kInvalid,
        kInvalid, kInvalid, kInvalid, kInvalid, kInvalid, kInvalid, kInvalid, kInvalid,
        kInvalid, kInvalid, kInvalid, kInvalid, kInvalid, kInvalid, kInvalid, kInvalid,
        kInvalid, kInvalid, kInvalid, kInvalid, kInvalid, kInvalid, kInvalid, kInvalid,
        kInvalid, kInvalid, kInvalid, kInvalid, kInvalid, kInvalid, kInvalid, kInvalid,
        kInvalid, kInvalid, kInvalid, kInvalid, kInvalid, kInvalid, kInvalid, kInvalid,
        kInvalid, kInvalid, kInvalid, kInvalid, kInvalid, kInvalid, kInvalid, kInvalid,
        kInvalid, kInvalid, kInvalid, kInvalid, kInvalid, kInvalid, kInvalid, kInvalid,
        kInvalid, kInvalid, kInvalid, kInvalid, kInvalid, kInvalid, kInvalid, kInvalid,
        kInvalid, kInvalid, kInvalid, kInvalid, kInvalid, kInvalid, kInvalid, kInvalid,
        kInvalid, kInvalid, kInvalid, kInvalid, kInvalid, kInvalid, kInvalid, kInvalid,
        kInvalid, kInvalid, kInvalid, kInvalid, kInvalid, kInvalid, kInvalid, kInvalid,
        kInvalid, kInvalid, kInvalid, kInvalid, kInvalid, kInvalid, kInvalid, kInvalid,
        kInvalid, kInvalid, kInvalid, kInvalid, kInvalid, kInvalid, kInvalid, kInvalid,
        kInvalid, kInvalid, kInvalid, kInvalid, kInvalid, kInvalid, kInvalid, kInvalid,
        kInvalid, kInvalid, kInvalid, kInvalid, kInvalid, kInvalid, kInvalid, kInvalid,
        kInvalid, kInvalid, kInvalid, kInvalid, kInvalid, kInvalid, kInvalid, kInvalid,
        kInvalid, kInvalid, kInvalid, kInvalid, kInvalid, kInvalid, kInvalid, kInvalid,
        kInvalid, kInvalid, kInvalid, kInvalid, kInvalid, kInvalid, kInvalid, kInvalid,
        kInvalid, kInvalid, kInvalid, kInvalid, kInvalid, kInvalid, kInvalid, kInvalid
    };

    std::string output;
    output.reserve(input.size() * 3 / 4);

    unsigned int buffer = 0;
    int bits_collected = 0;
    for (char c : input) {
        if (c == '=') {
            break;
        }
        unsigned char val = table[static_cast<unsigned char>(c)];
        if (val == kInvalid) {
            continue;
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

// byte pair encoding tokenizer
class TikTokenTokenizer {

public:
    explicit TikTokenTokenizer(const std::string& path) {
        path_ = path;
        if (!path_.empty()) {
            load_vocab(path_);
        } else {
            throw std::runtime_error("failed to load tokenizer file");
        }
    }

    ~TikTokenTokenizer() = default;

    std::vector<int> encode(std::string_view text) const {
        if (text.empty() || token_to_id_.empty()) {
            return {};
        }
        return bpe_encode(text);
    }

    std::string_view decode(int prev, int token) const {
        (void)prev;
        if (token < 0 || static_cast<size_t>(token) >= id_to_token_.size()) {
            return {};
        }
        return id_to_token_[static_cast<size_t>(token)];
    }

private:
    struct Part {
        size_t start;
        size_t end;
        int prev;
        int next;
        bool alive;
    };

    struct MergeCandidate {
        int rank;
        int left;
        int right;
    };

    struct CandidateCompare {
        bool operator()(const MergeCandidate& a, const MergeCandidate& b) const {
            return a.rank > b.rank;
        }
    };

    void load_vocab(const std::string& path) {
        std::ifstream file(path);

        std::string line;
        size_t max_id = 0;
        while (std::getline(file, line)) {
            if (line.empty()) {
                break;
            }
            const auto space_pos = line.find(' ');

            std::string_view b64_token(line.data(), space_pos);
            const int token_id = std::stoi(line.substr(space_pos + 1));
            std::string token = base64_decode(b64_token);
            token_to_id_[token] = token_id;
            if (static_cast<size_t>(token_id) > max_id) {
                max_id = static_cast<size_t>(token_id);
            }
            if (token_id >= 0) {
                if (id_to_token_.size() <= static_cast<size_t>(token_id)) {
                    id_to_token_.resize(static_cast<size_t>(token_id) + 1);
                }
                id_to_token_[static_cast<size_t>(token_id)] = std::move(token);
            }
        }
        if (id_to_token_.size() <= max_id) {
            id_to_token_.resize(max_id + 1);
        }
    }

    int rank_for(std::string_view piece) const {
        auto it = token_to_id_.find(std::string(piece));
        if (it == token_to_id_.end()) {
            return -1;
        }
        return it->second;
    }

    std::vector<int> bpe_encode(std::string_view text) const {
        std::string piece(text);
        if (piece.empty()) {
            return {};
        }

        std::vector<Part> parts;
        parts.reserve(piece.size());
        for (size_t i = 0; i < piece.size(); ++i) {
            Part part;
            part.start = i;
            part.end = i + 1;
            part.prev = (i == 0) ? -1 : static_cast<int>(i - 1);
            part.next = (i + 1 < piece.size()) ? static_cast<int>(i + 1) : -1;
            part.alive = true;
            parts.push_back(part);
        }

        std::priority_queue<MergeCandidate, std::vector<MergeCandidate>, CandidateCompare> queue;
        for (size_t i = 0; i + 1 < parts.size(); ++i) {
            const size_t start = parts[i].start;
            const size_t end = parts[i + 1].end;
            const int rank = rank_for(std::string_view(piece.data() + start, end - start));
            if (rank >= 0) {
                queue.push(MergeCandidate{rank, static_cast<int>(i), static_cast<int>(i + 1)});
            }
        }

        int head = parts.empty() ? -1 : 0;
        while (!queue.empty()) {
            MergeCandidate cand = queue.top();
            queue.pop();

            if (cand.left < 0 || cand.right < 0) {
                continue;
            }
            if (static_cast<size_t>(cand.left) >= parts.size() ||
                static_cast<size_t>(cand.right) >= parts.size()) {
                continue;
            }
            Part& left = parts[static_cast<size_t>(cand.left)];
            Part& right = parts[static_cast<size_t>(cand.right)];
            if (!left.alive || !right.alive || left.next != cand.right || right.prev != cand.left) {
                continue;
            }

            const size_t start = left.start;
            const size_t end = right.end;
            const int merged_rank = rank_for(std::string_view(piece.data() + start, end - start));
            if (merged_rank < 0) {
                continue;
            }

            Part merged;
            merged.start = start;
            merged.end = end;
            merged.prev = left.prev;
            merged.next = right.next;
            merged.alive = true;
            const int merged_index = static_cast<int>(parts.size());
            parts.push_back(merged);

            left.alive = false;
            right.alive = false;

            if (merged.prev != -1) {
                parts[static_cast<size_t>(merged.prev)].next = merged_index;
            } else {
                head = merged_index;
            }
            if (merged.next != -1) {
                parts[static_cast<size_t>(merged.next)].prev = merged_index;
            }

            if (merged.prev != -1) {
                const Part& prev = parts[static_cast<size_t>(merged.prev)];
                const int rank = rank_for(std::string_view(piece.data() + prev.start, merged.end - prev.start));
                if (rank >= 0) {
                    queue.push(MergeCandidate{rank, merged.prev, merged_index});
                }
            }
            if (merged.next != -1) {
                const Part& next = parts[static_cast<size_t>(merged.next)];
                const int rank = rank_for(std::string_view(piece.data() + merged.start, next.end - merged.start));
                if (rank >= 0) {
                    queue.push(MergeCandidate{rank, merged_index, merged.next});
                }
            }
        }

        std::vector<int> tokens;
        int current = head;
        while (current != -1) {
            const Part& part = parts[static_cast<size_t>(current)];
            if (!part.alive) {
                current = part.next;
                continue;
            }
            const std::string_view view(piece.data() + part.start, part.end - part.start);
            auto it = token_to_id_.find(std::string(view));
            if (it != token_to_id_.end()) {
                tokens.push_back(it->second);
            } else {
                for (size_t i = part.start; i < part.end; ++i) {
                    std::string byte_token(1, piece[i]);
                    auto byte_it = token_to_id_.find(byte_token);
                    if (byte_it != token_to_id_.end()) {
                        tokens.push_back(byte_it->second);
                    }
                }
            }
            current = part.next;
        }
        return tokens;
    }

    std::string path_;
    std::vector<std::string> id_to_token_;
    std::unordered_map<std::string, int> token_to_id_;

};
