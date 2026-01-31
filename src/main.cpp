#include <algorithm>
#include <cstdint>
#include <iostream>
#include <string>
#include <vector>

#include "checkpoint.h"
#include "model.h"
#include "tokenizer.h"

int main(int argc, char* argv[]) {
    const std::string model_path = "gpt-oss-20b-model/original/model.safetensors";
    const std::string tokenizer_path = "gpt-oss-20b-model/o200k_base.tiktoken";
    const std::size_t vocab_size = 201088;

    // inference params
    const std::string prompt = (argc > 1) ? argv[1] : "Hello";
    const std::size_t max_new_tokens = 16;
    std::cerr << "loading checkpoint\n";
    Checkpoint checkpoint(model_path);
    std::cerr << "loading tokenizer\n";
    Tokenizer tokenizer(tokenizer_path);

    std::cerr << "building model\n";
    GPTOSSModel model(checkpoint);

    std::vector<int> tokens = tokenizer.encode(prompt);
    std::cerr << "prompt tokens=" << tokens.size() << "\n";
    if (tokens.empty()) {
        std::cerr << "prompt produced no tokens; aborting.\n";
        return 1;
    }
    std::cout << prompt;
    for (std::size_t step = 0; step < max_new_tokens; ++step) {
        const std::size_t seq_len = tokens.size();
        std::vector<float> logits(seq_len * vocab_size, 0.0f);

        std::vector<std::int32_t> token_ids(tokens.begin(), tokens.end());
        model.forward(token_ids, logits, seq_len);

        const float* last_logits = logits.data() + (seq_len - 1) * vocab_size;
        const auto it = std::max_element(last_logits, last_logits + vocab_size);
        const int next_token = static_cast<int>(std::distance(last_logits, it));

        tokens.push_back(next_token);
        std::cout << tokenizer.decode(next_token);
        std::cout.flush();
    }

    std::cout << "\n";
    return 0;
}
