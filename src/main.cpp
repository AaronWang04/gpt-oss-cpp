#include <algorithm>
#include <cstdint>
#include <iostream>
#include <string>
#include <vector>

#include "checkpoint.h"
#include "kv_cache.h"
#include "model.h"
#include "tokenizer.h"
#include "util.h"

int main(int argc, char* argv[]) {
    const std::string model_path = "gpt-oss-20b-model/original/model.safetensors";
    const std::string tokenizer_path = "gpt-oss-20b-model/o200k_base.tiktoken";
    const std::size_t vocab_size = 201088;
    const std::size_t num_layers = 24;

    // inference params
    const std::string prompt = (argc > 1) ? argv[1] : "hello my name is bob";
    const std::size_t max_tokens = 16;

    std::cout << "loading checkpoint" << std::endl;
    Checkpoint checkpoint(model_path);
    std::cout << "loading tokenizer" << std::endl;
    Tokenizer tokenizer(tokenizer_path);
    std::cout << "building model" << std::endl;
    GPTOSSModel model(checkpoint);

    std::vector<std::int32_t> tokens = tokenizer.encode(prompt);

    std::cout << "prompt tokens=" << tokens.size() << "\n";
    if (tokens.empty()) {
        throw std::runtime_error("prompt produced no tokens");
    }

    std::cout << prompt << std::endl;

    KVCache kv_cache(num_layers);

    // Prefill: process the whole prompt in one shot.
    const std::size_t prefill_len = tokens.size();
    std::vector<float> prefill_logits(prefill_len * vocab_size, 0.0f);
    model.forward(tokens, prefill_logits, kv_cache);

    auto argmax = [&](const float* p) {
        return static_cast<int>(std::distance(p, std::max_element(p, p + vocab_size)));
    };

    // Argmax over the last prompt token's logits → first generated token.
    int next_token = argmax(prefill_logits.data() + (prefill_len - 1) * vocab_size);
    std::cout << "next token: " << next_token << ' ' << tokenizer.decode(next_token) << std::endl;

    // Decode: one token at a time, reading from the KV cache.
    std::vector<float> decode_logits(vocab_size, 0.0f);
    for (std::size_t step = 1; step < max_tokens; ++step) {
        std::vector<std::int32_t> single = {next_token};
        model.forward(single, decode_logits, kv_cache);

        next_token = argmax(decode_logits.data());
        std::cout << "next token: " << next_token << ' ' << tokenizer.decode(next_token) << std::endl;
        std::cout.flush();
    }

    std::cout << "\n";
    return 0;
}
