#include <iostream>
#include <string>

#include "checkpoint.cpp"
#include "tokenizer.cpp"

int main(int argc, char* argv[]) {
    const std::string model_path = "gpt-oss-20b-model/original/model.safetensors";
    const std::string tokenizer_path = "gpt-oss-20b-model/o200k_base.tiktoken";
    
    // inference params
    // float temperature = 1.0f;
    // float top_p = 0.9f;
    Checkpoint checkpoint(model_path);
    Tokenizer tokenizer(tokenizer_path);

    std::vector<int> tokens = tokenizer.encode("hello my name is aaron");
    for (int i : tokens) {
        std::cout << i << ' ';
    }
    std::cout << std::endl;
};
