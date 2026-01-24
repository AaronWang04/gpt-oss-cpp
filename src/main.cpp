#include <iostream>
#include <string>
#include "checkpoint.cpp"


int main(int argc, char* argv[]) {
    const std::string model_path = "gpt-oss-20b-model/original/model.safetensors";
    // const std::string tokenizer_path = "";
    
    // inference params
    // float temperature = 1.0f;
    // float top_p = 0.9f;
    Checkpoint checkpoint(model_path);
    
}
