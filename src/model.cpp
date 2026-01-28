#include <stdint.h>
#include "src/tokenizer.cpp"

class GPT_OSS_Model {

private:
    TikTokenTokenizer tokenizer;

    uint32_t context_length;
    uint32_t num_blocks;
    uint32_t num_experts;
    uint32_t num_active_experts;
    uint32_t embedding_dim;
    uint32_t mlp_dim;
    float swiglu_limit;
    uint32_t head_dim;
    uint32_t num_heads;
    uint32_t num_kv_heads;
    uint32_t attention_window;
    float rope_theta;
    float interpolation_scale;
    float yarn_offset;
    float yarn_scale;
    float yarn_multiplier;
    float rmsnorm_epsilon;

    uint32_t vocabulary_size;
    size_t max_batch_tokens;

public:
    GPT_OSS_Model() {
        
    }

};
