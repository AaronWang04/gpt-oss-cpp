#include "model.h"

#include "checkpoint.h"

#include <string>

namespace {

// icba parsing this from the json and its not like i can inferencing any other sizes anyways
// ram is expensive these days ._.
struct Config20B {
    int num_hidden_layers;
    int num_experts;
    int experts_per_token;
    int vocab_size;
    int hidden_size;
    int intermediate_size;
    int swiglu_limit;
    int head_dim;
    int num_attention_heads;
    int num_key_value_heads;
    int sliding_window;
    int initial_context_length;
    int rope_theta;
    int rope_scaling_factor;
    int rope_ntk_alpha;
    int rope_ntk_beta;
};

const Config20B kConfig20B = {
    24,
    32,
    4,
    201088,
    2880,
    2880,
    7,
    64,
    64,
    8,
    128,
    4096,
    150000,
    32,
    1,
    32,
};

}  // namespace

Embedding::Embedding(Checkpoint& checkpoint) {
    weight = checkpoint.get_bf16_ptr("embedding.weight");
    weight_count = checkpoint.get_bf16_count("embedding.weight");
    hidden_size = kConfig20B.hidden_size;
}

AttentionBlock::AttentionBlock(Checkpoint& checkpoint, int layer_idx) {
    std::string prefix = "block." + std::to_string(layer_idx) + ".attn.";
    norm_scale = checkpoint.get_bf16_ptr(prefix + "norm.scale");
    norm_scale_count = checkpoint.get_bf16_count(prefix + "norm.scale");
    qkv_weight = checkpoint.get_bf16_ptr(prefix + "qkv.weight");
    qkv_weight_count = checkpoint.get_bf16_count(prefix + "qkv.weight");
    qkv_bias = checkpoint.get_bf16_ptr(prefix + "qkv.bias");
    qkv_bias_count = checkpoint.get_bf16_count(prefix + "qkv.bias");
    out_weight = checkpoint.get_bf16_ptr(prefix + "out.weight");
    out_weight_count = checkpoint.get_bf16_count(prefix + "out.weight");
    out_bias = checkpoint.get_bf16_ptr(prefix + "out.bias");
    out_bias_count = checkpoint.get_bf16_count(prefix + "out.bias");
    sinks = checkpoint.get_bf16_ptr(prefix + "sinks");
    sinks_count = checkpoint.get_bf16_count(prefix + "sinks");
    hidden_size = kConfig20B.hidden_size;
}

MLPBlock::MLPBlock(Checkpoint& checkpoint, int layer_idx) {
    std::string prefix = "block." + std::to_string(layer_idx) + ".mlp.";
    norm_scale = checkpoint.get_bf16_ptr(prefix + "norm.scale");
    norm_scale_count = checkpoint.get_bf16_count(prefix + "norm.scale");
    gate_weight = checkpoint.get_bf16_ptr(prefix + "gate.weight");
    gate_weight_count = checkpoint.get_bf16_count(prefix + "gate.weight");
    gate_bias = checkpoint.get_bf16_ptr(prefix + "gate.bias");
    gate_bias_count = checkpoint.get_bf16_count(prefix + "gate.bias");
    mlp1_bias = checkpoint.get_bf16_ptr(prefix + "mlp1_bias");
    mlp1_bias_count = checkpoint.get_bf16_count(prefix + "mlp1_bias");
    mlp2_bias = checkpoint.get_bf16_ptr(prefix + "mlp2_bias");
    mlp2_bias_count = checkpoint.get_bf16_count(prefix + "mlp2_bias");
    mlp1_weight_blocks = checkpoint.get_u8_ptr(prefix + "mlp1_weight.blocks");
    mlp1_weight_blocks_count = checkpoint.get_u8_count(prefix + "mlp1_weight.blocks");
    mlp1_weight_scales = checkpoint.get_u8_ptr(prefix + "mlp1_weight.scales");
    mlp1_weight_scales_count = checkpoint.get_u8_count(prefix + "mlp1_weight.scales");
    mlp2_weight_blocks = checkpoint.get_u8_ptr(prefix + "mlp2_weight.blocks");
    mlp2_weight_blocks_count = checkpoint.get_u8_count(prefix + "mlp2_weight.blocks");
    mlp2_weight_scales = checkpoint.get_u8_ptr(prefix + "mlp2_weight.scales");
    mlp2_weight_scales_count = checkpoint.get_u8_count(prefix + "mlp2_weight.scales");
    hidden_size = kConfig20B.hidden_size;
}

TransformerBlock::TransformerBlock(Checkpoint& checkpoint, int layer_idx)
    : attn(checkpoint, layer_idx), mlp(checkpoint, layer_idx) {
    hidden_size = kConfig20B.hidden_size;
}

UnEmbedding::UnEmbedding(Checkpoint& checkpoint) {
    weight = checkpoint.get_bf16_ptr("unembedding.weight");
    weight_count = checkpoint.get_bf16_count("unembedding.weight");
    hidden_size = kConfig20B.hidden_size;
    vocab_size = kConfig20B.vocab_size;
}

GPTOSSModel::GPTOSSModel(Checkpoint& checkpoint) : embedding(checkpoint), unembedding(checkpoint) {
    norm_scale = checkpoint.get_bf16_ptr("norm.scale");
    norm_scale_count = checkpoint.get_bf16_count("norm.scale");
    blocks.reserve(kConfig20B.num_hidden_layers);
    for (int layer_idx = 0; layer_idx < kConfig20B.num_hidden_layers; ++layer_idx) {
        blocks.emplace_back(checkpoint, layer_idx);
    }
}

GPTOSSModel::~GPTOSSModel() = default;
