#include "model.h"

#include "checkpoint.h"
#include "kernels.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <utility>
#include <vector>

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

inline float bf16_to_float(std::uint16_t v) {
    std::uint32_t tmp = static_cast<std::uint32_t>(v) << 16;
    float out = 0.0f;
    std::memcpy(&out, &tmp, sizeof(out));
    return out;
}

void require_count(const char* name, std::size_t actual, std::size_t expected) {
    if (actual != expected) {
        throw std::runtime_error(std::string("tensor size mismatch: ") + name +
                                 " actual=" + std::to_string(actual) +
                                 " expected=" + std::to_string(expected));
    }
}

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
    sliding_window = (layer_idx % 2 == 0) ? static_cast<std::size_t>(kConfig20B.sliding_window) : 0;
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

void Embedding::forward(std::span<const std::int32_t> token_id,
                        std::span<float> out,
                        std::size_t seq_len) const {
    require_count("embedding.weight", weight_count,
                  kConfig20B.vocab_size * kConfig20B.hidden_size);
    embedding_lookup(weight, kConfig20B.vocab_size, hidden_size, token_id, out);
}

void AttentionBlock::forward(std::span<const float> x,
                             std::span<float> out,
                             std::size_t seq_len) const {
    const std::size_t hidden = hidden_size;
    const std::size_t num_heads = kConfig20B.num_attention_heads;
    const std::size_t num_kv_heads = kConfig20B.num_key_value_heads;
    const std::size_t head_dim = kConfig20B.head_dim;
    const std::size_t qkv_dim = head_dim * (num_heads + 2 * num_kv_heads);
    const float eps = 1e-5f;
    const float sm_scale = 1.0f / std::sqrt(static_cast<float>(head_dim));

    require_count("attn.norm.scale", norm_scale_count, hidden);
    require_count("attn.qkv.weight", qkv_weight_count, qkv_dim * hidden);
    require_count("attn.qkv.bias", qkv_bias_count, qkv_dim);
    require_count("attn.out.weight", out_weight_count, hidden * (num_heads * head_dim));
    require_count("attn.out.bias", out_bias_count, hidden);
    require_count("attn.sinks", sinks_count, num_heads);

    std::vector<float> normed(seq_len * hidden, 0.0f);
    rmsnorm(x, std::span<const std::uint16_t>(norm_scale, norm_scale_count), eps, hidden, normed);

    std::vector<float> qkv(seq_len * qkv_dim, 0.0f);
    linear_bf16(qkv_weight, qkv_bias, hidden, qkv_dim, normed, qkv);

    std::vector<float> q(seq_len * num_heads * head_dim, 0.0f);
    std::vector<float> k(seq_len * num_kv_heads * head_dim, 0.0f);
    std::vector<float> v(seq_len * num_kv_heads * head_dim, 0.0f);

    for (std::size_t t = 0; t < seq_len; ++t) {
        const float* row = qkv.data() + t * qkv_dim;
        float* q_row = q.data() + t * num_heads * head_dim;
        float* k_row = k.data() + t * num_kv_heads * head_dim;
        float* v_row = v.data() + t * num_kv_heads * head_dim;
        const std::size_t q_bytes = num_heads * head_dim;
        const std::size_t k_bytes = num_kv_heads * head_dim;
        std::copy(row, row + q_bytes, q_row);
        std::copy(row + q_bytes, row + q_bytes + k_bytes, k_row);
        std::copy(row + q_bytes + k_bytes, row + q_bytes + 2 * k_bytes, v_row);
    }

    apply_rope(q, k, seq_len, num_heads, num_kv_heads, head_dim,
               kConfig20B.initial_context_length, static_cast<float>(kConfig20B.rope_theta),
               static_cast<float>(kConfig20B.rope_scaling_factor),
               static_cast<float>(kConfig20B.rope_ntk_alpha),
               static_cast<float>(kConfig20B.rope_ntk_beta));

    std::vector<float> attn(seq_len * num_heads * head_dim, 0.0f);
    sdpa_with_sinks(q, k, v, std::span<const std::uint16_t>(sinks, sinks_count), seq_len,
                    num_heads, num_kv_heads, head_dim, sm_scale, sliding_window, attn);

    std::vector<float> projected(seq_len * hidden, 0.0f);
    linear_bf16(out_weight, out_bias, num_heads * head_dim, hidden, attn, projected);

    for (std::size_t i = 0; i < out.size(); ++i) {
        out[i] = x[i] + projected[i];
    }
}

void MLPBlock::forward(std::span<const float> x,
                       std::span<float> out,
                       std::size_t seq_len) const {
    const std::size_t hidden = hidden_size;
    const std::size_t num_experts = kConfig20B.num_experts;
    const std::size_t experts_per_token = kConfig20B.experts_per_token;
    const std::size_t intermediate = kConfig20B.intermediate_size;
    const float eps = 1e-5f;

    require_count("mlp.norm.scale", norm_scale_count, hidden);
    require_count("mlp.gate.weight", gate_weight_count, num_experts * hidden);
    require_count("mlp.gate.bias", gate_bias_count, num_experts);

    std::vector<float> normed(seq_len * hidden, 0.0f);
    rmsnorm(x, std::span<const std::uint16_t>(norm_scale, norm_scale_count), eps, hidden, normed);

    std::vector<float> gate_logits(seq_len * num_experts, 0.0f);
    linear_bf16(gate_weight, gate_bias, hidden, num_experts, normed, gate_logits);

    const std::size_t mlp1_out_features = intermediate * 2;
    const std::size_t mlp2_out_features = hidden;
    const std::size_t blocks_per_row_mlp1 = hidden / 32;
    const std::size_t blocks_per_row_mlp2 = intermediate / 32;
    const std::size_t mlp1_row_blocks = blocks_per_row_mlp1 * 16;
    const std::size_t mlp2_row_blocks = blocks_per_row_mlp2 * 16;

    require_count("mlp.mlp1_bias", mlp1_bias_count, num_experts * mlp1_out_features);
    require_count("mlp.mlp2_bias", mlp2_bias_count, num_experts * mlp2_out_features);
    require_count("mlp.mlp1_weight.blocks", mlp1_weight_blocks_count,
                  num_experts * mlp1_out_features * mlp1_row_blocks);
    require_count("mlp.mlp1_weight.scales", mlp1_weight_scales_count,
                  num_experts * mlp1_out_features * blocks_per_row_mlp1);
    require_count("mlp.mlp2_weight.blocks", mlp2_weight_blocks_count,
                  num_experts * mlp2_out_features * mlp2_row_blocks);
    require_count("mlp.mlp2_weight.scales", mlp2_weight_scales_count,
                  num_experts * mlp2_out_features * blocks_per_row_mlp2);

    std::vector<std::int32_t> topk_indices(experts_per_token);
    std::vector<float> topk_weights(experts_per_token);
    std::vector<float> expert_outputs(experts_per_token * hidden, 0.0f);
    std::vector<float> mlp1_out(mlp1_out_features, 0.0f);
    std::vector<float> swiglu_out(intermediate, 0.0f);
    std::vector<float> mlp2_out(hidden, 0.0f);

    for (std::size_t t = 0; t < seq_len; ++t) {
        const float* gate_row = gate_logits.data() + t * num_experts;
        moe_topk_gating(std::span<const float>(gate_row, num_experts), num_experts,
                        experts_per_token, topk_indices, topk_weights);

        for (std::size_t e = 0; e < experts_per_token; ++e) {
            const std::size_t expert_idx = static_cast<std::size_t>(topk_indices[e]);

            const std::uint8_t* mlp1_blocks = mlp1_weight_blocks +
                                             expert_idx * mlp1_out_features * mlp1_row_blocks;
            const std::uint8_t* mlp1_scales = mlp1_weight_scales +
                                             expert_idx * mlp1_out_features * blocks_per_row_mlp1;
            const std::uint16_t* mlp1_bias_row = mlp1_bias + expert_idx * mlp1_out_features;

            const float* x_row = normed.data() + t * hidden;
            mxfp4_gemm(mlp1_blocks, mlp1_scales, mlp1_out_features, hidden,
                      std::span<const float>(x_row, hidden), mlp1_out);
            for (std::size_t i = 0; i < mlp1_out_features; ++i) {
                mlp1_out[i] += bf16_to_float(mlp1_bias_row[i]);
            }

            swiglu(mlp1_out, 1.702f, static_cast<float>(kConfig20B.swiglu_limit), swiglu_out);

            const std::uint8_t* mlp2_blocks = mlp2_weight_blocks +
                                             expert_idx * mlp2_out_features * mlp2_row_blocks;
            const std::uint8_t* mlp2_scales = mlp2_weight_scales +
                                             expert_idx * mlp2_out_features * blocks_per_row_mlp2;
            const std::uint16_t* mlp2_bias_row = mlp2_bias + expert_idx * mlp2_out_features;

            mxfp4_gemm(mlp2_blocks, mlp2_scales, mlp2_out_features, intermediate,
                      swiglu_out, mlp2_out);
            for (std::size_t i = 0; i < mlp2_out_features; ++i) {
                mlp2_out[i] += bf16_to_float(mlp2_bias_row[i]);
            }

            float* expert_out = expert_outputs.data() + e * hidden;
            std::copy(mlp2_out.begin(), mlp2_out.end(), expert_out);
        }

        float* out_row = out.data() + t * hidden;
        moe_combine(expert_outputs, topk_weights, experts_per_token, hidden,
                    std::span<float>(out_row, hidden));
        for (std::size_t i = 0; i < hidden; ++i) {
            out_row[i] += x[t * hidden + i];
        }
    }
}

void TransformerBlock::forward(std::span<const float> x,
                               std::span<float> out,
                               std::size_t seq_len) const {
    std::vector<float> attn_out(seq_len * hidden_size, 0.0f);
    attn.forward(x, attn_out, seq_len);
    mlp.forward(attn_out, out, seq_len);
}

void UnEmbedding::forward(std::span<const float> x,
                          std::span<float> out,
                          std::size_t seq_len) const {
    unembedding_logits(weight, vocab_size, hidden_size, x, out);
}

void GPTOSSModel::forward(std::span<const std::int32_t> token_ids,
                          std::span<float> logits,
                          std::size_t seq_len) const {
    const std::size_t hidden = kConfig20B.hidden_size;
    const float eps = 1e-5f;
    std::vector<float> x(seq_len * hidden, 0.0f);
    std::vector<float> tmp(seq_len * hidden, 0.0f);

    embedding.forward(token_ids, x, seq_len);
    for (std::size_t i = 0; i < blocks.size(); ++i) {
        blocks[i].forward(x, tmp, seq_len);
        std::swap(x, tmp);
    }

    rmsnorm(x, std::span<const std::uint16_t>(norm_scale, norm_scale_count), eps, hidden, tmp);
    unembedding.forward(tmp, logits, seq_len);
}
