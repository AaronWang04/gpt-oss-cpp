#pragma once

#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

class Checkpoint;

class Embedding {
public:
    explicit Embedding(Checkpoint& checkpoint);
    void forward(std::span<const std::int32_t> token_id,
                 std::span<float> out,
                 std::size_t seq_len) const;

private:
    const std::uint16_t* weight{nullptr};
    std::size_t weight_count{0};
    std::size_t hidden_size{0};
};

class AttentionBlock {
public:
    AttentionBlock(Checkpoint& checkpoint, int layer_idx);

    void forward(std::span<const float> x,
                 std::span<float> out,
                 std::size_t seq_len) const;

private:
    const std::uint16_t* norm_scale{nullptr};
    std::size_t norm_scale_count{0};
    const std::uint16_t* qkv_weight{nullptr};
    std::size_t qkv_weight_count{0};
    const std::uint16_t* qkv_bias{nullptr};
    std::size_t qkv_bias_count{0};
    const std::uint16_t* out_weight{nullptr};
    std::size_t out_weight_count{0};
    const std::uint16_t* out_bias{nullptr};
    std::size_t out_bias_count{0};
    const std::uint16_t* sinks{nullptr};
    std::size_t sinks_count{0};
    std::size_t hidden_size{0};
};

class MLPBlock {
public:
    MLPBlock(Checkpoint& checkpoint, int layer_idx);

    void forward(std::span<const float> x,
                 std::span<float> out,
                 std::size_t seq_len) const;
private:
    const std::uint16_t* norm_scale{nullptr};
    std::size_t norm_scale_count{0};
    const std::uint16_t* gate_weight{nullptr};
    std::size_t gate_weight_count{0};
    const std::uint16_t* gate_bias{nullptr};
    std::size_t gate_bias_count{0};
    const std::uint16_t* mlp1_bias{nullptr};
    std::size_t mlp1_bias_count{0};
    const std::uint16_t* mlp2_bias{nullptr};
    std::size_t mlp2_bias_count{0};
    const std::uint8_t* mlp1_weight_blocks{nullptr};
    std::size_t mlp1_weight_blocks_count{0};
    const std::uint8_t* mlp1_weight_scales{nullptr};
    std::size_t mlp1_weight_scales_count{0};
    const std::uint8_t* mlp2_weight_blocks{nullptr};
    std::size_t mlp2_weight_blocks_count{0};
    const std::uint8_t* mlp2_weight_scales{nullptr};
    std::size_t mlp2_weight_scales_count{0};
    std::size_t hidden_size{0};
};

class TransformerBlock {
public:
    TransformerBlock(Checkpoint& checkpoint, int layer_idx);

    void forward(std::span<const float> x,
                std::span<float> out,
                std::size_t seq_len) const;
private:
    AttentionBlock attn;
    MLPBlock mlp;
    std::size_t hidden_size{0};
};

class UnEmbedding {
public:
    explicit UnEmbedding(Checkpoint& checkpoint);

    void forward(std::span<const float> x,
                 std::span<float> out,
                 std::size_t seq_len) const;

private:
    const std::uint16_t* weight{nullptr};
    std::size_t weight_count{0};
    std::size_t hidden_size{0};
    std::size_t vocab_size{0};
};

class GPTOSSModel {
public:
    explicit GPTOSSModel(Checkpoint& checkpoint);
    ~GPTOSSModel();

private:
    Embedding embedding;
    UnEmbedding unembedding;
    std::vector<TransformerBlock> blocks;
    const std::uint16_t* norm_scale{nullptr};
    std::size_t norm_scale_count{0};
};
