#pragma once

#include <cstddef>
#include <span>
#include <vector>

class KVCache {
public:
    explicit KVCache(std::size_t num_layers);

    // Append post-RoPE K and V for q_len new tokens to the layer's cache.
    // Does NOT update seq_len — GPTOSSModel::forward bumps seq_len once per forward pass
    // so all layers read a consistent position offset.
    void append(std::size_t layer,
                std::span<const float> k_new,
                std::span<const float> v_new);

    std::size_t seq_len = 0;
    // [layer][token_pos * num_kv_heads * head_dim + ...]  (flat)
    std::vector<std::vector<float>> k_cache;
    std::vector<std::vector<float>> v_cache;
};
