#pragma once

#include <cstddef>
#include <cstdint>
#include <span>

// Embedding / unembedding
void embedding_lookup(const std::uint16_t* weight_bf16,
                      std::size_t vocab_size,
                      std::size_t hidden_size,
                      std::span<const std::int32_t> token_ids,
                      std::span<float> out);

void unembedding_logits(const std::uint16_t* weight_bf16,
                        std::size_t vocab_size,
                        std::size_t hidden_size,
                        std::span<const float> x,
                        std::span<float> out);

// RMSNorm
void rmsnorm(std::span<const float> x,
             std::span<const std::uint16_t> scale_bf16,
             float eps,
             std::size_t hidden_size,
             std::span<float> out);

// Linear layers
void linear_bf16(const std::uint16_t* weight_bf16,
                 const std::uint16_t* bias_bf16,
                 std::size_t in_features,
                 std::size_t out_features,
                 std::span<const float> x,
                 std::span<float> out);

// Rotary embedding for Q/K
void apply_rope(std::span<float> q,
                std::span<float> k,
                std::size_t seq_len,
                std::size_t num_q_heads,
                std::size_t num_kv_heads,
                std::size_t head_dim,
                std::size_t initial_context_length,
                float rope_theta,
                float rope_scaling_factor,
                float rope_ntk_alpha,
                float rope_ntk_beta);

// Scaled dot-product attention with sinks and optional sliding window.
void sdpa_with_sinks(std::span<const float> q,
                     std::span<const float> k,
                     std::span<const float> v,
                     std::span<const std::uint16_t> sinks_bf16,
                     std::size_t seq_len,
                     std::size_t num_q_heads,
                     std::size_t num_kv_heads,
                     std::size_t head_dim,
                     float sm_scale,
                     std::size_t sliding_window,
                     std::span<float> out);

// MoE gating and top-k selection.
void moe_topk_gating(std::span<const float> gate_logits,
                     std::size_t num_experts,
                     std::size_t experts_per_token,
                     std::span<std::int32_t> topk_indices,
                     std::span<float> topk_weights);

// MXFP4 dequant + matmul for MLP1/MLP2
void mxfp4_gemm(const std::uint8_t* blocks,
                const std::uint8_t* scales,
                std::size_t out_features,
                std::size_t in_features,
                std::span<const float> x,
                std::span<float> out);

// SWIGLU activation for MLP1 output.
void swiglu(std::span<const float> x,
            float alpha,
            float limit,
            std::span<float> out);

// MoE expert combine (weighted sum).
void moe_combine(std::span<const float> expert_outputs,
                 std::span<const float> expert_weights,
                 std::size_t experts_per_token,
                 std::size_t hidden_size,
                 std::span<float> out);
