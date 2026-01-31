#include "kernels.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <vector>

namespace {

constexpr std::size_t kMxFp4BytesPerBlock = 16;
constexpr std::size_t kMxFp4ValuesPerBlock = 32;
constexpr float kPi = 3.14159265358979323846f;
constexpr float kFp4Values[16] = {
    +0.0f, +0.5f, +1.0f, +1.5f, +2.0f, +3.0f, +4.0f, +6.0f,
    -0.0f, -0.5f, -1.0f, -1.5f, -2.0f, -3.0f, -4.0f, -6.0f,
};

inline float bf16_to_float(std::uint16_t v) {
    std::uint32_t tmp = static_cast<std::uint32_t>(v) << 16;
    float out = 0.0f;
    std::memcpy(&out, &tmp, sizeof(out));
    return out;
}

inline void softmax_in_place(std::vector<float>& values) {
    float max_val = -std::numeric_limits<float>::infinity();
    for (float v : values) {
        if (v > max_val) {
            max_val = v;
        }
    }
    float sum = 0.0f;
    for (float& v : values) {
        v = std::exp(v - max_val);
        sum += v;
    }
    if (sum == 0.0f) {
        return;
    }
    for (float& v : values) {
        v /= sum;
    }
}

}  // namespace

void embedding_lookup(const std::uint16_t* weight_bf16,
                      std::size_t vocab_size,
                      std::size_t hidden_size,
                      std::span<const std::int32_t> token_ids,
                      std::span<float> out) {
    const std::size_t seq_len = token_ids.size();
    for (std::size_t t = 0; t < seq_len; ++t) {
        const std::int32_t token_id = token_ids[t];
        float* out_row = out.data() + t * hidden_size;
        if (token_id < 0 || static_cast<std::size_t>(token_id) >= vocab_size) {
            std::fill(out_row, out_row + hidden_size, 0.0f);
            continue;
        }
        const std::uint16_t* row = weight_bf16 + static_cast<std::size_t>(token_id) * hidden_size;
        for (std::size_t i = 0; i < hidden_size; ++i) {
            out_row[i] = bf16_to_float(row[i]);
        }
    }
}

void unembedding_logits(const std::uint16_t* weight_bf16,
                        std::size_t vocab_size,
                        std::size_t hidden_size,
                        std::span<const float> x,
                        std::span<float> out) {
    const std::size_t seq_len = x.size() / hidden_size;
    for (std::size_t t = 0; t < seq_len; ++t) {
        const float* x_row = x.data() + t * hidden_size;
        float* out_row = out.data() + t * vocab_size;
        for (std::size_t v = 0; v < vocab_size; ++v) {
            const std::uint16_t* w_row = weight_bf16 + v * hidden_size;
            float acc = 0.0f;
            for (std::size_t i = 0; i < hidden_size; ++i) {
                acc += x_row[i] * bf16_to_float(w_row[i]);
            }
            out_row[v] = acc;
        }
    }
}

void rmsnorm(std::span<const float> x,
             std::span<const std::uint16_t> scale_bf16,
             float eps,
             std::size_t hidden_size,
             std::span<float> out) {
    const std::size_t seq_len = x.size() / hidden_size;
    for (std::size_t t = 0; t < seq_len; ++t) {
        const float* x_row = x.data() + t * hidden_size;
        float* out_row = out.data() + t * hidden_size;
        float mean_sq = 0.0f;
        for (std::size_t i = 0; i < hidden_size; ++i) {
            mean_sq += x_row[i] * x_row[i];
        }
        mean_sq /= static_cast<float>(hidden_size);
        const float inv_rms = 1.0f / std::sqrt(mean_sq + eps);
        for (std::size_t i = 0; i < hidden_size; ++i) {
            out_row[i] = x_row[i] * inv_rms * bf16_to_float(scale_bf16[i]);
        }
    }
}

void linear_bf16(const std::uint16_t* weight_bf16,
                 const std::uint16_t* bias_bf16,
                 std::size_t in_features,
                 std::size_t out_features,
                 std::span<const float> x,
                 std::span<float> out) {
    const std::size_t seq_len = x.size() / in_features;
    for (std::size_t t = 0; t < seq_len; ++t) {
        const float* x_row = x.data() + t * in_features;
        float* out_row = out.data() + t * out_features;
        for (std::size_t o = 0; o < out_features; ++o) {
            const std::uint16_t* w_row = weight_bf16 + o * in_features;
            float acc = 0.0f;
            for (std::size_t i = 0; i < in_features; ++i) {
                acc += x_row[i] * bf16_to_float(w_row[i]);
            }
            if (bias_bf16) {
                acc += bf16_to_float(bias_bf16[o]);
            }
            out_row[o] = acc;
        }
    }
}

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
                float rope_ntk_beta) {
    const std::size_t half_dim = head_dim / 2;
    std::vector<float> inv_freq(half_dim, 0.0f);
    float concentration = 1.0f;
    for (std::size_t i = 0; i < half_dim; ++i) {
        const float exponent = static_cast<float>(2 * i) / static_cast<float>(head_dim);
        const float freq = std::pow(rope_theta, exponent);
        inv_freq[i] = 1.0f / freq;
    }

    if (rope_scaling_factor > 1.0f) {
        concentration = 0.1f * std::log(rope_scaling_factor) + 1.0f;
        const float d_half = static_cast<float>(head_dim) * 0.5f;
        const float low = d_half * std::log(static_cast<float>(initial_context_length) /
                                            (rope_ntk_beta * 2.0f * kPi)) /
                          std::log(rope_theta);
        const float high = d_half * std::log(static_cast<float>(initial_context_length) /
                                             (rope_ntk_alpha * 2.0f * kPi)) /
                           std::log(rope_theta);
        for (std::size_t i = 0; i < half_dim; ++i) {
            const float ramp = (static_cast<float>(i) - low) / (high - low);
            const float mask = 1.0f - std::clamp(ramp, 0.0f, 1.0f);
            const float interpolation = inv_freq[i] / rope_scaling_factor;
            const float extrapolation = inv_freq[i];
            inv_freq[i] = interpolation * (1.0f - mask) + extrapolation * mask;
        }
    }

    for (std::size_t t = 0; t < seq_len; ++t) {
        for (std::size_t h = 0; h < num_q_heads; ++h) {
            float* q_row = q.data() + (t * num_q_heads + h) * head_dim;
            for (std::size_t d = 0; d < half_dim; ++d) {
                const float angle = static_cast<float>(t) * inv_freq[d];
                const float c = std::cos(angle) * concentration;
                const float s = std::sin(angle) * concentration;
                const float x1 = q_row[d];
                const float x2 = q_row[d + half_dim];
                q_row[d] = x1 * c - x2 * s;
                q_row[d + half_dim] = x2 * c + x1 * s;
            }
        }
        for (std::size_t h = 0; h < num_kv_heads; ++h) {
            float* k_row = k.data() + (t * num_kv_heads + h) * head_dim;
            for (std::size_t d = 0; d < half_dim; ++d) {
                const float angle = static_cast<float>(t) * inv_freq[d];
                const float c = std::cos(angle) * concentration;
                const float s = std::sin(angle) * concentration;
                const float x1 = k_row[d];
                const float x2 = k_row[d + half_dim];
                k_row[d] = x1 * c - x2 * s;
                k_row[d + half_dim] = x2 * c + x1 * s;
            }
        }
    }
}

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
                     std::span<float> out) {
    const std::size_t q_mult = num_q_heads / num_kv_heads;
    for (std::size_t t = 0; t < seq_len; ++t) {
        const std::size_t min_k = sliding_window > 0
                                      ? (t + 1 > sliding_window ? t + 1 - sliding_window : 0)
                                      : 0;
        for (std::size_t h = 0; h < num_q_heads; ++h) {
            const std::size_t kv_head = h / q_mult;
            const float* q_row = q.data() + (t * num_q_heads + h) * head_dim;
            std::vector<float> logits;
            logits.reserve(seq_len - min_k + 1);
            for (std::size_t k_idx = min_k; k_idx <= t; ++k_idx) {
                const float* k_row = k.data() + (k_idx * num_kv_heads + kv_head) * head_dim;
                float acc = 0.0f;
                for (std::size_t d = 0; d < head_dim; ++d) {
                    acc += q_row[d] * k_row[d];
                }
                logits.push_back(acc * sm_scale);
            }
            const float sink = bf16_to_float(sinks_bf16[h]);
            logits.push_back(sink);
            softmax_in_place(logits);

            float* out_row = out.data() + (t * num_q_heads + h) * head_dim;
            std::fill(out_row, out_row + head_dim, 0.0f);
            std::size_t logit_idx = 0;
            for (std::size_t k_idx = min_k; k_idx <= t; ++k_idx) {
                const float* v_row = v.data() + (k_idx * num_kv_heads + kv_head) * head_dim;
                const float w = logits[logit_idx++];
                for (std::size_t d = 0; d < head_dim; ++d) {
                    out_row[d] += w * v_row[d];
                }
            }
        }
    }
}

void moe_topk_gating(std::span<const float> gate_logits,
                     std::size_t num_experts,
                     std::size_t experts_per_token,
                     std::span<std::int32_t> topk_indices,
                     std::span<float> topk_weights) {
    std::vector<std::pair<float, std::int32_t>> values;
    values.reserve(num_experts);
    for (std::size_t i = 0; i < num_experts; ++i) {
        values.emplace_back(gate_logits[i], static_cast<std::int32_t>(i));
    }
    const std::size_t k = experts_per_token;
    std::partial_sort(values.begin(), values.begin() + k, values.end(),
                      [](const auto& a, const auto& b) { return a.first > b.first; });
    std::vector<float> weights(k, 0.0f);
    for (std::size_t i = 0; i < k; ++i) {
        topk_indices[i] = values[i].second;
        weights[i] = values[i].first;
    }
    softmax_in_place(weights);
    for (std::size_t i = 0; i < k; ++i) {
        topk_weights[i] = weights[i];
    }
}

void mxfp4_gemm(const std::uint8_t* blocks,
                const std::uint8_t* scales,
                std::size_t out_features,
                std::size_t in_features,
                std::span<const float> x,
                std::span<float> out) {
    const std::size_t blocks_per_row = in_features / kMxFp4ValuesPerBlock;
    for (std::size_t o = 0; o < out_features; ++o) {
        float acc = 0.0f;
        const std::uint8_t* row_blocks = blocks + o * blocks_per_row * kMxFp4BytesPerBlock;
        const std::uint8_t* row_scales = scales + o * blocks_per_row;
        std::size_t x_idx = 0;
        for (std::size_t b = 0; b < blocks_per_row; ++b) {
            const int exp = static_cast<int>(row_scales[b]) - 127;
            const std::uint8_t* blk = row_blocks + b * kMxFp4BytesPerBlock;
            for (std::size_t i = 0; i < kMxFp4BytesPerBlock; ++i) {
                const std::uint8_t byte = blk[i];
                const std::uint8_t lo = byte & 0x0F;
                const std::uint8_t hi = byte >> 4;
                float v0 = std::ldexp(kFp4Values[lo], exp);
                float v1 = std::ldexp(kFp4Values[hi], exp);
                acc += x[x_idx++] * v0;
                acc += x[x_idx++] * v1;
            }
        }
        out[o] = acc;
    }
}

void swiglu(std::span<const float> x,
            float alpha,
            float limit,
            std::span<float> out) {
    const std::size_t half = x.size() / 2;
    for (std::size_t i = 0; i < half; ++i) {
        float x_glu = std::min(x[2 * i], limit);
        float x_lin = std::clamp(x[2 * i + 1], -limit, limit);
        float out_glu = x_glu * (1.0f / (1.0f + std::exp(-alpha * x_glu)));
        out[i] = out_glu * (x_lin + 1.0f);
    }
}

void moe_combine(std::span<const float> expert_outputs,
                 std::span<const float> expert_weights,
                 std::size_t experts_per_token,
                 std::size_t hidden_size,
                 std::span<float> out) {
    std::fill(out.begin(), out.end(), 0.0f);
    for (std::size_t e = 0; e < experts_per_token; ++e) {
        const float w = expert_weights[e];
        const float* expert = expert_outputs.data() + e * hidden_size;
        for (std::size_t i = 0; i < hidden_size; ++i) {
            out[i] += w * expert[i];
        }
    }
}
