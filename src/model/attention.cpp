#include "model/attention.h"

#include <cmath>
#include <fmt/format.h>
#include <mlx/mlx.h>

namespace marlin {

namespace mx = mlx::core;

static mx::array rms_norm(const mx::array& x, const mx::array& weight, float eps) {
    auto variance = mx::mean(mx::square(x), {-1}, true);
    // Qwen3.5 stores norm weight as offset: actual multiplier is (1 + weight)
    return x * mx::rsqrt(variance + eps) * (1.0f + weight);
}

mx::array mrope(
    const mx::array& x,
    int offset,
    int head_dim,
    float partial_rotary_factor,
    float theta,
    const std::array<int, 3>& sections,
    bool interleaved) {

    int rotary_dim = static_cast<int>(head_dim * partial_rotary_factor);
    if (rotary_dim == 0) return x;

    auto batch = x.shape(0);
    auto seq_len = x.shape(1);
    auto n_heads = x.shape(2);

    auto x_rot = mx::slice(x, {0, 0, 0, 0}, {batch, seq_len, n_heads, rotary_dim});
    auto x_pass = mx::slice(x, {0, 0, 0, rotary_dim}, {batch, seq_len, n_heads, head_dim});

    int half = rotary_dim / 2;
    std::vector<float> freqs_data(half);
    for (int i = 0; i < half; ++i) {
        freqs_data[i] = 1.0f / std::pow(theta, static_cast<float>(2 * i) / rotary_dim);
    }
    auto freqs = mx::array(freqs_data.data(), {half});

    std::vector<float> positions(seq_len);
    for (int i = 0; i < seq_len; ++i) {
        positions[i] = static_cast<float>(offset + i);
    }
    auto pos = mx::array(positions.data(), {seq_len});

    auto angles = mx::outer(pos, freqs);
    angles = mx::reshape(angles, {1, seq_len, 1, half});

    auto cos_vals = mx::cos(angles);
    auto sin_vals = mx::sin(angles);

    auto x1 = mx::slice(x_rot, {0, 0, 0, 0}, {batch, seq_len, n_heads, half});
    auto x2 = mx::slice(x_rot, {0, 0, 0, half}, {batch, seq_len, n_heads, rotary_dim});

    auto rotated = mx::concatenate({
        x1 * cos_vals - x2 * sin_vals,
        x2 * cos_vals + x1 * sin_vals
    }, -1);

    return mx::concatenate({rotated, x_pass}, -1);
}

mx::array attention_forward(
    const mx::array& hidden,
    const AttentionLayerWeights& w,
    KVCacheEntry& cache,
    int num_heads,
    int num_kv_heads,
    int head_dim,
    float partial_rotary_factor,
    float rope_theta,
    const std::array<int, 3>& mrope_sections,
    bool mrope_interleaved,
    int position) {

    auto batch = hidden.shape(0);
    auto seq_len = hidden.shape(1);

    auto q_gate_flat = mx::matmul(hidden, mx::transpose(w.q_proj));
    auto q_gate = mx::reshape(q_gate_flat, {batch, seq_len, num_heads, head_dim * 2});
    auto q_gate_split = mx::split(q_gate, 2, -1);
    auto q = mx::reshape(q_gate_split[0], {batch, seq_len, num_heads * head_dim});
    auto gate = mx::reshape(q_gate_split[1], {batch, seq_len, num_heads * head_dim});

    auto k = mx::matmul(hidden, mx::transpose(w.k_proj));
    auto v = mx::matmul(hidden, mx::transpose(w.v_proj));

    q = mx::reshape(q, {batch, seq_len, num_heads, head_dim});
    k = mx::reshape(k, {batch, seq_len, num_kv_heads, head_dim});
    v = mx::reshape(v, {batch, seq_len, num_kv_heads, head_dim});

    q = rms_norm(q, w.q_norm, 1e-6f);
    k = rms_norm(k, w.k_norm, 1e-6f);

    q = mrope(q, position, head_dim, partial_rotary_factor, rope_theta,
              mrope_sections, mrope_interleaved);
    k = mrope(k, position, head_dim, partial_rotary_factor, rope_theta,
              mrope_sections, mrope_interleaved);

    q = mx::transpose(q, {0, 2, 1, 3});
    k = mx::transpose(k, {0, 2, 1, 3});
    v = mx::transpose(v, {0, 2, 1, 3});

    if (cache.length > 0) {
        k = mx::concatenate({cache.keys, k}, 2);
        v = mx::concatenate({cache.values, v}, 2);
    }
    cache.keys = k;
    cache.values = v;
    cache.length += seq_len;

    if (num_kv_heads < num_heads) {
        int repeat = num_heads / num_kv_heads;
        k = mx::repeat(k, repeat, 1);
        v = mx::repeat(v, repeat, 1);
    }

    float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));
    auto scores = mx::matmul(q, mx::transpose(k, {0, 1, 3, 2})) * scale;

    if (seq_len > 1) {
        auto mask = mx::triu(
            mx::full({seq_len, cache.length}, -std::numeric_limits<float>::infinity()),
            cache.length - seq_len + 1);
        scores = scores + mask;
    }

    auto attn = mx::softmax(scores, {-1});
    auto out = mx::matmul(attn, v);

    out = mx::transpose(out, {0, 2, 1, 3});
    out = mx::reshape(out, {batch, seq_len, num_heads * head_dim});

    out = out * mx::sigmoid(gate);

    return mx::matmul(out, mx::transpose(w.o_proj));
}

}  // namespace marlin
