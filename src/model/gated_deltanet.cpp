#include "model/gated_deltanet.h"

#include <cmath>
#include <mlx/mlx.h>
#include <mlx/compile.h>

namespace marlin {

namespace mx = mlx::core;

static mx::array l2norm(const mx::array& x, float eps = 1e-6f) {
    auto norm = mx::sqrt(mx::sum(mx::square(x), {-1}, true) + eps);
    return x / norm;
}

static mx::array softplus(const mx::array& x) {
    return mx::log1p(mx::exp(x));
}

static const auto& compiled_recurrence_and_norm() {
    static auto fn = mx::compile(
        +[](const std::vector<mx::array>& inputs) -> std::vector<mx::array> {
            const auto& query = inputs[0];
            const auto& key = inputs[1];
            const auto& value = inputs[2];
            const auto& gate = inputs[3];
            const auto& beta = inputs[4];
            const auto& rec_state = inputs[5];
            const auto& z = inputs[6];
            const auto& norm_w = inputs[7];

            auto g = mx::expand_dims(mx::expand_dims(gate, -1), -1);
            auto new_state = rec_state * g;

            auto k_expanded = mx::expand_dims(key, -1);
            auto kv_mem = mx::sum(new_state * k_expanded, {-2});
            auto delta = (value - kv_mem) * beta;
            auto delta_expanded = mx::expand_dims(delta, -2);
            new_state = new_state + k_expanded * delta_expanded;

            auto q_expanded = mx::expand_dims(query, -1);
            auto output = mx::sum(new_state * q_expanded, {-2});

            auto norm_x = output / mx::sqrt(mx::mean(mx::square(output), {-1}, true) + 1e-6f);
            norm_x = norm_x * norm_w;
            auto gated = norm_x * (mx::sigmoid(z) * z);

            return {gated, new_state};
        },
        true);
    return fn;
}

mx::array causal_conv1d_step(
    const mx::array& x,
    const mx::array& weight,
    const mx::array& bias,
    mx::array& conv_state) {

    auto full_window = mx::concatenate({conv_state, mx::expand_dims(x, -1)}, -1);
    conv_state = mx::slice(full_window, {0, 0, 1}, full_window.shape());

    // PyTorch depthwise weight [C, 1, K] → squeeze to [C, K] for per-channel dot
    auto w = mx::squeeze(weight, 1);
    auto out = mx::sum(full_window * w, {-1});
    if (bias.size() > 0) {
        out = out + bias;
    }
    return mx::sigmoid(out) * out;
}

mx::array gated_delta_recurrence(
    const mx::array& query,
    const mx::array& key,
    const mx::array& value,
    const mx::array& gate,
    const mx::array& beta,
    mx::array& recurrent_state) {

    // state: [B, H, Dk, Dv], q/k: [B, H, Dk], v: [B, H, Dv], gate: [B, H], beta: [B, H, 1]
    auto g = mx::expand_dims(mx::expand_dims(gate, -1), -1);
    recurrent_state = recurrent_state * g;

    auto k_expanded = mx::expand_dims(key, -1);
    auto kv_mem = mx::sum(recurrent_state * k_expanded, {-2});
    auto delta = (value - kv_mem) * beta;
    auto delta_expanded = mx::expand_dims(delta, -2);
    recurrent_state = recurrent_state + k_expanded * delta_expanded;

    auto q_expanded = mx::expand_dims(query, -1);
    auto output = mx::sum(recurrent_state * q_expanded, {-2});
    return output;
}

mx::array gated_deltanet_forward(
    const mx::array& hidden,
    const DeltaNetLayerWeights& w,
    DeltaNetState& state,
    int num_k_heads,
    int num_v_heads,
    int head_k_dim,
    int head_v_dim,
    bool has_cache) {

    auto batch = hidden.shape(0);
    auto seq_len = hidden.shape(1);

    auto mixed_qkv = mx::matmul(hidden, mx::transpose(w.in_proj_qkv));
    auto z = mx::matmul(hidden, mx::transpose(w.in_proj_z));
    auto b = mx::matmul(hidden, mx::transpose(w.in_proj_b));
    auto a = mx::matmul(hidden, mx::transpose(w.in_proj_a));

    z = mx::reshape(z, {batch, seq_len, num_v_heads, head_v_dim});

    if (has_cache && seq_len == 1) {
        mixed_qkv = mx::squeeze(mixed_qkv, 1);
        mixed_qkv = causal_conv1d_step(mixed_qkv, w.conv1d_weight, w.conv1d_bias, state.conv_state);
        mixed_qkv = mx::expand_dims(mixed_qkv, 1);
    } else {
        int conv_dim = mixed_qkv.shape(2);
        // PyTorch conv1d weight: [C_out, C_in/groups, kernel] -> MLX: [C_out, kernel, C_in/groups]
        auto conv_w = mx::transpose(w.conv1d_weight, {0, 2, 1});
        auto conv_out = mx::conv1d(
            mixed_qkv,
            conv_w,
            1, 3, 1, conv_dim);
        conv_out = mx::slice(conv_out, {0, 0, 0}, {batch, seq_len, conv_dim});
        mixed_qkv = mx::sigmoid(conv_out) * conv_out;

        if (has_cache) {
            int kernel_minus_1 = 3;
            int start = std::max(0, static_cast<int>(seq_len) - kernel_minus_1);
            state.conv_state = mx::slice(
                mx::transpose(mixed_qkv, {0, 2, 1}),
                {0, 0, start},
                {batch, conv_dim, static_cast<int>(seq_len)});
        }
    }

    int key_dim = num_k_heads * head_k_dim;

    auto splits = mx::split(mixed_qkv, {key_dim, key_dim * 2}, -1);
    auto query = mx::reshape(splits[0], {batch, seq_len, num_k_heads, head_k_dim});
    auto key = mx::reshape(splits[1], {batch, seq_len, num_k_heads, head_k_dim});
    auto value = mx::reshape(splits[2], {batch, seq_len, num_v_heads, head_v_dim});

    query = l2norm(query);
    key = l2norm(key);

    float scale = 1.0f / std::sqrt(static_cast<float>(head_k_dim));
    query = query * scale;

    auto beta_val = mx::sigmoid(b);
    beta_val = mx::reshape(beta_val, {batch, seq_len, num_v_heads, 1});

    auto gate = mx::exp(
        mx::negative(mx::exp(mx::astype(w.A_log, mx::float32))) *
        softplus(mx::astype(a, mx::float32) + mx::astype(w.dt_bias, mx::float32))
    );
    gate = mx::reshape(gate, {batch, seq_len, num_v_heads});

    if (num_v_heads > num_k_heads) {
        int repeat = num_v_heads / num_k_heads;
        query = mx::repeat(query, repeat, 2);
        key = mx::repeat(key, repeat, 2);
    }

    if (has_cache && seq_len == 1) {
        query = mx::squeeze(query, 1);
        key = mx::squeeze(key, 1);
        value = mx::squeeze(value, 1);
        gate = mx::squeeze(gate, 1);
        beta_val = mx::squeeze(beta_val, 1);
        auto z_squeezed = mx::squeeze(z, 1);

        auto results = compiled_recurrence_and_norm()({
            query, key, value, gate, beta_val,
            state.recurrent_state, z_squeezed, w.norm_weight,
        });

        state.recurrent_state = results[1];
        auto gated = mx::reshape(results[0], {batch, 1, num_v_heads * head_v_dim});
        return mx::matmul(gated, mx::transpose(w.out_proj));
    }

    mx::array core_out = mx::zeros({batch, seq_len, num_v_heads, head_v_dim}, hidden.dtype());

    for (int i = 0; i < seq_len; ++i) {
        auto q_t = mx::squeeze(mx::slice(query, {0, i, 0, 0}, {batch, i + 1, query.shape(2), query.shape(3)}), 1);
        auto k_t = mx::squeeze(mx::slice(key, {0, i, 0, 0}, {batch, i + 1, key.shape(2), key.shape(3)}), 1);
        auto v_t = mx::squeeze(mx::slice(value, {0, i, 0, 0}, {batch, i + 1, value.shape(2), value.shape(3)}), 1);
        auto g_t = mx::squeeze(mx::slice(gate, {0, i, 0}, {batch, i + 1, gate.shape(2)}), 1);
        auto b_t = mx::squeeze(mx::slice(beta_val, {0, i, 0, 0}, {batch, i + 1, beta_val.shape(2), beta_val.shape(3)}), 1);

        auto out_t = gated_delta_recurrence(q_t, k_t, v_t, g_t, b_t, state.recurrent_state);

        core_out = mx::slice_update(
            core_out, mx::expand_dims(out_t, 1),
            {0, i, 0, 0}, {batch, i + 1, num_v_heads, head_v_dim});
    }

    auto flat = mx::reshape(core_out, {batch * seq_len, num_v_heads, head_v_dim});
    auto z_flat = mx::reshape(z, {batch * seq_len, num_v_heads, head_v_dim});

    auto norm_x = flat / mx::sqrt(mx::mean(mx::square(flat), {-1}, true) + 1e-6f);
    norm_x = norm_x * w.norm_weight;
    auto gated = norm_x * (mx::sigmoid(z_flat) * z_flat);
    gated = mx::reshape(gated, {batch, seq_len, num_v_heads * head_v_dim});

    return mx::matmul(gated, mx::transpose(w.out_proj));
}

}  // namespace marlin
