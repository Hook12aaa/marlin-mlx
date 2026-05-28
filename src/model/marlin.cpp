#include "model/marlin.h"
#include "model/weights.h"

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

mx::array qmatmul(const mx::array& x, const QuantizedTensor& w) {
    if (w.quantized) {
        return mx::quantized_matmul(
            x, w.weight, w.scales, w.biases,
            true, w.group_size, w.bits);
    }
    return mx::matmul(x, mx::transpose(w.weight));
}

static mx::array swiglu(const mx::array& gate, const mx::array& up) {
    return (mx::sigmoid(gate) * gate) * up;
}

static mx::array mlp_forward(const mx::array& x, const MLPWeights& w) {
    auto gate = qmatmul(x, w.gate_proj);
    auto up = qmatmul(x, w.up_proj);
    return qmatmul(swiglu(gate, up), w.down_proj);
}

void HybridCache::reset(const ModelConfig& config) {
    layers.clear();
    layers.reserve(config.text.num_hidden_layers);
    for (int i = 0; i < config.text.num_hidden_layers; ++i) {
        if (config.text.layer_types[i] == LayerType::LinearAttention) {
            int num_v = config.text.linear_num_value_heads;
            int dk = config.text.linear_key_head_dim;
            int dv = config.text.linear_value_head_dim;
            int num_k = config.text.linear_num_key_heads;
            int conv_dim = num_k * dk + num_k * dk + num_v * dv;
            int kernel = config.text.linear_conv_kernel_dim;

            layers.push_back(DeltaNetState{
                .conv_state = mx::zeros({1, conv_dim, kernel - 1}, mx::bfloat16),
                .recurrent_state = mx::zeros({1, num_v, dk, dv}, mx::float32),
            });
        } else {
            layers.push_back(KVCacheEntry{
                .keys = mx::zeros({1}),
                .values = mx::zeros({1}),
                .length = 0,
            });
        }
    }
}

mx::array MarlinModel::embed_tokens(const mx::array& token_ids) {
    return mx::take(weights_.embed_tokens, token_ids, 0);
}

mx::array MarlinModel::merge_vision_text(
    const mx::array& text_ids,
    const mx::array& vision_embeds,
    int video_token_id) {

    auto text_embeds = embed_tokens(text_ids);
    auto batch = text_ids.shape(0);
    auto seq_len = text_ids.shape(1);
    auto hidden = text_ids.shape().size() > 1 ? config_.text.hidden_size : 0;

    auto flat_ids = mx::reshape(text_ids, {-1});
    auto is_video = mx::equal(flat_ids, mx::array(video_token_id));

    mx::eval(is_video);
    auto is_video_data = is_video.data<bool>();

    int vision_idx = 0;
    int n_vision = vision_embeds.shape(1);
    auto result = text_embeds;

    for (int i = 0; i < flat_ids.shape(0); ++i) {
        if (is_video_data[i] && vision_idx < n_vision) {
            auto vision_tok = mx::slice(vision_embeds, {0, vision_idx, 0},
                                        {1, vision_idx + 1, vision_embeds.shape(2)});
            int row = i / seq_len;
            int col = i % seq_len;
            result = mx::slice_update(result, vision_tok,
                                      {row, col, 0},
                                      {row + 1, col + 1, config_.text.hidden_size});
            ++vision_idx;
        }
    }

    return result;
}

mx::array MarlinModel::text_forward_embeddings(
    const mx::array& embeddings,
    int position) {

    auto hidden = embeddings;
    auto& cfg = config_.text;

    for (int i = 0; i < cfg.num_hidden_layers; ++i) {
        auto& layer = weights_.layers[i];
        auto normed = rms_norm(hidden, layer.norms.input_norm, cfg.rms_norm_eps);

        mx::array attn_out = mx::zeros({1});
        if (layer.type == LayerType::LinearAttention) {
            auto& state = std::get<DeltaNetState>(cache_.layers[i]);
            auto& w = std::get<DeltaNetLayerWeights>(layer.attn_weights);
            attn_out = gated_deltanet_forward(
                normed, w, state,
                cfg.linear_num_key_heads,
                cfg.linear_num_value_heads,
                cfg.linear_key_head_dim,
                cfg.linear_value_head_dim,
                position > 0);
        } else {
            auto& kv = std::get<KVCacheEntry>(cache_.layers[i]);
            auto& w = std::get<AttentionLayerWeights>(layer.attn_weights);
            attn_out = attention_forward(
                normed, w, kv,
                cfg.num_attention_heads,
                cfg.num_key_value_heads,
                cfg.head_dim,
                cfg.partial_rotary_factor,
                cfg.rope_theta,
                cfg.mrope_section,
                cfg.mrope_interleaved,
                position);
        }

        hidden = hidden + attn_out;
        auto post_normed = rms_norm(hidden, layer.norms.post_attn_norm, cfg.rms_norm_eps);
        hidden = hidden + mlp_forward(post_normed, layer.mlp);
    }

    return rms_norm(hidden, weights_.final_norm, cfg.rms_norm_eps);
}

mx::array MarlinModel::text_forward(
    const mx::array& token_ids,
    int position) {
    return text_forward_embeddings(embed_tokens(token_ids), position);
}

mx::array MarlinModel::text_forward_token(
    const mx::array& token_id,
    int position) {
    auto ids = mx::reshape(token_id, {1, 1});
    return text_forward(ids, position);
}

mx::array MarlinModel::lm_head(const mx::array& hidden) {
    auto& lm = weights_.lm_head;
    if (lm.quantized) {
        return mx::quantized_matmul(
            hidden, lm.weight, lm.scales, lm.biases,
            true, lm.group_size, lm.bits);
    }
    return mx::matmul(hidden, mx::transpose(lm.weight));
}

static mx::array vit_layer_norm(
    const mx::array& x,
    const mx::array& weight,
    const mx::array& bias,
    float eps = 1e-6f) {
    auto mean = mx::mean(x, {-1}, true);
    auto var = mx::mean(mx::square(x - mean), {-1}, true);
    auto normed = (x - mean) * mx::rsqrt(var + eps);
    return normed * weight + bias;
}

static mx::array gelu_approx(const mx::array& x) {
    constexpr float k = 0.7978845608f;
    return x * 0.5f * (1.0f + mx::tanh(k * (x + 0.044715f * mx::power(x, mx::array(3)))));
}

mx::array MarlinModel::vision_forward(const mx::array& pixel_values) {
    auto& vw = weights_.vision;
    auto& cfg = config_.vision;

    // Safetensors stores PyTorch layout [C_out, C_in, D, H, W]; MLX needs C_in last
    auto conv_w = mx::transpose(vw.patch_embed_proj, {0, 2, 3, 4, 1});

    auto patches = mx::conv3d(
        pixel_values, conv_w,
        {cfg.temporal_patch_size, cfg.patch_size, cfg.patch_size});
    patches = patches + vw.patch_embed_bias;

    auto batch = patches.shape(0);
    auto d = patches.shape(1);
    auto h = patches.shape(2);
    auto w_dim = patches.shape(3);
    auto hidden_dim = patches.shape(4);
    auto seq_len = d * h * w_dim;

    int merge = cfg.spatial_merge_size;
    auto reordered = mx::reshape(patches, {batch, d, h / merge, merge, w_dim / merge, merge, hidden_dim});
    reordered = mx::transpose(reordered, {0, 1, 2, 4, 3, 5, 6});
    auto hidden = mx::reshape(reordered, {batch, seq_len, hidden_dim});
    int head_dim = cfg.hidden_size / cfg.num_heads;
    int rot_dim = head_dim / 2;
    int freq_dim = rot_dim / 2;
    float rope_theta = 10000.0f;

    std::vector<float> inv_freq_data(freq_dim);
    for (int i = 0; i < freq_dim; ++i) {
        inv_freq_data[i] = 1.0f / std::pow(rope_theta, static_cast<float>(2 * i) / rot_dim);
    }
    auto inv_freq = mx::array(inv_freq_data.data(), {freq_dim});

    int per_group = h * w_dim;
    std::vector<float> hpos_data(per_group);
    std::vector<float> wpos_data(per_group);
    int idx = 0;
    for (int mh = 0; mh < h / merge; ++mh) {
        for (int mw = 0; mw < w_dim / merge; ++mw) {
            for (int dh = 0; dh < merge; ++dh) {
                for (int dw = 0; dw < merge; ++dw) {
                    hpos_data[idx] = static_cast<float>(mh * merge + dh);
                    wpos_data[idx] = static_cast<float>(mw * merge + dw);
                    ++idx;
                }
            }
        }
    }
    auto hpos = mx::array(hpos_data.data(), {per_group});
    auto wpos = mx::array(wpos_data.data(), {per_group});

    auto h_angles = mx::outer(hpos, inv_freq);
    auto w_angles = mx::outer(wpos, inv_freq);
    auto angles = mx::concatenate({h_angles, w_angles}, -1);
    auto full_angles = mx::concatenate({angles, angles}, -1);
    auto rot_cos = mx::reshape(mx::cos(full_angles), {1, per_group, 1, head_dim});
    auto rot_sin = mx::reshape(mx::sin(full_angles), {1, per_group, 1, head_dim});

    for (int i = 0; i < cfg.depth; ++i) {
        auto& blk = vw.blocks[i];

        auto normed = vit_layer_norm(hidden, blk.norm1_weight, blk.norm1_bias);
        auto qkv = mx::matmul(normed, mx::transpose(blk.qkv_weight)) + blk.qkv_bias;

        auto qkv_split = mx::split(qkv, 3, -1);
        auto q = mx::reshape(qkv_split[0], {batch, seq_len, cfg.num_heads, head_dim});
        auto k = mx::reshape(qkv_split[1], {batch, seq_len, cfg.num_heads, head_dim});
        auto v = mx::reshape(qkv_split[2], {batch, seq_len, cfg.num_heads, head_dim});

        auto q_rot = q * rot_cos + mx::concatenate({
            mx::negative(mx::slice(q, {0,0,0,rot_dim}, {batch, seq_len, cfg.num_heads, head_dim})),
            mx::slice(q, {0,0,0,0}, {batch, seq_len, cfg.num_heads, rot_dim})
        }, -1) * rot_sin;
        auto k_rot = k * rot_cos + mx::concatenate({
            mx::negative(mx::slice(k, {0,0,0,rot_dim}, {batch, seq_len, cfg.num_heads, head_dim})),
            mx::slice(k, {0,0,0,0}, {batch, seq_len, cfg.num_heads, rot_dim})
        }, -1) * rot_sin;

        q_rot = mx::transpose(q_rot, {0, 2, 1, 3});
        k_rot = mx::transpose(k_rot, {0, 2, 1, 3});
        v = mx::transpose(v, {0, 2, 1, 3});

        float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));
        auto scores = mx::matmul(q_rot, mx::transpose(k_rot, {0, 1, 3, 2})) * scale;
        auto attn = mx::softmax(scores, {-1});
        auto attn_out = mx::matmul(attn, v);

        attn_out = mx::transpose(attn_out, {0, 2, 1, 3});
        attn_out = mx::reshape(attn_out, {batch, seq_len, cfg.hidden_size});
        attn_out = mx::matmul(attn_out, mx::transpose(blk.proj_weight)) + blk.proj_bias;

        hidden = hidden + attn_out;

        auto normed2 = vit_layer_norm(hidden, blk.norm2_weight, blk.norm2_bias);
        auto fc1_out = mx::matmul(normed2, mx::transpose(blk.fc1_weight)) + blk.fc1_bias;
        fc1_out = gelu_approx(fc1_out);
        auto fc2_out = mx::matmul(fc1_out, mx::transpose(blk.fc2_weight)) + blk.fc2_bias;

        hidden = hidden + fc2_out;
    }

    auto normed_for_merge = vit_layer_norm(hidden, vw.merger_norm, vw.merger_norm_bias);

    int msz = cfg.spatial_merge_size;
    int n_merge_groups = d * (h / msz) * (w_dim / msz);
    auto merged = mx::reshape(normed_for_merge,
        {batch, n_merge_groups, msz * msz, hidden_dim});
    merged = mx::reshape(merged,
        {batch, n_merge_groups, msz * msz * hidden_dim});

    merged = mx::matmul(merged, mx::transpose(vw.merger_fc1)) + vw.merger_fc1_bias;
    merged = gelu_approx(merged);
    merged = mx::matmul(merged, mx::transpose(vw.merger_fc2)) + vw.merger_fc2_bias;

    return merged;
}

Result<std::unique_ptr<MarlinModel>> MarlinModel::load(
    const std::string& model_path,
    const ModelConfig& config) {

    WeightLoadOptions opts{
        .model_path = model_path,
        .default_level = QuantLevel::Q4_64,
        .mixed_precision = true,
    };

    auto weights_result = load_weights(opts, config);
    if (!weights_result) {
        return std::unexpected(weights_result.error());
    }

    auto model = std::unique_ptr<MarlinModel>(new MarlinModel());
    model->weights_ = std::move(*weights_result);
    model->config_ = config;
    model->cache_.reset(config);

    return model;
}

}  // namespace marlin
