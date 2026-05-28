#include "model/weights.h"

#include <filesystem>
#include <fstream>
#include <mlx/mlx.h>
#include <nlohmann/json.hpp>

namespace marlin {

namespace mx = mlx::core;
namespace fs = std::filesystem;
using json = nlohmann::json;

using TensorMap = std::unordered_map<std::string, mx::array>;

static TensorMap load_all_tensors(const std::string& model_path) {
    auto index_path = fs::path(model_path) / "model.safetensors.index.json";
    std::ifstream f(index_path);
    auto idx = json::parse(f);

    std::set<std::string> shard_files;
    for (auto& [key, shard] : idx["weight_map"].items()) {
        shard_files.insert(shard.get<std::string>());
    }

    TensorMap all;
    for (auto& shard : shard_files) {
        auto path = (fs::path(model_path) / shard).string();
        auto [tensors, metadata] = mx::load_safetensors(path);
        for (auto& [k, a] : tensors) {
            all.emplace(k, std::move(a));
        }
    }
    return all;
}

static mx::array get(TensorMap& m, const std::string& key) {
    auto it = m.find(key);
    if (it == m.end()) {
        return mx::zeros({1});
    }
    return std::move(it->second);
}

static QuantizedTensor quantize_weight(mx::array w, int bits = 4, int group_size = 64) {
    auto [qw, scales, biases] = mx::quantize(w, group_size, bits);
    mx::eval(qw); mx::eval(scales); mx::eval(biases);
    return QuantizedTensor{
        .weight = std::move(qw),
        .scales = std::move(scales),
        .biases = std::move(biases),
        .group_size = group_size,
        .bits = bits,
        .quantized = true,
    };
}

static QuantizedTensor as_unquantized(mx::array w) {
    return QuantizedTensor{
        .weight = std::move(w),
        .scales = mx::zeros({1}),
        .biases = mx::zeros({1}),
        .group_size = 64,
        .bits = 4,
        .quantized = false,
    };
}

static DeltaNetLayerWeights load_ssm_layer(TensorMap& m, int i) {
    auto p = "model.language_model.layers." + std::to_string(i) + ".linear_attn.";
    return DeltaNetLayerWeights{
        .in_proj_qkv = get(m, p + "in_proj_qkv.weight"),
        .in_proj_a = get(m, p + "in_proj_a.weight"),
        .in_proj_b = get(m, p + "in_proj_b.weight"),
        .in_proj_z = get(m, p + "in_proj_z.weight"),
        .conv1d_weight = get(m, p + "conv1d.weight"),
        .conv1d_bias = mx::zeros({1}),
        .A_log = get(m, p + "A_log"),
        .dt_bias = get(m, p + "dt_bias"),
        .norm_weight = get(m, p + "norm.weight"),
        .out_proj = get(m, p + "out_proj.weight"),
    };
}

static AttentionLayerWeights load_attn_layer(TensorMap& m, int i) {
    auto p = "model.language_model.layers." + std::to_string(i) + ".self_attn.";
    return AttentionLayerWeights{
        .q_proj = get(m, p + "q_proj.weight"),
        .k_proj = get(m, p + "k_proj.weight"),
        .v_proj = get(m, p + "v_proj.weight"),
        .o_proj = get(m, p + "o_proj.weight"),
        .q_norm = get(m, p + "q_norm.weight"),
        .k_norm = get(m, p + "k_norm.weight"),
    };
}

static MLPWeights load_mlp(TensorMap& m, int i, bool quantize) {
    auto p = "model.language_model.layers." + std::to_string(i) + ".mlp.";
    if (quantize) {
        return MLPWeights{
            .gate_proj = quantize_weight(get(m, p + "gate_proj.weight")),
            .up_proj = quantize_weight(get(m, p + "up_proj.weight")),
            .down_proj = quantize_weight(get(m, p + "down_proj.weight")),
        };
    }
    return MLPWeights{
        .gate_proj = as_unquantized(get(m, p + "gate_proj.weight")),
        .up_proj = as_unquantized(get(m, p + "up_proj.weight")),
        .down_proj = as_unquantized(get(m, p + "down_proj.weight")),
    };
}

static LayerNormWeights load_norms(TensorMap& m, int i) {
    auto p = "model.language_model.layers." + std::to_string(i) + ".";
    return LayerNormWeights{
        .input_norm = get(m, p + "input_layernorm.weight"),
        .post_attn_norm = get(m, p + "post_attention_layernorm.weight"),
    };
}

static VisionBlockWeights load_vit_block(TensorMap& m, int i) {
    auto p = "model.visual.blocks." + std::to_string(i) + ".";
    return VisionBlockWeights{
        .qkv_weight = get(m, p + "attn.qkv.weight"),
        .qkv_bias = get(m, p + "attn.qkv.bias"),
        .proj_weight = get(m, p + "attn.proj.weight"),
        .proj_bias = get(m, p + "attn.proj.bias"),
        .norm1_weight = get(m, p + "norm1.weight"),
        .norm1_bias = get(m, p + "norm1.bias"),
        .norm2_weight = get(m, p + "norm2.weight"),
        .norm2_bias = get(m, p + "norm2.bias"),
        .fc1_weight = get(m, p + "mlp.linear_fc1.weight"),
        .fc1_bias = get(m, p + "mlp.linear_fc1.bias"),
        .fc2_weight = get(m, p + "mlp.linear_fc2.weight"),
        .fc2_bias = get(m, p + "mlp.linear_fc2.bias"),
    };
}

Result<MarlinWeights> load_weights(
    const WeightLoadOptions& opts,
    const ModelConfig& config) {

    auto m = load_all_tensors(opts.model_path);

    std::vector<TransformerLayer> layers;
    layers.reserve(config.text.num_hidden_layers);
    for (int i = 0; i < config.text.num_hidden_layers; ++i) {
        auto type = config.text.layer_types[i];
        AttnVariant attn = (type == LayerType::LinearAttention)
            ? AttnVariant{load_ssm_layer(m, i)}
            : AttnVariant{load_attn_layer(m, i)};

        layers.push_back(TransformerLayer{
            .attn_weights = std::move(attn),
            .mlp = load_mlp(m, i, opts.mixed_precision || opts.default_level == QuantLevel::Q4_64),
            .norms = load_norms(m, i),
            .type = type,
        });
    }

    std::vector<VisionBlockWeights> vit_blocks;
    vit_blocks.reserve(config.vision.depth);
    for (int i = 0; i < config.vision.depth; ++i) {
        vit_blocks.push_back(load_vit_block(m, i));
    }

    VisionWeights vision{
        .patch_embed_proj = get(m, "model.visual.patch_embed.proj.weight"),
        .patch_embed_bias = get(m, "model.visual.patch_embed.proj.bias"),
        .pos_embed = get(m, "model.visual.pos_embed.weight"),
        .blocks = std::move(vit_blocks),
        .merger_fc1 = get(m, "model.visual.merger.linear_fc1.weight"),
        .merger_fc1_bias = get(m, "model.visual.merger.linear_fc1.bias"),
        .merger_fc2 = get(m, "model.visual.merger.linear_fc2.weight"),
        .merger_fc2_bias = get(m, "model.visual.merger.linear_fc2.bias"),
        .merger_norm = get(m, "model.visual.merger.norm.weight"),
        .merger_norm_bias = get(m, "model.visual.merger.norm.bias"),
    };

    auto lm_head_raw = get(m, "lm_head.weight");
    QuantizedTensor lm_head_qt;
    if (opts.default_level == QuantLevel::Q4_64 || opts.mixed_precision) {
        auto [qw, scales, biases] = mx::quantize(lm_head_raw, 64, 4);
        mx::eval(qw);
        mx::eval(scales);
        mx::eval(biases);
        lm_head_qt = QuantizedTensor{
            .weight = std::move(qw),
            .scales = std::move(scales),
            .biases = std::move(biases),
            .group_size = 64,
            .bits = 4,
            .quantized = true,
        };
    } else {
        lm_head_qt = QuantizedTensor{
            .weight = std::move(lm_head_raw),
            .scales = mx::zeros({1}),
            .biases = mx::zeros({1}),
            .group_size = 64,
            .bits = 4,
            .quantized = false,
        };
    }

    MarlinWeights weights{
        .embed_tokens = get(m, "model.language_model.embed_tokens.weight"),
        .lm_head = std::move(lm_head_qt),
        .final_norm = get(m, "model.language_model.norm.weight"),
        .layers = std::move(layers),
        .vision = std::move(vision),
    };

    return weights;
}

}  // namespace marlin
