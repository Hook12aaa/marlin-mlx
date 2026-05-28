#pragma once

#include "core/config.h"
#include "core/types.h"
#include "model/attention.h"
#include "model/gated_deltanet.h"

#include <mlx/mlx.h>
#include <memory>
#include <variant>
#include <vector>

namespace marlin {

namespace mx = mlx::core;

struct MLPWeights {
    QuantizedTensor gate_proj;
    QuantizedTensor up_proj;
    QuantizedTensor down_proj;
};

struct LayerNormWeights {
    mx::array input_norm;
    mx::array post_attn_norm;
};

using AttnVariant = std::variant<DeltaNetLayerWeights, AttentionLayerWeights>;
using CacheVariant = std::variant<DeltaNetState, KVCacheEntry>;

struct TransformerLayer {
    AttnVariant attn_weights;
    MLPWeights mlp;
    LayerNormWeights norms;
    LayerType type;
};

struct VisionBlockWeights {
    mx::array qkv_weight;
    mx::array qkv_bias;
    mx::array proj_weight;
    mx::array proj_bias;
    mx::array norm1_weight;
    mx::array norm1_bias;
    mx::array norm2_weight;
    mx::array norm2_bias;
    mx::array fc1_weight;
    mx::array fc1_bias;
    mx::array fc2_weight;
    mx::array fc2_bias;
};

struct VisionWeights {
    mx::array patch_embed_proj{mx::zeros({1})};
    mx::array patch_embed_bias{mx::zeros({1})};
    mx::array pos_embed{mx::zeros({1})};
    std::vector<VisionBlockWeights> blocks;
    mx::array merger_fc1{mx::zeros({1})};
    mx::array merger_fc1_bias{mx::zeros({1})};
    mx::array merger_fc2{mx::zeros({1})};
    mx::array merger_fc2_bias{mx::zeros({1})};
    mx::array merger_norm{mx::zeros({1})};
    mx::array merger_norm_bias{mx::zeros({1})};
};

struct MarlinWeights {
    mx::array embed_tokens{mx::zeros({1})};
    QuantizedTensor lm_head;
    mx::array final_norm{mx::zeros({1})};
    std::vector<TransformerLayer> layers;
    VisionWeights vision;
};

struct HybridCache {
    std::vector<CacheVariant> layers;
    void reset(const ModelConfig& config);
};

class MarlinModel {
public:
    static Result<std::unique_ptr<MarlinModel>> load(
        const std::string& model_path,
        const ModelConfig& config);

    mx::array text_forward(
        const mx::array& token_ids,
        int position);

    mx::array text_forward_embeddings(
        const mx::array& embeddings,
        int position);

    mx::array text_forward_token(
        const mx::array& token_id,
        int position);

    mx::array vision_forward(const mx::array& pixel_values);

    mx::array embed_tokens(const mx::array& token_ids);

    mx::array merge_vision_text(
        const mx::array& text_ids,
        const mx::array& vision_embeds,
        int video_token_id);

    mx::array lm_head(const mx::array& hidden);

    HybridCache& cache() { return cache_; }
    const ModelConfig& config() const { return config_; }

    MarlinModel() = default;
    friend struct std::default_delete<MarlinModel>;

private:
    MarlinWeights weights_;
    HybridCache cache_;
    ModelConfig config_;
};

}  // namespace marlin
