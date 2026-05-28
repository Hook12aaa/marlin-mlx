#include "core/config.h"

#include <filesystem>
#include <fstream>
#include <nlohmann/json.hpp>

namespace marlin {

using json = nlohmann::json;
namespace fs = std::filesystem;

ModelConfig load_model_config(const std::string& model_path) {
    auto path = fs::path(model_path) / "config.json";
    std::ifstream f(path);
    auto j = json::parse(f);

    ModelConfig cfg;

    if (j.contains("text_config")) {
        auto& tc = j["text_config"];
        auto& t = cfg.text;
        t.hidden_size = tc.value("hidden_size", t.hidden_size);
        t.intermediate_size = tc.value("intermediate_size", t.intermediate_size);
        t.num_hidden_layers = tc.value("num_hidden_layers", t.num_hidden_layers);
        t.num_attention_heads = tc.value("num_attention_heads", t.num_attention_heads);
        t.num_key_value_heads = tc.value("num_key_value_heads", t.num_key_value_heads);
        t.head_dim = tc.value("head_dim", t.head_dim);
        t.vocab_size = tc.value("vocab_size", t.vocab_size);
        t.rms_norm_eps = tc.value("rms_norm_eps", t.rms_norm_eps);
        t.partial_rotary_factor = tc.value("partial_rotary_factor", t.partial_rotary_factor);
        t.tie_word_embeddings = tc.value("tie_word_embeddings", t.tie_word_embeddings);
        t.full_attention_interval = tc.value("full_attention_interval", t.full_attention_interval);
        t.linear_conv_kernel_dim = tc.value("linear_conv_kernel_dim", t.linear_conv_kernel_dim);
        t.linear_key_head_dim = tc.value("linear_key_head_dim", t.linear_key_head_dim);
        t.linear_num_key_heads = tc.value("linear_num_key_heads", t.linear_num_key_heads);
        t.linear_num_value_heads = tc.value("linear_num_value_heads", t.linear_num_value_heads);
        t.linear_value_head_dim = tc.value("linear_value_head_dim", t.linear_value_head_dim);

        if (tc.contains("rope_parameters")) {
            auto& rp = tc["rope_parameters"];
            t.rope_theta = rp.value("rope_theta", t.rope_theta);
            t.mrope_interleaved = rp.value("mrope_interleaved", t.mrope_interleaved);
            if (rp.contains("mrope_section")) {
                auto& s = rp["mrope_section"];
                t.mrope_section = {s[0].get<int>(), s[1].get<int>(), s[2].get<int>()};
            }
        }

        if (tc.contains("layer_types")) {
            t.layer_types.clear();
            for (auto& lt : tc["layer_types"]) {
                auto s = lt.get<std::string>();
                if (s == "full_attention") {
                    t.layer_types.push_back(LayerType::FullAttention);
                } else {
                    t.layer_types.push_back(LayerType::LinearAttention);
                }
            }
        }
    }

    if (j.contains("vision_config")) {
        auto& vc = j["vision_config"];
        auto& v = cfg.vision;
        v.hidden_size = vc.value("hidden_size", v.hidden_size);
        v.intermediate_size = vc.value("intermediate_size", v.intermediate_size);
        v.depth = vc.value("depth", v.depth);
        v.num_heads = vc.value("num_heads", v.num_heads);
        v.patch_size = vc.value("patch_size", v.patch_size);
        v.temporal_patch_size = vc.value("temporal_patch_size", v.temporal_patch_size);
        v.spatial_merge_size = vc.value("spatial_merge_size", v.spatial_merge_size);
        v.out_hidden_size = vc.value("out_hidden_size", v.out_hidden_size);
        v.in_channels = vc.value("in_channels", v.in_channels);
        v.num_position_embeddings = vc.value("num_position_embeddings", v.num_position_embeddings);
    }

    cfg.image_token_id = j.value("image_token_id", cfg.image_token_id);
    cfg.video_token_id = j.value("video_token_id", cfg.video_token_id);
    cfg.vision_start_token_id = j.value("vision_start_token_id", cfg.vision_start_token_id);
    cfg.vision_end_token_id = j.value("vision_end_token_id", cfg.vision_end_token_id);
    cfg.eos_token_id = j.value("eos_token_id", cfg.eos_token_id);
    cfg.pad_token_id = j.value("pad_token_id", cfg.pad_token_id);

    return cfg;
}

}  // namespace marlin
