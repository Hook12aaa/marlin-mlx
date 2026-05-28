#pragma once

#include <array>
#include <cstdint>
#include <string>
#include <vector>

namespace marlin {

enum class LayerType { LinearAttention, FullAttention };

struct TextConfig {
    int hidden_size = 2048;
    int intermediate_size = 6144;
    int num_hidden_layers = 24;
    int num_attention_heads = 8;
    int num_key_value_heads = 2;
    int head_dim = 256;
    int vocab_size = 248320;
    float rms_norm_eps = 1e-6f;
    float partial_rotary_factor = 0.25f;
    float rope_theta = 10000000.0f;
    bool tie_word_embeddings = true;
    bool mrope_interleaved = true;
    std::array<int, 3> mrope_section = {11, 11, 10};
    int full_attention_interval = 4;
    int linear_conv_kernel_dim = 4;
    int linear_key_head_dim = 128;
    int linear_num_key_heads = 16;
    int linear_num_value_heads = 16;
    int linear_value_head_dim = 128;
    std::vector<LayerType> layer_types;
};

struct VisionConfig {
    int hidden_size = 1024;
    int intermediate_size = 4096;
    int depth = 24;
    int num_heads = 16;
    int patch_size = 16;
    int temporal_patch_size = 2;
    int spatial_merge_size = 2;
    int out_hidden_size = 2048;
    int in_channels = 3;
    int num_position_embeddings = 2304;
    float rms_norm_eps = 1e-6f;
};

struct ModelConfig {
    TextConfig text;
    VisionConfig vision;
    int image_token_id = 248056;
    int video_token_id = 248057;
    int vision_start_token_id = 248053;
    int vision_end_token_id = 248054;
    int eos_token_id = 248046;
    int pad_token_id = 248044;
};

struct SamplingConfig {
    float temperature = 0.0f;
    float top_p = 1.0f;
    int top_k = 0;
    float min_p = 0.0f;
    int max_tokens = 512;
    int seed = -1;
    std::vector<std::string> stop;
    bool enable_thinking = false;
};

struct VideoConfig {
    float fps = 2.0f;
    int max_frames = 240;
    int min_frames = 4;
    int max_pixels = 200704;
    int patch_size = 16;
    int temporal_patch_size = 2;
    int merge_size = 2;
};

struct RunConfig {
    std::string model_path;
    int port = 8080;
    VideoConfig video;
    SamplingConfig sampling;
};

ModelConfig load_model_config(const std::string& model_path);

}  // namespace marlin
