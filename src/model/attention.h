#pragma once

#include <mlx/mlx.h>

namespace marlin {

namespace mx = mlx::core;

struct KVCacheEntry {
    mx::array keys;
    mx::array values;
    int length = 0;
};

struct AttentionLayerWeights {
    mx::array q_proj{mx::zeros({1})};
    mx::array k_proj{mx::zeros({1})};
    mx::array v_proj{mx::zeros({1})};
    mx::array o_proj{mx::zeros({1})};
    mx::array q_norm{mx::zeros({1})};
    mx::array k_norm{mx::zeros({1})};
};

mx::array mrope(
    const mx::array& x,
    int offset,
    int head_dim,
    float partial_rotary_factor,
    float theta,
    const std::array<int, 3>& sections,
    bool interleaved);

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
    int position);

}  // namespace marlin
