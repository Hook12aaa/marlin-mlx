#pragma once

#include <mlx/mlx.h>
#include <vector>

namespace marlin {

namespace mx = mlx::core;

struct DeltaNetState {
    mx::array conv_state;
    mx::array recurrent_state;
};

struct DeltaNetLayerWeights {
    mx::array in_proj_qkv{mx::zeros({1})};
    mx::array in_proj_a{mx::zeros({1})};
    mx::array in_proj_b{mx::zeros({1})};
    mx::array in_proj_z{mx::zeros({1})};
    mx::array conv1d_weight{mx::zeros({1})};
    mx::array conv1d_bias{mx::zeros({1})};
    mx::array A_log{mx::zeros({1})};
    mx::array dt_bias{mx::zeros({1})};
    mx::array norm_weight{mx::zeros({1})};
    mx::array out_proj{mx::zeros({1})};
};

mx::array gated_deltanet_forward(
    const mx::array& hidden,
    const DeltaNetLayerWeights& w,
    DeltaNetState& state,
    int num_k_heads,
    int num_v_heads,
    int head_k_dim,
    int head_v_dim,
    bool has_cache);

mx::array gated_delta_recurrence(
    const mx::array& query,
    const mx::array& key,
    const mx::array& value,
    const mx::array& gate,
    const mx::array& beta,
    mx::array& recurrent_state);

mx::array causal_conv1d_step(
    const mx::array& x,
    const mx::array& weight,
    const mx::array& bias,
    mx::array& conv_state);

}  // namespace marlin
