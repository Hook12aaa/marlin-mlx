#pragma once

#include "core/config.h"

#include <mlx/mlx.h>

namespace marlin {

namespace mx = mlx::core;

mx::array sample_token(const mx::array& logits, const SamplingConfig& config);

}  // namespace marlin
