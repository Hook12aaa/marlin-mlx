#pragma once

#include "core/config.h"
#include "core/types.h"
#include "model/marlin.h"

#include <string>

namespace marlin {

enum class QuantLevel { None, Q4_64, Q8_64 };

struct WeightLoadOptions {
    std::string model_path;
    QuantLevel default_level = QuantLevel::None;
    bool mixed_precision = false;
};

Result<MarlinWeights> load_weights(
    const WeightLoadOptions& opts,
    const ModelConfig& config);

}  // namespace marlin
