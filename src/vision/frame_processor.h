#pragma once

#include "core/config.h"
#include "vision/video_decoder.h"

#include <mlx/mlx.h>

namespace marlin {

namespace mx = mlx::core;

mx::array process_frames_to_patches(
    const DecodedFrames& frames,
    const VisionConfig& config);

}  // namespace marlin
