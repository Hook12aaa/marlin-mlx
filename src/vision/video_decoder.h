#pragma once

#include "core/types.h"

#include <mlx/mlx.h>
#include <string>
#include <vector>

namespace marlin {

namespace mx = mlx::core;

struct DecodedFrames {
    mx::array pixels;
    int num_frames;
    int width;
    int height;
    float fps_sampled;
};

Result<DecodedFrames> decode_video(
    const std::string& path,
    float target_fps,
    int min_frames,
    int max_frames,
    int max_pixels);

}  // namespace marlin
