#include "vision/frame_processor.h"

namespace marlin {

namespace mx = mlx::core;

mx::array process_frames_to_patches(
    const DecodedFrames& frames,
    const VisionConfig& config) {

    auto pixels = frames.pixels;
    int n = frames.num_frames;
    int h = frames.height;
    int w = frames.width;
    int tp = config.temporal_patch_size;

    int n_groups = (n / tp) * tp;
    if (n_groups == 0) n_groups = tp;

    pixels = mx::slice(pixels, {0, 0, 0, 0}, {n_groups, h, w, 3});
    pixels = mx::reshape(pixels, {n_groups / tp, tp, h, w, 3});

    return pixels;
}

}  // namespace marlin
