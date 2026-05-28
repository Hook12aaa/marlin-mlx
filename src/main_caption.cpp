#include "core/engine.h"

#include <fmt/format.h>
#include <iostream>

int main(int argc, char* argv[]) {
    if (argc < 3) {
        fmt::print(stderr, "Usage: marlin-caption <model-path> <video-path> [event]\n");
        return 1;
    }

    std::string model_path = argv[1];
    std::string video_path = argv[2];

    marlin::RunConfig config;
    config.model_path = model_path;

    auto engine_result = marlin::Engine::create(config);
    if (!engine_result) {
        fmt::print(stderr, "Error: {}\n", engine_result.error().message);
        return 1;
    }
    auto& engine = *engine_result;

    marlin::SamplingConfig sampling;
    sampling.temperature = 0.0f;
    sampling.max_tokens = 512;

    if (argc >= 4) {
        std::string event = argv[3];
        auto result = engine->find(video_path, event, sampling);
        if (!result) {
            fmt::print(stderr, "Error: {}\n", result.error().message);
            return 1;
        }
        fmt::print("{}\n", result->raw);
        fmt::print("span: {}\n",
                   result->span ? fmt::format("{:.1f}-{:.1f}",
                                              result->span->first,
                                              result->span->second)
                                : "none");
        fmt::print("decode: {:.1f} tok/s\n", result->metrics.decode_tok_s);
    } else {
        auto result = engine->caption(video_path, sampling);
        if (!result) {
            fmt::print(stderr, "Error: {}\n", result.error().message);
            return 1;
        }
        fmt::print("{}\n", result->caption);
        fmt::print("decode: {:.1f} tok/s\n", result->metrics.decode_tok_s);
    }

    return 0;
}
