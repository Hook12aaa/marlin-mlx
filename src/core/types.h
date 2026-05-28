#pragma once

#include <cstdint>
#include <expected>
#include <functional>
#include <optional>
#include <string>
#include <variant>
#include <vector>

namespace marlin {

struct Error {
    std::string message;
    static Error from(std::string msg) { return {std::move(msg)}; }
};

template <typename T>
using Result = std::expected<T, Error>;

struct SpeedMetrics {
    double prefill_tok_s{};
    double decode_tok_s{};
    double total_wall_s{};
    int prompt_tokens{};
    int gen_tokens{};
};

struct ChatMessage {
    std::string role;
    std::string content;
};

struct VideoInput {
    std::string path;
};

struct Event {
    float start;
    float end;
    std::string description;
};

struct CaptionResult {
    std::string caption;
    std::string scene;
    std::vector<Event> events;
    std::string raw;
    SpeedMetrics metrics;
};

struct FindResult {
    std::string raw;
    std::optional<std::pair<float, float>> span;
    bool format_ok;
    SpeedMetrics metrics;
};

struct GenerateResult {
    std::string text;
    std::string thinking;
    SpeedMetrics metrics;
};

using TokenCallback = std::function<void(std::string_view text)>;

}  // namespace marlin

#include <mlx/mlx.h>

namespace marlin {

namespace mx = mlx::core;

struct QuantizedTensor {
    mx::array weight{mx::zeros({1})};
    mx::array scales{mx::zeros({1})};
    mx::array biases{mx::zeros({1})};
    int group_size = 64;
    int bits = 4;
    bool quantized = false;
};

mx::array qmatmul(const mx::array& x, const QuantizedTensor& w);

}  // namespace marlin
