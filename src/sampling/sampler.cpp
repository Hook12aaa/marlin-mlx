#include "sampling/sampler.h"

#include <limits>

namespace marlin {

namespace mx = mlx::core;

mx::array sample_token(const mx::array& logits, const SamplingConfig& config) {
    auto last_logits = mx::slice(logits, {0, logits.shape(1) - 1, 0},
                                 {1, logits.shape(1), logits.shape(2)});
    last_logits = mx::squeeze(last_logits, 0);

    if (config.temperature <= 0.0f) {
        return mx::argmax(last_logits, -1);
    }

    auto scaled = last_logits / config.temperature;

    if (config.top_k > 0) {
        auto topk = mx::topk(scaled, config.top_k, -1);
        auto min_val = mx::min(topk, {-1}, true);
        auto neg_inf = mx::full(scaled.shape(), -std::numeric_limits<float>::infinity());
        scaled = mx::where(scaled < min_val, neg_inf, scaled);
    }

    if (config.top_p < 1.0f && config.top_p > 0.0f) {
        auto probs = mx::softmax(scaled, {-1});
        auto sorted_idx = mx::argsort(probs, -1);
        auto sorted_probs = mx::take_along_axis(probs, sorted_idx, -1);
        auto cumulative = mx::cumsum(sorted_probs, -1);
        auto mask = cumulative - sorted_probs > config.top_p;
        auto scatter_mask = mx::take_along_axis(mask, mx::argsort(sorted_idx, -1), -1);
        auto neg_inf = mx::full(scaled.shape(), -std::numeric_limits<float>::infinity());
        scaled = mx::where(scatter_mask, neg_inf, scaled);
    }

    auto probs = mx::softmax(scaled, {-1});
    return mx::random::categorical(mx::log(probs));
}

}  // namespace marlin
