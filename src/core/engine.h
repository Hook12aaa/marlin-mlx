#pragma once

#include "core/config.h"
#include "core/types.h"
#include "model/marlin.h"
#include "tokenizer/tokenizer.h"

#include <atomic>
#include <memory>

namespace marlin {

class Engine {
public:
    static Result<std::unique_ptr<Engine>> create(const RunConfig& config);

    Result<CaptionResult> caption(
        const std::string& video_path,
        const SamplingConfig& sampling);

    Result<FindResult> find(
        const std::string& video_path,
        const std::string& event,
        const SamplingConfig& sampling);

    Result<GenerateResult> generate(
        const std::vector<int32_t>& token_ids,
        const SamplingConfig& sampling,
        TokenCallback on_token = nullptr,
        std::atomic<bool>* cancel = nullptr);

    Result<GenerateResult> generate_with_vision(
        const std::vector<int32_t>& token_ids,
        const mx::array& vision_embeds,
        const SamplingConfig& sampling,
        TokenCallback on_token = nullptr,
        std::atomic<bool>* cancel = nullptr);

    void cancel() { cancel_.store(true); }

private:
    std::unique_ptr<MarlinModel> model_;
    std::unique_ptr<Tokenizer> tokenizer_;
    RunConfig config_;
    ModelConfig model_config_;
    std::atomic<bool> cancel_{false};

    Result<mx::array> encode_video(const std::string& video_path);
};

}  // namespace marlin
