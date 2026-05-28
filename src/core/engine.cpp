#include "core/engine.h"
#include "model/streaming_detokenizer.h"
#include "sampling/sampler.h"
#include "tokenizer/chat_template.h"
#include "vision/frame_processor.h"
#include "vision/video_decoder.h"

#include <chrono>
#include <fmt/format.h>
#include <mlx/mlx.h>

namespace marlin {

namespace mx = mlx::core;

Result<std::unique_ptr<Engine>> Engine::create(const RunConfig& config) {
    auto model_config = load_model_config(config.model_path);

    auto model_result = MarlinModel::load(config.model_path, model_config);
    if (!model_result) return std::unexpected(model_result.error());

    auto tok_result = Tokenizer::load(config.model_path);
    if (!tok_result) return std::unexpected(tok_result.error());

    auto engine = std::make_unique<Engine>();
    engine->model_ = std::move(*model_result);
    engine->tokenizer_ = std::move(*tok_result);
    engine->config_ = config;
    engine->model_config_ = model_config;

    return engine;
}

Result<mx::array> Engine::encode_video(const std::string& video_path) {
    auto frames_result = decode_video(
        video_path,
        config_.video.fps,
        config_.video.min_frames,
        config_.video.max_frames,
        config_.video.max_pixels);

    if (!frames_result) return std::unexpected(frames_result.error());

    auto patches = process_frames_to_patches(*frames_result, model_config_.vision);
    auto vision_out = model_->vision_forward(patches);
    int n_groups = vision_out.shape(0);
    int tokens_per_group = vision_out.shape(1);
    int hidden = vision_out.shape(2);
    return mx::reshape(vision_out, {1, n_groups * tokens_per_group, hidden});
}

static GenerateResult run_decode_loop(
    MarlinModel& model,
    const mx::array& prefill_hidden,
    const SamplingConfig& sampling,
    const Tokenizer& tokenizer,
    int prompt_len,
    int eos_id,
    TokenCallback on_token,
    std::atomic<bool>* cancel,
    std::atomic<bool>& cancel_flag,
    std::chrono::high_resolution_clock::time_point t_start) {

    auto logits = model.lm_head(prefill_hidden);
    auto token = sample_token(logits, sampling);
    mx::async_eval({token});

    auto t_first = std::chrono::high_resolution_clock::now();

    int position = prompt_len;
    int gen_count = 0;
    std::vector<int32_t> generated;

    StreamingDetokenizer detok(tokenizer);
    if (on_token) detok.set_callback(on_token);

    int max_tok = sampling.max_tokens;

    while (gen_count < max_tok) {
        int32_t tok_val = token.item<int32_t>();
        if (tok_val == eos_id) break;
        if (cancel && cancel->load()) break;
        if (cancel_flag.load()) break;

        generated.push_back(tok_val);
        detok.add_token(tok_val);
        ++gen_count;

        auto hidden = model.text_forward_token(token, position);
        logits = model.lm_head(hidden);
        token = sample_token(logits, sampling);
        mx::async_eval({token});
        ++position;
    }

    detok.flush();

    auto t_end = std::chrono::high_resolution_clock::now();
    double prefill_s = std::chrono::duration<double>(t_first - t_start).count();
    double total_s = std::chrono::duration<double>(t_end - t_start).count();
    double decode_s = total_s - prefill_s;

    auto text = tokenizer.decode(generated);

    return GenerateResult{
        .text = text,
        .thinking = "",
        .metrics = SpeedMetrics{
            .prefill_tok_s = prompt_len / prefill_s,
            .decode_tok_s = gen_count > 1 ? (gen_count - 1) / decode_s : 0,
            .total_wall_s = total_s,
            .prompt_tokens = prompt_len,
            .gen_tokens = gen_count,
        },
    };
}

Result<GenerateResult> Engine::generate(
    const std::vector<int32_t>& token_ids,
    const SamplingConfig& sampling,
    TokenCallback on_token,
    std::atomic<bool>* cancel) {

    cancel_.store(false);
    auto t_start = std::chrono::high_resolution_clock::now();

    auto ids = mx::array(token_ids.data(),
                         {1, static_cast<int>(token_ids.size())},
                         mx::int32);

    model_->cache().reset(model_config_);

    auto hidden = model_->text_forward(ids, 0);
    int prompt_len = static_cast<int>(token_ids.size());

    return run_decode_loop(*model_, hidden, sampling, *tokenizer_,
                           prompt_len, model_config_.eos_token_id,
                           on_token, cancel, cancel_, t_start);
}

Result<GenerateResult> Engine::generate_with_vision(
    const std::vector<int32_t>& token_ids,
    const mx::array& vision_embeds,
    const SamplingConfig& sampling,
    TokenCallback on_token,
    std::atomic<bool>* cancel) {

    cancel_.store(false);
    auto t_start = std::chrono::high_resolution_clock::now();

    auto ids = mx::array(token_ids.data(),
                         {1, static_cast<int>(token_ids.size())},
                         mx::int32);

    model_->cache().reset(model_config_);

    auto merged = model_->merge_vision_text(ids, vision_embeds,
                                             model_config_.video_token_id);
    auto hidden = model_->text_forward_embeddings(merged, 0);
    int prompt_len = static_cast<int>(token_ids.size());

    return run_decode_loop(*model_, hidden, sampling, *tokenizer_,
                           prompt_len, model_config_.eos_token_id,
                           on_token, cancel, cancel_, t_start);
}

static std::string build_video_placeholder(int n_vision_tokens, int tokens_per_group) {
    int n_groups = n_vision_tokens / tokens_per_group;
    if (n_groups == 0) n_groups = 1;
    int toks_per = n_vision_tokens / n_groups;

    std::string result;
    for (int g = 0; g < n_groups; ++g) {
        result += "<|vision_start|>";
        for (int t = 0; t < toks_per; ++t) {
            result += "<|video_pad|>";
        }
        result += "<|vision_end|>";
    }
    return result;
}

Result<CaptionResult> Engine::caption(
    const std::string& video_path,
    const SamplingConfig& sampling) {

    auto vision_result = encode_video(video_path);
    if (!vision_result) return std::unexpected(vision_result.error());

    int n_vision_tokens = vision_result->shape(1);
    int merge = model_config_.vision.spatial_merge_size;
    int ph = model_config_.vision.patch_size;
    int tokens_per_frame = (448 / ph / merge) * (448 / ph / merge);

    auto video_placeholder = build_video_placeholder(n_vision_tokens, tokens_per_frame);
    auto prompt = build_caption_prompt(video_placeholder);
    auto prompt_ids = tokenizer_->encode(prompt);

    auto gen_result = generate_with_vision(prompt_ids, *vision_result, sampling);
    if (!gen_result) return std::unexpected(gen_result.error());

    return CaptionResult{
        .caption = gen_result->text,
        .scene = "",
        .events = {},
        .raw = gen_result->text,
        .metrics = gen_result->metrics,
    };
}

Result<FindResult> Engine::find(
    const std::string& video_path,
    const std::string& event,
    const SamplingConfig& sampling) {

    auto vision_result = encode_video(video_path);
    if (!vision_result) return std::unexpected(vision_result.error());

    int n_vision_tokens = vision_result->shape(1);
    int merge = model_config_.vision.spatial_merge_size;
    int ph = model_config_.vision.patch_size;
    int tokens_per_frame = (448 / ph / merge) * (448 / ph / merge);

    auto video_placeholder = build_video_placeholder(n_vision_tokens, tokens_per_frame);
    auto prompt = build_find_prompt(video_placeholder, event);
    auto prompt_ids = tokenizer_->encode(prompt);

    SamplingConfig find_sampling = sampling;
    find_sampling.max_tokens = 64;

    auto gen_result = generate_with_vision(prompt_ids, *vision_result, find_sampling);
    if (!gen_result) return std::unexpected(gen_result.error());

    return FindResult{
        .raw = gen_result->text,
        .span = std::nullopt,
        .format_ok = false,
        .metrics = gen_result->metrics,
    };
}

}  // namespace marlin
