#pragma once

#include <optional>
#include <string>
#include <vector>

namespace marlin {
namespace openai {

struct Message {
    std::string role;
    std::string content;
    bool has_video = false;
    std::string video_path;
};

struct ChatRequest {
    std::string model;
    std::vector<Message> messages;
    bool stream = false;
    int max_tokens = 512;
    double temperature = 0.0;
    double top_p = 1.0;
    int top_k = 0;
    double min_p = 0.0;
    int seed = -1;
    std::vector<std::string> stop;
    bool thinking = false;
};

struct ParseResult {
    ChatRequest req;
    bool valid = false;
    std::string error_message;
    std::string error_type;
    std::string error_param;
    bool ok() const { return valid; }
};

ParseResult parse_chat_request(const std::string& body);

std::string build_stream_chunk(
    const std::string& id, const std::string& model, int64_t created,
    std::string_view delta_content, std::string_view delta_reasoning,
    const std::string& finish_reason);

std::string build_full_response(
    const std::string& id, const std::string& model, int64_t created,
    const std::string& content, const std::string& reasoning,
    const std::string& finish_reason,
    int prompt_tokens, int completion_tokens);

std::string build_error(int http_status, const std::string& type,
                        const std::string& param, const std::string& message);

}  // namespace openai
}  // namespace marlin
