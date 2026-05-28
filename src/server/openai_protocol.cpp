#include "server/openai_protocol.h"
#include <nlohmann/json.hpp>

namespace marlin {
namespace openai {

using json = nlohmann::json;

static ParseResult fail(const std::string& param, const std::string& msg) {
    ParseResult r;
    r.valid = false;
    r.error_type = "invalid_request_error";
    r.error_param = param;
    r.error_message = msg;
    return r;
}

ParseResult parse_chat_request(const std::string& body) {
    json j;
    try { j = json::parse(body); } catch (...) {
        return fail("", "request body is not valid JSON");
    }
    if (!j.is_object()) return fail("", "request must be a JSON object");

    ChatRequest req;

    if (j.contains("model") && j["model"].is_string())
        req.model = j["model"].get<std::string>();

    if (!j.contains("messages") || !j["messages"].is_array() || j["messages"].empty())
        return fail("messages", "messages is required and must be a non-empty array");

    for (const auto& m : j["messages"]) {
        if (!m.is_object()) return fail("messages", "each message must be an object");
        if (!m.contains("role") || !m["role"].is_string())
            return fail("messages", "each message must have a string role");
        if (!m.contains("content"))
            return fail("messages", "each message must have content");

        Message msg;
        msg.role = m["role"].get<std::string>();

        if (m["content"].is_string()) {
            msg.content = m["content"].get<std::string>();
        } else if (m["content"].is_array()) {
            for (const auto& part : m["content"]) {
                if (!part.is_object() || !part.contains("type"))
                    return fail("messages", "content part must have a type field");
                auto type = part["type"].get<std::string>();
                if (type == "text") {
                    if (!part.contains("text") || !part["text"].is_string())
                        return fail("messages", "text part must have a text field");
                    msg.content += part["text"].get<std::string>();
                } else if (type == "video_url") {
                    if (!part.contains("video_url") || !part["video_url"].is_object())
                        return fail("messages", "video_url part must have a video_url object");
                    auto url = part["video_url"]["url"].get<std::string>();
                    if (url.substr(0, 7) == "file://") {
                        msg.video_path = url.substr(7);
                    } else {
                        msg.video_path = url;
                    }
                    msg.has_video = true;
                }
            }
        } else {
            return fail("messages", "content must be a string or array");
        }

        req.messages.push_back(std::move(msg));
    }

    if (j.contains("stream") && j["stream"].is_boolean()) req.stream = j["stream"].get<bool>();
    if (j.contains("max_tokens") && j["max_tokens"].is_number_integer()) req.max_tokens = j["max_tokens"].get<int>();
    if (j.contains("temperature") && j["temperature"].is_number()) req.temperature = j["temperature"].get<double>();
    if (j.contains("top_p") && j["top_p"].is_number()) req.top_p = j["top_p"].get<double>();
    if (j.contains("seed") && j["seed"].is_number_integer()) req.seed = j["seed"].get<int>();
    if (j.contains("top_k") && j["top_k"].is_number_integer()) req.top_k = j["top_k"].get<int>();
    if (j.contains("min_p") && j["min_p"].is_number()) req.min_p = j["min_p"].get<double>();
    if (j.contains("thinking") && j["thinking"].is_boolean()) req.thinking = j["thinking"].get<bool>();

    if (j.contains("stop")) {
        const auto& s = j["stop"];
        if (s.is_string()) req.stop.push_back(s.get<std::string>());
        else if (s.is_array()) {
            for (const auto& v : s) {
                if (!v.is_string()) return fail("stop", "stop entries must be strings");
                req.stop.push_back(v.get<std::string>());
            }
        }
    }

    ParseResult r;
    r.req = std::move(req);
    r.valid = true;
    return r;
}

std::string build_stream_chunk(
    const std::string& id, const std::string& model, int64_t created,
    std::string_view delta_content, std::string_view delta_reasoning,
    const std::string& finish_reason) {

    json delta = json::object();
    bool is_first = delta_content.empty() && delta_reasoning.empty() && finish_reason.empty();
    if (is_first) delta["role"] = "assistant";
    if (!delta_content.empty()) delta["content"] = std::string(delta_content);
    if (!delta_reasoning.empty()) delta["reasoning_content"] = std::string(delta_reasoning);

    json choice = {
        {"index", 0},
        {"delta", delta},
        {"finish_reason", finish_reason.empty() ? json(nullptr) : json(finish_reason)}
    };
    json out = {
        {"id", id},
        {"object", "chat.completion.chunk"},
        {"created", created},
        {"model", model},
        {"choices", json::array({choice})}
    };
    return out.dump();
}

std::string build_full_response(
    const std::string& id, const std::string& model, int64_t created,
    const std::string& content, const std::string& reasoning,
    const std::string& finish_reason,
    int prompt_tokens, int completion_tokens) {

    json message = {{"role", "assistant"}, {"content", content}};
    if (!reasoning.empty()) message["reasoning_content"] = reasoning;
    json out = {
        {"id", id},
        {"object", "chat.completion"},
        {"created", created},
        {"model", model},
        {"choices", json::array({{
            {"index", 0},
            {"message", message},
            {"finish_reason", finish_reason}
        }})},
        {"usage", {
            {"prompt_tokens", prompt_tokens},
            {"completion_tokens", completion_tokens},
            {"total_tokens", prompt_tokens + completion_tokens}
        }}
    };
    return out.dump();
}

std::string build_error(int, const std::string& type,
                        const std::string& param, const std::string& message) {
    json err = {{"type", type}, {"message", message}};
    if (!param.empty()) err["param"] = param;
    err["code"] = nullptr;
    json out = {{"error", err}};
    return out.dump();
}

}  // namespace openai
}  // namespace marlin
