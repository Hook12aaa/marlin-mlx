#include "tokenizer/chat_template.h"

namespace marlin {

static const char* CAPTION_PROMPT =
    "Provide a spatial description of this clip followed by time-ranged events.\n"
    "For each event, give the time range as <start - end> and a short description.";

std::string build_caption_prompt(const std::string& video_token) {
    return "<|im_start|>user\n" + video_token + "\n" +
           CAPTION_PROMPT + "<|im_end|>\n<|im_start|>assistant\n";
}

std::string build_find_prompt(const std::string& video_token, const std::string& event) {
    std::string prompt =
        "Identify the timestamps during which \"" + event + "\" takes place. "
        "Output the time range as \"From <start> to <end>.\" (numbers in seconds).";
    return "<|im_start|>user\n" + video_token + "\n" +
           prompt + "<|im_end|>\n<|im_start|>assistant\n";
}

}  // namespace marlin
