#pragma once

#include <string>
#include <vector>

namespace marlin {

struct ChatMessage;

std::string build_caption_prompt(const std::string& video_token);
std::string build_find_prompt(const std::string& video_token, const std::string& event);

}  // namespace marlin
