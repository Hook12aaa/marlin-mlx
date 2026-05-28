#pragma once

#include <cstdint>
#include <functional>
#include <string>
#include <vector>

namespace marlin {

class Tokenizer;

class StreamingDetokenizer {
public:
    explicit StreamingDetokenizer(const Tokenizer& tokenizer);

    void add_token(int32_t id);
    void flush();
    void set_callback(std::function<void(std::string_view)> cb) { callback_ = std::move(cb); }

private:
    const Tokenizer& tokenizer_;
    std::vector<int32_t> pending_;
    std::string buffer_;
    std::function<void(std::string_view)> callback_;
};

}  // namespace marlin
