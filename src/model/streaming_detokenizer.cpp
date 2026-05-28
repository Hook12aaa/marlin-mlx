#include "model/streaming_detokenizer.h"
#include "tokenizer/tokenizer.h"

namespace marlin {

StreamingDetokenizer::StreamingDetokenizer(const Tokenizer& tokenizer)
    : tokenizer_(tokenizer) {}

void StreamingDetokenizer::add_token(int32_t id) {
    pending_.push_back(id);
    auto decoded = tokenizer_.decode(pending_);

    size_t valid_end = 0;
    for (size_t i = 0; i < decoded.size(); ++i) {
        uint8_t c = decoded[i];
        if (c < 0x80 || (c >= 0xC0)) {
            int expected_len = 1;
            if (c >= 0xF0) expected_len = 4;
            else if (c >= 0xE0) expected_len = 3;
            else if (c >= 0xC0) expected_len = 2;

            if (i + expected_len <= decoded.size()) {
                valid_end = i + expected_len;
            }
        }
    }

    if (valid_end > buffer_.size()) {
        auto new_text = decoded.substr(buffer_.size(), valid_end - buffer_.size());
        buffer_ = decoded.substr(0, valid_end);
        if (callback_ && !new_text.empty()) {
            callback_(new_text);
        }
    }
}

void StreamingDetokenizer::flush() {
    if (pending_.empty()) return;
    auto decoded = tokenizer_.decode(pending_);
    if (decoded.size() > buffer_.size()) {
        auto remainder = decoded.substr(buffer_.size());
        if (callback_ && !remainder.empty()) {
            callback_(remainder);
        }
    }
    pending_.clear();
    buffer_.clear();
}

}  // namespace marlin
