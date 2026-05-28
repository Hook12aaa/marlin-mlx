#pragma once

#include "core/types.h"

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace marlin {

class Tokenizer {
public:
    static Result<std::unique_ptr<Tokenizer>> load(const std::string& model_path);

    std::vector<int32_t> encode(const std::string& text) const;
    std::string decode(const std::vector<int32_t>& ids) const;
    std::string id_to_piece(int32_t id) const;
    int32_t eos_id() const { return eos_id_; }

    Tokenizer();
    ~Tokenizer();
    Tokenizer(Tokenizer&&) noexcept;
    Tokenizer& operator=(Tokenizer&&) noexcept;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
    int32_t eos_id_ = 248044;
};

}  // namespace marlin
