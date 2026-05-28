#include "tokenizer/tokenizer.h"

#include <filesystem>
#include <fstream>
#include <tokenizers_cpp.h>

namespace marlin {

namespace fs = std::filesystem;

struct Tokenizer::Impl {
    std::unique_ptr<tokenizers::Tokenizer> inner;
};

Tokenizer::Tokenizer() = default;
Tokenizer::~Tokenizer() = default;
Tokenizer::Tokenizer(Tokenizer&&) noexcept = default;
Tokenizer& Tokenizer::operator=(Tokenizer&&) noexcept = default;

Result<std::unique_ptr<Tokenizer>> Tokenizer::load(const std::string& model_path) {
    auto tok_path = fs::path(model_path) / "tokenizer.json";
    if (!fs::exists(tok_path)) {
        return std::unexpected(Error::from("tokenizer.json not found in " + model_path));
    }

    std::ifstream f(tok_path);
    std::string blob((std::istreambuf_iterator<char>(f)),
                     std::istreambuf_iterator<char>());

    auto tok = std::make_unique<Tokenizer>();
    tok->impl_ = std::make_unique<Tokenizer::Impl>();
    tok->impl_->inner = tokenizers::Tokenizer::FromBlobJSON(blob);

    return tok;
}

std::vector<int32_t> Tokenizer::encode(const std::string& text) const {
    auto ids = impl_->inner->Encode(text);
    return std::vector<int32_t>(ids.begin(), ids.end());
}

std::string Tokenizer::decode(const std::vector<int32_t>& ids) const {
    std::vector<int> int_ids(ids.begin(), ids.end());
    return impl_->inner->Decode(int_ids);
}

std::string Tokenizer::id_to_piece(int32_t id) const {
    return impl_->inner->Decode({static_cast<int>(id)});
}

}  // namespace marlin
