#pragma once

#include <string>
#include <string_view>

namespace marlin::sse {

void write_data(std::string& out, std::string_view payload);
void write_done(std::string& out);

}  // namespace marlin::sse
