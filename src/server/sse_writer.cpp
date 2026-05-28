#include "server/sse_writer.h"

namespace marlin::sse {

void write_data(std::string& out, std::string_view payload) {
    size_t start = 0;
    for (size_t i = 0; i < payload.size(); ++i) {
        if (payload[i] == '\n') {
            out += "data: ";
            out.append(payload.data() + start, i - start);
            out += '\n';
            start = i + 1;
        }
    }
    if (start < payload.size()) {
        out += "data: ";
        out.append(payload.data() + start, payload.size() - start);
    }
    out += "\n\n";
}

void write_done(std::string& out) {
    out += "data: [DONE]\n\n";
}

}  // namespace marlin::sse
