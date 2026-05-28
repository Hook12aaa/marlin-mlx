#include "core/config.h"
#include "core/engine.h"
#include "server/http_server.h"
#include "server/session_manager.h"

#include <fmt/format.h>
#include <cstdlib>
#include <string>

int main(int argc, char* argv[]) {
    if (argc < 2) {
        fmt::print(stderr, "Usage: marlin-serve <model-path> [--port PORT]\n");
        return 1;
    }

    std::string model_path = argv[1];
    int port = 8080;
    for (int i = 2; i < argc - 1; ++i) {
        if (std::string(argv[i]) == "--port") port = std::atoi(argv[i + 1]);
    }

    marlin::RunConfig config;
    config.model_path = model_path;
    config.port = port;

    fmt::print("Loading model from {}...\n", model_path);
    auto engine_result = marlin::Engine::create(config);
    if (!engine_result) {
        fmt::print(stderr, "Error: {}\n", engine_result.error().message);
        return 1;
    }
    fmt::print("Model loaded.\n");

    marlin::SessionManager sm(std::move(*engine_result));
    marlin::HttpConfig http_cfg{.host = "127.0.0.1", .port = port};
    marlin::HttpServer server(sm, std::move(http_cfg));
    return server.run();
}
