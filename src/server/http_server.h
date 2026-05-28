#pragma once

#include "server/session_manager.h"
#include <memory>
#include <string>

namespace marlin {

struct HttpConfig {
    std::string host = "127.0.0.1";
    int port = 8080;
    std::string api_key;
};

class HttpServer {
public:
    HttpServer(SessionManager& sm, HttpConfig cfg);
    ~HttpServer();
    int run();
    void stop();
private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace marlin
