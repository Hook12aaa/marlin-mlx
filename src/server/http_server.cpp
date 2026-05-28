#include "server/http_server.h"
#include "server/openai_protocol.h"
#include "server/sse_writer.h"

#include <httplib.h>
#include <fmt/format.h>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <deque>
#include <mutex>
#include <thread>

namespace marlin {

namespace {

int64_t epoch_s() {
    return std::chrono::duration_cast<std::chrono::seconds>(
        std::chrono::system_clock::now().time_since_epoch()).count();
}

std::string gen_id() {
    static std::atomic<uint64_t> n{0};
    return fmt::format("chatcmpl-{}", n.fetch_add(1));
}

bool auth_ok(const httplib::Request& req, const std::string& key) {
    if (key.empty()) return true;
    auto h = req.get_header_value("Authorization");
    return h == "Bearer " + key;
}

void add_cors(httplib::Response& res) {
    res.set_header("Access-Control-Allow-Origin", "*");
    res.set_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS");
    res.set_header("Access-Control-Allow-Headers", "Content-Type, Authorization");
}

SamplingConfig request_to_sampling(const openai::ChatRequest& r) {
    SamplingConfig s;
    s.temperature = static_cast<float>(r.temperature);
    s.top_p = static_cast<float>(r.top_p);
    s.top_k = r.top_k;
    s.min_p = static_cast<float>(r.min_p);
    s.max_tokens = r.max_tokens;
    s.seed = r.seed;
    s.stop = r.stop;
    s.enable_thinking = r.thinking;
    return s;
}

}  // namespace

struct HttpServer::Impl {
    SessionManager& sm;
    HttpConfig cfg;
    httplib::Server svr;

    Impl(SessionManager& s, HttpConfig c) : sm(s), cfg(std::move(c)) {
        svr.Options(R"(.*)", [](const httplib::Request&, httplib::Response& res) {
            add_cors(res);
            res.status = 204;
        });

        svr.Get("/health", [](const httplib::Request&, httplib::Response& res) {
            add_cors(res);
            res.set_content(R"({"status":"ok"})", "application/json");
        });

        svr.Get("/v1/models", [this](const httplib::Request& req, httplib::Response& res) {
            add_cors(res);
            if (!auth_ok(req, cfg.api_key)) {
                res.status = 401;
                res.set_content(openai::build_error(401, "auth_error", "", "invalid api key"),
                                "application/json");
                return;
            }
            res.set_content(
                R"({"object":"list","data":[{"id":"marlin-2b","object":"model","created":0,"owned_by":"local"}]})",
                "application/json");
        });

        svr.Post("/v1/chat/completions", [this](const httplib::Request& req, httplib::Response& res) {
            handle_chat(req, res);
        });
    }

    void handle_chat(const httplib::Request& req, httplib::Response& res) {
        add_cors(res);
        if (!auth_ok(req, cfg.api_key)) {
            res.status = 401;
            res.set_content(openai::build_error(401, "auth_error", "", "invalid api key"),
                            "application/json");
            return;
        }

        auto parsed = openai::parse_chat_request(req.body);
        if (!parsed.ok()) {
            res.status = 400;
            res.set_content(openai::build_error(400, parsed.error_type,
                                                 parsed.error_param, parsed.error_message),
                            "application/json");
            return;
        }

        if (!sm.try_acquire()) {
            res.status = 429;
            res.set_header("Retry-After", "1");
            res.set_content(openai::build_error(429, "rate_limit_error", "",
                                                "server is busy with another request"),
                            "application/json");
            return;
        }

        if (parsed.req.stream) {
            handle_stream(parsed.req, res);
        } else {
            handle_non_stream(parsed.req, res);
            sm.release();
        }
    }

    void handle_stream(const openai::ChatRequest& req, httplib::Response& res) {
        const std::string id = gen_id();
        const int64_t created = epoch_s();
        const std::string model = "marlin-2b";

        struct Buf {
            std::deque<std::string> q;
            std::mutex m;
            std::condition_variable cv;
            bool done = false;
        };
        auto buf = std::make_shared<Buf>();

        std::thread worker([this, req, id, created, model, buf]() {
            auto sampling = request_to_sampling(req);

            std::string video_path;
            for (auto& msg : req.messages) {
                if (msg.has_video) video_path = msg.video_path;
            }

            auto on_token = [&](std::string_view piece) {
                if (piece.empty()) return;
                std::string chunk = openai::build_stream_chunk(id, model, created, piece, "", "");
                std::string framed;
                sse::write_data(framed, chunk);
                {
                    std::lock_guard<std::mutex> g(buf->m);
                    buf->q.push_back(std::move(framed));
                }
                buf->cv.notify_one();
            };

            std::string final_reason = "stop";
            if (!video_path.empty()) {
                auto cr = sm.engine().caption(video_path, sampling);
                if (!cr) final_reason = "error";
            } else {
                auto text = req.messages.back().content;
                auto token_ids = std::vector<int32_t>{};
                auto gr = sm.engine().generate(token_ids, sampling, on_token, sm.cancel_flag());
                if (!gr) final_reason = "error";
            }
            std::string tail = openai::build_stream_chunk(id, model, created, "", "", final_reason);
            std::string framed;
            sse::write_data(framed, tail);
            sse::write_done(framed);
            {
                std::lock_guard<std::mutex> g(buf->m);
                buf->q.push_back(std::move(framed));
                buf->done = true;
            }
            buf->cv.notify_one();
        });
        worker.detach();

        std::string initial;
        sse::write_data(initial, openai::build_stream_chunk(id, model, created, "", "", ""));

        res.set_chunked_content_provider(
            "text/event-stream",
            [buf, sm_ref = &sm, initial_data = std::move(initial)](size_t, httplib::DataSink& sink) {
                if (!initial_data.empty()) {
                    if (!sink.write(initial_data.data(), initial_data.size())) {
                        sm_ref->signal_cancel();
                        sm_ref->release();
                        return false;
                    }
                }
                while (true) {
                    std::unique_lock<std::mutex> g(buf->m);
                    buf->cv.wait(g, [&]{ return !buf->q.empty() || buf->done; });
                    while (!buf->q.empty()) {
                        auto s = std::move(buf->q.front());
                        buf->q.pop_front();
                        g.unlock();
                        if (!sink.write(s.data(), s.size())) {
                            sm_ref->signal_cancel();
                            sm_ref->release();
                            return false;
                        }
                        g.lock();
                    }
                    if (buf->done) {
                        sink.done();
                        sm_ref->release();
                        return true;
                    }
                }
            });
    }

    void handle_non_stream(const openai::ChatRequest& req, httplib::Response& res) {
        const std::string id = gen_id();
        const int64_t created = epoch_s();
        const std::string model = "marlin-2b";

        auto sampling = request_to_sampling(req);

        std::string video_path;
        for (auto& msg : req.messages) {
            if (msg.has_video) video_path = msg.video_path;
        }

        Result<GenerateResult> r = std::unexpected(Error::from("no input"));
        if (!video_path.empty()) {
            auto caption_r = sm.engine().caption(video_path, sampling);
            if (caption_r) {
                r = GenerateResult{
                    .text = caption_r->caption,
                    .thinking = "",
                    .metrics = caption_r->metrics,
                };
            } else {
                r = std::unexpected(caption_r.error());
            }
        } else {
            auto text = req.messages.back().content;
            auto token_ids = std::vector<int32_t>{};
            r = sm.engine().generate(token_ids, sampling, nullptr, sm.cancel_flag());
        }

        if (!r) {
            res.status = 500;
            res.set_content(openai::build_error(500, "server_error", "", r.error().message),
                            "application/json");
            return;
        }

        std::string body = openai::build_full_response(
            id, model, created, r->text, r->thinking, "stop",
            r->metrics.prompt_tokens, r->metrics.gen_tokens);
        res.set_content(body, "application/json");
    }
};

HttpServer::HttpServer(SessionManager& sm, HttpConfig cfg)
    : impl_(std::make_unique<Impl>(sm, std::move(cfg))) {}
HttpServer::~HttpServer() = default;

int HttpServer::run() {
    fmt::print("marlin-serve listening on http://{}:{}\n", impl_->cfg.host, impl_->cfg.port);
    if (!impl_->svr.listen(impl_->cfg.host, impl_->cfg.port)) {
        fmt::print(stderr, "bind failed\n");
        return 1;
    }
    return 0;
}

void HttpServer::stop() { impl_->svr.stop(); }

}  // namespace marlin
