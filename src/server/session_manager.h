#pragma once

#include "core/engine.h"
#include <atomic>
#include <memory>

namespace marlin {

class SessionManager {
public:
    explicit SessionManager(std::unique_ptr<Engine> engine)
        : engine_(std::move(engine)) {}

    bool try_acquire() {
        bool expected = false;
        return busy_.compare_exchange_strong(expected, true);
    }

    void release() {
        cancel_flag_.store(false);
        busy_.store(false);
    }

    void signal_cancel() {
        cancel_flag_.store(true, std::memory_order_release);
    }

    std::atomic<bool>* cancel_flag() { return &cancel_flag_; }
    Engine& engine() { return *engine_; }

private:
    std::unique_ptr<Engine> engine_;
    std::atomic<bool> busy_{false};
    std::atomic<bool> cancel_flag_{false};
};

}  // namespace marlin
