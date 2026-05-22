#pragma once

#include <atomic>
#include <functional>
#include <memory>
#include <string>
#include <thread>
#include <iostream>

#include "pipeline/shared_memory.hpp"

using FrameCallback = std::function<void(const uint8_t*, uint32_t, uint32_t)>;

class CameraSource {
public:
    virtual ~CameraSource() = default;
    virtual void start(FrameCallback cb) = 0;
    virtual void stop()                  = 0;
    virtual uint32_t width()  const      = 0;
    virtual uint32_t height() const      = 0;
};

class MuJoCoSource : public CameraSource {
public:
    MuJoCoSource(const std::string& shm_name, int fps)
        : shm_name_(shm_name), fps_(fps)
    {
        int retries = 20;
        while (retries-- > 0) {
            try {
                reader_ = std::make_unique<SharedMemoryReader>(shm_name_);
                if (reader_->width() > 0 && reader_->height() > 0) break;
            } catch (...) {}
            std::cerr << "[MuJoCoSource] waiting for shared memory..." << std::endl;
            std::this_thread::sleep_for(std::chrono::milliseconds(500));
            reader_.reset();
        }
        if (!reader_ || reader_->width() == 0)
            throw std::runtime_error("shared memory not available or has zero dimensions");
    }

    ~MuJoCoSource() { stop(); }

    void start(FrameCallback cb) override {
        reader_ = std::make_unique<SharedMemoryReader>(shm_name_);
        cb_     = cb;
        bRunning_ = true;
        thread_ = std::thread(&MuJoCoSource::run, this);
    }

    void stop() override {
        bRunning_ = false;
        if (thread_.joinable()) thread_.join();
    }

    uint32_t width()  const override { return reader_ ? reader_->width()  : 0; }
    uint32_t height() const override { return reader_ ? reader_->height() : 0; }

private:
    void run() {
        auto period = std::chrono::microseconds(1000000 / fps_);
        auto next   = std::chrono::steady_clock::now();

        while (bRunning_) {
            if (reader_->hasNewFrame()) {
                const uint8_t* frame = reader_->read();
                cb_(frame, reader_->width(), reader_->height());
            }
            next += period;
            std::this_thread::sleep_until(next);
        }
    }

    std::string                       shm_name_;
    int                               fps_;
    FrameCallback                     cb_;
    std::unique_ptr<SharedMemoryReader> reader_;
    std::thread                       thread_;
    std::atomic<bool>                 bRunning_{false};
};