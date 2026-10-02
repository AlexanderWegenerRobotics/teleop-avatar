#pragma once

#include <algorithm>
#include <atomic>
#include <cstring>
#include <functional>
#include <memory>
#include <string>
#include <thread>
#include <vector>
#include <iostream>

#include "pipeline/shared_memory.hpp"

// capture_time_ns: ns since Unix epoch (system_clock), every source converts to this
using FrameCallback = std::function<void(const uint8_t*, uint32_t, uint32_t, uint64_t)>;

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
                uint64_t capture_time_ns = 0;
                const uint8_t* frame = reader_->read(&capture_time_ns);
                cb_(frame, reader_->width(), reader_->height(), capture_time_ns);
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

// Reads both eye shm buffers in one poll and composites them left|right (2x width).

class StereoMuJoCoSource : public CameraSource {
public:
    StereoMuJoCoSource(const std::string& left_shm,
                       const std::string& right_shm,
                       int fps)
        : left_shm_(left_shm), right_shm_(right_shm), fps_(fps)
    {
        auto tryOpen = [](const std::string& name)
            -> std::unique_ptr<SharedMemoryReader>
        {
            int retries = 20;
            while (retries-- > 0) {
                try {
                    auto r = std::make_unique<SharedMemoryReader>(name);
                    if (r->width() > 0 && r->height() > 0) return r;
                } catch (...) {}
                std::cerr << "[StereoMuJoCoSource] waiting for shm: " << name << std::endl;
                std::this_thread::sleep_for(std::chrono::milliseconds(500));
            }
            throw std::runtime_error("shm not available: " + name);
        };

        left_reader_  = tryOpen(left_shm_);
        right_reader_ = tryOpen(right_shm_);
    }

    ~StereoMuJoCoSource() { stop(); }

    void start(FrameCallback cb) override {
        left_reader_  = std::make_unique<SharedMemoryReader>(left_shm_);
        right_reader_ = std::make_unique<SharedMemoryReader>(right_shm_);
        cb_       = cb;
        bRunning_ = true;
        thread_   = std::thread(&StereoMuJoCoSource::run, this);
    }

    void stop() override {
        bRunning_ = false;
        if (thread_.joinable()) thread_.join();
    }

    uint32_t width()  const override {
        return left_reader_ ? left_reader_->width() * 2 : 0;
    }
    uint32_t height() const override {
        return left_reader_ ? left_reader_->height() : 0;
    }

private:
    void run() {
        auto period = std::chrono::microseconds(1000000 / fps_);
        auto next   = std::chrono::steady_clock::now();

        while (bRunning_) {
            // take latest of both eyes even if only one is new, so a lagging eye can't stall
            if (left_reader_->hasNewFrame() || right_reader_->hasNewFrame()) {
                uint64_t left_ts = 0, right_ts = 0;
                const uint8_t* left  = left_reader_->read(&left_ts);
                const uint8_t* right = right_reader_->read(&right_ts);
                uint64_t capture_time_ns = std::max(left_ts, right_ts);

                const uint32_t w        = left_reader_->width();
                const uint32_t h        = left_reader_->height();
                const size_t   row_src  = static_cast<size_t>(w) * 3;
                const size_t   row_dst  = row_src * 2;

                composite_.resize(row_dst * h);

                for (uint32_t y = 0; y < h; ++y) {
                    uint8_t* dst = composite_.data() + y * row_dst;
                    std::memcpy(dst,           left  + y * row_src, row_src);
                    std::memcpy(dst + row_src, right + y * row_src, row_src);
                }

                cb_(composite_.data(), w * 2, h, capture_time_ns);
            }

            next += period;
            std::this_thread::sleep_until(next);
        }
    }

    std::string  left_shm_;
    std::string  right_shm_;
    int          fps_;
    FrameCallback cb_;
    std::unique_ptr<SharedMemoryReader> left_reader_;
    std::unique_ptr<SharedMemoryReader> right_reader_;
    std::thread       thread_;
    std::atomic<bool> bRunning_{false};
    std::vector<uint8_t> composite_;
};