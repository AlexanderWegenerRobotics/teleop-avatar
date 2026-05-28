#pragma once

#include <atomic>
#include <cstring>
#include <functional>
#include <memory>
#include <string>
#include <thread>
#include <vector>
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

// ---------------------------------------------------------------------------
// StereoMuJoCoSource
// Reads left and right shm buffers in the SAME poll iteration so both eyes
// are always from the same simulation step.  Composites them side-by-side
// (left | right) into a single RGB buffer of width 2×w, then fires the
// FrameCallback with that combined frame.  This guarantees temporal sync and
// correlated compression artifacts, eliminating the independent-stream drift.
// ---------------------------------------------------------------------------

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

    // Reports combined width (2 × single-eye width).
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
            // Read both eyes in the same iteration — same simulation frame.
            // Accept the most recent frame from each shm, even if only one
            // has a strictly-new flag, so we never stall on one lagging.
            if (left_reader_->hasNewFrame() || right_reader_->hasNewFrame()) {
                const uint8_t* left  = left_reader_->read();
                const uint8_t* right = right_reader_->read();

                const uint32_t w        = left_reader_->width();
                const uint32_t h        = left_reader_->height();
                const size_t   row_src  = static_cast<size_t>(w) * 3;   // bytes per eye row
                const size_t   row_dst  = row_src * 2;                   // bytes per combined row

                composite_.resize(row_dst * h);

                for (uint32_t y = 0; y < h; ++y) {
                    uint8_t* dst = composite_.data() + y * row_dst;
                    std::memcpy(dst,           left  + y * row_src, row_src);
                    std::memcpy(dst + row_src, right + y * row_src, row_src);
                }

                cb_(composite_.data(), w * 2, h);
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