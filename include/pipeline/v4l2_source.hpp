#pragma once

#include "pipeline/camera_source.hpp"

#include <atomic>
#include <string>
#include <thread>

class V4L2Source : public CameraSource {
public:
    V4L2Source(const std::string& device, int width, int height, int fps);
    ~V4L2Source();

    void     start(FrameCallback cb) override;
    void     stop()                  override;
    uint32_t width()  const          override;
    uint32_t height() const          override;

private:
    void run();
    void initDevice();
    void uninitDevice();

    std::string device_;
    int         width_;
    int         height_;
    int         fps_;

    int           fd_      = -1;
    void*         buffers_[4]{};
    size_t        buf_lengths_[4]{};
    int           n_buffers_ = 0;

    FrameCallback     cb_;
    std::thread       thread_;
    std::atomic<bool> bRunning_{false};

    std::vector<uint8_t> rgb_buf_;
};