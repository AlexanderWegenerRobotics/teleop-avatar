#pragma once
#ifdef WITH_V4L2
#include "pipeline/camera_source.hpp"

#include <atomic>
#include <string>
#include <thread>

class V4L2Source : public CameraSource {
public:
    // exposure_100us: 0 = auto, >0 = manual in 100 us units
    V4L2Source(const std::string& device, int width, int height, int fps,
               const std::string& format = "yuyv", int exposure_100us = 0);
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
    bool        mjpeg_ = false;
    int         exposure_100us_ = 0;
    void*       tj_    = nullptr;
    bool        decode_warned_ = false;

    int           fd_      = -1;
    void*         buffers_[4]{};
    size_t        buf_lengths_[4]{};
    int           n_buffers_ = 0;

    FrameCallback     cb_;
    std::thread       thread_;
    std::atomic<bool> bRunning_{false};

    std::vector<uint8_t> rgb_buf_;

    // CLOCK_REALTIME - CLOCK_MONOTONIC, sampled once, to map buffer timestamps to system_clock
    int64_t clock_offset_ns_ = 0;
};

#endif // WITH_V4L2