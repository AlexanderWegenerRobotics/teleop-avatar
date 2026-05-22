#pragma once

#include "pipeline/camera_source.hpp"

#include <librealsense2/rs.hpp>

#include <atomic>
#include <string>
#include <thread>

class RealSenseSource : public CameraSource {
public:
    RealSenseSource(const std::string& serial, int width, int height, int fps);
    ~RealSenseSource();

    void start(FrameCallback cb) override;
    void stop()                  override;
    uint32_t width()  const      override;
    uint32_t height() const      override;

private:
    void run();

    std::string serial_;
    int         width_;
    int         height_;
    int         fps_;

    FrameCallback         cb_;
    rs2::pipeline         pipe_;
    std::thread           thread_;
    std::atomic<bool>     bRunning_{false};
};
