#pragma once

#include "pipeline/camera_source.hpp"

#include <librealsense2/rs.hpp>

#include <atomic>
#include <string>
#include <thread>

class RealSenseSource : public CameraSource {
public:
    // exposure_100us: 0 = keep auto-exposure, >0 = manual RGB exposure in 100 us units.
    RealSenseSource(const std::string& serial, int width, int height, int fps,
                    int exposure_100us = 0, bool auto_exposure = true, int gain = -1);
    ~RealSenseSource();

    void start(FrameCallback cb) override;
    void stop()                  override;
    uint32_t width()  const      override;
    uint32_t height() const      override;

private:
    void run();
    void configureColorSensor(rs2::pipeline_profile& profile);

    std::string serial_;
    int         width_;
    int         height_;
    int         fps_;
    int         exposure_100us_ = 0;
    bool        auto_exposure_  = true;
    int         gain_           = -1;
    rs2::sensor color_sensor_;
    bool        have_color_sensor_ = false;

    FrameCallback         cb_;
    rs2::pipeline         pipe_;
    std::thread           thread_;
    std::atomic<bool>     bRunning_{false};
};
