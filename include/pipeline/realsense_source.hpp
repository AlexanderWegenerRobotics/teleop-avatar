#pragma once

#include "pipeline/camera_source.hpp"

#include <librealsense2/rs.hpp>

#include <atomic>
#include <string>
#include <thread>

class RealSenseSource : public CameraSource {
public:
    // exposure_100us: 0 = auto, >0 = manual in 100 us units
    RealSenseSource(const std::string& serial, int width, int height, int fps,
                    int exposure_100us = 0, bool auto_exposure = true, int gain = -1);
    ~RealSenseSource();

    void start(FrameCallback cb) override;
    void stop()                  override;
    uint32_t width()  const      override;
    uint32_t height() const      override;

private:
    struct Stats {
        int      n = 0;
        double   sum_ms = 0.0, min_ms = 1e9, max_ms = 0.0;
        double   sum_exp_us = 0.0; int n_exp = 0;
        double   sum_gain = 0.0;   int n_gain = 0;
        double   sum_ts_backend = 0.0;  int n_ts_backend = 0;
        double   sum_backend_toa = 0.0; int n_backend_toa = 0;
        double   sum_toa_app = 0.0;     int n_toa_app = 0;
        double   sum_mid_frame = 0.0;   int n_mid_frame = 0;
        uint64_t first_arrival_ns = 0;
        unsigned long long last_frame_number = 0;
        int      dropped = 0;
        bool     have_sensor_time = false;
    };

    void run();
    void startPipeline();
    void startSensor();
    void configureColorSensor(rs2::device dev);
    void handleFrame(const rs2::video_frame& color);
    void resetStats();

    std::string serial_;
    int         width_;
    int         height_;
    int         fps_;
    int         exposure_100us_ = 0;
    bool        auto_exposure_  = true;
    int         gain_           = -1;
    rs2::sensor color_sensor_;
    bool        have_color_sensor_ = false;

    bool          sensor_mode_    = true;
    bool          sensor_started_ = false;
    rs2::context  ctx_;
    rs2::device   dev_;

    Stats         stats_;
    bool          domain_logged_ = false;
    bool          time_warned_   = false;

    FrameCallback         cb_;
    rs2::pipeline         pipe_;
    std::thread           thread_;
    std::atomic<bool>     bRunning_{false};
};
