#include "pipeline/realsense_source.hpp"

#ifdef WITH_REALSENSE

#include <chrono>
#include <iostream>
#include <stdexcept>

RealSenseSource::RealSenseSource(const std::string& serial, int width, int height, int fps)
    : serial_(serial), width_(width), height_(height), fps_(fps)
{
    std::cout << "[RealSenseSource] configured "
              << (serial_.empty() ? "first available device" : serial_)
              << " " << width_ << "x" << height_ << " @ " << fps_ << "fps" << std::endl;
}

RealSenseSource::~RealSenseSource() {
    // Stop streaming thread and pipeline cleanly.
    stop();
}

void RealSenseSource::start(FrameCallback cb) {
    // Open pipeline once and launch capture thread.
    cb_ = cb;

    rs2::config cfg;
    if (!serial_.empty())
        cfg.enable_device(serial_);
    cfg.enable_stream(RS2_STREAM_COLOR, width_, height_, RS2_FORMAT_RGB8, fps_);

    rs2::pipeline_profile profile = pipe_.start(cfg);
    auto color_stream = profile.get_stream(RS2_STREAM_COLOR).as<rs2::video_stream_profile>();
    width_  = color_stream.width();
    height_ = color_stream.height();

    std::cout << "[RealSenseSource] started "
              << (serial_.empty() ? "first available device" : serial_)
              << " negotiated " << width_ << "x" << height_ << " @ " << fps_ << "fps" << std::endl;

    bRunning_ = true;
    thread_ = std::thread(&RealSenseSource::run, this);
}

void RealSenseSource::stop() {
    // Signal thread to exit, join, then stop pipeline.
    bRunning_ = false;
    if (thread_.joinable()) thread_.join();
    try { pipe_.stop(); } catch (...) {}
}

uint32_t RealSenseSource::width()  const { return static_cast<uint32_t>(width_);  }
uint32_t RealSenseSource::height() const { return static_cast<uint32_t>(height_); }

void RealSenseSource::run() {
    // Pull color frames from pipeline and forward via callback.
    while (bRunning_) {
        rs2::frameset frames;
        if (!pipe_.try_wait_for_frames(&frames, 100)){ 
            std::cout << "Timeout in realsense camera" << std::endl;
            continue;
        }

        rs2::video_frame color = frames.get_color_frame();
        if (!color) continue;

        // RealSense timestamps are only directly comparable to host wall-clock
        // (system_clock) when the domain is SYSTEM_TIME. HARDWARE_CLOCK/GLOBAL_TIME
        // domains use the device's free-running clock and would need a calibrated
        // offset to translate - not attempted here. Fall back to time-of-retrieval
        // in that case (loses the capture->USB-delivery portion of the latency, but
        // stays clock-consistent with the rest of the pipeline).
        uint64_t capture_time_ns;
        if (color.get_frame_timestamp_domain() == RS2_TIMESTAMP_DOMAIN_SYSTEM_TIME) {
            capture_time_ns = static_cast<uint64_t>(color.get_timestamp() * 1e6);  // ms -> ns
        } else {
            capture_time_ns = static_cast<uint64_t>(
                std::chrono::duration_cast<std::chrono::nanoseconds>(
                    std::chrono::system_clock::now().time_since_epoch()).count());
            static bool warned = false;
            if (!warned) {
                std::cout << "[RealSenseSource] WARNING: frame timestamp domain is not "
                             "SYSTEM_TIME - capture_time_ns falls back to time-of-retrieval "
                             "(missing the sensor->USB delivery portion of capture latency)."
                          << std::endl;
                warned = true;
            }
        }

        const uint8_t* data = static_cast<const uint8_t*>(color.get_data());
        cb_(data, static_cast<uint32_t>(color.get_width()),
                  static_cast<uint32_t>(color.get_height()),
                  capture_time_ns);
    }
}

#endif