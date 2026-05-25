#include "pipeline/realsense_source.hpp"

#ifdef WITH_REALSENSE

#include <iostream>
#include <stdexcept>

RealSenseSource::RealSenseSource(const std::string& serial, int width, int height, int fps)
    : serial_(serial), width_(width), height_(height), fps_(fps)
{
    rs2::config cfg;
    if (!serial_.empty())
        cfg.enable_device(serial_);
    cfg.enable_stream(RS2_STREAM_COLOR, width_, height_, RS2_FORMAT_RGB8, fps_);

    rs2::pipeline_profile profile = pipe_.start(cfg);

    auto color_stream = profile.get_stream(RS2_STREAM_COLOR)
                               .as<rs2::video_stream_profile>();
    width_  = color_stream.width();
    height_ = color_stream.height();

    std::cout << "[RealSenseSource] opened "
              << (serial_.empty() ? "first available device" : serial_)
              << " " << width_ << "x" << height_ << " @ " << fps_ << "fps" << std::endl;

    pipe_.stop();
}

RealSenseSource::~RealSenseSource() {
    stop();
}

void RealSenseSource::start(FrameCallback cb) {
    cb_ = cb;

    rs2::config cfg;
    if (!serial_.empty())
        cfg.enable_device(serial_);
    cfg.enable_stream(RS2_STREAM_COLOR, width_, height_, RS2_FORMAT_RGB8, fps_);
    pipe_.start(cfg);

    bRunning_ = true;
    thread_ = std::thread(&RealSenseSource::run, this);
}

void RealSenseSource::stop() {
    bRunning_ = false;
    if (thread_.joinable()) thread_.join();
    try { pipe_.stop(); } catch (...) {}
}

uint32_t RealSenseSource::width()  const { return static_cast<uint32_t>(width_);  }
uint32_t RealSenseSource::height() const { return static_cast<uint32_t>(height_); }

void RealSenseSource::run() {
    while (bRunning_) {
        rs2::frameset frames;
        if (!pipe_.try_wait_for_frames(&frames, 100))
            continue;

        rs2::video_frame color = frames.get_color_frame();
        if (!color) continue;

        const uint8_t* data = static_cast<const uint8_t*>(color.get_data());
        cb_(data, static_cast<uint32_t>(color.get_width()),
                  static_cast<uint32_t>(color.get_height()));
    }
}

#endif