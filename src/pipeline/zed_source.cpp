#include "pipeline/zed_source.hpp"

#include <algorithm>
#include <chrono>
#include <cstring>
#include <iostream>
#include <stdexcept>

static sl::RESOLUTION parseZedResolution(const std::string& r) {
    if (r == "HD2K")   return sl::RESOLUTION::HD2K;
    if (r == "HD1080") return sl::RESOLUTION::HD1080;
    if (r == "VGA")    return sl::RESOLUTION::VGA;
    return sl::RESOLUTION::HD720;
}

static uint64_t nowNs() {
    return static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::system_clock::now().time_since_epoch()).count());
}

ZedSource::ZedSource(int serial, const std::string& resolution, int fps,
                     bool auto_exposure, int exposure_100us, int gain)
    : fps_(fps)
{
    sl::InitParameters init;
    init.camera_resolution = parseZedResolution(resolution);
    init.camera_fps        = fps;
    init.depth_mode        = sl::DEPTH_MODE::NONE;
    init.sdk_verbose       = 1;
    if (serial > 0) init.input.setFromSerialNumber(static_cast<unsigned int>(serial));

    const sl::ERROR_CODE err = zed_.open(init);
    if (err != sl::ERROR_CODE::SUCCESS)
        throw std::runtime_error(std::string("[ZedSource] open failed: ") + sl::toString(err).c_str());

    const sl::CameraInformation info = zed_.getCameraInformation();
    const sl::Resolution res = info.camera_configuration.resolution;
    width_  = static_cast<uint32_t>(res.width) * 2;
    height_ = static_cast<uint32_t>(res.height);

    if (auto_exposure || exposure_100us <= 0) {
        zed_.setCameraSettings(sl::VIDEO_SETTINGS::AEC_AGC, 1);
    } else {
        const double frame_us = 1e6 / fps;
        const int pct = std::clamp(static_cast<int>(exposure_100us * 100.0 / frame_us * 100.0 + 0.5), 1, 100);
        zed_.setCameraSettings(sl::VIDEO_SETTINGS::EXPOSURE, pct);
        if (gain >= 0) zed_.setCameraSettings(sl::VIDEO_SETTINGS::GAIN, std::clamp(gain, 0, 100));
    }

    std::cout << "[ZedSource] " << sl::toString(info.camera_model).c_str()
              << " SN " << info.serial_number
              << " " << res.width << "x" << res.height
              << " @ " << info.camera_configuration.fps << " fps -> SBS BGRA "
              << width_ << "x" << height_ << std::endl;
}

ZedSource::~ZedSource() {
    stop();
    zed_.close();
}

void ZedSource::start(FrameCallback cb) {
    cb_       = cb;
    bRunning_ = true;
    thread_   = std::thread(&ZedSource::run, this);
}

void ZedSource::stop() {
    bRunning_ = false;
    if (thread_.joinable()) thread_.join();
}

void ZedSource::run() {
    sl::RuntimeParameters rt;
    rt.enable_depth = false;

    const size_t row = static_cast<size_t>(width_) * 4;
    packed_.resize(row * height_);

    const int report_every = std::max(fps_, 1) * 5;
    int       n = 0;
    double    sum_ms = 0.0, max_ms = 0.0;
    auto      window_start = std::chrono::steady_clock::now();

    while (bRunning_) {
        const sl::ERROR_CODE err = zed_.grab(rt);
        if (err != sl::ERROR_CODE::SUCCESS) {
            std::cerr << "[ZedSource] grab: " << sl::toString(err).c_str() << std::endl;
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
            continue;
        }
        if (zed_.retrieveImage(sbs_, sl::VIEW::SIDE_BY_SIDE, sl::MEM::CPU) != sl::ERROR_CODE::SUCCESS)
            continue;
        if (sbs_.getWidth() != width_ || sbs_.getHeight() != height_ || sbs_.getChannels() != 4) {
            std::cerr << "[ZedSource] unexpected image " << sbs_.getWidth() << "x" << sbs_.getHeight()
                      << "x" << sbs_.getChannels() << std::endl;
            continue;
        }

        const uint64_t capture_ns = zed_.getTimestamp(sl::TIME_REFERENCE::IMAGE).getNanoseconds();
        const uint8_t* src  = sbs_.getPtr<sl::uchar1>(sl::MEM::CPU);
        const size_t   step = sbs_.getStepBytes(sl::MEM::CPU);
        const uint8_t* frame = src;
        if (step != row) {
            for (uint32_t y = 0; y < height_; ++y)
                std::memcpy(packed_.data() + y * row, src + y * step, row);
            frame = packed_.data();
        }

        cb_(frame, width_, height_, capture_ns);

        const double ms = (static_cast<double>(nowNs()) - static_cast<double>(capture_ns)) * 1e-6;
        sum_ms += ms;
        max_ms  = std::max(max_ms, ms);
        if (++n >= report_every) {
            const double secs = std::chrono::duration<double>(std::chrono::steady_clock::now() - window_start).count();
            std::cout << "[ZedSource] " << n / secs << " fps | image ts -> pushed mean "
                      << sum_ms / n << " ms max " << max_ms << " ms" << std::endl;
            n = 0; sum_ms = 0.0; max_ms = 0.0;
            window_start = std::chrono::steady_clock::now();
        }
    }
}
