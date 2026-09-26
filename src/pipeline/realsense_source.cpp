#include "pipeline/realsense_source.hpp"

#ifdef WITH_REALSENSE

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <iostream>
#include <stdexcept>

static uint64_t nowNs() {
    return static_cast<uint64_t>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::system_clock::now().time_since_epoch()).count());
}

static const char* domainName(rs2_timestamp_domain d) {
    switch (d) {
        case RS2_TIMESTAMP_DOMAIN_HARDWARE_CLOCK: return "HARDWARE_CLOCK";
        case RS2_TIMESTAMP_DOMAIN_SYSTEM_TIME:    return "SYSTEM_TIME";
        case RS2_TIMESTAMP_DOMAIN_GLOBAL_TIME:    return "GLOBAL_TIME";
        default:                                  return "UNKNOWN";
    }
}

RealSenseSource::RealSenseSource(const std::string& serial, int width, int height, int fps,
                                 int exposure_100us, bool auto_exposure, int gain)
    : serial_(serial), width_(width), height_(height), fps_(fps), exposure_100us_(exposure_100us),
      auto_exposure_(auto_exposure), gain_(gain)
{
    std::cout << "[RealSenseSource] configured "
              << (serial_.empty() ? "first available device" : serial_)
              << " " << width_ << "x" << height_ << " @ " << fps_ << "fps"
              << " auto_exposure=" << (auto_exposure_ ? "on" : "off");
    if (!auto_exposure_)
        std::cout << " exposure=" << (exposure_100us_ > 0 ? std::to_string(exposure_100us_) : std::string("unchanged"))
                  << " gain=" << (gain_ >= 0 ? std::to_string(gain_) : std::string("unchanged"));
    std::cout << std::endl;
}

static float readOpt(const rs2::sensor& s, rs2_option o) {
    try { return s.supports(o) ? s.get_option(o) : -1.f; } catch (...) { return -1.f; }
}

RealSenseSource::~RealSenseSource() {
    // Stop streaming thread and pipeline cleanly.
    stop();
}

// Latency-relevant colour sensor settings. Every option is checked with supports()
// and wrapped in try/catch: D435i and D455 expose slightly different option sets.
void RealSenseSource::configureColorSensor(rs2::pipeline_profile& profile) {
    try {
        rs2::device dev = profile.get_device();
        std::cout << "[RealSenseSource] device: "
                  << (dev.supports(RS2_CAMERA_INFO_NAME) ? dev.get_info(RS2_CAMERA_INFO_NAME) : "?")
                  << "  fw " << (dev.supports(RS2_CAMERA_INFO_FIRMWARE_VERSION) ? dev.get_info(RS2_CAMERA_INFO_FIRMWARE_VERSION) : "?")
                  << "  usb " << (dev.supports(RS2_CAMERA_INFO_USB_TYPE_DESCRIPTOR) ? dev.get_info(RS2_CAMERA_INFO_USB_TYPE_DESCRIPTOR) : "?")
                  << std::endl;

        for (rs2::sensor& s : dev.query_sensors()) {
            if (!s.is<rs2::color_sensor>()) continue;

            // Auto-exposure priority ON lets the camera drop below the requested fps in
            // dim light to lengthen exposure. Always off: fps must stay fixed.
            if (s.supports(RS2_OPTION_AUTO_EXPOSURE_PRIORITY)) {
                s.set_option(RS2_OPTION_AUTO_EXPOSURE_PRIORITY, 0.f);
                std::cout << "[RealSenseSource] auto_exposure_priority=0" << std::endl;
            }

            if (!auto_exposure_) {
                if (s.supports(RS2_OPTION_ENABLE_AUTO_EXPOSURE))
                    s.set_option(RS2_OPTION_ENABLE_AUTO_EXPOSURE, 0.f);
                if (exposure_100us_ > 0 && s.supports(RS2_OPTION_EXPOSURE)) {
                    rs2::option_range r = s.get_option_range(RS2_OPTION_EXPOSURE);
                    s.set_option(RS2_OPTION_EXPOSURE, std::clamp(static_cast<float>(exposure_100us_), r.min, r.max));
                }
                if (gain_ >= 0 && s.supports(RS2_OPTION_GAIN)) {
                    rs2::option_range r = s.get_option_range(RS2_OPTION_GAIN);
                    s.set_option(RS2_OPTION_GAIN, std::clamp(static_cast<float>(gain_), r.min, r.max));
                }
            } else if (s.supports(RS2_OPTION_ENABLE_AUTO_EXPOSURE)) {
                s.set_option(RS2_OPTION_ENABLE_AUTO_EXPOSURE, 1.f);
            }

            std::cout << "[RealSenseSource] applied: auto_exposure=" << readOpt(s, RS2_OPTION_ENABLE_AUTO_EXPOSURE)
                      << " exposure=" << readOpt(s, RS2_OPTION_EXPOSURE)
                      << " gain=" << readOpt(s, RS2_OPTION_GAIN)
                      << " auto_exposure_priority=" << readOpt(s, RS2_OPTION_AUTO_EXPOSURE_PRIORITY);
            if (s.supports(RS2_OPTION_EXPOSURE)) {
                rs2::option_range re = s.get_option_range(RS2_OPTION_EXPOSURE);
                std::cout << " (exposure range " << re.min << "-" << re.max;
                if (s.supports(RS2_OPTION_GAIN)) {
                    rs2::option_range rg = s.get_option_range(RS2_OPTION_GAIN);
                    std::cout << ", gain range " << rg.min << "-" << rg.max;
                }
                std::cout << ")";
            }
            std::cout << std::endl;

            color_sensor_      = s;
            have_color_sensor_ = true;
            break;
        }
    } catch (const rs2::error& e) {
        std::cout << "[RealSenseSource] WARNING: colour sensor configuration failed: "
                  << e.what() << std::endl;
    }
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

    configureColorSensor(profile);

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

    // Periodic camera-latency report: sensor timestamp -> frame in our hands.
    // This is the part of the video path that the embedded pixel timestamp
    // (set at encode push) cannot see.
    constexpr int kReportEvery = 150;              // ~5 s at 30 fps
    int      n = 0;
    double   sum_ms = 0.0, min_ms = 1e9, max_ms = 0.0;
    double   sum_exp_us = 0.0; int n_exp = 0;
    double   sum_gain = 0.0;   int n_gain = 0;
    uint64_t first_arrival_ns = 0;
    bool     domain_logged = false;

    while (bRunning_) {
        rs2::frameset frames;
        if (!pipe_.try_wait_for_frames(&frames, 100)){
            std::cout << "Timeout in realsense camera" << std::endl;
            continue;
        }

        rs2::video_frame color = frames.get_color_frame();
        if (!color) continue;

        const uint64_t arrival_ns = nowNs();
        const rs2_timestamp_domain domain = color.get_frame_timestamp_domain();
        if (!domain_logged) {
            std::cout << "[RealSenseSource] frame timestamp domain: " << domainName(domain) << std::endl;
            domain_logged = true;
        }

        // SYSTEM_TIME and GLOBAL_TIME are both host wall-clock (system_clock) in ms:
        // GLOBAL_TIME is the device's hardware timestamp translated into host time by
        // librealsense. Either one therefore includes exposure/readout/USB delivery.
        // HARDWARE_CLOCK is the device's free-running clock -> fall back to arrival.
        // Sanity bound: if the translated time is >1 s away from now, don't trust it.
        uint64_t capture_time_ns = arrival_ns;
        bool     have_sensor_time = false;
        if (domain == RS2_TIMESTAMP_DOMAIN_SYSTEM_TIME || domain == RS2_TIMESTAMP_DOMAIN_GLOBAL_TIME) {
            const uint64_t ts_ns = static_cast<uint64_t>(color.get_timestamp() * 1e6);   // ms -> ns
            const int64_t  diff  = static_cast<int64_t>(arrival_ns) - static_cast<int64_t>(ts_ns);
            if (diff > -1000000000LL && diff < 1000000000LL) {
                capture_time_ns  = ts_ns;
                have_sensor_time = true;
            }
        }
        if (!have_sensor_time) {
            static bool warned = false;
            if (!warned) {
                std::cout << "[RealSenseSource] WARNING: no usable host-clock frame timestamp "
                             "(domain " << domainName(domain) << ") - capture_time_ns falls back to "
                             "time-of-retrieval (misses sensor->USB delivery)." << std::endl;
                warned = true;
            }
        }

        if (have_sensor_time) {
            const double lat_ms = (static_cast<int64_t>(arrival_ns) - static_cast<int64_t>(capture_time_ns)) / 1e6;
            sum_ms += lat_ms; min_ms = std::min(min_ms, lat_ms); max_ms = std::max(max_ms, lat_ms);
        }
        if (color.supports_frame_metadata(RS2_FRAME_METADATA_ACTUAL_EXPOSURE)) {
            sum_exp_us += static_cast<double>(color.get_frame_metadata(RS2_FRAME_METADATA_ACTUAL_EXPOSURE));
            ++n_exp;
        }
        if (color.supports_frame_metadata(RS2_FRAME_METADATA_GAIN_LEVEL)) {
            sum_gain += static_cast<double>(color.get_frame_metadata(RS2_FRAME_METADATA_GAIN_LEVEL));
            ++n_gain;
        }
        if (n == 0) first_arrival_ns = arrival_ns;
        if (++n >= kReportEvery) {
            const double span_s = (arrival_ns - first_arrival_ns) / 1e9;
            std::cout << "[RealSenseSource] cam latency (sensor ts -> app) ";
            if (have_sensor_time)
                std::cout << "mean " << sum_ms / n << " ms  min " << min_ms << "  max " << max_ms;
            else
                std::cout << "n/a";
            std::cout << " | fps " << (span_s > 0 ? (n - 1) / span_s : 0.0);
            if (n_exp > 0) std::cout << " | exposure " << (sum_exp_us / n_exp) / 1000.0 << " ms";
            if (n_gain > 0) std::cout << " | gain " << sum_gain / n_gain;
            if (have_color_sensor_)
                std::cout << " | option exposure " << readOpt(color_sensor_, RS2_OPTION_EXPOSURE)
                          << " gain " << readOpt(color_sensor_, RS2_OPTION_GAIN)
                          << " ae " << readOpt(color_sensor_, RS2_OPTION_ENABLE_AUTO_EXPOSURE);
            std::cout << std::endl;
            n = 0; sum_ms = 0.0; min_ms = 1e9; max_ms = 0.0; sum_exp_us = 0.0; n_exp = 0;
            sum_gain = 0.0; n_gain = 0;
        }

        const uint8_t* data = static_cast<const uint8_t*>(color.get_data());
        cb_(data, static_cast<uint32_t>(color.get_width()),
                  static_cast<uint32_t>(color.get_height()),
                  capture_time_ns);
    }
}

#endif
