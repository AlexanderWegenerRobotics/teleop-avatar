#include "pipeline/realsense_source.hpp"

#ifdef WITH_REALSENSE

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdlib>
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

static bool plausibleMs(double d) {
    return d > -1000.0 && d < 1000.0;
}

RealSenseSource::RealSenseSource(const std::string& serial, int width, int height, int fps,
                                 int exposure_100us, bool auto_exposure, int gain)
    : serial_(serial), width_(width), height_(height), fps_(fps), exposure_100us_(exposure_100us),
      auto_exposure_(auto_exposure), gain_(gain)
{
    sensor_mode_ = std::getenv("AVATAR_RS_PIPELINE") == nullptr;

    std::cout << "[RealSenseSource] configured "
              << (serial_.empty() ? "first available device" : serial_)
              << " " << width_ << "x" << height_ << " @ " << fps_ << "fps"
              << " auto_exposure=" << (auto_exposure_ ? "on" : "off");
    if (!auto_exposure_)
        std::cout << " exposure=" << (exposure_100us_ > 0 ? std::to_string(exposure_100us_) : std::string("unchanged"))
                  << " gain=" << (gain_ >= 0 ? std::to_string(gain_) : std::string("unchanged"));
    std::cout << " capture=" << (sensor_mode_ ? "sensor-callback" : "pipeline");
    std::cout << std::endl;
}

static bool isColorSensor(const rs2::sensor& s) {
    if (s.is<rs2::color_sensor>()) return true;
    for (const rs2::stream_profile& p : s.get_stream_profiles())
        if (p.stream_type() == RS2_STREAM_COLOR) return true;
    return false;
}

static float readOpt(const rs2::sensor& s, rs2_option o) {
    try { return s.supports(o) ? s.get_option(o) : -1.f; } catch (...) { return -1.f; }
}

RealSenseSource::~RealSenseSource() {
    stop();
}

// Latency-related color sensor options. D435i and D455 support different sets, so check each.
void RealSenseSource::configureColorSensor(rs2::device dev) {
    try {
        std::cout << "[RealSenseSource] device: "
                  << (dev.supports(RS2_CAMERA_INFO_NAME) ? dev.get_info(RS2_CAMERA_INFO_NAME) : "?")
                  << "  fw " << (dev.supports(RS2_CAMERA_INFO_FIRMWARE_VERSION) ? dev.get_info(RS2_CAMERA_INFO_FIRMWARE_VERSION) : "?")
                  << "  usb " << (dev.supports(RS2_CAMERA_INFO_USB_TYPE_DESCRIPTOR) ? dev.get_info(RS2_CAMERA_INFO_USB_TYPE_DESCRIPTOR) : "?")
                  << std::endl;

        for (rs2::sensor& s : dev.query_sensors()) {
            if (!isColorSensor(s)) continue;

            // off so the camera never drops fps to lengthen exposure
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
                      << " auto_exposure_priority=" << readOpt(s, RS2_OPTION_AUTO_EXPOSURE_PRIORITY)
                      << " frames_queue_size=" << readOpt(s, RS2_OPTION_FRAMES_QUEUE_SIZE);
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
    cb_ = cb;
    resetStats();
    bRunning_ = true;

    if (sensor_mode_) {
        startSensor();
    } else {
        startPipeline();
        thread_ = std::thread(&RealSenseSource::run, this);
    }

    std::cout << "[RealSenseSource] started "
              << (serial_.empty() ? "first available device" : serial_)
              << " negotiated " << width_ << "x" << height_ << " @ " << fps_ << "fps"
              << " via " << (sensor_mode_ ? "sensor callback" : "rs2::pipeline") << std::endl;
}

void RealSenseSource::startPipeline() {
    rs2::config cfg;
    if (!serial_.empty())
        cfg.enable_device(serial_);
    cfg.enable_stream(RS2_STREAM_COLOR, width_, height_, RS2_FORMAT_RGB8, fps_);

    rs2::pipeline_profile profile = pipe_.start(cfg);
    auto color_stream = profile.get_stream(RS2_STREAM_COLOR).as<rs2::video_stream_profile>();
    width_  = color_stream.width();
    height_ = color_stream.height();

    configureColorSensor(profile.get_device());
}

void RealSenseSource::startSensor() {
    rs2::sensor         color;
    rs2::stream_profile chosen;
    bool                found = false;

    rs2::device_list list = ctx_.query_devices();
    for (uint32_t i = 0; i < list.size() && !found; ++i) {
        rs2::device d = list[i];
        const std::string name   = d.supports(RS2_CAMERA_INFO_NAME) ? d.get_info(RS2_CAMERA_INFO_NAME) : "?";
        const std::string serial = d.supports(RS2_CAMERA_INFO_SERIAL_NUMBER) ? d.get_info(RS2_CAMERA_INFO_SERIAL_NUMBER) : "";
        std::cout << "[RealSenseSource] found device " << i << ": " << name
                  << (serial.empty() ? "" : " serial " + serial) << std::endl;
        if (!serial_.empty() && serial != serial_) continue;

        for (rs2::sensor& s : d.query_sensors()) {
            for (const rs2::stream_profile& p : s.get_stream_profiles()) {
                if (p.stream_type() != RS2_STREAM_COLOR || p.format() != RS2_FORMAT_RGB8 || p.fps() != fps_) continue;
                auto v = p.as<rs2::video_stream_profile>();
                if (v && v.width() == width_ && v.height() == height_) {
                    dev_   = d;
                    color  = s;
                    chosen = p;
                    found  = true;
                    break;
                }
            }
            if (found) break;
        }
    }
    if (!found)
        throw std::runtime_error("[RealSenseSource] no device with an RGB8 colour stream " + std::to_string(width_) + "x" +
                                 std::to_string(height_) + "@" + std::to_string(fps_));

    std::cout << "[RealSenseSource] using sensor "
              << (color.supports(RS2_CAMERA_INFO_NAME) ? color.get_info(RS2_CAMERA_INFO_NAME) : "?") << std::endl;

    try {
        if (color.supports(RS2_OPTION_FRAMES_QUEUE_SIZE))
            color.set_option(RS2_OPTION_FRAMES_QUEUE_SIZE, 1.f);
    } catch (const rs2::error& e) {
        std::cout << "[RealSenseSource] WARNING: frames_queue_size not set: " << e.what() << std::endl;
    }

    color_sensor_      = color;
    have_color_sensor_ = true;

    color.open(chosen);
    color.start([this](rs2::frame f) {
        if (!bRunning_) return;
        rs2::video_frame v = f.as<rs2::video_frame>();
        if (v) handleFrame(v);
    });
    sensor_started_ = true;

    configureColorSensor(dev_);
}

void RealSenseSource::stop() {
    bRunning_ = false;
    if (thread_.joinable()) thread_.join();
    if (sensor_started_) {
        try { color_sensor_.stop(); }  catch (...) {}
        try { color_sensor_.close(); } catch (...) {}
        sensor_started_ = false;
    } else {
        try { pipe_.stop(); } catch (...) {}
    }
}

uint32_t RealSenseSource::width()  const { return static_cast<uint32_t>(width_);  }
uint32_t RealSenseSource::height() const { return static_cast<uint32_t>(height_); }

void RealSenseSource::resetStats() {
    const unsigned long long last = stats_.last_frame_number;
    stats_ = Stats{};
    stats_.last_frame_number = last;
}

void RealSenseSource::run() {
    while (bRunning_) {
        rs2::frameset frames;
        if (!pipe_.try_wait_for_frames(&frames, 100)){
            std::cout << "Timeout in realsense camera" << std::endl;
            continue;
        }

        rs2::video_frame color = frames.get_color_frame();
        if (!color) continue;
        handleFrame(color);
    }
}

void RealSenseSource::handleFrame(const rs2::video_frame& color) {
    const int kReportEvery = std::max(30, fps_ * 5);
    Stats& st = stats_;

    const uint64_t arrival_ns = nowNs();
    const double   arrival_ms = arrival_ns / 1e6;
    const rs2_timestamp_domain domain = color.get_frame_timestamp_domain();
    if (!domain_logged_) {
        std::cout << "[RealSenseSource] frame timestamp domain: " << domainName(domain)
                  << "  metadata: backend_ts=" << color.supports_frame_metadata(RS2_FRAME_METADATA_BACKEND_TIMESTAMP)
                  << " time_of_arrival=" << color.supports_frame_metadata(RS2_FRAME_METADATA_TIME_OF_ARRIVAL)
                  << " frame_ts=" << color.supports_frame_metadata(RS2_FRAME_METADATA_FRAME_TIMESTAMP)
                  << " sensor_ts=" << color.supports_frame_metadata(RS2_FRAME_METADATA_SENSOR_TIMESTAMP)
                  << std::endl;
        domain_logged_ = true;
    }

    // SYSTEM/GLOBAL_TIME are host wall clock in ms, HARDWARE_CLOCK is not -> use arrival.
    // ignore sensor time if it is more than 1 s off from now
    uint64_t capture_time_ns  = arrival_ns;
    bool     have_sensor_time = false;
    const double ts_ms = color.get_timestamp();
    if (domain == RS2_TIMESTAMP_DOMAIN_SYSTEM_TIME || domain == RS2_TIMESTAMP_DOMAIN_GLOBAL_TIME) {
        const uint64_t ts_ns = static_cast<uint64_t>(ts_ms * 1e6);
        const int64_t  diff  = static_cast<int64_t>(arrival_ns) - static_cast<int64_t>(ts_ns);
        if (diff > -1000000000LL && diff < 1000000000LL) {
            capture_time_ns  = ts_ns;
            have_sensor_time = true;
        }
    }
    if (!have_sensor_time && !time_warned_) {
        std::cout << "[RealSenseSource] WARNING: no usable host-clock frame timestamp "
                     "(domain " << domainName(domain) << ") - capture_time_ns falls back to "
                     "time-of-retrieval (misses sensor->USB delivery)." << std::endl;
        time_warned_ = true;
    }
    st.have_sensor_time = have_sensor_time;

    if (have_sensor_time) {
        const double lat_ms = (static_cast<int64_t>(arrival_ns) - static_cast<int64_t>(capture_time_ns)) / 1e6;
        st.sum_ms += lat_ms; st.min_ms = std::min(st.min_ms, lat_ms); st.max_ms = std::max(st.max_ms, lat_ms);
    }

    const bool has_backend = color.supports_frame_metadata(RS2_FRAME_METADATA_BACKEND_TIMESTAMP);
    const bool has_toa     = color.supports_frame_metadata(RS2_FRAME_METADATA_TIME_OF_ARRIVAL);
    const double backend_ms = has_backend ? static_cast<double>(color.get_frame_metadata(RS2_FRAME_METADATA_BACKEND_TIMESTAMP)) : 0.0;
    const double toa_ms     = has_toa     ? static_cast<double>(color.get_frame_metadata(RS2_FRAME_METADATA_TIME_OF_ARRIVAL))   : 0.0;
    if (has_backend && have_sensor_time && plausibleMs(backend_ms - ts_ms)) {
        st.sum_ts_backend += backend_ms - ts_ms; ++st.n_ts_backend;
    }
    if (has_backend && has_toa && plausibleMs(toa_ms - backend_ms)) {
        st.sum_backend_toa += toa_ms - backend_ms; ++st.n_backend_toa;
    }
    if (has_toa && plausibleMs(arrival_ms - toa_ms)) {
        st.sum_toa_app += arrival_ms - toa_ms; ++st.n_toa_app;
    }
    if (color.supports_frame_metadata(RS2_FRAME_METADATA_FRAME_TIMESTAMP) &&
        color.supports_frame_metadata(RS2_FRAME_METADATA_SENSOR_TIMESTAMP)) {
        const double d_ms = (static_cast<double>(color.get_frame_metadata(RS2_FRAME_METADATA_FRAME_TIMESTAMP)) -
                             static_cast<double>(color.get_frame_metadata(RS2_FRAME_METADATA_SENSOR_TIMESTAMP))) / 1000.0;
        if (plausibleMs(d_ms)) { st.sum_mid_frame += d_ms; ++st.n_mid_frame; }
    }

    if (color.supports_frame_metadata(RS2_FRAME_METADATA_ACTUAL_EXPOSURE)) {
        st.sum_exp_us += static_cast<double>(color.get_frame_metadata(RS2_FRAME_METADATA_ACTUAL_EXPOSURE));
        ++st.n_exp;
    }
    if (color.supports_frame_metadata(RS2_FRAME_METADATA_GAIN_LEVEL)) {
        st.sum_gain += static_cast<double>(color.get_frame_metadata(RS2_FRAME_METADATA_GAIN_LEVEL));
        ++st.n_gain;
    }

    const unsigned long long fn = color.get_frame_number();
    if (st.last_frame_number != 0 && fn > st.last_frame_number + 1)
        st.dropped += static_cast<int>(fn - st.last_frame_number - 1);
    st.last_frame_number = fn;

    if (st.n == 0) st.first_arrival_ns = arrival_ns;
    if (++st.n >= kReportEvery) {
        const double span_s = (arrival_ns - st.first_arrival_ns) / 1e9;
        std::cout << "[RealSenseSource] cam latency (sensor ts -> app) ";
        if (st.have_sensor_time)
            std::cout << "mean " << st.sum_ms / st.n << " ms  min " << st.min_ms << "  max " << st.max_ms;
        else
            std::cout << "n/a";
        std::cout << " | fps " << (span_s > 0 ? (st.n - 1) / span_s : 0.0)
                  << " | dropped " << st.dropped;
        if (st.n_exp > 0) std::cout << " | exposure " << (st.sum_exp_us / st.n_exp) / 10.0 << " ms";
        if (st.n_gain > 0) std::cout << " | gain " << st.sum_gain / st.n_gain;
        if (have_color_sensor_)
            std::cout << " | option exposure " << readOpt(color_sensor_, RS2_OPTION_EXPOSURE)
                      << " gain " << readOpt(color_sensor_, RS2_OPTION_GAIN)
                      << " ae " << readOpt(color_sensor_, RS2_OPTION_ENABLE_AUTO_EXPOSURE);
        std::cout << std::endl;

        std::cout << "[RealSenseSource] split:";
        if (st.n_mid_frame > 0)    std::cout << " sensor_ts->frame_ts " << st.sum_mid_frame / st.n_mid_frame << " ms |";
        if (st.n_ts_backend > 0)   std::cout << " ts->backend " << st.sum_ts_backend / st.n_ts_backend << " ms |";
        else                       std::cout << " ts->backend n/a |";
        if (st.n_backend_toa > 0)  std::cout << " backend->librealsense " << st.sum_backend_toa / st.n_backend_toa << " ms |";
        else                       std::cout << " backend->librealsense n/a |";
        if (st.n_toa_app > 0)      std::cout << " librealsense->app " << st.sum_toa_app / st.n_toa_app << " ms";
        else                       std::cout << " librealsense->app n/a";
        std::cout << std::endl;

        resetStats();
    }

    const uint8_t* data = static_cast<const uint8_t*>(color.get_data());
    cb_(data, static_cast<uint32_t>(color.get_width()),
              static_cast<uint32_t>(color.get_height()),
              capture_time_ns);
}

#endif
