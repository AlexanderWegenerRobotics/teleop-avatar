#include "pipeline/camera_channel.hpp"
#ifdef WITH_REALSENSE
    #include "pipeline/realsense_source.hpp"
#endif

#ifdef WITH_V4L2
    #include "pipeline/v4l2_source.hpp"
#endif

#include <chrono>
#include <cstdio>
#include <filesystem>
#include <iostream>
#include <stdexcept>

CameraChannel::CameraChannel(const CameraChannelConfig& config)
    : config_(config)
{
    if (!config_.stream_enabled && !config_.log_enabled) {
        std::cerr << "[CameraChannel:" << config_.name
                  << "] WARNING: both streaming and logging are disabled."
                  << " This channel will receive frames but not process them.\n";
    }

    if (config_.stereo_combined) {
        // stream_width must be set to 2x eye width in the YAML
        source_ = std::make_unique<StereoMuJoCoSource>(
            config_.shm_name, config_.stereo_partner_shm, config_.fps);
    } else if (config_.source_type == "realsense") {
    #ifdef WITH_REALSENSE
        source_ = std::make_unique<RealSenseSource>(
            config_.realsense_serial,
            config_.source_width,
            config_.source_height,
            config_.fps,
            config_.exposure_100us,
            config_.auto_exposure,
            config_.gain);
    #else
        throw std::runtime_error("Built without RealSense support. Rebuild with -DBUILD_WITH_REALSENSE=ON");
    #endif
    #ifdef WITH_V4L2
    } else if (config_.source_type == "v4l2") {
        source_ = std::make_unique<V4L2Source>(
            config_.v4l2_device,
            config_.source_width,
            config_.source_height,
            config_.fps,
            config_.v4l2_format,
            config_.auto_exposure ? 0 : config_.exposure_100us);
    #endif
    } else {
        source_ = std::make_unique<MuJoCoSource>(config_.shm_name, config_.fps);
    }

    // if streaming and logging, the streamer logs its encoded H.264 directly
    if (config_.stream_enabled) {
        config_.stream.log_enabled = config_.log_enabled;
        streamer_ = std::make_unique<VideoStreamer>(config_.stream);
    }

    // raw HDF5 logger only for log-only channels
    if (config_.log_enabled && !config_.stream_enabled) {
        LoggerConfig lcfg   = config_.log;
        lcfg.camera_name    = config_.name;
        logger_ = std::make_unique<VideoLogger>(lcfg);
    }
}

CameraChannel::~CameraChannel() { stop(); }

void CameraChannel::start() {
    if (streamer_) streamer_->start();

    source_->start([this](const uint8_t* rgb, uint32_t w, uint32_t h, uint64_t capture_time_ns) {
        const uint64_t frame_id = frame_count_.fetch_add(1, std::memory_order_relaxed);

        if (!config_.raw_shm.empty()) {
            if (!raw_writer_) {
#ifndef _WIN32
                shm_unlink(config_.raw_shm.c_str());
#endif
                raw_writer_ = std::make_unique<SharedMemoryWriter>(config_.raw_shm, w, h);
                std::cout << "[CameraChannel:" << config_.name << "] raw frames -> shm " << config_.raw_shm << std::endl;
            }
            raw_writer_->write(rgb, static_cast<size_t>(w) * h * 3, capture_time_ns);
        }
        if (!undistort_checked_) {
            undistort_checked_ = true;
            if (!config_.undistort_file.empty()) {
                LensModel cam, out;
                if (!loadUndistortFile(config_.undistort_file, cam, out))
                    std::cerr << "[CameraChannel:" << config_.name << "] no undistortion file "
                              << config_.undistort_file << ", streaming raw frames" << std::endl;
                else if (cam.width != static_cast<int>(w) || cam.height != static_cast<int>(h))
                    std::cerr << "[CameraChannel:" << config_.name << "] undistortion file is for " << cam.width
                              << "x" << cam.height << " but frames are " << w << "x" << h
                              << ", streaming raw frames" << std::endl;
                else {
                    undistort_ = RemapTable::undistort(cam, out);
                    undistorted_.resize(static_cast<size_t>(out.width) * out.height * 3);
                    std::cout << "[CameraChannel:" << config_.name << "] undistorting to pinhole fx="
                              << out.fx << " (" << config_.undistort_file << ")" << std::endl;
                }
            }
        }
        if (!undistort_.empty()) {
            undistort_.apply(rgb, undistorted_.data());
            rgb = undistorted_.data();
            w = static_cast<uint32_t>(undistort_.width());
            h = static_cast<uint32_t>(undistort_.height());
        }

        if (streamer_) streamer_->pushFrame(rgb, w, h, capture_time_ns);

        // fall back to now if the source gave no capture time
        if (logger_) {
            const uint64_t ts = capture_time_ns != 0 ? capture_time_ns
                : static_cast<uint64_t>(
                    std::chrono::duration_cast<std::chrono::nanoseconds>(
                        std::chrono::system_clock::now().time_since_epoch()).count());
            logger_->writeFrame(rgb, w, h, ts, frame_id);
        }
    });

    std::cout << "[CameraChannel:" << config_.name << "] Started"
              << " stream=" << (config_.stream_enabled ? "on" : "off")
              << " log="    << (config_.log_enabled    ? "on" : "off")
              << std::endl;
}

void CameraChannel::stop() {
    if (source_)   source_->stop();
    if (streamer_) streamer_->stop();
    if (logger_ && logger_->isActive())
        logger_->stopEpisode("channel_stop");
}

void CameraChannel::onEpisodeStart(const std::string& session_id, int episode_index,
                                    const std::string& log_dir) {
    if (!config_.log_enabled) return;
    if (episode_index == logging_idx_) return;   // duplicate start
    logging_idx_ = episode_index;

    if (streamer_) {
        namespace fs = std::filesystem;
        fs::path dir;
        if (!log_dir.empty()) {
            dir = fs::path(log_dir);
        } else {
            char idx_buf[8];
            std::snprintf(idx_buf, sizeof(idx_buf), "%03d", episode_index);
            dir = fs::path(config_.log.output_dir) / idx_buf;
        }
        std::error_code ec;
        fs::create_directories(dir, ec);
        streamer_->startEncodedLog((dir / ("video_" + config_.name + ".h264")).string());
        return;
    }

    if (logger_) {
        try {
            logger_->startEpisode(session_id, episode_index, log_dir);
        } catch (const std::exception& e) {
            std::cerr << "[CameraChannel:" << config_.name << "] startEpisode failed: " << e.what() << std::endl;
        }
    }
}

void CameraChannel::onEpisodeEnd(const std::string& /*session_id*/,
                                  int /*episode_index*/,
                                  const std::string& reason) {
    logging_idx_ = -1;
    if (streamer_ && config_.log_enabled) { streamer_->stopEncodedLog(); return; }
    if (logger_) logger_->stopEpisode(reason);
}
