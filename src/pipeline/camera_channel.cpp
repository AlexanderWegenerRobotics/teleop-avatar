#include "pipeline/camera_channel.hpp"
#ifdef WITH_REALSENSE
    #include "pipeline/realsense_source.hpp"
#endif

#ifdef WITH_V4L2
    #include "pipeline/v4l2_source.hpp"
#endif

#include <chrono>
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

    // ── Source ────────────────────────────────────────────────────────────
    if (config_.stereo_combined) {
        // Side-by-side stereo: both eyes composited into one 2×width stream.
        // stream_width is set to 2×single-eye width in the YAML config directly.
        source_ = std::make_unique<StereoMuJoCoSource>(
            config_.shm_name, config_.stereo_partner_shm, config_.fps);
    } else if (config_.source_type == "realsense") {
    #ifdef WITH_REALSENSE
        source_ = std::make_unique<RealSenseSource>(
            config_.realsense_serial,
            config_.source_width,
            config_.source_height,
            config_.fps);
    #else
        throw std::runtime_error("Built without RealSense support. Rebuild with -DBUILD_WITH_REALSENSE=ON");
    #endif
    #ifdef WITH_V4L2
    } else if (config_.source_type == "v4l2") {
        source_ = std::make_unique<V4L2Source>(
            config_.shm_name,
            config_.source_width,
            config_.source_height,
            config_.fps);
    #endif
    } else {
        source_ = std::make_unique<MuJoCoSource>(config_.shm_name, config_.fps);
    }

    // ── Streamer (optional) ───────────────────────────────────────────────
    if (config_.stream_enabled) {
        streamer_ = std::make_unique<VideoStreamer>(config_.stream);
    }

    // ── Logger (optional) ─────────────────────────────────────────────────
    if (config_.log_enabled) {
        LoggerConfig lcfg   = config_.log;
        lcfg.camera_name    = config_.name;
        logger_ = std::make_unique<VideoLogger>(lcfg);
    }
}

CameraChannel::~CameraChannel() { stop(); }

void CameraChannel::start() {
    if (streamer_) streamer_->start();

    source_->start([this](const uint8_t* rgb, uint32_t w, uint32_t h) {
        const uint64_t frame_id = frame_count_.fetch_add(1, std::memory_order_relaxed);

        // Streamer gets the raw frame — it appends its own timestamp rows internally.
        if (streamer_) streamer_->pushFrame(rgb, w, h);

        // Logger gets the clean frame without any embedded timestamp rows,
        // plus an explicit wall-clock timestamp.
        if (logger_) {
            const uint64_t ts = static_cast<uint64_t>(
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
    // Logger is closed via episode lifecycle (stopEpisode); force-close here if still active.
    if (logger_ && logger_->isActive())
        logger_->stopEpisode("channel_stop");
}

void CameraChannel::onEpisodeStart(const std::string& session_id, int episode_index) {
    if (logger_) logger_->startEpisode(session_id, episode_index);
}

void CameraChannel::onEpisodeEnd(const std::string& /*session_id*/,
                                  int /*episode_index*/,
                                  const std::string& reason) {
    if (logger_) logger_->stopEpisode(reason);
}
