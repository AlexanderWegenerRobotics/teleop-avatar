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
    // When this channel both streams and logs, the streamer tees its already-encoded
    // H.264 to a per-episode file (no separate CPU encode/resize/gzip).
    if (config_.stream_enabled) {
        config_.stream.log_enabled = config_.log_enabled;
        streamer_ = std::make_unique<VideoStreamer>(config_.stream);
    }

    // ── Logger (raw-HDF5 fallback, only for log-only channels with no stream) ──
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

        // Streamer gets the raw frame — it appends its own timestamp rows internally,
        // and uses capture_time_ns to log the capture->encode latency leg.
        if (streamer_) streamer_->pushFrame(rgb, w, h, capture_time_ns);

        // Logger gets the clean frame without any embedded timestamp rows. Prefer the
        // source's real capture_time_ns over a receipt-time stamp; fall back to "now"
        // only if a source hasn't got a real one yet (shouldn't happen once frames flow).
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
    // Logger is closed via episode lifecycle (stopEpisode); force-close here if still active.
    if (logger_ && logger_->isActive())
        logger_->stopEpisode("channel_stop");
}

void CameraChannel::onEpisodeStart(const std::string& session_id, int episode_index,
                                    const std::string& log_dir) {
    if (!config_.log_enabled) return;
    if (episode_index == logging_idx_) return;   // duplicate start (boot re-announce); ignore
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
