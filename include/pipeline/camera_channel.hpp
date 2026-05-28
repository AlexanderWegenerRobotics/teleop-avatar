#pragma once

#include <atomic>
#include <memory>
#include <string>

#include "pipeline/camera_source.hpp"
#include "pipeline/video_streamer.hpp"
#include "pipeline/video_logger.hpp"

// Per-camera config parsed from the YAML cameras list.
struct CameraChannelConfig {
    std::string name;

    // ── Source ────────────────────────────────────────────────────────────
    std::string source_type     = "mujoco";  // "mujoco" | "realsense"
    std::string shm_name;                    // mujoco only
    std::string realsense_serial;            // realsense only (empty = first device)
    int         fps             = 30;
    int         source_width    = 640;       // realsense only (mujoco reads from shm)
    int         source_height   = 480;

    // ── Stereo-combined mode ──────────────────────────────────────────────
    // When true, this channel reads shm_name (left) AND stereo_partner_shm
    // (right) in the same poll loop, composites them side-by-side, and sends
    // a single 2×width stream.  The right-eye CameraChannel should have
    // stream_enabled=false.  Only valid for source_type="mujoco".
    bool        stereo_combined    = false;
    std::string stereo_partner_shm = "";

    // ── Streaming ─────────────────────────────────────────────────────────
    bool         stream_enabled = false;
    StreamerConfig stream;

    // ── Logging ───────────────────────────────────────────────────────────
    bool         log_enabled    = false;
    LoggerConfig log;
};

// Owns one CameraSource and fans its frames to an optional VideoStreamer and/or
// VideoLogger. Episode lifecycle is forwarded in from the EpisodeController.
class CameraChannel {
public:
    explicit CameraChannel(const CameraChannelConfig& config);
    ~CameraChannel();

    void start();
    void stop();

    // Called by EpisodeController callbacks — thread-safe.
    void onEpisodeStart(const std::string& session_id, int episode_index);
    void onEpisodeEnd(const std::string& session_id, int episode_index,
                      const std::string& reason);

    const std::string& name() const { return config_.name; }

private:
    CameraChannelConfig           config_;
    std::unique_ptr<CameraSource> source_;
    std::unique_ptr<VideoStreamer> streamer_;
    std::unique_ptr<VideoLogger>  logger_;

    std::atomic<uint64_t> frame_count_{0};
};
