#pragma once

#include <atomic>
#include <memory>
#include <string>

#include "pipeline/camera_source.hpp"
#include "pipeline/lens.hpp"
#include "pipeline/video_streamer.hpp"
#include "pipeline/video_logger.hpp"

// Per-camera config parsed from the YAML cameras list.
struct CameraChannelConfig {
    std::string name;

    std::string source_type     = "mujoco";  // "mujoco" | "realsense" | "v4l2"
    std::string shm_name;                    // mujoco only
    std::string realsense_serial;            // realsense only (empty = first device)
    std::string v4l2_device     = "/dev/video0";  // v4l2 only
    std::string v4l2_format     = "yuyv";         // v4l2 only: "yuyv" | "mjpeg"
    int         fps             = 30;
    int         source_width    = 640;       // realsense/v4l2 only (mujoco reads from shm)
    int         source_height   = 480;
    // 0 = auto-exposure, >0 = manual exposure in 100 us units (100 = 10 ms)
    int         exposure_100us  = 0;
    bool        auto_exposure   = true;
    int         gain            = -1;

    // calibration file from teleop-perception; frames are resampled to its ideal pinhole before stream/log
    std::string undistort_file;
    // optional shm name where the raw (not undistorted) frames are published, e.g. for calibration
    std::string raw_shm;

    // mujoco only: send left|right side by side as one 2x width stream
    bool        stereo_combined    = false;
    std::string stereo_partner_shm = "";

    bool         stream_enabled = false;
    StreamerConfig stream;

    bool         log_enabled    = false;
    LoggerConfig log;
};

// One camera source fanned out to an optional streamer and logger.
class CameraChannel {
public:
    explicit CameraChannel(const CameraChannelConfig& config);
    ~CameraChannel();

    void start();
    void stop();

    // Called by EpisodeController callbacks — thread-safe.
    void onEpisodeStart(const std::string& session_id, int episode_index,
                        const std::string& log_dir = "");
    void onEpisodeEnd(const std::string& session_id, int episode_index,
                      const std::string& reason);

    const std::string& name() const { return config_.name; }

private:
    CameraChannelConfig           config_;
    std::unique_ptr<CameraSource> source_;
    std::unique_ptr<VideoStreamer> streamer_;
    std::unique_ptr<VideoLogger>  logger_;
    std::unique_ptr<SharedMemoryWriter> raw_writer_;
    RemapTable                    undistort_;
    std::vector<uint8_t>          undistorted_;
    bool                          undistort_checked_ = false;

    std::atomic<uint64_t> frame_count_{0};
    int logging_idx_ = -1;   // -1 = not logging
};
