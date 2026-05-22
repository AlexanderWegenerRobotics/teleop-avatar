#pragma once

#include <cstdint>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

struct LoggerConfig {
    std::string output_dir  = "logs";   // episodes are written to {output_dir}/{index:03d}/images.hdf5
    std::string camera_name = "camera"; // used as HDF5 dataset path: observations/images/{camera_name}
    int         width       = 224;
    int         height      = 224;

    // Center-crop before resize (recommended for square outputs from non-square sources).
    // When true: crops the largest W×H rectangle centred in the source, then resizes.
    // When false (default): stretches the full source to W×H (may distort aspect ratio).
    bool        center_crop = false;

    // Optional explicit crop window (pixels in source space, applied before resize).
    // If all four are 0 and center_crop is true, the crop is computed automatically.
    // If non-zero, this crop is used regardless of center_crop.
    int         crop_x = 0, crop_y = 0, crop_w = 0, crop_h = 0;
};

// Writes per-episode HDF5 files containing:
//   observations/images/{camera_name}   (T, H, W, 3)  uint8
//   observations/timestamp_ns           (T,)           uint64
//   observations/frame_id               (T,)           uint64
//
// File attributes: session_id, episode_index, end_reason, frame_count
//
// Usage:
//   logger.startEpisode(session_id, episode_index);
//   // per frame:
//   logger.writeFrame(rgb, src_w, src_h, timestamp_ns, frame_id);
//   logger.stopEpisode(reason);
class VideoLogger {
public:
    explicit VideoLogger(const LoggerConfig& config);
    ~VideoLogger();

    // Opens a new HDF5 file and creates extendable datasets.
    // If an episode is already active it is closed first.
    void startEpisode(const std::string& session_id, int episode_index);

    // Finalises the HDF5 file and writes closing attributes.
    void stopEpisode(const std::string& reason);

    // Appends one frame. If log resolution differs from (src_w, src_h) the frame
    // is area-averaged downscaled before writing. No-op when no episode is active.
    void writeFrame(const uint8_t* rgb, uint32_t src_w, uint32_t src_h,
                    uint64_t timestamp_ns, uint64_t frame_id);

    bool isActive() const { std::lock_guard<std::mutex> lk(mutex_); return episode_active_; }

private:
    // Internal stop — caller must already hold mutex_.
    void stopEpisodeLocked(const std::string& reason);

    // Area-weighted downscale. Handles arbitrary src→dst ratios.
    void resizeFrame(const uint8_t* src, uint32_t src_w, uint32_t src_h,
                     uint8_t* dst,       uint32_t dst_w, uint32_t dst_h);

    LoggerConfig config_;

    mutable std::mutex mutex_;   // guards episode_active_ + impl_ handle access
    bool         episode_active_ = false;
    std::vector<uint8_t> crop_buf_;
    std::vector<uint8_t> resize_buf_;

    // HDF5 handles hidden behind PIMPL to avoid pulling hdf5.h into every TU.
    struct Impl;
    std::unique_ptr<Impl> impl_;
};
