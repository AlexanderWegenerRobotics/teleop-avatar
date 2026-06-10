#pragma once

#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <deque>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

struct LoggerConfig {
    std::string output_dir  = "../logs";   // episodes are written to {output_dir}/{index:03d}/images_{camera_name}.hdf5
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
// Crop/resize, compression and HDF5 I/O run on a dedicated writer thread; the caller's
// writeFrame() only copies the frame into a bounded queue and returns, dropping frames
// when the writer falls behind so logging never blocks the capture/stream pipeline.
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

    // Queues a new-episode command; the writer thread opens the HDF5 file. If log_dir is
    // non-empty it is the episode directory (absolute, supplied by the avatar); otherwise
    // {output_dir}/{episode_index:03d}/ is used. Any active episode is closed first.
    void startEpisode(const std::string& session_id, int episode_index,
                      const std::string& log_dir = "");

    // Queues an end-of-episode command and blocks until the writer has drained the queue
    // and finalised the file, so the episode file is complete on return.
    void stopEpisode(const std::string& reason);

    // Copies one frame into the writer queue and returns immediately; drops the frame if
    // the queue is full. No-op when no episode is active.
    void writeFrame(const uint8_t* rgb, uint32_t src_w, uint32_t src_h,
                    uint64_t timestamp_ns, uint64_t frame_id);

    bool isActive() const { return episode_active_.load(); }

private:
    // Writer-thread main loop: serially opens, writes and closes the HDF5 file.
    void writerLoop();

    // Writer thread: creates the HDF5 file and extendable datasets for one episode.
    void openEpisodeImpl(const std::string& session_id, int episode_index,
                         const std::string& log_dir);

    // Writer thread: writes closing attributes and closes the HDF5 file.
    void closeEpisodeImpl(const std::string& reason);

    // Writer thread: crops/resizes one frame and appends it to the open datasets.
    void writeFrameImpl(const uint8_t* rgb, uint32_t src_w, uint32_t src_h,
                        uint64_t timestamp_ns, uint64_t frame_id);

    // Area-weighted downscale. Handles arbitrary src→dst ratios.
    void resizeFrame(const uint8_t* src, uint32_t src_w, uint32_t src_h,
                     uint8_t* dst,       uint32_t dst_w, uint32_t dst_h);

    LoggerConfig config_;

    // One unit of work handed from the capture thread to the writer thread.
    struct Job {
        enum class Kind { Open, Frame, Close };
        Kind                 kind = Kind::Frame;
        std::vector<uint8_t> rgb;
        uint32_t             src_w = 0, src_h = 0;
        uint64_t             timestamp_ns = 0, frame_id = 0;
        std::string          session_id, log_dir, reason;
        int                  episode_index = 0;
    };

    std::atomic<bool> episode_active_{false};
    std::atomic<bool> running_{false};

    mutable std::mutex      queue_mutex_;
    std::condition_variable queue_cv_;   // wakes the writer when work arrives
    std::condition_variable drain_cv_;   // wakes stopEpisode when the writer is idle
    std::deque<Job>         queue_;
    size_t                  queued_frames_  = 0;   // # of Frame jobs currently queued
    bool                    processing_     = false;
    std::atomic<uint64_t>   dropped_frames_{0};
    static constexpr size_t kMaxQueuedFrames = 16;

    std::thread writer_;

    // Writer-thread scratch buffers.
    std::vector<uint8_t> crop_buf_;
    std::vector<uint8_t> resize_buf_;

    // HDF5 handles hidden behind PIMPL to avoid pulling hdf5.h into every TU.
    struct Impl;
    std::unique_ptr<Impl> impl_;
};
