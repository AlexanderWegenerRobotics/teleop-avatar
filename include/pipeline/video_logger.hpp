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
    std::string output_dir  = "../logs";   // {output_dir}/{index:03d}/images_{camera_name}.hdf5
    std::string camera_name = "camera";
    int         width       = 224;
    int         height      = 224;

    // true: center-crop to output aspect before resize, false: stretch
    bool        center_crop = false;

    // explicit crop in source pixels, overrides center_crop when non-zero
    int         crop_x = 0, crop_y = 0, crop_w = 0, crop_h = 0;
};

// Writes per-episode HDF5 files, images as (T, H, W, 3) uint8. Heavy work runs on a
// writer thread, writeFrame() only queues and drops frames when the queue is full.
class VideoLogger {
public:
    explicit VideoLogger(const LoggerConfig& config);
    ~VideoLogger();

    // Empty log_dir -> {output_dir}/{episode_index:03d}/. Closes any active episode first.
    void startEpisode(const std::string& session_id, int episode_index,
                      const std::string& log_dir = "");

    // Blocks until the writer has finished the file.
    void stopEpisode(const std::string& reason);

    // Non-blocking, drops the frame if the queue is full.
    void writeFrame(const uint8_t* rgb, uint32_t src_w, uint32_t src_h,
                    uint64_t timestamp_ns, uint64_t frame_id);

    bool isActive() const { return episode_active_.load(); }

private:
    void writerLoop();

    void openEpisodeImpl(const std::string& session_id, int episode_index,
                         const std::string& log_dir);

    void closeEpisodeImpl(const std::string& reason);

    void writeFrameImpl(const uint8_t* rgb, uint32_t src_w, uint32_t src_h,
                        uint64_t timestamp_ns, uint64_t frame_id);

    // area-weighted downscale
    void resizeFrame(const uint8_t* src, uint32_t src_w, uint32_t src_h,
                     uint8_t* dst,       uint32_t dst_w, uint32_t dst_h);

    LoggerConfig config_;

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
    std::condition_variable queue_cv_;
    std::condition_variable drain_cv_;   // wakes stopEpisode when writer is idle
    std::deque<Job>         queue_;
    size_t                  queued_frames_  = 0;
    bool                    processing_     = false;
    std::atomic<uint64_t>   dropped_frames_{0};
    static constexpr size_t kMaxQueuedFrames = 16;

    std::thread writer_;

    std::vector<uint8_t> crop_buf_;
    std::vector<uint8_t> resize_buf_;

    // PIMPL so hdf5.h stays out of other TUs
    struct Impl;
    std::unique_ptr<Impl> impl_;
};
