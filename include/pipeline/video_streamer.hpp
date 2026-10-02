#pragma once

#include <atomic>
#include <cstdint>
#include <cstdio>
#include <deque>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include <gst/gst.h>
#include <gst/app/gstappsrc.h>
#include <gst/app/gstappsink.h>

#include "pipeline/stream_quality_controller.hpp"

// Streaming-only config. Source selection lives in CameraChannelConfig.
struct StreamerConfig {
    std::string host;
    int         port               = 5004;
    int         feedback_port      = 5005;
    int         status_port        = 5007;
    int         status_interval_ms = 500;
    int         fps                = 30;
    int         bitrate_kbps       = 2000;
    int         fec_percentage     = 10;
    int         stream_width       = 640;
    int         stream_height      = 480;
    // 0 = same as stream size, otherwise pushFrame rescales
    int         source_width       = 0;
    int         source_height      = 0;
    bool        log_enabled        = false;
};

class VideoStreamer {
public:
    static constexpr size_t kTimestampBytes = 8;

    // gst_init() must be called before constructing
    explicit VideoStreamer(const StreamerConfig& config);
    ~VideoStreamer();

    void start();
    void stop();

    // capture_time_ns in system_clock ns, 0 if unknown
    void pushFrame(const uint8_t* rgb, uint32_t width, uint32_t height, uint64_t capture_time_ns);

    // Logs the already-encoded H.264 stream per episode.
    void startEncodedLog(const std::string& path);
    void stopEncodedLog();

    uint64_t frameId() const { return frame_count_; }

private:
    void buildPipeline();
    void requestKeyframe();
    static GstFlowReturn onNewSample(GstAppSink* sink, gpointer user);

private:
    StreamerConfig config_;

    GstElement*       pipeline_    = nullptr;
    GstElement*       appsrc_      = nullptr;
    GMainLoop*        loop_        = nullptr;
    std::thread       loop_thread_;
    std::atomic<bool> bRunning_{false};
    uint64_t          frame_count_ = 0;

    GstElement* encoder_ = nullptr;
    GstElement* fec_     = nullptr;
    std::unique_ptr<StreamQualityController> quality_;
    std::atomic<int> target_fps_{0};

    GstElement* logsink_  = nullptr;
    FILE*       enc_file_ = nullptr;
    FILE*       ts_file_  = nullptr;
    std::mutex  enc_mutex_;
    std::atomic<bool>     await_keyframe_{false};  // wait for first IDR of episode

    struct PendingFrameTs {
        uint64_t encode_ns;
        uint64_t capture_ns;
    };
    std::deque<PendingFrameTs> ts_queue_;
    uint64_t              log_frame_idx_ = 0;
};
