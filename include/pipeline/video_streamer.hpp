#pragma once

#include <atomic>
#include <memory>
#include <string>
#include <thread>

#include <gst/gst.h>
#include <gst/app/gstappsrc.h>

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
};

class VideoStreamer {
public:
    static constexpr size_t kTimestampBytes = 8;

    // gst_init() must be called by the caller (main) before constructing any VideoStreamer.
    explicit VideoStreamer(const StreamerConfig& config);
    ~VideoStreamer();

    void start();
    void stop();

    // Push one raw RGB frame from CameraChannel. w/h must match config stream_width/height.
    void pushFrame(const uint8_t* rgb, uint32_t width, uint32_t height);

    uint64_t frameId() const { return frame_count_; }

private:
    void buildPipeline();

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
};
