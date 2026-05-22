#pragma once

#include <atomic>
#include <mutex>
#include <thread>
#include <string>
#include <chrono>
#include <functional>

#include "pipeline/stream_feedback.hpp"
#include "network/platform_socket.hpp"

enum class StreamState {
    NO_RECEIVER,
    NORMAL,
    DEGRADED,
    CRITICAL,
    STALE
};

struct StreamQualityParams {
    int bitrate_kbps;
    int fps;
    int width;
    int height;
    int fec_percentage;
};

struct StreamQualityConfig {
    std::string listen_ip   = "0.0.0.0";
    int         listen_port = 5005;

    std::string status_host;
    int         status_port        = 5007;
    int         status_interval_ms = 500;

    int bitrate_normal   = 2000;
    int bitrate_degraded = 1000;
    int bitrate_critical = 500;

    int fps_normal   = 30;
    int fps_degraded = 20;
    int fps_critical = 10;

    int width_normal    = 1280;
    int height_normal   = 720;
    int width_critical  = 640;
    int height_critical = 360;

    int fec_normal   = 10;
    int fec_degraded = 20;
    int fec_critical = 30;

    float loss_threshold_degraded         = 0.05f;
    float loss_threshold_critical         = 0.15f;
    float loss_threshold_recover_normal   = 0.02f;
    float loss_threshold_recover_degraded = 0.10f;

    float jitter_threshold_critical_ms = 50.0f;

    int degrade_count = 3;
    int recover_count = 5;

    int stale_timeout_ms       = 2000;
    int no_receiver_timeout_ms = 10000;
};

using QualityChangeCallback = std::function<void(const StreamQualityParams&)>;

class StreamQualityController {
public:
    explicit StreamQualityController(const StreamQualityConfig& config);
    ~StreamQualityController();

    void start();
    void stop();

    StreamState         getState()  const;
    StreamQualityParams getParams() const;

    void setOnQualityChange(QualityChangeCallback cb);

private:
    void run();
    void processReport(const StreamFeedbackMsg& report);
    void updateState();
    void sendStatusHeartbeat(socket_t fd);
    StreamQualityParams paramsForState(StreamState state) const;

    StreamQualityConfig config_;

    std::thread       thread_;
    std::atomic<bool> bRunning_{false};

    mutable std::mutex  state_mtx_;
    StreamState         state_ = StreamState::NO_RECEIVER;
    StreamQualityParams current_params_;

    int degrade_counter_ = 0;
    int recover_counter_ = 0;

    float last_loss_   = 0.0f;
    float last_jitter_ = 0.0f;

    std::chrono::steady_clock::time_point last_report_time_;
    std::chrono::steady_clock::time_point last_heartbeat_time_;
    bool ever_received_ = false;

    QualityChangeCallback on_change_;
};
