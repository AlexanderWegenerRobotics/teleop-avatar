#include "pipeline/stream_quality_controller.hpp"

#include <iostream>
#include <cstring>

#include "network/platform_socket.hpp"

#ifdef _WIN32
    // WSAPoll mirrors POSIX poll; pollfd is WSAPOLLFD but layout-compatible
    #define poll WSAPoll
#else
    #include <unistd.h>
    #include <fcntl.h>
    #include <poll.h>
#endif


StreamQualityController::StreamQualityController(const StreamQualityConfig& config)
    : config_(config)
{
    current_params_ = paramsForState(StreamState::NO_RECEIVER);
    last_report_time_ = std::chrono::steady_clock::now();
}

StreamQualityController::~StreamQualityController() {
    stop();
}

void StreamQualityController::start() {
    bRunning_ = true;
    thread_ = std::thread(&StreamQualityController::run, this);
}

void StreamQualityController::stop() {
    bRunning_ = false;
    if (thread_.joinable()) thread_.join();
}

StreamState StreamQualityController::getState() const {
    std::lock_guard<std::mutex> lock(state_mtx_);
    return state_;
}

StreamQualityParams StreamQualityController::getParams() const {
    std::lock_guard<std::mutex> lock(state_mtx_);
    return current_params_;
}

void StreamQualityController::setOnQualityChange(QualityChangeCallback cb) {
    on_change_ = std::move(cb);
}

StreamQualityParams StreamQualityController::paramsForState(StreamState state) const {
    switch (state) {
        case StreamState::DEGRADED:
            return {
                config_.bitrate_degraded,
                config_.fps_degraded,
                config_.width_normal,
                config_.height_normal,
                config_.fec_degraded
            };
        case StreamState::CRITICAL:
            return {
                config_.bitrate_critical,
                config_.fps_critical,
                config_.width_critical,
                config_.height_critical,
                config_.fec_critical
            };
        case StreamState::NO_RECEIVER:
        case StreamState::NORMAL:
        case StreamState::STALE:
        default:
            return {
                config_.bitrate_normal,
                config_.fps_normal,
                config_.width_normal,
                config_.height_normal,
                config_.fec_normal
            };
    }
}

void StreamQualityController::run() {
    socket_t recv_fd = socket(AF_INET, SOCK_DGRAM, 0);
    if (recv_fd == kInvalidSocket) {
        std::cerr << "[QualityController] socket creation failed" << std::endl;
        return;
    }

    // Set non-blocking
#ifdef _WIN32
    u_long nb = 1;
    ioctlsocket(recv_fd, FIONBIO, &nb);
#else
    int flags = fcntl(recv_fd, F_GETFL, 0);
    fcntl(recv_fd, F_SETFL, flags | O_NONBLOCK);
#endif

    sockaddr_in addr{};
    addr.sin_family      = AF_INET;
    addr.sin_port        = htons(static_cast<u_short>(config_.listen_port));
    addr.sin_addr.s_addr = inet_addr(config_.listen_ip.c_str());

    if (bind(recv_fd, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) < 0) {
        std::cerr << "[QualityController] bind failed on port " << config_.listen_port << std::endl;
        close_socket(recv_fd);
        return;
    }

    socket_t send_fd = socket(AF_INET, SOCK_DGRAM, 0);
    if (send_fd == kInvalidSocket) {
        std::cerr << "[QualityController] send socket creation failed" << std::endl;
        close_socket(recv_fd);
        return;
    }

    sockaddr_in dest{};
    dest.sin_family      = AF_INET;
    dest.sin_port        = htons(static_cast<u_short>(config_.status_port));
    dest.sin_addr.s_addr = inet_addr(config_.status_host.c_str());

    if (connect(send_fd, reinterpret_cast<sockaddr*>(&dest), sizeof(dest)) < 0) {
        std::cerr << "[QualityController] connect failed for status destination" << std::endl;
        close_socket(recv_fd);
        close_socket(send_fd);
        return;
    }

    std::cout << "[QualityController] listening on " << config_.listen_ip
              << ":" << config_.listen_port << std::endl;
    std::cout << "[QualityController] status heartbeat -> "
              << config_.status_host << ":" << config_.status_port << std::endl;

    last_heartbeat_time_ = std::chrono::steady_clock::now();

    while (bRunning_) {
        struct pollfd pfd{};
#ifdef _WIN32
        pfd.fd = static_cast<SOCKET>(recv_fd);
#else
        pfd.fd = recv_fd;
#endif
        pfd.events = POLLIN;

        int ret = poll(&pfd, 1, 100);
        if (ret > 0 && (pfd.revents & POLLIN)) {
            StreamFeedbackMsg report{};
            ssize_t n = recv(recv_fd,
                             reinterpret_cast<char*>(&report),
                             static_cast<int>(sizeof(report)), 0);
            if (n == static_cast<ssize_t>(sizeof(StreamFeedbackMsg))) {
                processReport(report);
            }
        }

        updateState();

        auto now     = std::chrono::steady_clock::now();
        auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
            now - last_heartbeat_time_).count();

        if (elapsed >= config_.status_interval_ms) {
            sendStatusHeartbeat(send_fd);
            last_heartbeat_time_ = now;
        }
    }

    close_socket(recv_fd);
    close_socket(send_fd);
}

void StreamQualityController::sendStatusHeartbeat(socket_t fd) {
    StreamStatusMsg msg{};
    msg.magic        = kStreamStatusMagic;
    msg.state        = static_cast<uint8_t>(getState());
    msg.timestamp_ns = static_cast<uint64_t>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::system_clock::now().time_since_epoch()).count());

    send(fd, reinterpret_cast<const char*>(&msg), static_cast<int>(sizeof(msg)), 0);
}

void StreamQualityController::processReport(const StreamFeedbackMsg& report) {
    std::lock_guard<std::mutex> lock(state_mtx_);

    last_loss_   = report.loss_rate;
    last_jitter_ = report.jitter_ms;
    last_report_time_ = std::chrono::steady_clock::now();
    ever_received_ = true;

    if (state_ == StreamState::NO_RECEIVER || state_ == StreamState::STALE) {
        state_ = StreamState::NORMAL;
        current_params_ = paramsForState(StreamState::NORMAL);
        degrade_counter_ = 0;
        recover_counter_ = 0;
        std::cout << "[QualityController] receiver connected → NORMAL" << std::endl;
        if (on_change_) on_change_(current_params_);
    }
}

void StreamQualityController::updateState() {
    std::lock_guard<std::mutex> lock(state_mtx_);

    if (!ever_received_) return;

    auto now = std::chrono::steady_clock::now();
    auto since_last = std::chrono::duration_cast<std::chrono::milliseconds>(
        now - last_report_time_).count();

    if (state_ != StreamState::NO_RECEIVER && state_ != StreamState::STALE) {
        if (since_last > config_.stale_timeout_ms) {
            state_ = StreamState::STALE;
            std::cout << "[QualityController] feedback lost → STALE" << std::endl;
            degrade_counter_ = 0;
            recover_counter_ = 0;
            return;
        }
    }

    if (state_ == StreamState::STALE) {
        if (since_last > config_.no_receiver_timeout_ms) {
            state_ = StreamState::NO_RECEIVER;
            current_params_ = paramsForState(StreamState::NO_RECEIVER);
            std::cout << "[QualityController] timeout → NO_RECEIVER" << std::endl;
            if (on_change_) on_change_(current_params_);
            ever_received_ = false;
        }
        return;
    }

    StreamState prev = state_;
    bool should_degrade = false;
    bool should_recover = false;

    switch (state_) {
        case StreamState::NORMAL:
            if (last_loss_ > config_.loss_threshold_degraded) {
                degrade_counter_++;
                recover_counter_ = 0;
            } else {
                degrade_counter_ = 0;
            }
            if (degrade_counter_ >= config_.degrade_count)
                should_degrade = true;
            break;

        case StreamState::DEGRADED:
            if (last_loss_ > config_.loss_threshold_critical ||
                last_jitter_ > config_.jitter_threshold_critical_ms) {
                degrade_counter_++;
                recover_counter_ = 0;
            } else if (last_loss_ < config_.loss_threshold_recover_normal) {
                recover_counter_++;
                degrade_counter_ = 0;
            } else {
                degrade_counter_ = 0;
                recover_counter_ = 0;
            }
            if (degrade_counter_ >= config_.degrade_count)
                should_degrade = true;
            if (recover_counter_ >= config_.recover_count)
                should_recover = true;
            break;

        case StreamState::CRITICAL:
            if (last_loss_ < config_.loss_threshold_recover_degraded) {
                recover_counter_++;
                degrade_counter_ = 0;
            } else {
                recover_counter_ = 0;
            }
            if (recover_counter_ >= config_.recover_count)
                should_recover = true;
            break;

        default:
            break;
    }

    if (should_degrade) {
        if (state_ == StreamState::NORMAL) state_ = StreamState::DEGRADED;
        else if (state_ == StreamState::DEGRADED) state_ = StreamState::CRITICAL;
        degrade_counter_ = 0;
        recover_counter_ = 0;
    }

    if (should_recover) {
        if (state_ == StreamState::CRITICAL) state_ = StreamState::DEGRADED;
        else if (state_ == StreamState::DEGRADED) state_ = StreamState::NORMAL;
        degrade_counter_ = 0;
        recover_counter_ = 0;
    }

    if (state_ != prev) {
        current_params_ = paramsForState(state_);
        std::cout << "[QualityController] state change → "
                  << (state_ == StreamState::NORMAL   ? "NORMAL"   :
                      state_ == StreamState::DEGRADED ? "DEGRADED" :
                      state_ == StreamState::CRITICAL ? "CRITICAL" : "UNKNOWN")
                  << " (loss=" << last_loss_ << " jitter=" << last_jitter_ << "ms)"
                  << std::endl;
        if (on_change_) on_change_(current_params_);
    }
}
