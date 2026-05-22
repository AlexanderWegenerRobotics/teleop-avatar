#include "pipeline/episode_controller.hpp"
#include "pipeline/episode_msg.hpp"
#include "network/platform_socket.hpp"

#include <msgpack.hpp>

#include <iostream>
#include <stdexcept>

#ifdef _WIN32
#  include <winsock2.h>
#  include <ws2tcpip.h>
#else
#  include <arpa/inet.h>
#  include <sys/socket.h>
#  include <unistd.h>
#endif

EpisodeController::EpisodeController(int port) : port_(port) {}

EpisodeController::~EpisodeController() { stop(); }

void EpisodeController::onEpisodeStart(StartCallback cb) { start_cb_ = std::move(cb); }
void EpisodeController::onEpisodeEnd(EndCallback cb)     { end_cb_   = std::move(cb); }

void EpisodeController::start() {
    sock_ = socket(AF_INET, SOCK_DGRAM, 0);
    if (sock_ == kInvalidSocket)
        throw std::runtime_error("[EpisodeController] socket() failed");

    sockaddr_in addr{};
    addr.sin_family      = AF_INET;
    addr.sin_addr.s_addr = INADDR_ANY;
    addr.sin_port        = htons(static_cast<uint16_t>(port_));

    if (bind(sock_, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) < 0) {
        close_socket(sock_);
        sock_ = kInvalidSocket;
        throw std::runtime_error("[EpisodeController] bind() failed on port "
                                 + std::to_string(port_));
    }

    // 500 ms receive timeout so the run loop can check bRunning_
#ifdef _WIN32
    DWORD tv_ms = 500;
    setsockopt(sock_, SOL_SOCKET, SO_RCVTIMEO,
               reinterpret_cast<const char*>(&tv_ms), sizeof(tv_ms));
#else
    struct timeval tv{0, 500'000};
    setsockopt(sock_, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof(tv));
#endif

    bRunning_ = true;
    thread_   = std::thread(&EpisodeController::run, this);
    std::cout << "[EpisodeController] Listening on port " << port_ << std::endl;
}

void EpisodeController::stop() {
    bRunning_ = false;
    if (thread_.joinable()) thread_.join();
    if (sock_ != kInvalidSocket) {
        close_socket(sock_);
        sock_ = kInvalidSocket;
    }
}

void EpisodeController::run() {
    char buf[2048];

    while (bRunning_) {
        ssize_t n = recvfrom(sock_, buf, sizeof(buf), 0, nullptr, nullptr);
        if (n <= 0) continue;   // timeout or error — loop again

        try {
            auto oh  = msgpack::unpack(buf, static_cast<size_t>(n));
            EpisodeEventMsg msg;
            oh.get().convert(msg);

            if (msg.type == "episode_start") {
                std::cout << "[EpisodeController] episode_start"
                          << " session=" << msg.session_id
                          << " idx="     << msg.episode_index << std::endl;
                if (start_cb_) start_cb_(msg.session_id, msg.episode_index);

            } else if (msg.type == "episode_end") {
                std::cout << "[EpisodeController] episode_end"
                          << " session=" << msg.session_id
                          << " idx="     << msg.episode_index
                          << " reason="  << msg.reason << std::endl;
                if (end_cb_) end_cb_(msg.session_id, msg.episode_index, msg.reason);
            }
        } catch (const std::exception& e) {
            std::cerr << "[EpisodeController] parse error: " << e.what() << std::endl;
        }
    }
}
