#pragma once

#include <atomic>
#include <functional>
#include <string>
#include <thread>

#include "network/platform_socket.hpp"

// Listens for episode start/end packets from the avatar and fires callbacks.
class EpisodeController {
public:
    using StartCallback = std::function<void(const std::string& session_id, int episode_index,
                                             const std::string& log_dir)>;
    using EndCallback   = std::function<void(const std::string& session_id, int episode_index,
                                             const std::string& reason)>;

    explicit EpisodeController(int listen_port);
    ~EpisodeController();

    void onEpisodeStart(StartCallback cb);
    void onEpisodeEnd(EndCallback cb);

    void start();
    void stop();

private:
    void run();

    int      port_;
    socket_t sock_     = kInvalidSocket;
    std::thread       thread_;
    std::atomic<bool> bRunning_{false};

    StartCallback start_cb_;
    EndCallback   end_cb_;
};
