#pragma once

// Avatar-side UDP sender for joint telemetry to the twin's reconciler. Config: avatar.twin_telemetry.

#include <chrono>
#include <string>

#include <yaml-cpp/yaml.h>

#include "network/platform_socket.hpp"
#include "twin/telemetry_msg.hpp"

class TelemetryForwarder {
public:
    // avatar_node is sys_config["avatar"]
    explicit TelemetryForwarder(const YAML::Node& avatar_node);
    ~TelemetryForwarder();

    TelemetryForwarder(const TelemetryForwarder&)            = delete;
    TelemetryForwarder& operator=(const TelemetryForwarder&) = delete;

    bool enabled() const { return enabled_; }

    // sends if enabled and the send period has elapsed, safe to call every tick
    void maybeSend(const TwinTelemetryMsg& msg);

private:
    bool                                  enabled_ = false;
    std::string                           host_;
    int                                   port_ = 0;
    std::chrono::microseconds             period_{10000};  // 100 Hz default
    socket_t                              sock_ = kInvalidSocket;
    std::chrono::steady_clock::time_point last_send_{};
};
