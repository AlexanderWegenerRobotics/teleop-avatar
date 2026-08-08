#pragma once

// Avatar-side sender for TwinTelemetryMsg (y(t_s)) -- the real hardware
// joint telemetry a paired role: twin process's Reconciler corrects
// against (docs/twin_concept.md section 4). Deliberately trivial (raw UDP,
// no ack/retry -- telemetry is a stream of samples, dropping one is fine)
// and independent of WITH_MUJOCO: it must work whether this avatar process
// is a real-hardware build (WITH_FRANKA) or a local sim-as-avatar build.
//
// Config lives under robot_config's `avatar.twin_telemetry` block:
//   enabled:   true|false
//   host/port: twin's reconciler_config.yaml listen_port
//   frequency: Hz, best-effort throttle (Avatar's control loop runs faster)

#include <chrono>
#include <string>

#include <yaml-cpp/yaml.h>

#include "network/platform_socket.hpp"
#include "twin/telemetry_msg.hpp"

class TelemetryForwarder {
public:
    // avatar_node is sys_config["avatar"] (robot_config's top-level
    // `avatar:` block); looks for a `twin_telemetry` child.
    explicit TelemetryForwarder(const YAML::Node& avatar_node);
    ~TelemetryForwarder();

    TelemetryForwarder(const TelemetryForwarder&)            = delete;
    TelemetryForwarder& operator=(const TelemetryForwarder&) = delete;

    bool enabled() const { return enabled_; }

    // Sends msg if enabled and the configured send period has elapsed since
    // the last send; otherwise a no-op. Safe to call every control tick.
    void maybeSend(const TwinTelemetryMsg& msg);

private:
    bool                                  enabled_ = false;
    std::string                           host_;
    int                                   port_ = 0;
    std::chrono::microseconds             period_{10000};  // 100 Hz default
    socket_t                              sock_ = kInvalidSocket;
    std::chrono::steady_clock::time_point last_send_{};
};
