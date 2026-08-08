#include "twin/telemetry_forward.hpp"

#include <iostream>

TelemetryForwarder::TelemetryForwarder(const YAML::Node& avatar_node) {
    if (!avatar_node || !avatar_node["twin_telemetry"]) return;

    const auto& tt = avatar_node["twin_telemetry"];
    enabled_ = tt["enabled"].as<bool>(false);
    if (!enabled_) return;

    host_ = tt["host"].as<std::string>("127.0.0.1");
    port_ = tt["port"].as<int>(7400);
    int hz = tt["frequency"].as<int>(100);
    if (hz <= 0) hz = 100;
    period_ = std::chrono::microseconds(1000000 / hz);

    sock_ = socket(AF_INET, SOCK_DGRAM, 0);
    if (sock_ == kInvalidSocket) {
        std::cerr << "[AVATAR-WARN] TelemetryForwarder: failed to create socket, twin telemetry disabled\n";
        enabled_ = false;
        return;
    }

    std::cout << "[AVATAR-INFO] Twin telemetry -> " << host_ << ":" << port_
              << " (" << hz << " Hz)" << std::endl;
}

TelemetryForwarder::~TelemetryForwarder() {
    if (sock_ != kInvalidSocket) close_socket(sock_);
}

void TelemetryForwarder::maybeSend(const TwinTelemetryMsg& msg) {
    if (!enabled_ || sock_ == kInvalidSocket) return;

    auto now = std::chrono::steady_clock::now();
    if (now - last_send_ < period_) return;
    last_send_ = now;

    sockaddr_in dst{};
    dst.sin_family = AF_INET;
    dst.sin_port   = htons(static_cast<uint16_t>(port_));
    inet_pton(AF_INET, host_.c_str(), &dst.sin_addr);
    sendto(sock_, reinterpret_cast<const char*>(&msg), static_cast<int>(sizeof(msg)), 0,
           reinterpret_cast<sockaddr*>(&dst), sizeof(dst));
}
