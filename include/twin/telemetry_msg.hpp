#pragma once

// Wire format for hardware joint telemetry avatar -> twin reconciler (UDP, 2 arms x 7 DoF).

#include <cstdint>
#include <type_traits>

#include "common.hpp"

#pragma pack(push, 1)

struct TwinTelemetryMsg {
    MsgHeader header;   // timestamp_ns = sample time t_s (not send time), NTP-synced clocks
    float    q_left[7];
    float    dq_left[7];
    float    q_right[7];
    float    dq_right[7];
    uint8_t  valid_left;   // 1 if arm_left is enabled on the sending avatar
    uint8_t  valid_right;  // 1 if arm_right is enabled on the sending avatar
};

#pragma pack(pop)

static_assert(std::is_trivially_copyable_v<TwinTelemetryMsg>,
              "TwinTelemetryMsg must be trivially copyable (raw UDP wire format)");
