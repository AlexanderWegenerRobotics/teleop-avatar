#pragma once

// Wire format for y(t_s): real hardware joint telemetry, published by a
// role: avatar process and consumed by a paired role: twin process's
// Reconciler (docs/twin_concept.md section 4). One-way, best-effort UDP,
// same "trivially copyable, MsgHeader-first" convention as the existing
// ArmCommandMsg/ArmStateMsg in common.hpp.
//
// Fixed to two arms (arm_left, arm_right / 7 DoF each -> x in R^14) to match
// the rest of this codebase's bimanual assumption (see common.hpp's
// ArmStateMsg, robot_config's devices list). Extend to N arms only if a
// third manipulator is ever added to the rig.

#include <cstdint>
#include <type_traits>

#include "common.hpp"

#pragma pack(push, 1)

struct TwinTelemetryMsg {
    MsgHeader header;   // timestamp_ns here is t_s -- the sample's wall-clock
                         // stamp on the avatar side, NOT the send time. Both
                         // machines are NTP-synchronized (docs/twin_concept.md
                         // section 2), so this is directly comparable to the
                         // twin's local clock without a separate offset.
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
