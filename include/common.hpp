#pragma once

#include <chrono>
#include <cstdint>
#include <Eigen/Dense>
#include <yaml-cpp/yaml.h>

enum class SysState : uint8_t {
    OFFLINE  = 0,
    IDLE     = 1,
    HOMING   = 2,
    AWAITING = 3,
    ENGAGED  = 4,
    PAUSED   = 5,
    FAULT    = 6,
    STOP     = 7,
    RECOVERING = 8,
    UNDEFINED   = 255
};

enum class GraspState : uint8_t {
    OPEN = 0,
    HELD = 1,
    LOST = 2
};

// Which channel may move an arm, per arm. Values match EControlAuthority in the VR interface.
// UNSET = no request yet, both channels pass; the first request latches enforcement on.
enum class CommandAuthority : uint8_t {
    POLICY = 0,   // orchestrator (absolute channel)
    HUMAN  = 1,   // VR interface
    HOLD   = 2,   // neither, arm holds last target
    UNSET  = 255
};

inline const char* toString(CommandAuthority a) {
    switch (a) {
        case CommandAuthority::POLICY: return "POLICY";
        case CommandAuthority::HUMAN:  return "HUMAN";
        case CommandAuthority::HOLD:   return "HOLD";
        default:                       return "UNSET";
    }
}

enum class DeviceId : uint8_t {
    LEFT_ARM  = 1,
    RIGHT_ARM = 2,
    HEAD      = 3,
    AVATAR    = 4
};

enum class TransmissionRole : uint8_t {
    ARM    = 0,
    HEAD   = 1,
    AVATAR = 2
};

enum class FaultCode : uint8_t {
    NONE                = 0,
    JOINT_LIMIT         = 1,
    JOINT_LOCKED        = 2,
    HIGH_EXTERNAL_FORCE = 3,
    VELOCITY_LIMIT      = 4,
    IMPLAUSIBLE_COMMAND = 5,
    COMM_LOSS           = 6,
    INTERNAL_ERROR      = 7,
    HMD_NOT_WORN        = 8,
    COLLISION_RISK      = 9,
    WORKSPACE_LIMIT     = 10
};

inline uint64_t timestamp_ns() {
    return static_cast<uint64_t>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::system_clock::now().time_since_epoch()).count());
}

using Matrix6x7 = Eigen::Matrix<double, 6, 7>;
using Matrix7   = Eigen::Matrix<double, 7, 7>;
using Matrix4   = Eigen::Matrix<double, 4, 4>;
using Vector7   = Eigen::Matrix<double, 7, 1>;
using Vector2   = Eigen::Matrix<double, 2, 1>;

#pragma pack(push, 1)
 
struct MsgHeader {
    uint32_t sequence;
    uint64_t timestamp_ns;
    // When the data was sampled (timestamp_ns is send time). Use for staleness; 0 = unknown.
    uint64_t sample_time_ns;
    SysState state;
    FaultCode fault_code;
    DeviceId device_id;
};
 
struct ArmCommandMsg {
    MsgHeader header;
    float position[3];
    float quaternion[4];
    float gripper;
    uint8_t clutch;   // 1 = clutched, only sent for logging
};
 
struct ArmStateMsg {
    MsgHeader header;
    float position[3];
    float quaternion[4];
    float joint_positions[7];
    float tau_ext[7];
    uint8_t recovering;
    float gripper_width;
    GraspState grasp_state;
    // Sequence of the last consumed command, for single-clock RTT. 0 = none yet.
    uint32_t applied_cmd_sequence;
    // Network-delay echo: newest received cmd seq + how long it sat here (us, avatar clock).
    // network_rtt = (recv_time - send_time(echo_cmd_sequence)) - echo_hold_us. 0/0 = unknown.
    uint32_t echo_cmd_sequence;
    uint32_t echo_hold_us;
};
 
struct HeadCommandMsg {
    MsgHeader header;
    float pan;
    float tilt;
};
 
struct HeadStateMsg {
    MsgHeader header;
    float pan;
    float tilt;
};
 
#pragma pack(pop)

// Wire contract, must match protocol.hpp (VR interface) and live/wire.py (orchestrator).
static_assert(sizeof(MsgHeader)      == 23,  "MsgHeader size mismatch");
static_assert(sizeof(ArmCommandMsg)  == 56,  "ArmCommandMsg size mismatch");
static_assert(sizeof(ArmStateMsg)    == 125, "ArmStateMsg size mismatch");
static_assert(sizeof(HeadCommandMsg) == 31,  "HeadCommandMsg size mismatch");
static_assert(sizeof(HeadStateMsg)   == 31,  "HeadStateMsg size mismatch");

// per-device bookkeeping entry on the avatar side
struct DeviceRecord {
    bool active = true;
};

template<int N>
Eigen::Matrix<double, N, 1> yamlToVector(const YAML::Node& node) {
    auto vec = node.as<std::vector<double>>();
    return Eigen::Map<const Eigen::Matrix<double, N, 1>>(vec.data());
}