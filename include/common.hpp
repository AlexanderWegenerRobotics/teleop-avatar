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
    // Wall-clock instant at which the DATA in this message was sampled, as
    // opposed to timestamp_ns, which is stamped when the packet is handed to
    // the socket.
    //
    // The two are normally within a millisecond of each other and the
    // distinction looks academic. It is not. Arm state is published from the
    // 200 Hz state thread while the robot is read by the 1 kHz control
    // thread. If the control thread stops -- franka::ControlException,
    // automaticErrorRecovery(), a blocking FAULT wait -- the state thread
    // happily keeps transmitting the last pose it saw, with a fresh
    // timestamp_ns and an incrementing sequence every time. Every
    // transport-level metric on the receiving side then reports a healthy
    // link, because there IS a healthy link; it is carrying stale data.
    //
    // This happened on 2026-08-09: the avatar's control loop died at
    // t=404.7 s and the operator kept commanding it for another 2.5 s with
    // data_msg_rate_hz pinned at 200.1 and data_latency_ms at 50.4.
    //
    // Consumers should compute staleness as (now - sample_time_ns) and alarm
    // on it. Zero means the sender predates this field; treat as unknown
    // rather than as "very stale".
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

// per-device bookkeeping entry on the avatar side
struct DeviceRecord {
    bool active = true;
};

template<int N>
Eigen::Matrix<double, N, 1> yamlToVector(const YAML::Node& node) {
    auto vec = node.as<std::vector<double>>();
    return Eigen::Map<const Eigen::Matrix<double, N, 1>>(vec.data());
}