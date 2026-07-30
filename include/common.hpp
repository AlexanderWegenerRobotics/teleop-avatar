#pragma once

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