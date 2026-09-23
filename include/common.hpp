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

// Which command channel is allowed to move an arm. PER ARM, deliberately: the
// clutch is already per-arm at the interface, so the operator can correct one
// hand while the policy keeps driving the other.
//
// The avatar owns this. It is the only process both the VR interface and the
// orchestrator talk to, and the only place transmission_ and
// transmission_absolute_ physically converge, so it is the only place that can
// ENFORCE a mutex rather than politely request one.
//
// Values match EControlAuthority (teleop_vr_interface Public/Shared/AvatarTypes.h)
// and the `authority` column in arm.csv, so one vocabulary runs end to end.
//
// UNSET is not a fourth mode, it is the absence of the feature: until someone
// sends an authority_request for this arm, both channels behave exactly as they
// did before authority existed. Without it, switching enforcement on would
// break every existing autonomous and playback run -- they never ask for
// authority, so a HUMAN default would gate the policy out and a POLICY default
// would gate the operator out. The first request latches enforcement on for the
// session and it never returns to UNSET, because returning would re-open both
// gates at exactly the moment something has gone wrong.
enum class CommandAuthority : uint8_t {
    POLICY = 0,   // transmission_absolute_ only (orchestrator)
    HUMAN  = 1,   // transmission_ only (VR interface)
    HOLD   = 2,   // neither; the arm holds its last target
    UNSET  = 255  // unclaimed; both channels behave as they did before
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
    // Operator clutch, 1 = CLUTCHED. While clutched the operator's hand is
    // decoupled from the setpoint: the retarget origin follows the hand, the
    // commanded pose stops advancing, and the operator is repositioning their
    // arm rather than demonstrating anything. Commands are still sent, so the
    // sequence numbering stays continuous through a clutch.
    //
    // On the wire purely so the avatar can log it. Reconstructing it later
    // means joining the avatar's per-episode files against one continuous
    // operator-side stream recorded on another continent's clock, which is
    // not a join worth trusting for training data.
    //
    // Senders that are never clutched (orchestrator playback, the autonomous
    // policy path) send 0.
    uint8_t clutch;
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
    // header.sequence of the most recent ArmCommandMsg this arm actually
    // CONSUMED (not merely received). Echoed back so the operator side can
    // measure round-trip latency against its own clock.
    //
    // Every other latency figure in this system is a difference between
    // timestamps taken on two different machines, so it carries the hosts'
    // clock offset as an unknown additive error -- and that offset is the same
    // order of magnitude as the latency being measured. Echoing the sequence
    // lets the interface compute
    //     rtt = receive_time - send_time(applied_cmd_sequence)
    // entirely on one clock, where the offset cancels exactly.
    //
    // 0 = no command consumed yet (e.g. not ENGAGED). Treat as unknown rather
    // than as zero latency.
    uint32_t applied_cmd_sequence;
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

// The wire contract, not a description of the structs above. These same five
// numbers appear in teleop_vr_interface's Public/Shared/protocol.hpp and in
// teleop_orchestrator/live/wire.py; changing a struct in one place without the
// other two is the 2026-09-14 failure. UdpStream::runRecv accepts a packet only
// when n == sizeof(TRecv), so a stale sender is dropped on size and the device
// reads as absent rather than misconfigured.
//
// This copy had no asserts at all until the clutch field was added, which is
// why it was the one place a field could move silently.
static_assert(sizeof(MsgHeader)      == 23,  "MsgHeader size mismatch");
static_assert(sizeof(ArmCommandMsg)  == 56,  "ArmCommandMsg size mismatch");
static_assert(sizeof(ArmStateMsg)    == 117, "ArmStateMsg size mismatch");
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