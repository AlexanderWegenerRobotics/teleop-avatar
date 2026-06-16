#pragma once

#include <array>
#include <cstdint>
#include <string>
#include <vector>

#include <Eigen/Geometry>
#include <msgpack.hpp>

struct CameraIntrinsics {
    float fx, fy;
    float cx, cy;
    int   width, height;
};

struct CameraExtrinsics {
    Eigen::Vector3d    position;
    Eigen::Quaterniond orientation;

    Eigen::Isometry3d asIsometry() const {
        Eigen::Isometry3d T = Eigen::Isometry3d::Identity();
        T.translation() = position;
        T.linear()      = orientation.toRotationMatrix();
        return T;
    }
};

enum class SlotType : uint8_t {
    EE_LEFT    = 0,
    EE_RIGHT   = 1,
    PICK_OBJ   = 2,
    PLACE_POSE = 3
};

struct ObjectSlot {
    std::string       name;
    SlotType          type;
    Eigen::Isometry3d T_world;
    Eigen::Vector3d   half_extents = Eigen::Vector3d::Zero();
};

struct StateSnapshot {
    uint64_t frame_id      = 0;
    uint64_t timestamp_ns  = 0;

    Eigen::Isometry3d T_ee_left;
    Eigen::Isometry3d T_ee_right;
    float gripper_left  = 0.0f;
    float gripper_right = 0.0f;

    float head_pan  = 0.0f;
    float head_tilt = 0.0f;

    std::vector<ObjectSlot> slots;   // max 10: 2 EEF + up to 4 pick + 4 place
};

struct GazeSampleMsg {
    uint64_t frame_id             = 0;
    float    gaze_px_x            = 0.0f;
    float    gaze_px_y            = 0.0f;
    uint64_t timestamp_ns         = 0;  // operator-side capture time (sent over network)
    uint64_t timestamp_arrival_ns = 0;  // avatar-side receive time (stamped locally, not serialised)

    MSGPACK_DEFINE_MAP(frame_id, gaze_px_x, gaze_px_y, timestamp_ns)
};

struct IntentionSample {
    uint64_t frame_id             = 0;
    uint64_t timestamp_ns         = 0;  // operator-side gaze capture time
    uint64_t timestamp_arrival_ns = 0;  // avatar-side gaze receive time
    bool     gaze_valid           = false;
    float    gaze_px_x    = 0.0f;
    float    gaze_px_y    = 0.0f;
    std::vector<float>       slot_belief;
    std::vector<uint8_t>     slot_types;
    std::vector<std::string> slot_names;
    std::vector<float>       slot_px_u;      // projected center pixel u per slot (-1 if behind camera)
    std::vector<float>       slot_px_v;      // projected center pixel v per slot (-1 if behind camera)
    Eigen::Isometry3d T_ee_left;
    Eigen::Isometry3d T_ee_right;
    float gripper_left  = 0.0f;
    float gripper_right = 0.0f;
    std::vector<float> slot_distances;
};

struct IntentionLogEntry {
    double   time;
    uint64_t frame_id;
    uint64_t timestamp_ns;          // operator-side gaze capture time
    uint64_t timestamp_arrival_ns;  // avatar-side gaze receive time
    uint8_t  gaze_valid;
    float    gaze_px_x;
    float    gaze_px_y;
    std::array<float,       11> slot_belief;
    std::array<uint8_t,     10> slot_types;
    std::array<std::string, 10> slot_names;  // object/ee name per slot (episode-config order)
    std::array<float,       10> slot_px_u;   // projected center pixel u per slot
    std::array<float,       10> slot_px_v;   // projected center pixel v per slot
    std::array<double, 3> ee_left_pos;
    std::array<double, 3> ee_right_pos;
    float gripper_left;
    float gripper_right;
    std::array<float, 20> slot_distances;
    uint8_t n_slots;
};

inline std::string intentionLogHeader() {
    std::string h = "time;frame_id;timestamp_ns;timestamp_arrival_ns;gaze_valid;gaze_px_x;gaze_px_y;";
    for (int i = 0; i < 11; ++i) h += "slot_belief_" + std::to_string(i) + ";";
    for (int i = 0; i < 10; ++i) h += "slot_type_"   + std::to_string(i) + ";";
    for (int i = 0; i < 10; ++i) h += "slot_name_"   + std::to_string(i) + ";";
    for (int i = 0; i < 10; ++i) h += "slot_px_u_"   + std::to_string(i) + ";";
    for (int i = 0; i < 10; ++i) h += "slot_px_v_"   + std::to_string(i) + ";";
    h += "ee_left_x;ee_left_y;ee_left_z;";
    h += "ee_right_x;ee_right_y;ee_right_z;";
    h += "gripper_left;gripper_right;";
    for (int i = 0; i < 20; ++i) h += "slot_dist_" + std::to_string(i) + ";";
    h += "n_slots\n";
    return h;
}

inline std::string intentionLogRow(const IntentionLogEntry& e) {
    std::string r = std::to_string(e.time)                 + ";";
    r += std::to_string(e.frame_id)                        + ";";
    r += std::to_string(e.timestamp_ns)                    + ";";
    r += std::to_string(e.timestamp_arrival_ns)            + ";";
    r += std::to_string(e.gaze_valid)                      + ";";
    r += std::to_string(e.gaze_px_x)              + ";";
    r += std::to_string(e.gaze_px_y)              + ";";
    for (auto v : e.slot_belief)    r += std::to_string(v) + ";";
    for (auto v : e.slot_types)     r += std::to_string(v) + ";";
    for (auto& s : e.slot_names)    r += s + ";";
    for (auto v : e.slot_px_u)      r += std::to_string(v) + ";";
    for (auto v : e.slot_px_v)      r += std::to_string(v) + ";";
    for (auto v : e.ee_left_pos)    r += std::to_string(v) + ";";
    for (auto v : e.ee_right_pos)   r += std::to_string(v) + ";";
    r += std::to_string(e.gripper_left)           + ";";
    r += std::to_string(e.gripper_right)          + ";";
    for (auto v : e.slot_distances) r += std::to_string(v) + ";";
    r += std::to_string(e.n_slots) + "\n";
    return r;
}
