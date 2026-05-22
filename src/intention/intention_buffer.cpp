#include "intention/intention_buffer.hpp"

#include <cmath>
#include <iostream>
#include <numeric>

// Remaps from robot body frame to OpenCV frame:
//   body X → CV Z (forward), body Y → CV -X (left), body Z → CV -Y (down)
static const Eigen::Matrix3d R_body2cv = (Eigen::Matrix3d() <<
     0.0, -1.0,  0.0,
     0.0,  0.0, -1.0,
     1.0,  0.0,  0.0).finished();

// ---------------------------------------------------------------------------
// Constructor
// ---------------------------------------------------------------------------

IntentionBuffer::IntentionBuffer(const IntentionBufferConfig& config)
    : config_(config)
{}

void IntentionBuffer::setCallback(SampleCallback cb) {
    std::lock_guard<std::mutex> lock(cb_mtx_);
    callback_ = std::move(cb);
}

// ---------------------------------------------------------------------------
// Snapshot — called at frame-capture time
// ---------------------------------------------------------------------------

void IntentionBuffer::snapshot(const StateSnapshot& state) {
    std::lock_guard<std::mutex> lock(buf_mtx_);
    buffer_.push_back(state);
    if (static_cast<int>(buffer_.size()) > config_.max_frames)
        buffer_.pop_front();
}

// ---------------------------------------------------------------------------
// Gaze fusion — called when a gaze packet arrives from the operator
// ---------------------------------------------------------------------------

void IntentionBuffer::fuseGaze(const GazeSampleMsg& gaze) {
    auto snap_opt = lookup(gaze.frame_id);
    if (!snap_opt)
        snap_opt = interpolate(gaze.frame_id);

    IntentionSample sample;
    sample.frame_id     = gaze.frame_id;
    sample.timestamp_ns = gaze.timestamp_ns;

    if (!snap_opt) {
        sample.gaze_valid = false;
        //std::cerr << "[IntentionBuffer] frame_id " << gaze.frame_id << " not in buffer\n";
    } else {
        const StateSnapshot& snap = *snap_opt;

        sample.gaze_valid    = true;
        sample.T_ee_left     = snap.T_ee_left;
        sample.T_ee_right    = snap.T_ee_right;
        sample.gripper_left  = snap.gripper_left;
        sample.gripper_right = snap.gripper_right;

        // Head rotation — tilt * pan matches ProjectWorldToScreen (R_CH = R_tilt * R_pan)
        Eigen::Matrix3d R_pan  = Eigen::AngleAxisd(snap.head_pan,  Eigen::Vector3d::UnitZ()).toRotationMatrix();
        Eigen::Matrix3d R_tilt = Eigen::AngleAxisd(snap.head_tilt, Eigen::Vector3d::UnitY()).toRotationMatrix();
        Eigen::Matrix3d R_CH   = R_tilt * R_pan;

        // Build slot list: EEFs first (from snap.T_ee_*), then pick/place slots
        std::vector<Eigen::Vector3d> all_positions;
        all_positions.push_back(snap.T_ee_left.translation());
        all_positions.push_back(snap.T_ee_right.translation());
        sample.slot_types.push_back(static_cast<uint8_t>(SlotType::EE_LEFT));
        sample.slot_types.push_back(static_cast<uint8_t>(SlotType::EE_RIGHT));
        sample.slot_names.push_back("ee_left");
        sample.slot_names.push_back("ee_right");

        for (const auto& slot : snap.slots) {
            all_positions.push_back(slot.T_world.translation());
            sample.slot_types.push_back(static_cast<uint8_t>(slot.type));
            sample.slot_names.push_back(slot.name);
        }

        sample.slot_belief = computeBelief(
            gaze.gaze_px_x, gaze.gaze_px_y,
            all_positions,
            R_CH,
            config_.head_position);

        // Distances: 2 EEFs x N pick/place slots, interleaved [left_slot0, right_slot0, ...]
        for (const auto& slot : snap.slots) {
            float dl = static_cast<float>((snap.T_ee_left.translation()  - slot.T_world.translation()).norm());
            float dr = static_cast<float>((snap.T_ee_right.translation() - slot.T_world.translation()).norm());
            sample.slot_distances.push_back(dl);
            sample.slot_distances.push_back(dr);
        }
    }

    SampleCallback cb;
    {
        std::lock_guard<std::mutex> lock(cb_mtx_);
        cb = callback_;
    }
    if (cb) cb(sample);
}

// ---------------------------------------------------------------------------
// Buffer lookup helpers
// ---------------------------------------------------------------------------

std::optional<StateSnapshot> IntentionBuffer::lookup(uint64_t frame_id) const {
    std::lock_guard<std::mutex> lock(buf_mtx_);
    for (auto it = buffer_.rbegin(); it != buffer_.rend(); ++it) {
        if (it->frame_id == frame_id)
            return *it;
    }
    return std::nullopt;
}

std::optional<StateSnapshot> IntentionBuffer::interpolate(uint64_t frame_id) const {
    std::lock_guard<std::mutex> lock(buf_mtx_);
    if (buffer_.size() < 2) return std::nullopt;

    const StateSnapshot* before = nullptr;
    const StateSnapshot* after  = nullptr;

    for (const auto& s : buffer_) {
        if (s.frame_id <= frame_id) before = &s;
        if (s.frame_id >= frame_id && !after) after = &s;
    }

    if (!before || !after) return std::nullopt;
    if (before->frame_id == after->frame_id) return *before;

    double t = static_cast<double>(frame_id - before->frame_id) /
               static_cast<double>(after->frame_id - before->frame_id);

    auto lerpIso = [&](const Eigen::Isometry3d& a, const Eigen::Isometry3d& b) {
        Eigen::Isometry3d out = Eigen::Isometry3d::Identity();
        out.translation() = a.translation() + t * (b.translation() - a.translation());
        out.linear()      = Eigen::Quaterniond(a.rotation())
                                .slerp(t, Eigen::Quaterniond(b.rotation()))
                                .toRotationMatrix();
        return out;
    };

    StateSnapshot interp;
    interp.frame_id      = frame_id;
    interp.timestamp_ns  = before->timestamp_ns +
        static_cast<uint64_t>(t * static_cast<double>(after->timestamp_ns - before->timestamp_ns));
    interp.T_ee_left     = lerpIso(before->T_ee_left,  after->T_ee_left);
    interp.T_ee_right    = lerpIso(before->T_ee_right, after->T_ee_right);
    interp.gripper_left  = static_cast<float>(before->gripper_left  + t * (after->gripper_left  - before->gripper_left));
    interp.gripper_right = static_cast<float>(before->gripper_right + t * (after->gripper_right - before->gripper_right));
    interp.head_pan      = static_cast<float>(before->head_pan  + t * (after->head_pan  - before->head_pan));
    interp.head_tilt     = static_cast<float>(before->head_tilt + t * (after->head_tilt - before->head_tilt));

    for (const auto& slot_b : before->slots) {
        ObjectSlot os;
        os.name = slot_b.name;
        os.type = slot_b.type;
        auto it = std::find_if(after->slots.begin(), after->slots.end(),
            [&](const ObjectSlot& s){ return s.name == slot_b.name; });
        os.T_world = (it != after->slots.end()) ? lerpIso(slot_b.T_world, it->T_world) : slot_b.T_world;
        interp.slots.push_back(std::move(os));
    }

    return interp;
}

// ---------------------------------------------------------------------------
// Projection + belief
// ---------------------------------------------------------------------------

bool IntentionBuffer::projectToImage(const Eigen::Vector3d& p_world,
                                     const Eigen::Matrix3d& R_CH,
                                     const Eigen::Vector3d& t_WH,
                                     float& u, float& v) const {
    // Match ProjectWorldToScreen chain exactly:
    //   p_H      = R_CH * (p_W - t_WH)     — into head frame
    //   p_C_body = p_H - t_HC              — subtract cam offset in head frame
    //   p_CV     = R_body2cv * p_C_body    — remap to OpenCV axes
    Eigen::Vector3d p_H      = R_CH * (p_world - t_WH);
    Eigen::Vector3d p_C_body = p_H - config_.extrinsics.position;
    Eigen::Vector3d p_CV     = R_body2cv * p_C_body;

    if (p_CV.z() <= 0.0) return false;

    u = static_cast<float>(config_.intrinsics.fx * p_CV.x() / p_CV.z() + config_.intrinsics.cx);
    v = static_cast<float>(config_.intrinsics.fy * p_CV.y() / p_CV.z() + config_.intrinsics.cy);
    return true;
}

std::vector<float> IntentionBuffer::computeBelief(
    float gaze_u, float gaze_v,
    const std::vector<Eigen::Vector3d>& p_world,
    const Eigen::Matrix3d& R_CH,
    const Eigen::Vector3d& t_WH) const
{
    int N = static_cast<int>(p_world.size());
    std::vector<float> belief(N + 1, 0.0f);

    float sigma2 = config_.gaze_sigma_px * config_.gaze_sigma_px;

    for (int i = 0; i < N; ++i) {
        float u, v;
        if (!projectToImage(p_world[i], R_CH, t_WH, u, v)) {
            belief[i] = 0.0f;
            continue;
        }
        float du = gaze_u - u;
        float dv = gaze_v - v;
        belief[i] = std::exp(-(du * du + dv * dv) / (2.0f * sigma2));
    }

    //belief[N] = 1.0f / static_cast<float>(N + 1);
    belief[N] = 0.1f;

    float total = std::accumulate(belief.begin(), belief.end(), 0.0f);
    if (total > 1e-6f)
        for (auto& b : belief) b /= total;

    return belief;
}
