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
    sample.frame_id             = gaze.frame_id;
    sample.timestamp_ns         = gaze.timestamp_ns;
    sample.timestamp_arrival_ns = gaze.timestamp_arrival_ns;

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

        // Head rotation: tilt around +Y (R_Y(q_tilt)), pan world→head requires -q_pan (passive rotation)
        Eigen::Matrix3d R_pan  = Eigen::AngleAxisd(-snap.head_pan, Eigen::Vector3d::UnitZ()).toRotationMatrix();
        Eigen::Matrix3d R_tilt = Eigen::AngleAxisd(snap.head_tilt,  Eigen::Vector3d::UnitY()).toRotationMatrix();
        Eigen::Matrix3d R_CH   = R_tilt * R_pan;

        std::vector<SlotKernel> kernels;
        kernels.push_back({snap.T_ee_left.translation()});
        kernels.push_back({snap.T_ee_right.translation()});
        sample.slot_types.push_back(static_cast<uint8_t>(SlotType::EE_LEFT));
        sample.slot_types.push_back(static_cast<uint8_t>(SlotType::EE_RIGHT));
        sample.slot_names.push_back("ee_left");
        sample.slot_names.push_back("ee_right");

        for (const auto& slot : snap.slots) {
            kernels.push_back({slot.T_world.translation(), slot.half_extents});
            sample.slot_types.push_back(static_cast<uint8_t>(slot.type));
            sample.slot_names.push_back(slot.name);
        }

        // gaze_px_x comes in as GazeUV.X * 2560 (full stereo width).
        // projectToImage works in single-camera coordinates (cx=640, width=1280).
        // Halve u to align coordinate spaces; v height is the same in both frames.
        const float gaze_u_cam = gaze.gaze_px_x * 0.5f;
        const float gaze_v_cam = gaze.gaze_px_y;

        sample.gaze_px_x = gaze.gaze_px_x;
        sample.gaze_px_y = gaze.gaze_px_y;

        sample.slot_belief = computeBelief(
            gaze_u_cam, gaze_v_cam,
            kernels,
            R_CH,
            config_.head_position);

        for (const auto& k : kernels) {
            float u = -1.0f, v = -1.0f;
            projectToImage(k.center, R_CH, config_.head_position, u, v);
            sample.slot_px_u.push_back(u);
            sample.slot_px_v.push_back(v);
        }

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

float IntentionBuffer::slotLikelihood(float gaze_u, float gaze_v,
                                      const SlotKernel& kernel,
                                      const Eigen::Matrix3d& R_CH,
                                      const Eigen::Vector3d& t_WH) const
{
    float sigma2 = config_.gaze_sigma_px * config_.gaze_sigma_px;

    auto gaussian = [&](const Eigen::Vector3d& p) -> float {
        float u, v;
        if (!projectToImage(p, R_CH, t_WH, u, v)) return 0.0f;
        float du = gaze_u - u;
        float dv = gaze_v - v;
        return std::exp(-(du * du + dv * dv) / (2.0f * sigma2));
    };

    float best = gaussian(kernel.center);

    const Eigen::Vector3d& h = kernel.half_extents;
    if (h.squaredNorm() < 1e-8) return best;

    const double dx = h.x();
    const double dy = h.y();
    const double dz = h.z();

    const std::array<Eigen::Vector3d, 8> corners = {{
        kernel.center + Eigen::Vector3d( dx,  dy,  dz),
        kernel.center + Eigen::Vector3d(-dx,  dy,  dz),
        kernel.center + Eigen::Vector3d( dx, -dy,  dz),
        kernel.center + Eigen::Vector3d(-dx, -dy,  dz),
        kernel.center + Eigen::Vector3d( dx,  dy, -dz),
        kernel.center + Eigen::Vector3d(-dx,  dy, -dz),
        kernel.center + Eigen::Vector3d( dx, -dy, -dz),
        kernel.center + Eigen::Vector3d(-dx, -dy, -dz),
    }};

    for (const auto& c : corners)
        best = std::max(best, gaussian(c));

    return best;
}

std::vector<float> IntentionBuffer::computeBelief(
    float gaze_u, float gaze_v,
    const std::vector<SlotKernel>& kernels,
    const Eigen::Matrix3d& R_CH,
    const Eigen::Vector3d& t_WH) const
{
    int N = static_cast<int>(kernels.size());
    // belief has N+1 entries: indices 0..(N-1) are real slots (EE_LEFT, EE_RIGHT,
    // then PICK_OBJ slots, then PLACE_POSE/bin slots); index N is the null/no-target
    // slot (operator not looking at any tracked object).  The intent model's argmax
    // should land on the PICK_OBJ range (slots 2..N_objects+1 in the default 4-parcel
    // + 2-bin scene: indices 2–5 for parcels, 6–7 for bins, 8 = null).
    std::vector<float> belief(N + 1, 0.0f);

    for (int i = 0; i < N; ++i)
        belief[i] = slotLikelihood(gaze_u, gaze_v, kernels[i], R_CH, t_WH);

    // Null/no-target prior — kept in raw likelihood space before temperature scaling.
    belief[N] = 0.1f;

    // Temperature scaling: raise each raw likelihood to 1/T before normalising.
    // T=1  → original behaviour (saturates to argmax≈1.0 when gaze is on-target).
    // T>1  → softer posterior; graded uncertainty during early reach, sharpens on commit.
    const float inv_T = 1.0f / config_.belief_temperature;
    if (config_.belief_temperature != 1.0f) {
        for (auto& b : belief)
            b = std::pow(b, inv_T);
    }

    float total = std::accumulate(belief.begin(), belief.end(), 0.0f);
    if (total > 1e-6f)
        for (auto& b : belief) b /= total;

    return belief;
}
