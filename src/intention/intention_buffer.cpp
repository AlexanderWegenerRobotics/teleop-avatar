#include "intention/intention_buffer.hpp"

#include <cmath>
#include <iostream>
#include <numeric>

// body -> OpenCV: body X -> CV Z, body Y -> CV -X, body Z -> CV -Y
static const Eigen::Matrix3d R_body2cv = (Eigen::Matrix3d() <<
     0.0, -1.0,  0.0,
     0.0,  0.0, -1.0,
     1.0,  0.0,  0.0).finished();

static const Eigen::Matrix3d R_mjcam2cv = (Eigen::Matrix3d() <<
     1.0,  0.0,  0.0,
     0.0, -1.0,  0.0,
     0.0,  0.0, -1.0).finished();

IntentionBuffer::IntentionBuffer(const IntentionBufferConfig& config)
    : config_(config)
{}

void IntentionBuffer::setCallback(SampleCallback cb) {
    std::lock_guard<std::mutex> lock(cb_mtx_);
    callback_ = std::move(cb);
}

void IntentionBuffer::snapshot(const StateSnapshot& state) {
    std::lock_guard<std::mutex> lock(buf_mtx_);
    buffer_.push_back(state);
    if (static_cast<int>(buffer_.size()) > config_.max_frames)
        buffer_.pop_front();
}

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

        // tilt about +Y, pan world->head needs -q_pan (passive)
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

        const float gaze_u_cam = gaze.gaze_px_x * config_.gaze_scale_u;
        const float gaze_v_cam = gaze.gaze_px_y * config_.gaze_scale_v;

        // stored gaze is a normalized ray coord (u-cx)/fx, (v-cy)/fy, not pixels
        sample.gaze_px_x = (gaze_u_cam - config_.intrinsics.cx) / config_.intrinsics.fx;
        sample.gaze_px_y = (gaze_v_cam - config_.intrinsics.cy) / config_.intrinsics.fy;

        // empty on first packet -> uniform prior
        std::vector<float> prev;
        {
            std::lock_guard<std::mutex> bl(belief_mtx_);
            prev = prev_belief_;
        }

        sample.slot_belief = computeBelief(
            gaze_u_cam, gaze_v_cam,
            kernels,
            R_CH,
            config_.head_position,
            prev);

        {
            std::lock_guard<std::mutex> bl(belief_mtx_);
            prev_belief_ = sample.slot_belief;
        }

        // normalized ray coords; -1000 = not projected (-1 is a valid ray coord on wide FOV)
        constexpr float kNotProjected = -1000.0f;
        for (const auto& k : kernels) {
            float u = kNotProjected, v = kNotProjected;
            if (projectToImage(k.center, R_CH, config_.head_position, u, v)) {
                sample.slot_px_u.push_back((u - config_.intrinsics.cx) / config_.intrinsics.fx);
                sample.slot_px_v.push_back((v - config_.intrinsics.cy) / config_.intrinsics.fy);
            } else {
                sample.slot_px_u.push_back(kNotProjected);
                sample.slot_px_v.push_back(kNotProjected);
            }
        }

        // interleaved [left_slot0, right_slot0, ...]
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

bool IntentionBuffer::projectToImage(const Eigen::Vector3d& p_world,
                                     const Eigen::Matrix3d& R_CH,
                                     const Eigen::Vector3d& t_WH,
                                     float& u, float& v) const {
    Eigen::Vector3d p_CV;
    if (config_.static_camera) {
        p_CV = R_mjcam2cv * (config_.R_world_cam.transpose() * (p_world - config_.cam_pos_world));
    } else {
        // same chain as ProjectWorldToScreen: world -> head -> cam offset -> OpenCV axes
        Eigen::Vector3d p_H      = R_CH * (p_world - t_WH);
        Eigen::Vector3d p_C_body = p_H - config_.extrinsics.position;
        p_CV                     = R_body2cv * p_C_body;
    }

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
    const Eigen::Vector3d& t_WH,
    const std::vector<float>& prev_belief) const
{
    // slots: [ee_left, ee_right, objs..., bins..., null]; first 2 use rho_ee, rest rho_tgt
    int N  = static_cast<int>(kernels.size());
    int NB = N + 1;  // +1 null slot

    // raw likelihoods
    std::vector<float> likelihood(NB, 0.0f);
    for (int i = 0; i < N; ++i)
        likelihood[i] = slotLikelihood(gaze_u, gaze_v, kernels[i], R_CH, t_WH);
    likelihood[N] = 0.1f;  // fixed null likelihood

    // temperature scaling
    const float inv_T = 1.0f / config_.belief_temperature;
    if (config_.belief_temperature != 1.0f) {
        for (auto& l : likelihood)
            l = std::pow(l, inv_T);
    }

    // sticky prior: rho * prev + (1 - rho) / NB
    const float uniform = 1.0f / static_cast<float>(NB);
    std::vector<float> prior(NB);

    bool has_prev = (static_cast<int>(prev_belief.size()) == NB);
    for (int i = 0; i < NB; ++i) {
        float rho  = (i < 2) ? config_.rho_ee : config_.rho_tgt;
        float prev = has_prev ? prev_belief[i] : uniform;
        prior[i]   = rho * prev + (1.0f - rho) * uniform;
    }

    // posterior = prior * likelihood, normalized
    std::vector<float> belief(NB);
    for (int i = 0; i < NB; ++i)
        belief[i] = prior[i] * likelihood[i];

    float total = std::accumulate(belief.begin(), belief.end(), 0.0f);
    if (total > 1e-6f)
        for (auto& b : belief) b /= total;
    else
        belief = prior;  // degenerate, fall back to prior

    return belief;
}
