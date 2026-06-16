#pragma once

#include <deque>
#include <functional>
#include <mutex>
#include <optional>
#include <string>
#include <vector>

#include "intention/intention_sample.hpp"

struct IntentionBufferConfig {
    int    max_frames    = 300;
    float  gaze_sigma_px = 60.0f;
    float  belief_temperature = 1.0f;
    CameraIntrinsics  intrinsics;
    CameraExtrinsics  extrinsics;
    Eigen::Vector3d head_position = Eigen::Vector3d(0.0, 0.0, 1.844);  // tilt joint: base_pose(1.704)+link1(0.08)+link2(0.06)
};

class IntentionBuffer {
public:
    using SampleCallback = std::function<void(const IntentionSample&)>;

    explicit IntentionBuffer(const IntentionBufferConfig& config);
    void snapshot(const StateSnapshot& state);
    void fuseGaze(const GazeSampleMsg& gaze);
    void setCallback(SampleCallback cb);

private:
    std::optional<StateSnapshot> lookup(uint64_t frame_id) const;
    std::optional<StateSnapshot> interpolate(uint64_t frame_id) const;

    bool projectToImage(const Eigen::Vector3d& p_world,
                    const Eigen::Matrix3d& R_CH,
                    const Eigen::Vector3d& t_WH,
                    float& u, float& v) const;

    struct SlotKernel {
        Eigen::Vector3d center;
        Eigen::Vector3d half_extents = Eigen::Vector3d::Zero();
    };

    float slotLikelihood(float gaze_u, float gaze_v,
                         const SlotKernel& kernel,
                         const Eigen::Matrix3d& R_CH,
                         const Eigen::Vector3d& t_WH) const;

    std::vector<float> computeBelief(float gaze_u, float gaze_v,
                                    const std::vector<SlotKernel>& kernels,
                                    const Eigen::Matrix3d& R_CH,
                                    const Eigen::Vector3d& t_WH) const;

    IntentionBufferConfig config_;

    mutable std::mutex   buf_mtx_;
    std::deque<StateSnapshot> buffer_;

    std::mutex     cb_mtx_;
    SampleCallback callback_;
};
