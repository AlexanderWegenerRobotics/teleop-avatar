#pragma once

#include <thread>
#include <atomic>
#include <mutex>
#include <random>
#include <algorithm>
#include <cmath>
#include <vector>
#include <string>

#include <mujoco/mujoco.h>
#include <GLFW/glfw3.h>
#include <mujoco/mjvisualize.h>
#include <mujoco/mjrender.h>
#include <Eigen/Dense>
#include <Eigen/Geometry>

#include "pipeline/shared_memory.hpp"
#include "pipeline/lens.hpp"
#include "sim_env/scene_builder.hpp"
#include "intention/intention_sample.hpp"
#include "twin/role.hpp"
#include "sim_env/wrench_truth.hpp"

struct LightingConfig {
    float main_pos[3]           = {1.3f,  0.0f,  1.85f};
    float main_target[3]        = {0.8f,  0.0f,  0.72f};
    float main_diffuse[3]       = {0.35f, 0.35f, 0.35f};
    float main_specular[3]      = {0.09f, 0.09f, 0.09f};
    float fill_diffuse[3]       = {0.2f,  0.2f,  0.2f};
    float headlight_diffuse[3]  = {0.15f, 0.15f, 0.15f};
    float headlight_ambient[3]  = {0.55f, 0.55f, 0.55f};
    float main_cutoff           = 30.0f;
    float main_exponent         = 30.0f;
    float fill_cutoff           = 80.0f;
    float fill_exponent         = 1.0f;

    void randomize(int seed) {
        std::mt19937 rng(static_cast<uint32_t>(seed));
        auto u = [&](float lo, float hi) {
            return std::uniform_real_distribution<float>(lo, hi)(rng);
        };
        constexpr float kDeg = 3.14159265f / 180.0f;

        float dist = u(1.2f, 1.5f);
        float elev = u(45.0f, 65.0f) * kDeg;
        float azim = u(-135.0f, 135.0f) * kDeg;
        main_pos[0] = main_target[0] + dist * std::cos(elev) * std::cos(azim);
        main_pos[1] = main_target[1] + dist * std::cos(elev) * std::sin(azim);
        main_pos[2] = main_target[2] + dist * std::sin(elev);

        float kd   = u(0.25f, 0.45f);
        float warm = u(-1.0f, 1.0f);
        float fd   = u(0.15f, 0.3f);
        float hd   = u(0.1f, 0.2f);
        float ha   = u(0.5f, 0.6f);

        const float fill_pos[3] = {1.2f, 0.5f, 1.0f};
        float fdx = main_target[0] - fill_pos[0];
        float fdy = main_target[1] - fill_pos[1];
        float fdz = main_target[2] - fill_pos[2];
        float fill_up = std::max(0.0f, -fdz / std::sqrt(fdx * fdx + fdy * fdy + fdz * fdz));

        float up = kd * std::sin(elev) + fd * fill_up + hd * 0.7f + ha;
        float k  = std::min(1.0f, 1.4f / up);
        kd *= k; fd *= k; hd *= k; ha *= k;

        auto clamp01 = [](float v) { return std::max(0.0f, std::min(1.0f, v)); };
        main_diffuse[0] = clamp01(kd * (1.0f + 0.15f * warm));
        main_diffuse[1] = clamp01(kd * (1.0f + 0.05f * warm));
        main_diffuse[2] = clamp01(kd * (1.0f - 0.20f * warm));
        for (int i = 0; i < 3; ++i) {
            main_specular[i]     = clamp01(main_diffuse[i] * 0.25f);
            fill_diffuse[i]      = fd;
            headlight_diffuse[i] = hd;
            headlight_ambient[i] = ha;
        }
    }
};

struct DeviceState {
    std::vector<double> q;
    std::vector<double> dq;
    std::vector<double> tau_J;
    std::vector<double> tau_ext;
    // sim time (mjData::time), not wall time; differentiate with this clock only
    double time = 0.0;
};

// sim vs wall clock stats, rtf < 1 = slower than real time (reported, not enforced)
struct SimTimingStats {
    double   sim_seconds        = 0.0;  // steps * opt.timestep
    double   wall_seconds       = 0.0;
    double   rtf                = 0.0;  // sim_seconds / wall_seconds
    uint64_t steps              = 0;
    uint64_t deadline_misses    = 0;    // steps that overran their wall deadline
    double   step_ms_mean       = 0.0;  // mj_step cost alone
    double   step_ms_max        = 0.0;
};


class Simulation {
public:
    // for Twin role the streamer_overlay is applied so avatar and twin can run side by side
    explicit Simulation(const YAML::Node& config, Role role = Role::Avatar);
    ~Simulation();

    void start();
    void stop();
    bool isRunning() const;
    void setCtrl(const std::string& deviceName, const std::vector<double>& values);
    void setGripper(const std::string& deviceName, double value);
    double getGripperWidth(const std::string& deviceName);
    DeviceState getDeviceState(const std::string& deviceName);
    void setDeviceActive(const std::string& deviceName, bool state);
    void setFramePose(const std::string& name, const Eigen::Vector3d& pos, const Eigen::Quaterniond& quat, double z_offset=0.0);
    void setFreeBodyPose(const std::string& bodyName, const Eigen::Vector3d& pos, const Eigen::Quaterniond& quat);
    bool getFreeBodyPose(const std::string& bodyName, Eigen::Vector3d& pos, Eigen::Quaterniond& quat);
    CameraIntrinsics  getCameraIntrinsics(const std::string& cam_name) const;
    CameraExtrinsics  getCameraExtrinsics(const std::string& cam_name) const;
    void setLighting(const LightingConfig& lc);
    void setBodyScale(const std::string& bodyName, double scale);
    uint64_t         getFrameId() const { return stream_frame_count_.load(); }

    // effective timestep after the sim_config override, don't assume 0.001
    double           getTimestep() const { return model ? model->opt.timestep : 0.0; }
    SimTimingStats   getTimingStats() const;

    // no-op when wrench truth is off
    void restartWrenchTruthLoggers(const std::string& folder) {
        if (wrench_truth_) wrench_truth_->restartLoggers(folder);
    }

    // twin/reconciler support; model is immutable after construction so sharing it is safe
    const mjModel* mjModelPtr() const { return model; }

    // writes qpos/qvel deltas (rad, rad/s) directly, bypasses actuators
    void applyJointCorrection(const std::string& deviceName,
                               const std::vector<double>& dq_delta,
                               const std::vector<double>& ddq_delta);

    // ctrl values run_model() applies this tick, same order as q/dq
    std::vector<double> getDeviceCtrl(const std::string& deviceName);

    // replay helpers on a caller-owned scratch mjData, never touch the live data
    void replaySeed(mjData* replay_data, const std::string& deviceName,
                     const std::vector<double>& q, const std::vector<double>& dq);
    void replaySetCtrl(mjData* replay_data, const std::string& deviceName,
                        const std::vector<double>& ctrl);
    void replayAdvance(mjData* replay_data);
    std::vector<double> replayReadQ(mjData* replay_data, const std::string& deviceName) const;
    std::vector<double> replayReadDq(mjData* replay_data, const std::string& deviceName) const;

private:
    mjModel* model = nullptr;
    mjData*  data  = nullptr;

    // MuJoCo ground truth for the momentum observer, null when disabled
    std::unique_ptr<WrenchTruth> wrench_truth_;

    std::vector<DeviceConfig> devices_;
    std::vector<ObjectConfig> objects_;
    std::vector<CameraConfig> cameras_;
    std::unordered_map<std::string, int> mocap_index_;

private:
    void run_model();
    void run_rendering();
    void applyInitialPositions();
    void buildActuatorIndex();
    std::mutex data_mtx;
    std::vector<double> ctrl_buffer_;
    std::mutex          ctrl_mtx_;
    std::unordered_map<std::string, std::vector<int>> actuator_ids_;
    std::unordered_map<std::string, int>              gripper_ids_;
    std::unordered_map<std::string, std::vector<int>> joint_ids_;
    std::unordered_map<std::string, bool> active_devices_;
    // brake for inactive devices: damp until below kBrakeRestVel, then hold pose
    struct BrakeState {
        bool                braked = false;
        std::vector<double> q_hold;
    };
    std::unordered_map<std::string, BrakeState> brake_;
    static constexpr double kBrakeRestVel   = 0.05;  // rad/s, all joints
    static constexpr double kBrakeDampFrac  = 0.30;  // D_i = frac * ctrl_max_i  [Nm s/rad]
    static constexpr double kBrakeStiffFrac = 10.0;  // K_i = frac * ctrl_max_i  [Nm/rad]
    // written by the model thread only
    std::atomic<uint64_t> sim_steps_{0};
    std::atomic<uint64_t> deadline_misses_{0};
    std::atomic<double>   wall_seconds_{0.0};
    std::atomic<double>   step_ns_sum_{0.0};
    std::atomic<double>   step_ns_max_{0.0};

    std::atomic<bool> bModelIsRunning{false};
    std::atomic<bool> bRenderingIsRunning{false};
    std::thread       model_thread;
    std::thread       rendering_thread;

private:
    struct CamEntry { std::string name; int id; };
    struct StreamCamEntry {
        std::string camera_name;
        std::string shm_name;
        int width  = 0;   // 0 = use global stream_width_
        int height = 0;   // 0 = use global stream_height_
        std::vector<double> distortion;   // optional lens model (k1 k2 p1 p2 k3) applied before the shm write
    };
    std::vector<std::unique_ptr<SharedMemoryWriter>> shm_writers_;
    std::vector<RemapTable> stream_lens_;
    std::vector<uint8_t>    lens_pixels_;

    mjvScene    scn_;
    mjvOption   vopt_;
    mjrContext  con_;
    mjData*          snap_[2]    = {nullptr, nullptr};
    std::atomic<int> snap_write_ {0};
    std::atomic<int> snap_read_  {1};

    // mjv_updateScene writes mjData stack pointers, so each renderer copies snap_ into its own mjData.
    // snap_mtx_ is only taken inside data_mtx or alone, no lock-order inversion.
    std::mutex snap_mtx_;
    mjData*    render_data_ = nullptr;   // owned by the rendering thread
    mjData*    stream_data_ = nullptr;   // owned by the streaming thread
    void latchSnapshot(mjData* dst);
    int  render_fps_ = 20;
    void buildCameraList();
    void initRendering();
    void renderFrame();
    void swapSnapshots();

public:
    GLFWwindow* window_          = nullptr;
    GLFWwindow* offscreen_window_ = nullptr;
    bool        render_enabled_  = false;
    bool        shm_enabled_     = false;
    bool        stereo_          = false;
    std::vector<StreamCamEntry> stream_cameras_;
    std::vector<CamEntry> render_cams_;

private:
    mjvScene    stream_scn_;
    mjvOption   stream_vopt_;
    mjrContext  stream_con_;
    int         stream_width_  = 1280;
    int         stream_height_ = 720;
    int         stream_fps_    = 30;
    std::thread stream_thread_;
    std::atomic<bool>     bStreamingIsRunning{false};
    std::atomic<uint64_t> stream_frame_count_{0};

    void run_streaming();
    void initOffscreenStreaming();
    void renderStreamFrame();

    // original geom/inertial values, filled on first setBodyScale per body
    struct BodyScaleCache {
        mjtNum             original_mass;
        mjtNum             original_inertia[3];
        std::vector<int>   geom_ids;
        std::vector<std::array<mjtNum, 3>> original_geom_size;
        std::vector<std::array<mjtNum, 3>> original_geom_pos;
        // bounding volumes are not refreshed by MuJoCo, must be scaled too or contacts get culled
        std::vector<mjtNum>                original_geom_rbound;
        std::vector<std::array<mjtNum, 6>> original_geom_aabb;   // center[3], half-size[3]
        int                                bvh_adr = -1;
        std::vector<std::array<mjtNum, 6>> original_bvh_aabb;    // body BVH nodes
    };
    std::unordered_map<std::string, BodyScaleCache> body_scale_cache_;
};