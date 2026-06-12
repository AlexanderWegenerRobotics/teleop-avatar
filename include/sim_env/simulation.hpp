#pragma once

#include <thread>
#include <atomic>
#include <mutex>
#include <random>
#include <vector>
#include <string>

#include <mujoco/mujoco.h>
#include <GLFW/glfw3.h>
#include <mujoco/mjvisualize.h>
#include <mujoco/mjrender.h>
#include <Eigen/Dense>
#include <Eigen/Geometry>

#include "pipeline/shared_memory.hpp"
#include "sim_env/scene_builder.hpp"
#include "intention/intention_sample.hpp"

struct LightingConfig {
    float main_pos[3]           = {0.5f,  0.0f,  1.8f};
    float main_diffuse[3]       = {0.8f,  0.8f,  0.8f};
    float main_specular[3]      = {0.2f,  0.2f,  0.2f};
    float fill_diffuse[3]       = {0.25f, 0.25f, 0.25f};
    float headlight_diffuse[3]  = {0.4f,  0.4f,  0.4f};
    float headlight_ambient[3]  = {0.25f, 0.25f, 0.25f};

    void randomize(int seed) {
        std::mt19937 rng(static_cast<uint32_t>(seed));
        auto u = [&](float lo, float hi) {
            return std::uniform_real_distribution<float>(lo, hi)(rng);
        };

        // Key light: vary x/y position, keep height fixed for consistent shadow direction
        main_pos[0] = u(0.3f, 0.8f);
        main_pos[1] = u(-0.3f, 0.3f);
        main_pos[2] = 1.8f;

        float kd = u(0.6f, 1.4f);
        float tint = u(-0.05f, 0.05f);  // subtle warm/cool shift
        main_diffuse[0] = kd + tint;
        main_diffuse[1] = kd;
        main_diffuse[2] = kd - tint;

        float ks = u(0.1f, 0.3f);
        main_specular[0] = main_specular[1] = main_specular[2] = ks;

        float fd = u(0.1f, 0.55f);
        fill_diffuse[0] = fill_diffuse[1] = fill_diffuse[2] = fd;

        float hd = u(0.2f, 0.7f);
        headlight_diffuse[0] = headlight_diffuse[1] = headlight_diffuse[2] = hd;

        float ha = u(0.15f, 0.5f);
        headlight_ambient[0] = headlight_ambient[1] = headlight_ambient[2] = ha;
    }
};

struct DeviceState {
    std::vector<double> q;
    std::vector<double> dq;
    std::vector<double> tau_J;
    std::vector<double> tau_ext;
};


class Simulation {
public:
    explicit Simulation(const YAML::Node& config);
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

private:
    mjModel* model = nullptr;
    mjData*  data  = nullptr;

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
    };
    std::vector<std::unique_ptr<SharedMemoryWriter>> shm_writers_;

    mjvScene    scn_;
    mjvOption   vopt_;
    mjrContext  con_;
    mjData*          snap_[2]    = {nullptr, nullptr};
    std::atomic<int> snap_write_ {0};
    std::atomic<int> snap_read_  {1};
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

    // Per-body scale cache: original geom sizes/positions and inertial params,
    // populated lazily on first setBodyScale call for each body.
    struct BodyScaleCache {
        mjtNum             original_mass;
        mjtNum             original_inertia[3];
        std::vector<int>   geom_ids;
        std::vector<std::array<mjtNum, 3>> original_geom_size;
        std::vector<std::array<mjtNum, 3>> original_geom_pos;
    };
    std::unordered_map<std::string, BodyScaleCache> body_scale_cache_;
};