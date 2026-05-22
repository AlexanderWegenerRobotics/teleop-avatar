#pragma once

#include <thread>
#include <atomic>
#include <mutex>
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
    void setFramePose(const std::string& name, const Eigen::Vector3d& pos, const Eigen::Quaterniond& quat);
    void setFreeBodyPose(const std::string& bodyName, const Eigen::Vector3d& pos, const Eigen::Quaterniond& quat);
    bool getFreeBodyPose(const std::string& bodyName, Eigen::Vector3d& pos, Eigen::Quaterniond& quat);
    CameraIntrinsics getCameraIntrinsics(const std::string& cam_name) const;
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
    std::unique_ptr<SharedMemoryWriter> shm_writer_;

private:
    struct CamEntry { std::string name; int id; };

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
    std::string stream_camera_;
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
};