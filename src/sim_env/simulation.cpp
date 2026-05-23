#include "sim_env/simulation.hpp"

#include <stdexcept>
#include <iostream>
#include <chrono>
#include <thread>
#include <cmath>
#include <unordered_map>

#include <GLFW/glfw3.h>
#include <yaml-cpp/yaml.h>


// ---------------------------------------------------------------------------
// Constructor / Destructor
// ---------------------------------------------------------------------------

Simulation::Simulation(const YAML::Node& config) {
    YAML::Node sim_config    = YAML::LoadFile(config["sim_config"].as<std::string>());
    YAML::Node robot_config  = YAML::LoadFile(config["robot_config"].as<std::string>());
    YAML::Node stream_config = YAML::LoadFile(config["streamer_config"].as<std::string>());

    BuiltScene scene = SceneBuilder::build(sim_config, robot_config);

    devices_ = std::move(scene.devices);
    objects_ = std::move(scene.objects);
    cameras_ = std::move(scene.cameras);

    char err[2000] = {};
    model = mj_loadXML(scene.xml_path.string().c_str(), nullptr, err, sizeof(err));
    if (!model)
        throw std::runtime_error(std::string("mj_loadXML failed: ") + err);

    if (sim_config["simulation"] && sim_config["simulation"]["timestep"])
        model->opt.timestep = sim_config["simulation"]["timestep"].as<double>();

    data = mj_makeData(model);
    if (!data)
        throw std::runtime_error("mj_makeData failed");

    for (int i = 0; i < model->nbody; ++i) {
        int mocap_id = model->body_mocapid[i];
        if (mocap_id >= 0) {
            const char* name = mj_id2name(model, mjOBJ_BODY, i);
            if (name){
                mocap_index_[std::string(name)] = mocap_id;
            }
        }
    }

    ctrl_buffer_.assign(model->nu, 0.0);
    buildActuatorIndex();
    applyInitialPositions();

    render_enabled_ = sim_config["rendering"] && sim_config["rendering"]["enabled"].as<bool>(false);
    if (sim_config["rendering"] && sim_config["rendering"]["fps"])
        render_fps_ = sim_config["rendering"]["fps"].as<int>();

    shm_enabled_   = stream_config["enabled"].as<bool>(false);
    stream_camera_ = stream_config["stream_camera"].as<std::string>("");
    stream_fps_    = stream_config["stream_fps"].as<int>(30);
    stream_width_  = stream_config["stream_width"].as<int>(model->vis.global.offwidth);
    stream_height_ = stream_config["stream_height"].as<int>(model->vis.global.offheight);

    snap_[0] = mj_copyData(nullptr, model, data);
    snap_[1] = mj_copyData(nullptr, model, data);
    buildCameraList();
}

Simulation::~Simulation() {
    if (snap_[0]) mj_deleteData(snap_[0]);
    if (snap_[1]) mj_deleteData(snap_[1]);
    if (data)     mj_deleteData(data);
    if (model)    mj_deleteModel(model);
}


// ---------------------------------------------------------------------------
// Threading
// ---------------------------------------------------------------------------

void Simulation::start() {
    if (render_enabled_) {
        rendering_thread = std::thread(&Simulation::run_rendering, this);
        while (!bRenderingIsRunning)
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }

    if (shm_enabled_) {
        stream_thread_ = std::thread(&Simulation::run_streaming, this);
        while (!bStreamingIsRunning)
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }

    model_thread = std::thread(&Simulation::run_model, this);
    while (!bModelIsRunning)
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    std::cout << "[INFO]: Simulation started." << std::endl;
}

void Simulation::stop() {
    bModelIsRunning     = false;
    bRenderingIsRunning = false;
    bStreamingIsRunning = false;
    if (model_thread.joinable())     model_thread.join();
    if (rendering_thread.joinable()) rendering_thread.join();
    if (stream_thread_.joinable())   stream_thread_.join();
    std::cout << "[INFO]: Simulation stopped." << std::endl;
}

bool Simulation::isRunning() const {
    return bModelIsRunning || bRenderingIsRunning || bStreamingIsRunning;
}

void Simulation::run_model() {
    bModelIsRunning = true;
    while (bModelIsRunning) {
        {
            std::lock_guard<std::mutex> lock(data_mtx);
            {
                std::lock_guard<std::mutex> ctrl_lock(ctrl_mtx_);

                for (const auto& [name, active] : active_devices_) {
                    if (active) continue;
                    const auto& act_ids  = actuator_ids_[name];
                    const auto& jnt_ids  = joint_ids_[name];
                    for (size_t i = 0; i < act_ids.size() && i < jnt_ids.size(); ++i) {
                        int vadr = model->jnt_dofadr[jnt_ids[i]];
                        ctrl_buffer_[act_ids[i]] = data->qfrc_bias[vadr];
                    }
                }

                for (int i = 0; i < model->nu; ++i)
                    data->ctrl[i] = ctrl_buffer_[i];
            }
            mj_step(model, data);
            swapSnapshots();
        }
        std::this_thread::sleep_for(
            std::chrono::microseconds(
                static_cast<int>(model->opt.timestep * 1e6)));
    }
}


// ---------------------------------------------------------------------------
// Rendering
// ---------------------------------------------------------------------------

void Simulation::buildCameraList() {
    render_cams_.clear();
    for (int i = 0; i < model->ncam; ++i) {
        const char* name = mj_id2name(model, mjOBJ_CAMERA, i);
        render_cams_.push_back({name ? name : ("cam" + std::to_string(i)), i});
    }
}

static void gridDims(int n, int& cols, int& rows) {
    cols = static_cast<int>(std::ceil(std::sqrt(static_cast<double>(n))));
    rows = (n + cols - 1) / cols;
}

void Simulation::initRendering() {
    if (!window_)
        throw std::runtime_error("[Rendering] window_ is null — must be created on main thread");

    glfwMakeContextCurrent(window_);
    glfwSwapInterval(0);

    mjv_defaultOption(&vopt_);
    mjv_defaultScene(&scn_);
    mjv_makeScene(model, &scn_, 2000);
    mjr_defaultContext(&con_);
    mjr_makeContext(model, &con_, mjFONTSCALE_100);
}

void Simulation::run_rendering() {
    if (!render_enabled_) {
        bRenderingIsRunning = true;
        return;
    }

    try {
        initRendering();
    } catch (const std::exception& e) {
        std::cerr << e.what() << "\n";
        bRenderingIsRunning = true;
        return;
    }

    bRenderingIsRunning = true;

    auto frame_duration = std::chrono::microseconds(1000000 / render_fps_);
    auto next_frame     = std::chrono::steady_clock::now();

    while (bRenderingIsRunning && !glfwWindowShouldClose(window_)) {
        renderFrame();
        next_frame += frame_duration;
        std::this_thread::sleep_until(next_frame);
    }

    mjv_freeScene(&scn_);
    mjr_freeContext(&con_);
    bRenderingIsRunning = false;
}

void Simulation::renderFrame() {
    int ncam = static_cast<int>(render_cams_.size());
    if (ncam == 0) return;

    int win_w, win_h;
    glfwGetFramebufferSize(window_, &win_w, &win_h);

    int cols, rows;
    gridDims(ncam, cols, rows);
    int cell_w = win_w / cols;
    int cell_h = win_h / rows;

    int r        = snap_read_.load(std::memory_order_acquire);
    mjData* snap = snap_[r];

    for (int i = 0; i < ncam; ++i) {
        int col = i % cols;
        int row = i / cols;

        int x = col * cell_w;
        int y = (rows - 1 - row) * cell_h;
        mjrRect viewport = {x, y, cell_w, cell_h};

        mjvCamera mjcam;
        mjv_defaultCamera(&mjcam);
        mjcam.type       = mjCAMERA_FIXED;
        mjcam.fixedcamid = render_cams_[i].id;

        mjv_updateScene(model, snap, &vopt_, nullptr, &mjcam, mjCAT_ALL, &scn_);
        mjr_render(viewport, &scn_, &con_);
        mjr_overlay(mjFONT_NORMAL, mjGRID_TOPLEFT, viewport,
                    render_cams_[i].name.c_str(), nullptr, &con_);
    }

    glfwSwapBuffers(window_);
}

void Simulation::swapSnapshots() {
    int w = snap_write_.load(std::memory_order_relaxed);
    mj_copyData(snap_[w], model, data);
    int next = 1 - w;
    snap_write_.store(next, std::memory_order_release);
    snap_read_.store(w,    std::memory_order_release);
}


// ---------------------------------------------------------------------------
// Offscreen streaming
// ---------------------------------------------------------------------------

void Simulation::initOffscreenStreaming() {
    if (!offscreen_window_)
        throw std::runtime_error("[Streaming] offscreen_window_ is null");

    glfwMakeContextCurrent(offscreen_window_);

    model->vis.global.offwidth  = stream_width_;
    model->vis.global.offheight = stream_height_;

    mjv_defaultOption(&stream_vopt_);
    mjv_defaultScene(&stream_scn_);
    mjv_makeScene(model, &stream_scn_, 2000);
    mjr_defaultContext(&stream_con_);
    mjr_makeContext(model, &stream_con_, mjFONTSCALE_100);

#ifndef _WIN32
    shm_unlink("/avatar_cam");
#endif
    shm_writer_ = std::make_unique<SharedMemoryWriter>("/avatar_cam", stream_width_, stream_height_);
}

void Simulation::renderStreamFrame() {
    int cam_id = -1;
    for (const auto& c : render_cams_)
        if (c.name == stream_camera_) { cam_id = c.id; break; }
    if (cam_id < 0) return;

    int r = snap_read_.load(std::memory_order_acquire);
    mjData* snap = snap_[r];

    mjvCamera mjcam;
    mjv_defaultCamera(&mjcam);
    mjcam.type       = mjCAMERA_FIXED;
    mjcam.fixedcamid = cam_id;

    mjrRect viewport = {0, 0, stream_con_.offWidth, stream_con_.offHeight};
    mjr_setBuffer(mjFB_OFFSCREEN, &stream_con_);
    mjv_updateScene(model, snap, &stream_vopt_, nullptr, &mjcam, mjCAT_ALL, &stream_scn_);
    mjr_render(viewport, &stream_scn_, &stream_con_);

    std::vector<uint8_t> pixels(stream_width_ * stream_height_ * 3);
    mjr_readPixels(pixels.data(), nullptr, viewport, &stream_con_);

    for (int row = 0; row < stream_height_ / 2; ++row) {
        uint8_t* top = pixels.data() + row * stream_width_ * 3;
        uint8_t* bot = pixels.data() + (stream_height_ - 1 - row) * stream_width_ * 3;
        std::swap_ranges(top, top + stream_width_ * 3, bot);
    }

    shm_writer_->write(pixels.data(), pixels.size());
}

void Simulation::run_streaming() {
    try {
        initOffscreenStreaming();
    } catch (const std::exception& e) {
        std::cerr << "[Streaming] Init failed: " << e.what() << "\n";
        bStreamingIsRunning = true;
        return;
    } catch (...) {
        std::cerr << "[Streaming] Init failed: unknown exception\n";
        bStreamingIsRunning = true;
        return;
    }

    bStreamingIsRunning = true;

    auto period = std::chrono::microseconds(1000000 / stream_fps_);
    auto next   = std::chrono::steady_clock::now();

    while (bStreamingIsRunning) {
        try {
            renderStreamFrame();
            stream_frame_count_.fetch_add(1, std::memory_order_relaxed);
        } catch (const std::exception& e) {
            std::cerr << "[Streaming] Frame error: " << e.what() << "\n";
            break;
        } catch (...) {
            std::cerr << "[Streaming] Frame error: unknown exception\n";
            break;
        }
        next += period;
        std::this_thread::sleep_until(next);
    }

    mjv_freeScene(&stream_scn_);
    mjr_freeContext(&stream_con_);
    if (offscreen_window_) {
        glfwDestroyWindow(offscreen_window_);
        offscreen_window_ = nullptr;
    }
}


// ---------------------------------------------------------------------------
// Control API
// ---------------------------------------------------------------------------

void Simulation::setCtrl(const std::string& deviceName,
                          const std::vector<double>& values) {
    auto it = actuator_ids_.find(deviceName);
    if (it == actuator_ids_.end()) {
        std::cerr << "[Simulation] setCtrl: unknown device '" << deviceName << "'\n";
        return;
    }
    const auto& ids = it->second;
    std::lock_guard<std::mutex> lock(ctrl_mtx_);
    for (size_t i = 0; i < values.size() && i < ids.size(); ++i)
        ctrl_buffer_[ids[i]] = values[i];
}

void Simulation::setGripper(const std::string& deviceName, double width) {
    auto it = gripper_ids_.find(deviceName);
    if (it == gripper_ids_.end() || it->second < 0) {
        std::cerr << "[Simulation] setGripper: device '" << deviceName << "' has no gripper actuator defined\n";
        return;
    }

    double half_width = std::clamp(width, 0.0, 0.08) / 2.0;
    double ctrl_value = (half_width / 0.04) * 255.0;

    std::lock_guard<std::mutex> lock(ctrl_mtx_);
    ctrl_buffer_[it->second] = ctrl_value;
}

double Simulation::getGripperWidth(const std::string& deviceName) {
    auto it = joint_ids_.find(deviceName);
    if (it == joint_ids_.end() || it->second.empty()) return 0.0;

    int r        = snap_read_.load(std::memory_order_acquire);
    mjData* snap = snap_[r];

    double total = 0.0;
    for (int j : it->second) {
        int qadr = model->jnt_qposadr[j];
        total   += snap->qpos[qadr];
    }
    return total;
}

// ---------------------------------------------------------------------------
// Actuator index tables
// ---------------------------------------------------------------------------

void Simulation::buildActuatorIndex() {
    for (const auto& dev : devices_) {
        std::string gripperFullName;
        if (!dev.gripper_actuator.empty())
            gripperFullName = dev.name + "_" + dev.gripper_actuator;

        std::vector<int> jointActuators;
        int gripperId = -1;

        for (int i = 0; i < model->nu; ++i) {
            const char* aname = mj_id2name(model, mjOBJ_ACTUATOR, i);
            if (!aname) continue;
            std::string name(aname);
            if (name.rfind(dev.name + "_", 0) != 0) continue;
            if (!gripperFullName.empty() && name == gripperFullName)
                gripperId = i;
            else
                jointActuators.push_back(i);
        }

        std::vector<int> joints;
        for (int j = 0; j < model->njnt; ++j) {
            const char* jname = mj_id2name(model, mjOBJ_JOINT, j);
            if (jname && std::string(jname).rfind(dev.name + "_", 0) == 0)
                joints.push_back(j);
        }

        actuator_ids_[dev.name] = std::move(jointActuators);
        gripper_ids_[dev.name]  = gripperId;
        joint_ids_[dev.name]    = std::move(joints);
        active_devices_[dev.name] = false;
    }
}


// ---------------------------------------------------------------------------
// Initial joint positions
// ---------------------------------------------------------------------------

void Simulation::applyInitialPositions() {
    for (const auto& dev : devices_) {
        std::vector<int> joints;
        for (int j = 0; j < model->njnt; ++j) {
            const char* jname = mj_id2name(model, mjOBJ_JOINT, j);
            if (jname && std::string(jname).rfind(dev.name + "_", 0) == 0)
                joints.push_back(j);
        }
        for (size_t i = 0; i < dev.q0.size() && i < joints.size(); ++i)
            data->qpos[model->jnt_qposadr[joints[i]]] = dev.q0[i];
    }
    mj_forward(model, data);
}

DeviceState Simulation::getDeviceState(const std::string& deviceName) {
    auto it = joint_ids_.find(deviceName);
    if (it == joint_ids_.end()) {
        std::cerr << "[Simulation] getDeviceState: unknown device '" << deviceName << "'\n";
        return {};
    }

    int r = snap_read_.load(std::memory_order_acquire);
    mjData* snap = snap_[r];

    DeviceState state;
    for (int j : it->second) {
        int qadr = model->jnt_qposadr[j];
        int vadr = model->jnt_dofadr[j];
        state.q.push_back(snap->qpos[qadr]);
        state.dq.push_back(snap->qvel[vadr]);
        state.tau_J.push_back(snap->qfrc_actuator[vadr]);
        state.tau_ext.push_back(snap->qfrc_constraint[vadr]);
    }

    return state;
}

void Simulation::setDeviceActive(const std::string& deviceName, bool state){
    active_devices_[deviceName] = state;
}

void Simulation::setFramePose(const std::string& name, const Eigen::Vector3d& pos, const Eigen::Quaterniond& quat, double z_offset) {
    auto it = mocap_index_.find(name);
    if (it == mocap_index_.end()) return;
    int id = it->second;

    Eigen::Vector3d offset_pos = pos + quat * Eigen::Vector3d(0, 0, z_offset);

    data->mocap_pos[id * 3 + 0] = offset_pos.x();
    data->mocap_pos[id * 3 + 1] = offset_pos.y();
    data->mocap_pos[id * 3 + 2] = offset_pos.z();
    data->mocap_quat[id * 4 + 0] = quat.w();
    data->mocap_quat[id * 4 + 1] = quat.x();
    data->mocap_quat[id * 4 + 2] = quat.y();
    data->mocap_quat[id * 4 + 3] = quat.z();
}

void Simulation::setFreeBodyPose(const std::string& bodyName, const Eigen::Vector3d& pos, const Eigen::Quaterniond& quat){
    // Find the body by name
    int body_id = mj_name2id(model, mjOBJ_BODY, bodyName.c_str());
    if (body_id < 0) {
        std::cerr << "[Simulation] setFreeBodyPose: body '" << bodyName << "' not found\n";
        return;
    }

    // Find the freejoint attached to this body (must be the first joint)
    int jnt_id = -1;
    for (int j = 0; j < model->njnt; ++j) {
        if (model->jnt_bodyid[j] == body_id && model->jnt_type[j] == mjJNT_FREE) {
            jnt_id = j;
            break;
        }
    }
    if (jnt_id < 0) {
        std::cerr << "[Simulation] setFreeBodyPose: body '" << bodyName << "' has no freejoint\n";
        return;
    }

    int qadr = model->jnt_qposadr[jnt_id];

    std::lock_guard<std::mutex> lock(data_mtx);
    // Position (3 values)
    data->qpos[qadr + 0] = pos.x();
    data->qpos[qadr + 1] = pos.y();
    data->qpos[qadr + 2] = pos.z();
    // Quaternion — MuJoCo stores w,x,y,z
    data->qpos[qadr + 3] = quat.w();
    data->qpos[qadr + 4] = quat.x();
    data->qpos[qadr + 5] = quat.y();
    data->qpos[qadr + 6] = quat.z();
    // Zero out velocities so the body doesn't carry momentum from last episode
    int vadr = model->jnt_dofadr[jnt_id];
    for (int i = 0; i < 6; ++i)
        data->qvel[vadr + i] = 0.0;

    mj_forward(model, data);
}

bool Simulation::getFreeBodyPose(const std::string& bodyName, Eigen::Vector3d& pos, Eigen::Quaterniond& quat) {
    int body_id = mj_name2id(model, mjOBJ_BODY, bodyName.c_str());
    if (body_id < 0) return false;

    int jnt_id = -1;
    for (int j = 0; j < model->njnt; ++j) {
        if (model->jnt_bodyid[j] == body_id && model->jnt_type[j] == mjJNT_FREE) {
            jnt_id = j; break;
        }
    }
    if (jnt_id < 0) return false;

    int qadr = model->jnt_qposadr[jnt_id];
    int r        = snap_read_.load(std::memory_order_acquire);
    mjData* snap = snap_[r];

    pos = Eigen::Vector3d(snap->qpos[qadr+0], snap->qpos[qadr+1], snap->qpos[qadr+2]);
    quat = Eigen::Quaterniond(snap->qpos[qadr+3], snap->qpos[qadr+4],
                               snap->qpos[qadr+5], snap->qpos[qadr+6]);
    return true;
}

CameraIntrinsics Simulation::getCameraIntrinsics(const std::string& cam_name) const {
    int cam_id = mj_name2id(model, mjOBJ_CAMERA, cam_name.c_str());
    if (cam_id < 0)
        throw std::runtime_error("[Simulation] getCameraIntrinsics: camera '" + cam_name + "' not found");

    // fovy is stored in radians in mjModel
    float fovy_rad = static_cast<float>(model->cam_fovy[cam_id]) * static_cast<float>(M_PI) / 180.0f;
    float fy = (stream_height_ / 2.0f) / std::tan(fovy_rad / 2.0f);
    float fx = fy;  // square pixels in MuJoCo

    return CameraIntrinsics{
        .fx     = fx,
        .fy     = fy,
        .cx     = stream_width_  / 2.0f,
        .cy     = stream_height_ / 2.0f,
        .width  = stream_width_,
        .height = stream_height_
    };
}
