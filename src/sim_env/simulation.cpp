#include "sim_env/simulation.hpp"

#include <algorithm>

#include <stdexcept>
#include <iostream>
#include <chrono>
#include <thread>
#include <cmath>
#include <unordered_map>

#include <GLFW/glfw3.h>
#include <yaml-cpp/yaml.h>

#include "twin/config_overlay.hpp"


// Constructor / Destructor

Simulation::Simulation(const YAML::Node& config, Role role) {
    YAML::Node sim_config    = SceneBuilder::loadMergedSimConfig(config["sim_config"].as<std::string>());
    YAML::Node robot_config  = YAML::LoadFile(config["robot_config"].as<std::string>());
    YAML::Node stream_config = YAML::LoadFile(config["streamer_config"].as<std::string>());

    // twin overlay avoids shm/port collisions when avatar and twin run on one machine
    if (role == Role::Twin && config["streamer_overlay"]) {
        applyTwinStreamerOverlay(stream_config, config["streamer_overlay"].as<std::string>());
        std::cout << "[SIM-INFO] Applied twin streamer overlay: "
                  << config["streamer_overlay"].as<std::string>() << std::endl;
    }

    BuiltScene scene = SceneBuilder::build(sim_config, robot_config);

    devices_ = std::move(scene.devices);
    objects_ = std::move(scene.objects);
    cameras_ = std::move(scene.cameras);

    // refuse to run with a mismatched mujoco DLL
    if (mj_version() != mjVERSION_HEADER)
        throw std::runtime_error("MuJoCo version mismatch: built against "
            + std::to_string(mjVERSION_HEADER) + " headers but loaded mujoco.dll "
            + mj_versionString() + ". Fix PATH (launch.bat) or MUJOCO_ROOT.");
    std::cout << "[SIM-INFO] MuJoCo " << mj_versionString() << std::endl;

    char err[2000] = {};
    model = mj_loadXML(scene.xml_path.string().c_str(), nullptr, err, sizeof(err));
    if (!model)
        throw std::runtime_error(std::string("mj_loadXML failed: ") + err);

    // scene XMLs say 0.005, so warn if the config doesn't set the timestep
    const double xml_timestep = model->opt.timestep;
    const bool   from_config  = sim_config["simulation"] && sim_config["simulation"]["timestep"];
    if (from_config)
        model->opt.timestep = sim_config["simulation"]["timestep"].as<double>();

    std::cout << "[SIM-INFO] timestep " << model->opt.timestep * 1e3 << " ms ("
              << 1.0 / model->opt.timestep << " Hz), from "
              << (from_config ? "sim_config" : "scene XML") << std::endl;
    if (!from_config)
        std::cout << "[SIM-WARN] sim_config has no simulation.timestep -- inherited "
                  << xml_timestep * 1e3 << " ms from the scene XML. Set it explicitly."
                  << std::endl;

    // ~2 mm near clip so wrist cams don't see through close objects
    model->vis.map.znear = 0.001;

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
    for (const auto& [name, gid] : gripper_ids_)
        if (gid >= 0) ctrl_buffer_[gid] = 255.0;
    applyInitialPositions();

    render_enabled_ = sim_config["rendering"] && sim_config["rendering"]["enabled"].as<bool>(false);
    if (sim_config["rendering"] && sim_config["rendering"]["fps"])
        render_fps_ = sim_config["rendering"]["fps"].as<int>();

    shm_enabled_  = stream_config["enabled"].as<bool>(false);
    stereo_       = stream_config["stereo"].as<bool>(false);
    stream_fps_   = stream_config["stream_fps"].as<int>(30);
    stream_width_ = stream_config["stream_width"].as<int>(model->vis.global.offwidth);
    stream_height_= stream_config["stream_height"].as<int>(model->vis.global.offheight);

    // mono: eye=="mono" or missing, stereo: eye=="left"/"right"
    if (stream_config["stream_cameras"] && stream_config["stream_cameras"].IsSequence()) {
        for (const auto& entry : stream_config["stream_cameras"]) {
            std::string eye = entry["eye"].as<std::string>("mono");
            bool active = (eye == "aux") ||
                          (stereo_ ? (eye == "left" || eye == "right")
                                   : (eye == "mono"));
            if (!active) continue;
            StreamCamEntry sc;
            sc.camera_name = entry["camera"].as<std::string>("");
            sc.shm_name    = entry["shm_name"].as<std::string>("");
            sc.width       = entry["width"].as<int>(0);
            sc.height      = entry["height"].as<int>(0);
            if (entry["distortion"]) sc.distortion = entry["distortion"].as<std::vector<double>>();
            if (!sc.camera_name.empty() && !sc.shm_name.empty())
                stream_cameras_.push_back(std::move(sc));
        }
    }

    // needs joint_ids_, so after buildActuatorIndex()
    wrench_truth_ = WrenchTruth::create(
        model, sim_config, robot_config, joint_ids_,
        std::to_string(std::chrono::duration_cast<std::chrono::seconds>(
            std::chrono::system_clock::now().time_since_epoch()).count()));

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


// Threading

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

    using clock = std::chrono::steady_clock;
    const auto step_period = std::chrono::duration_cast<clock::duration>(
        std::chrono::duration<double>(model->opt.timestep));

    // sleep overshoots by a scheduler quantum, so sleep close to the deadline then spin
    const auto spin_margin = std::min(
        std::chrono::duration_cast<clock::duration>(std::chrono::microseconds(400)),
        step_period / 2);

    auto loop_start  = clock::now();
    auto next        = loop_start + step_period;
    auto last_report = loop_start;

    while (bModelIsRunning) {
        const auto step_t0 = clock::now();
        {
            std::lock_guard<std::mutex> lock(data_mtx);
            {
                std::lock_guard<std::mutex> ctrl_lock(ctrl_mtx_);

                for (const auto& [name, active] : active_devices_) {
                    if (active) continue;
                    const auto& act_ids  = actuator_ids_[name];
                    const auto& jnt_ids  = joint_ids_[name];
                    const size_t n = std::min(act_ids.size(), jnt_ids.size());
                    BrakeState& br = brake_[name];

                    if (!br.braked) {
                        bool at_rest = true;
                        for (size_t i = 0; i < n; ++i)
                            if (std::abs(data->qvel[model->jnt_dofadr[jnt_ids[i]]]) > kBrakeRestVel) { at_rest = false; break; }
                        if (at_rest) {
                            br.q_hold.resize(n);
                            for (size_t i = 0; i < n; ++i)
                                br.q_hold[i] = data->qpos[model->jnt_qposadr[jnt_ids[i]]];
                            br.braked = true;
                        }
                    }

                    for (size_t i = 0; i < n; ++i) {
                        const int a    = act_ids[i];
                        const int vadr = model->jnt_dofadr[jnt_ids[i]];
                        const int qadr = model->jnt_qposadr[jnt_ids[i]];
                        const double ctrl_max = model->actuator_ctrllimited[a]
                            ? std::max(std::abs(model->actuator_ctrlrange[2*a]), std::abs(model->actuator_ctrlrange[2*a+1]))
                            : 87.0;
                        double u = data->qfrc_bias[vadr] - kBrakeDampFrac * ctrl_max * data->qvel[vadr];
                        if (br.braked)
                            u -= kBrakeStiffFrac * ctrl_max * (data->qpos[qadr] - br.q_hold[i]);
                        ctrl_buffer_[a] = std::clamp(u, -ctrl_max, ctrl_max);
                    }
                }

                for (int i = 0; i < model->nu; ++i)
                    data->ctrl[i] = ctrl_buffer_[i];
            }
            mj_step(model, data);
            swapSnapshots();
            if (wrench_truth_) wrench_truth_->sample(data);
        }
        const double step_ns =
            std::chrono::duration<double, std::nano>(clock::now() - step_t0).count();

        sim_steps_.fetch_add(1, std::memory_order_relaxed);
        step_ns_sum_.store(step_ns_sum_.load(std::memory_order_relaxed) + step_ns,
                           std::memory_order_relaxed);
        if (step_ns > step_ns_max_.load(std::memory_order_relaxed))
            step_ns_max_.store(step_ns, std::memory_order_relaxed);

        // one mj_step per period, never multi-step to catch up; sim slips instead
        auto now = clock::now();
        if (now < next - spin_margin)
            std::this_thread::sleep_until(next - spin_margin);
        while (clock::now() < next) { }

        next += step_period;
        now = clock::now();
        if (now > next) {
            deadline_misses_.fetch_add(1, std::memory_order_relaxed);
            next = now + step_period;          // resync rather than spiral
        }

        wall_seconds_.store(std::chrono::duration<double>(now - loop_start).count(),
                            std::memory_order_relaxed);

        if (now - last_report >= std::chrono::seconds(10)) {
            const SimTimingStats st = getTimingStats();
            std::cout << "[SIM-TIMING] rtf " << st.rtf << "  steps " << st.steps
                      << "  mj_step " << st.step_ms_mean << " ms mean / "
                      << st.step_ms_max << " ms max  deadline misses "
                      << st.deadline_misses << std::endl;
            if (st.rtf < 0.95)
                std::cout << "[SIM-WARN] running at " << st.rtf
                          << "x real time -- episodes stay valid (resample on sim_time), "
                          << "but operator feel and any wall-clock latency number are distorted."
                          << std::endl;
            last_report = now;
        }
    }
}

SimTimingStats Simulation::getTimingStats() const {
    SimTimingStats s;
    s.steps           = sim_steps_.load(std::memory_order_relaxed);
    s.deadline_misses = deadline_misses_.load(std::memory_order_relaxed);
    s.wall_seconds    = wall_seconds_.load(std::memory_order_relaxed);
    s.sim_seconds     = static_cast<double>(s.steps) * (model ? model->opt.timestep : 0.0);
    s.rtf             = s.wall_seconds > 0.0 ? s.sim_seconds / s.wall_seconds : 0.0;
    s.step_ms_mean    = s.steps ? step_ns_sum_.load(std::memory_order_relaxed)
                                      / static_cast<double>(s.steps) * 1e-6
                                : 0.0;
    s.step_ms_max     = step_ns_max_.load(std::memory_order_relaxed) * 1e-6;
    return s;
}


// Rendering

void Simulation::buildCameraList() {
    render_cams_.clear();
    std::cout << "[Simulation] Available cameras (" << model->ncam << "):" << std::endl;
    for (int i = 0; i < model->ncam; ++i) {
        const char* name = mj_id2name(model, mjOBJ_CAMERA, i);
        std::string n = name ? name : ("cam" + std::to_string(i));
        render_cams_.push_back({n, i});
        std::cout << "  [" << i << "] " << n << std::endl;
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

    render_data_ = mj_makeData(model);
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
    mj_deleteData(render_data_);
    render_data_ = nullptr;
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

    // private copy, mjv_updateScene writes the mjData stack
    latchSnapshot(render_data_);
    mjData* snap = render_data_;

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
    std::lock_guard<std::mutex> lock(snap_mtx_);
    int w = snap_write_.load(std::memory_order_relaxed);
    mj_copyData(snap_[w], model, data);
    int next = 1 - w;
    snap_write_.store(next, std::memory_order_release);
    snap_read_.store(w,    std::memory_order_release);
}

// copies latest snapshot into a renderer-private mjData under snap_mtx_
void Simulation::latchSnapshot(mjData* dst) {
    std::lock_guard<std::mutex> lock(snap_mtx_);
    mj_copyData(dst, model, snap_[snap_read_.load(std::memory_order_acquire)]);
}


// Offscreen streaming

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

    shm_writers_.clear();
    stream_lens_.assign(stream_cameras_.size(), RemapTable());
    for (const auto& sc : stream_cameras_) {
#ifndef _WIN32
        shm_unlink(sc.shm_name.c_str());
#endif
        int cw = sc.width  > 0 ? sc.width  : stream_width_;
        int ch = sc.height > 0 ? sc.height : stream_height_;
        shm_writers_.push_back(
            std::make_unique<SharedMemoryWriter>(sc.shm_name, cw, ch));
        std::cout << "[Streaming] Opened SHM writer: " << sc.shm_name
                  << "  camera=" << sc.camera_name
                  << "  res=" << cw << "x" << ch << std::endl;
    }
    if (shm_writers_.empty())
        std::cerr << "[Streaming] Warning: no stream cameras configured — nothing will be written\n";
}

void Simulation::renderStreamFrame() {
    if (stream_cameras_.empty() || shm_writers_.empty()) return;

    // latch once so all cameras render the same state
    latchSnapshot(stream_data_);
    mjData* snap = stream_data_;

    mjr_setBuffer(mjFB_OFFSCREEN, &stream_con_);

    std::vector<uint8_t> pixels;

    for (size_t i = 0; i < stream_cameras_.size(); ++i) {
        const auto& sc = stream_cameras_[i];
        const int cw = sc.width  > 0 ? sc.width  : stream_width_;
        const int ch = sc.height > 0 ? sc.height : stream_height_;

        // sub-viewport, offscreen context is sized to the max resolution
        mjrRect viewport = {0, 0, cw, ch};

        int cam_id = -1;
        for (const auto& c : render_cams_)
            if (c.name == sc.camera_name) { cam_id = c.id; break; }
        if (cam_id < 0) continue;

        mjvCamera mjcam;
        mjv_defaultCamera(&mjcam);
        mjcam.type       = mjCAMERA_FIXED;
        mjcam.fixedcamid = cam_id;

        mjv_updateScene(model, snap, &stream_vopt_, nullptr, &mjcam, mjCAT_ALL, &stream_scn_);
        mjr_render(viewport, &stream_scn_, &stream_con_);

        pixels.resize(cw * ch * 3);
        mjr_readPixels(pixels.data(), nullptr, viewport, &stream_con_);

        // MuJoCo pixels are bottom-up
        for (int row = 0; row < ch / 2; ++row) {
            uint8_t* top = pixels.data() + row * cw * 3;
            uint8_t* bot = pixels.data() + (ch - 1 - row) * cw * 3;
            std::swap_ranges(top, top + cw * 3, bot);
        }

        const uint64_t capture_time_ns = static_cast<uint64_t>(
            std::chrono::duration_cast<std::chrono::nanoseconds>(
                std::chrono::system_clock::now().time_since_epoch()).count());

        // simulated lens: same distortion model the perception calibration estimates
        if (!sc.distortion.empty() && stream_lens_[i].empty()) {
            LensModel lens = LensModel::pinholeFromFovy(cw, ch, model->cam_fovy[cam_id]);
            for (size_t k = 0; k < std::min<size_t>(5, sc.distortion.size()); ++k) lens.dist[k] = sc.distortion[k];
            stream_lens_[i] = RemapTable::distort(lens);
            std::cout << "[Streaming] " << sc.camera_name << ": lens distortion applied" << std::endl;
        }
        if (!stream_lens_[i].empty()) {
            lens_pixels_.resize(pixels.size());
            stream_lens_[i].apply(pixels.data(), lens_pixels_.data());
            shm_writers_[i]->write(lens_pixels_.data(), lens_pixels_.size(), capture_time_ns);
            continue;
        }

        shm_writers_[i]->write(pixels.data(), pixels.size(), capture_time_ns);
    }
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

    stream_data_ = mj_makeData(model);
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
    mj_deleteData(stream_data_);
    stream_data_ = nullptr;
    if (offscreen_window_) {
        glfwDestroyWindow(offscreen_window_);
        offscreen_window_ = nullptr;
    }
}


// Control API

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
        std::cerr << "[Simulation] setGripper: no actuator found for device '" << deviceName << "' (id=" << (it != gripper_ids_.end() ? it->second : -999) << ")\n";
        return;
    }

    constexpr double kMinWidth = 0.006;   // m, avoids finger mesh penetration
    double half_width = std::clamp(width, kMinWidth, 0.08) / 2.0;
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

// Actuator index tables

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


// Initial joint positions

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
    for (int j = 0; j < model->njnt; ++j) {
        const char* jname = mj_id2name(model, mjOBJ_JOINT, j);
        if (jname && std::string(jname).find("finger_joint") != std::string::npos)
            data->qpos[model->jnt_qposadr[j]] = 0.04;
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
    state.time = snap->time;   // sim time
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
    std::lock_guard<std::mutex> lock(ctrl_mtx_);
    active_devices_[deviceName] = state;
    brake_[deviceName] = BrakeState{};
}

std::vector<double> Simulation::getDeviceCtrl(const std::string& deviceName) {
    auto it = actuator_ids_.find(deviceName);
    if (it == actuator_ids_.end()) return {};

    std::lock_guard<std::mutex> lock(ctrl_mtx_);
    std::vector<double> out;
    out.reserve(it->second.size());
    for (int aid : it->second)
        out.push_back(ctrl_buffer_[aid]);
    return out;
}

void Simulation::replaySeed(mjData* replay_data, const std::string& deviceName,
                             const std::vector<double>& q, const std::vector<double>& dq) {
    auto it = joint_ids_.find(deviceName);
    if (it == joint_ids_.end() || !replay_data) return;
    const auto& joints = it->second;
    for (size_t i = 0; i < joints.size() && i < q.size(); ++i)
        replay_data->qpos[model->jnt_qposadr[joints[i]]] = q[i];
    for (size_t i = 0; i < joints.size() && i < dq.size(); ++i)
        replay_data->qvel[model->jnt_dofadr[joints[i]]] = dq[i];
    mj_forward(model, replay_data);
}

void Simulation::replaySetCtrl(mjData* replay_data, const std::string& deviceName,
                                const std::vector<double>& ctrl) {
    auto it = actuator_ids_.find(deviceName);
    if (it == actuator_ids_.end() || !replay_data) return;
    const auto& ids = it->second;
    for (size_t i = 0; i < ids.size() && i < ctrl.size(); ++i)
        replay_data->ctrl[ids[i]] = ctrl[i];
}

void Simulation::replayAdvance(mjData* replay_data) {
    if (!replay_data) return;
    mj_step(model, replay_data);
}

std::vector<double> Simulation::replayReadQ(mjData* replay_data, const std::string& deviceName) const {
    auto it = joint_ids_.find(deviceName);
    if (it == joint_ids_.end() || !replay_data) return {};
    std::vector<double> out;
    out.reserve(it->second.size());
    for (int j : it->second) out.push_back(replay_data->qpos[model->jnt_qposadr[j]]);
    return out;
}

std::vector<double> Simulation::replayReadDq(mjData* replay_data, const std::string& deviceName) const {
    auto it = joint_ids_.find(deviceName);
    if (it == joint_ids_.end() || !replay_data) return {};
    std::vector<double> out;
    out.reserve(it->second.size());
    for (int j : it->second) out.push_back(replay_data->qvel[model->jnt_dofadr[j]]);
    return out;
}

void Simulation::applyJointCorrection(const std::string& deviceName,
                                       const std::vector<double>& dq_delta,
                                       const std::vector<double>& ddq_delta) {
    auto it = joint_ids_.find(deviceName);
    if (it == joint_ids_.end() || it->second.empty()) return;

    const auto& joints = it->second;

    std::lock_guard<std::mutex> lock(data_mtx);
    for (size_t i = 0; i < joints.size() && i < dq_delta.size(); ++i) {
        int qadr = model->jnt_qposadr[joints[i]];
        data->qpos[qadr] += dq_delta[i];
    }
    for (size_t i = 0; i < joints.size() && i < ddq_delta.size(); ++i) {
        int vadr = model->jnt_dofadr[joints[i]];
        data->qvel[vadr] += ddq_delta[i];
    }
    // state write only, so mj_forward not mj_step
    mj_forward(model, data);
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

void Simulation::setFreeBodyPoses(const std::vector<FreeBodyPose>& poses) {
    struct Target { int qadr; int vadr; const FreeBodyPose* p; };
    std::vector<Target> targets;
    targets.reserve(poses.size());
    for (const auto& p : poses) {
        int body_id = mj_name2id(model, mjOBJ_BODY, p.body.c_str());
        if (body_id < 0) {
            std::cerr << "[Simulation] setFreeBodyPoses: body '" << p.body << "' not found\n";
            continue;
        }
        int jnt_id = -1;
        for (int j = 0; j < model->njnt; ++j) {
            if (model->jnt_bodyid[j] == body_id && model->jnt_type[j] == mjJNT_FREE) { jnt_id = j; break; }
        }
        if (jnt_id < 0) {
            std::cerr << "[Simulation] setFreeBodyPoses: body '" << p.body << "' has no freejoint\n";
            continue;
        }
        targets.push_back({model->jnt_qposadr[jnt_id], model->jnt_dofadr[jnt_id], &p});
    }

    std::lock_guard<std::mutex> lock(data_mtx);
    for (const auto& t : targets) {
        data->qpos[t.qadr + 0] = t.p->pos.x();
        data->qpos[t.qadr + 1] = t.p->pos.y();
        data->qpos[t.qadr + 2] = t.p->pos.z();
        data->qpos[t.qadr + 3] = t.p->quat.w();
        data->qpos[t.qadr + 4] = t.p->quat.x();
        data->qpos[t.qadr + 5] = t.p->quat.y();
        data->qpos[t.qadr + 6] = t.p->quat.z();
        for (int i = 0; i < 6; ++i) data->qvel[t.vadr + i] = 0.0;
    }
    mj_forward(model, data);
}

void Simulation::setFreeBodyPose(const std::string& bodyName, const Eigen::Vector3d& pos, const Eigen::Quaterniond& quat){
    int body_id = mj_name2id(model, mjOBJ_BODY, bodyName.c_str());
    if (body_id < 0) {
        std::cerr << "[Simulation] setFreeBodyPose: body '" << bodyName << "' not found\n";
        return;
    }

    // freejoint must be the body's first joint
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
    data->qpos[qadr + 0] = pos.x();
    data->qpos[qadr + 1] = pos.y();
    data->qpos[qadr + 2] = pos.z();
    // MuJoCo quat is w,x,y,z
    data->qpos[qadr + 3] = quat.w();
    data->qpos[qadr + 4] = quat.x();
    data->qpos[qadr + 5] = quat.y();
    data->qpos[qadr + 6] = quat.z();
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

void Simulation::setLighting(const LightingConfig& lc) {
    int main_id = mj_name2id(model, mjOBJ_LIGHT, "light_main");
    int fill_id = mj_name2id(model, mjOBJ_LIGHT, "light_fill");

    auto aimAtTarget = [&](int id) {
        double d[3];
        for (int i = 0; i < 3; ++i) d[i] = lc.main_target[i] - model->light_pos[id * 3 + i];
        double n = std::sqrt(d[0] * d[0] + d[1] * d[1] + d[2] * d[2]);
        if (n < 1e-6) return;
        for (int i = 0; i < 3; ++i) model->light_dir[id * 3 + i] = d[i] / n;
    };

    if (main_id >= 0) {
        for (int i = 0; i < 3; ++i) {
            model->light_pos     [main_id * 3 + i] = lc.main_pos[i];
            model->light_diffuse [main_id * 3 + i] = lc.main_diffuse[i];
            model->light_specular[main_id * 3 + i] = lc.main_specular[i];
        }
        aimAtTarget(main_id);
        model->light_cutoff    [main_id] = lc.main_cutoff;
        model->light_exponent  [main_id] = lc.main_exponent;
        model->light_castshadow[main_id] = 1;
    }
    if (fill_id >= 0) {
        for (int i = 0; i < 3; ++i) {
            model->light_diffuse[fill_id * 3 + i] = lc.fill_diffuse[i];
        }
        aimAtTarget(fill_id);
        model->light_cutoff    [fill_id] = lc.fill_cutoff;
        model->light_exponent  [fill_id] = lc.fill_exponent;
        model->light_castshadow[fill_id] = 0;
    }

    model->vis.headlight.active = 1;
    for (int i = 0; i < 3; ++i) {
        model->vis.headlight.diffuse[i]  = lc.headlight_diffuse[i];
        model->vis.headlight.ambient[i]  = lc.headlight_ambient[i];
        model->vis.headlight.specular[i] = 0.0f;
    }
}

void Simulation::setBodyScale(const std::string& bodyName, double scale) {
    int body_id = mj_name2id(model, mjOBJ_BODY, bodyName.c_str());
    if (body_id < 0) {
        std::cerr << "[Simulation] setBodyScale: body '" << bodyName << "' not found\n";
        return;
    }

    if (body_scale_cache_.find(bodyName) == body_scale_cache_.end()) {
        BodyScaleCache cache;
        cache.original_mass = model->body_mass[body_id];
        for (int i = 0; i < 3; ++i)
            cache.original_inertia[i] = model->body_inertia[body_id * 3 + i];
        for (int g = 0; g < model->ngeom; ++g) {
            if (model->geom_bodyid[g] != body_id) continue;
            cache.geom_ids.push_back(g);
            cache.original_geom_size.push_back({
                model->geom_size[g * 3 + 0],
                model->geom_size[g * 3 + 1],
                model->geom_size[g * 3 + 2]
            });
            cache.original_geom_pos.push_back({
                model->geom_pos[g * 3 + 0],
                model->geom_pos[g * 3 + 1],
                model->geom_pos[g * 3 + 2]
            });
            cache.original_geom_rbound.push_back(model->geom_rbound[g]);
            std::array<mjtNum, 6> aabb;
            for (int j = 0; j < 6; ++j) aabb[j] = model->geom_aabb[g * 6 + j];
            cache.original_geom_aabb.push_back(aabb);
        }
        cache.bvh_adr = model->body_bvhadr[body_id];
        if (cache.bvh_adr >= 0) {
            for (int n = 0; n < model->body_bvhnum[body_id]; ++n) {
                std::array<mjtNum, 6> node;
                for (int j = 0; j < 6; ++j) node[j] = model->bvh_aabb[(cache.bvh_adr + n) * 6 + j];
                cache.original_bvh_aabb.push_back(node);
            }
        }
        body_scale_cache_[bodyName] = std::move(cache);
    }

    const BodyScaleCache& cache = body_scale_cache_.at(bodyName);

    // always from original values so scaling doesn't compound
    for (size_t i = 0; i < cache.geom_ids.size(); ++i) {
        int g = cache.geom_ids[i];
        for (int j = 0; j < 3; ++j) {
            model->geom_size[g * 3 + j] = cache.original_geom_size[i][j] * scale;
            model->geom_pos [g * 3 + j] = cache.original_geom_pos [i][j] * scale;
        }
        // bounding volumes must be scaled too
        model->geom_rbound[g] = cache.original_geom_rbound[i] * scale;
        for (int j = 0; j < 6; ++j)
            model->geom_aabb[g * 6 + j] = cache.original_geom_aabb[i][j] * scale;
    }
    for (size_t n = 0; n < cache.original_bvh_aabb.size(); ++n)
        for (int j = 0; j < 6; ++j)
            model->bvh_aabb[(cache.bvh_adr + n) * 6 + j] = cache.original_bvh_aabb[n][j] * scale;

    // mass ~ scale^3, inertia ~ scale^5
    double s3 = scale * scale * scale;
    double s5 = s3 * scale * scale;
    model->body_mass[body_id] = cache.original_mass * s3;
    for (int i = 0; i < 3; ++i)
        model->body_inertia[body_id * 3 + i] = cache.original_inertia[i] * s5;
}

CameraIntrinsics Simulation::getCameraIntrinsics(const std::string& cam_name) const {
    int cam_id = mj_name2id(model, mjOBJ_CAMERA, cam_name.c_str());
    if (cam_id < 0)
        throw std::runtime_error("[Simulation] getCameraIntrinsics: camera '" + cam_name + "' not found");

    // cam_fovy is in degrees
    float fovy_rad = static_cast<float>(model->cam_fovy[cam_id]) * static_cast<float>(M_PI) / 180.0f;
    float fy = (stream_height_ / 2.0f) / std::tan(fovy_rad / 2.0f);
    float fx = fy;  // square pixels

    return CameraIntrinsics{
        .fx     = fx,
        .fy     = fy,
        .cx     = stream_width_  / 2.0f,
        .cy     = stream_height_ / 2.0f,
        .width  = stream_width_,
        .height = stream_height_
    };
}

CameraExtrinsics Simulation::getCameraExtrinsics(const std::string& cam_name) const {
    int cam_id = mj_name2id(model, mjOBJ_CAMERA, cam_name.c_str());
    if (cam_id < 0)
        throw std::runtime_error("[Simulation] getCameraExtrinsics: camera '" + cam_name + "' not found");

    // default camera pose in world frame
    const mjtNum* p = model->cam_pos  + cam_id * 3;
    const mjtNum* q = model->cam_quat + cam_id * 4;  // w, x, y, z

    CameraExtrinsics ext;
    ext.position    = Eigen::Vector3d(p[0], p[1], p[2]);
    ext.orientation = Eigen::Quaterniond(q[0], q[1], q[2], q[3]);
    return ext;
}
