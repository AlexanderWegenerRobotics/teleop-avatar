#pragma once

#include <array>
#include <atomic>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <functional>
#include <mutex>
#include <string>
#include <thread>

#include "common.hpp"

struct ArmLogEntry {
    double                 time;          // s since logger start
    uint64_t               wall_clock_ns; // UNIX epoch ns (system_clock)
    // mjData::time (sim seconds), resample on this. 0 on hardware
    double                 sim_time;
    std::array<double, 7>  q;
    std::array<double, 7>  q_cmd;
    std::array<double, 7>  dq;
    std::array<double, 7>  tau_J;
    // torque actually sent to franka::Torques (after rate limit + saturation)
    std::array<double, 7>  tau_cmd;
    std::array<double, 7>  tau_ext;
    std::array<double, 16> O_T_EE;
    std::array<double, 16> O_T_EE_cmd;
    std::array<double, 16> O_T_EE_world;      // world frame, T_base_ * O_T_EE
    std::array<double, 16> O_T_EE_cmd_world;  // world frame, T_base_ * O_T_EE_cmd
    std::array<double, 6>  F_ext;
    double                 gripper_width;
    double                 gripper_cmd;
    uint8_t                grasp_state;
    // nullspace posture ref used this tick (q0 if optimizer off) + optimizer outputs
    std::array<double, 7>  q_null_ref;
    double                 posture_s;
    double                 posture_cost;
    double                 posture_margin;
    double                 posture_swivel;
    SysState               state;
    // 1 if O_T_EE_cmd is a real command, 0 = columns hold the measured pose
    uint8_t                cmd_valid;
    // 0 = control callback (~1 kHz), 1 = state thread (~200 Hz) while control() isn't running.
    // File is not uniformly sampled, don't interpolate across log_src = 1.
    uint8_t                log_src;
    // raw operator grip input (1 = held) in every state, unlike gripper_cmd
    uint8_t                grasp_cmd;
    // 1 = operator clutched (dead time, cut before training). 1 before first command
    uint8_t                clutch;
    // sequence of last consumed ArmCommandMsg, for aligning with operator-side logs. 0 until first command
    uint32_t               applied_cmd_sequence;
    // CommandAuthority: 0 POLICY, 1 HUMAN, 2 HOLD, 255 UNSET. Appended at the end to keep old column order
    uint8_t                authority;
    // friction feedforward part of tau_cmd, Nm
    std::array<double, 7>  tau_friction;
};

// One row per state-thread tick (~200 Hz), keeps logging through faults when arm.csv stops.
struct ArmStateTraceEntry {
    double   time;                  // s since logger start
    uint64_t wall_clock_ns;         // state-thread tick time
    uint64_t control_sample_ns;     // last robot read by the control thread
    double   control_age_ms;        // grows if control stalls
    uint32_t control_loop_entries;  // increments on every fault
    uint32_t fault_count;           // consecutive faults
    SysState state;
    uint8_t  recovering;
};

// no sim_time here, map wall->sim time through the arm CSV
struct HeadLogEntry {
    double                time;          // s since logger start
    uint64_t              wall_clock_ns; // UNIX epoch ns (system_clock)
    std::array<double, 2> q;
    std::array<double, 2> q_cmd;
    std::array<double, 2> dq;
    std::array<double, 2> tau_J;
    SysState              state;
};

template<typename T>
class DataLogger {
public:
    DataLogger(const std::string& path,
               std::function<std::string()>         headerFn,
               std::function<std::string(const T&)> rowFn,
               const std::string& session_id)
        : headerFn_(headerFn)
        , rowFn_(rowFn)
        , session_id_(session_id)
        , bRunning_(false)
        , bEnabled_(false)
        , bHasNewData_(false)
    {
        openFiles(path);
    }

    ~DataLogger() { stop(); }

    void start() {
        file_ << headerFn_();
        bRunning_ = true;
        startTime_ = std::chrono::high_resolution_clock::now();
        thread_ = std::thread(&DataLogger::run, this);
    }

    void stop() {
        bRunning_ = false;
        if (thread_.joinable()) thread_.join();
        file_.close();
        meta_file_.close();
    }

    // reopens files at new_path and restarts the logging thread, used between episodes
    void restart(const std::string& new_path) {
        bEnabled_ = false;
        bRunning_ = false;
        if (thread_.joinable()) thread_.join();
        file_.close();
        meta_file_.close();

        episode_id_ = 0;
        openFiles(new_path);

        file_ << headerFn_();
        bRunning_ = true;
        startTime_ = std::chrono::high_resolution_clock::now();
        thread_ = std::thread(&DataLogger::run, this);
        bEnabled_ = true; 
    }

    void enable(bool e) { 
        bEnabled_ = e; 
    }

    void write(const T& data) {
        if (!bEnabled_) return;
        std::lock_guard<std::mutex> lock(mtx_);
        data_        = data;
        bHasNewData_ = true;
    }

    void markEpisodeStart() {
        writeMarker("episode_start", "");
        episode_id_++;
    }

    void markEpisodeEnd(const std::string& reason) {
        writeMarker("episode_end", reason);
    }

    // writes an annotation row to the meta file
    void writeAnnotation(const std::string& label, uint8_t atype,
                         float confidence, float score, uint64_t frame_id) {
        double t = std::chrono::duration<double>(
            std::chrono::high_resolution_clock::now() - startTime_).count();
        std::lock_guard<std::mutex> lock(meta_mtx_);
        meta_file_ << session_id_ << ";"
                   << episode_id_ << ";"
                   << "annotation" << ";"
                   << t << ";"
                   << ";" << ";" << ";"
                   << label << ";"
                   << static_cast<int>(atype) << ";"
                   << confidence << ";"
                   << score << ";"
                   << frame_id << "\n";
        meta_file_.flush();
    }

    void writeEpisodeConfig(int seed, int mode, const std::string& color_bin_mapping)
    {
        double t = std::chrono::duration<double>(
            std::chrono::high_resolution_clock::now() - startTime_).count();
        std::lock_guard<std::mutex> lock(meta_mtx_);
        meta_file_ << session_id_ << ";"
                   << episode_id_ << ";"
                   << "episode_config" << ";"
                   << t << ";"
                   << seed << ";"
                   << mode << ";"
                   << color_bin_mapping << ";"
                   << ";;;;\n";
        meta_file_.flush();
    }

private:
    void openFiles(const std::string& path) {
        std::filesystem::path p(path);
        if (p.has_parent_path())
            std::filesystem::create_directories(p.parent_path());
        file_.open(path);
        if (!file_)
            throw std::runtime_error("DataLogger: failed to open file: " + path);

        std::string meta_path = path.substr(0, path.rfind('.')) + "_meta.csv";
        meta_file_.open(meta_path);
        if (!meta_file_)
            throw std::runtime_error("DataLogger: failed to open meta file: " + meta_path);

        meta_file_ << "session_id;episode_id;event;time_s;seed;mode;color_bin_mapping;ann_label;ann_atype;ann_confidence;ann_score;ann_frame_id\n";
        meta_file_.flush();
    }

    void writeMarker(const std::string& event, const std::string& reason) {
        double t = std::chrono::duration<double>(
            std::chrono::high_resolution_clock::now() - startTime_).count();
        std::lock_guard<std::mutex> lock(meta_mtx_);
        meta_file_ << session_id_ << ";"
                   << episode_id_ << ";"
                   << event       << ";"
                   << t           << ";"
                   << ";" << ";"
                   << reason      << ";"
                   << ";;;;\n";
        meta_file_.flush();
    }

    void run() {
        std::string buffer;
        buffer.reserve(1024 * 1024);
        int rowCount = 0;
        while (bRunning_) {
            if (bHasNewData_ && bEnabled_) {
                T snapshot;
                {
                    std::lock_guard<std::mutex> lock(mtx_);
                    snapshot     = data_;
                    bHasNewData_ = false;
                }
                buffer += rowFn_(snapshot);
                if (++rowCount >= 50) {
                    file_.write(buffer.data(), buffer.size());
                    buffer.clear();
                    rowCount = 0;
                }
            }
            std::this_thread::sleep_for(std::chrono::microseconds(500));
        }
        if (!buffer.empty())
            file_.write(buffer.data(), buffer.size());
    }

    std::function<std::string()>         headerFn_;
    std::function<std::string(const T&)> rowFn_;
    std::string                          session_id_;

    std::ofstream file_;
    std::ofstream meta_file_;
    std::thread   thread_;

    std::atomic<bool> bRunning_;
    std::atomic<bool> bEnabled_;
    std::atomic<bool> bHasNewData_;

    std::mutex mtx_;
    std::mutex meta_mtx_;
    T          data_{};
    int        episode_id_ = 0;

    std::chrono::high_resolution_clock::time_point startTime_;
};

inline std::string armLogHeader() {
    std::string h = "time;wall_clock_ns;sim_time;";
    for (int i = 0; i < 7;  ++i) h += "q_"          + std::to_string(i) + ";";
    for (int i = 0; i < 7;  ++i) h += "q_cmd_"      + std::to_string(i) + ";";
    for (int i = 0; i < 7;  ++i) h += "dq_"         + std::to_string(i) + ";";
    for (int i = 0; i < 7;  ++i) h += "tau_J_"      + std::to_string(i) + ";";
    for (int i = 0; i < 7;  ++i) h += "tau_cmd_"    + std::to_string(i) + ";";
    for (int i = 0; i < 7;  ++i) h += "tau_ext_"    + std::to_string(i) + ";";
    for (int i = 0; i < 16; ++i) h += "O_T_EE_"           + std::to_string(i) + ";";
    for (int i = 0; i < 16; ++i) h += "O_T_EE_cmd_"       + std::to_string(i) + ";";
    for (int i = 0; i < 16; ++i) h += "O_T_EE_world_"     + std::to_string(i) + ";";
    for (int i = 0; i < 16; ++i) h += "O_T_EE_cmd_world_" + std::to_string(i) + ";";
    for (int i = 0; i < 6;  ++i) h += "F_ext_"            + std::to_string(i) + ";";
    h += "gripper_width;";
    h += "gripper_cmd;";
    h += "grasp_state;";
    for (int i = 0; i < 7;  ++i) h += "q_null_ref_" + std::to_string(i) + ";";
    h += "posture_s;posture_cost;posture_margin;posture_swivel;";
    h += "state;cmd_valid;log_src;grasp_cmd;clutch;applied_cmd_sequence;authority;";
    for (int i = 0; i < 7;  ++i) h += "tau_friction_" + std::to_string(i) + (i < 6 ? ";" : "\n");
    return h;
}

inline std::string armLogRow(const ArmLogEntry& e) {
    std::string r = std::to_string(e.time) + ";" + std::to_string(e.wall_clock_ns) + ";"
                  + std::to_string(e.sim_time) + ";";
    for (auto v : e.q)          r += std::to_string(v) + ";";
    for (auto v : e.q_cmd)      r += std::to_string(v) + ";";
    for (auto v : e.dq)         r += std::to_string(v) + ";";
    for (auto v : e.tau_J)      r += std::to_string(v) + ";";
    for (auto v : e.tau_cmd)    r += std::to_string(v) + ";";
    for (auto v : e.tau_ext)    r += std::to_string(v) + ";";
    for (auto v : e.O_T_EE)           r += std::to_string(v) + ";";
    for (auto v : e.O_T_EE_cmd)       r += std::to_string(v) + ";";
    for (auto v : e.O_T_EE_world)     r += std::to_string(v) + ";";
    for (auto v : e.O_T_EE_cmd_world) r += std::to_string(v) + ";";
    for (auto v : e.F_ext)            r += std::to_string(v) + ";";
    r += std::to_string(e.gripper_width) + ";";
    r += std::to_string(e.gripper_cmd) + ";";
    r += std::to_string(e.grasp_state) + ";";
    for (auto v : e.q_null_ref) r += std::to_string(v) + ";";
    r += std::to_string(e.posture_s) + ";" + std::to_string(e.posture_cost) + ";"
       + std::to_string(e.posture_margin) + ";" + std::to_string(e.posture_swivel) + ";";
    r += std::to_string(static_cast<uint8_t>(e.state)) + ";";
    r += std::to_string(e.cmd_valid) + ";";
    r += std::to_string(e.log_src) + ";";
    r += std::to_string(e.grasp_cmd) + ";";
    r += std::to_string(e.clutch) + ";";
    r += std::to_string(e.applied_cmd_sequence) + ";";
    r += std::to_string(e.authority) + ";";
    for (int i = 0; i < 7; ++i) r += std::to_string(e.tau_friction[i]) + (i < 6 ? ";" : "\n");
    return r;
}

// One row per twin control tick, logs Reconciler::getStats().
struct ReconcilerLogEntry {
    double   time;
    uint64_t wall_clock_ns;
    double   innovation_norm_rad;   // ||e|| = ||y - x_hat(t_s - d_f)||
    double   measured_d_f_s;        // forward delay
    double   measured_d_b_s;
    uint8_t  regime;                // 0 = Soft, 1 = Hard
    uint64_t hard_resync_count;     // cumulative
    uint64_t packets_received;      // cumulative
};

inline std::string reconcilerLogHeader() {
    return "time;wall_clock_ns;innovation_norm_rad;measured_d_f_s;measured_d_b_s;"
           "regime;hard_resync_count;packets_received\n";
}

inline std::string reconcilerLogRow(const ReconcilerLogEntry& e) {
    std::string r;
    r += std::to_string(e.time) + ";";
    r += std::to_string(e.wall_clock_ns) + ";";
    r += std::to_string(e.innovation_norm_rad) + ";";
    r += std::to_string(e.measured_d_f_s) + ";";
    r += std::to_string(e.measured_d_b_s) + ";";
    r += std::to_string(static_cast<int>(e.regime)) + ";";
    r += std::to_string(e.hard_resync_count) + ";";
    r += std::to_string(e.packets_received) + "\n";
    return r;
}

inline std::string armStateTraceHeader() {
    return "time;wall_clock_ns;control_sample_ns;control_age_ms;"
           "control_loop_entries;fault_count;state;recovering\n";
}

inline std::string armStateTraceRow(const ArmStateTraceEntry& e) {
    std::string r;
    r += std::to_string(e.time) + ";";
    r += std::to_string(e.wall_clock_ns) + ";";
    r += std::to_string(e.control_sample_ns) + ";";
    r += std::to_string(e.control_age_ms) + ";";
    r += std::to_string(e.control_loop_entries) + ";";
    r += std::to_string(e.fault_count) + ";";
    r += std::to_string(static_cast<uint8_t>(e.state)) + ";";
    r += std::to_string(static_cast<int>(e.recovering)) + "\n";
    return r;
}

inline std::string headLogHeader() {
    std::string h = "time;wall_clock_ns;";
    for (int i = 0; i < 2; ++i) h += "q_"     + std::to_string(i) + ";";
    for (int i = 0; i < 2; ++i) h += "q_cmd_" + std::to_string(i) + ";";
    for (int i = 0; i < 2; ++i) h += "dq_"    + std::to_string(i) + ";";
    for (int i = 0; i < 2; ++i) h += "tau_J_" + std::to_string(i) + ";";
    h += "state\n";
    return h;
}

inline std::string headLogRow(const HeadLogEntry& e) {
    std::string r = std::to_string(e.time) + ";" + std::to_string(e.wall_clock_ns) + ";";
    for (auto v : e.q)     r += std::to_string(v) + ";";
    for (auto v : e.q_cmd) r += std::to_string(v) + ";";
    for (auto v : e.dq)    r += std::to_string(v) + ";";
    for (auto v : e.tau_J) r += std::to_string(v) + ";";
    r += std::to_string(static_cast<uint8_t>(e.state)) + "\n";
    return r;
}

struct SceneLogEntry {
    static constexpr int MAX_OBJECTS = 4;
    static constexpr int MAX_BINS    = 4;

    double   time          = 0.0;  // s since logger start
    uint64_t wall_clock_ns = 0;    // UNIX epoch ns (system_clock)
    int      mode          = 0;
    int      seed          = 0;

    int n_objects = 0;
    std::array<std::string,           MAX_OBJECTS> object_names;
    std::array<std::array<double, 3>, MAX_OBJECTS> object_pos{};
    std::array<std::array<double, 4>, MAX_OBJECTS> object_quat{};

    int n_bins = 0;
    std::array<std::string,           MAX_BINS> bin_names;
    std::array<std::array<double, 3>, MAX_BINS> bin_pos{};
    std::array<std::array<double, 4>, MAX_BINS> bin_quat{};

    // spawn config, constant per episode
    std::array<std::string, MAX_OBJECTS> object_colors;
    std::array<double, MAX_OBJECTS> object_spawn_yaw{};    // rad
    std::array<double, MAX_OBJECTS> object_scale{1.0, 1.0, 1.0, 1.0};

    // lighting, constant per episode
    std::array<float, 3> light_main_pos        = {0.5f,  0.0f,  1.8f};
    std::array<float, 3> light_main_diffuse    = {0.8f,  0.8f,  0.8f};
    std::array<float, 3> light_main_specular   = {0.2f,  0.2f,  0.2f};
    std::array<float, 3> light_fill_diffuse    = {0.25f, 0.25f, 0.25f};
    std::array<float, 3> light_headlight_diffuse = {0.4f, 0.4f, 0.4f};
    std::array<float, 3> light_headlight_ambient = {0.25f, 0.25f, 0.25f};
};

inline std::string sceneLogHeader() {
    std::string h = "time;wall_clock_ns;mode;seed;";
    for (int i = 0; i < SceneLogEntry::MAX_OBJECTS; ++i) {
        std::string p = "obj" + std::to_string(i) + "_";
        h += p + "name;" + p + "color;" + p + "x;" + p + "y;" + p + "z;"
           + p + "qw;" + p + "qx;" + p + "qy;" + p + "qz;"
           + p + "spawn_yaw;" + p + "scale;";
    }
    for (int i = 0; i < SceneLogEntry::MAX_BINS; ++i) {
        std::string p = "bin" + std::to_string(i) + "_";
        h += p + "name;" + p + "x;" + p + "y;" + p + "z;"
           + p + "qw;" + p + "qx;" + p + "qy;" + p + "qz;";
    }
    h += "n_objects;n_bins;"
         "light_main_pos_x;light_main_pos_y;light_main_pos_z;"
         "light_main_diffuse_r;light_main_diffuse_g;light_main_diffuse_b;"
         "light_main_specular_r;light_main_specular_g;light_main_specular_b;"
         "light_fill_diffuse_r;light_fill_diffuse_g;light_fill_diffuse_b;"
         "light_headlight_diffuse_r;light_headlight_diffuse_g;light_headlight_diffuse_b;"
         "light_headlight_ambient_r;light_headlight_ambient_g;light_headlight_ambient_b\n";
    return h;
}

inline std::string sceneLogRow(const SceneLogEntry& e) {
    std::string r = std::to_string(e.time) + ";" + std::to_string(e.wall_clock_ns) + ";"
                  + std::to_string(e.mode) + ";" + std::to_string(e.seed) + ";";
    for (int i = 0; i < SceneLogEntry::MAX_OBJECTS; ++i) {
        const auto& p = e.object_pos[i];
        const auto& q = e.object_quat[i];
        r += (i < e.n_objects ? e.object_names[i] : "") + ";"
           + (i < e.n_objects ? e.object_colors[i] : "") + ";"
           + std::to_string(p[0]) + ";" + std::to_string(p[1]) + ";" + std::to_string(p[2]) + ";"
           + std::to_string(q[0]) + ";" + std::to_string(q[1]) + ";" + std::to_string(q[2]) + ";" + std::to_string(q[3]) + ";"
           + std::to_string(e.object_spawn_yaw[i]) + ";"
           + std::to_string(e.object_scale[i]) + ";";
    }
    for (int i = 0; i < SceneLogEntry::MAX_BINS; ++i) {
        const auto& p = e.bin_pos[i];
        const auto& q = e.bin_quat[i];
        r += (i < e.n_bins ? e.bin_names[i] : "") + ";"
           + std::to_string(p[0]) + ";" + std::to_string(p[1]) + ";" + std::to_string(p[2]) + ";"
           + std::to_string(q[0]) + ";" + std::to_string(q[1]) + ";" + std::to_string(q[2]) + ";" + std::to_string(q[3]) + ";";
    }
    r += std::to_string(e.n_objects) + ";" + std::to_string(e.n_bins) + ";";
    for (float v : e.light_main_pos)           r += std::to_string(v) + ";";
    for (float v : e.light_main_diffuse)       r += std::to_string(v) + ";";
    for (float v : e.light_main_specular)      r += std::to_string(v) + ";";
    for (float v : e.light_fill_diffuse)       r += std::to_string(v) + ";";
    for (float v : e.light_headlight_diffuse)  r += std::to_string(v) + ";";
    r += std::to_string(e.light_headlight_ambient[0]) + ";"
       + std::to_string(e.light_headlight_ambient[1]) + ";"
       + std::to_string(e.light_headlight_ambient[2]) + "\n";
    return r;
}
