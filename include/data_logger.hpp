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
    double                 time;          // seconds since logger start (relative)
    uint64_t               wall_clock_ns; // UNIX epoch nanoseconds (system_clock, same source as intention timestamp_arrival_ns)
    std::array<double, 7>  q;
    std::array<double, 7>  q_cmd;
    std::array<double, 7>  dq;
    std::array<double, 7>  tau_J;
    std::array<double, 7>  tau_ext;
    std::array<double, 16> O_T_EE;
    std::array<double, 16> O_T_EE_cmd;
    std::array<double, 6>  F_ext;
    double                 gripper_width;
    double                 gripper_cmd;
    SysState               state;
};

struct HeadLogEntry {
    double                time;          // seconds since logger start (relative)
    uint64_t              wall_clock_ns; // UNIX epoch nanoseconds (system_clock, same source as intention timestamp_arrival_ns)
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

    // Close current files, open new ones at new_path, restart logging thread.
    // Call this between episodes when you want a fresh file in a new folder.
    void restart(const std::string& new_path) {
        // Stop logging thread and flush
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

    // Write episode config (pick/place pose, mode) to meta file once per episode
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
    std::string h = "time;wall_clock_ns;";
    for (int i = 0; i < 7;  ++i) h += "q_"          + std::to_string(i) + ";";
    for (int i = 0; i < 7;  ++i) h += "q_cmd_"      + std::to_string(i) + ";";
    for (int i = 0; i < 7;  ++i) h += "dq_"         + std::to_string(i) + ";";
    for (int i = 0; i < 7;  ++i) h += "tau_J_"      + std::to_string(i) + ";";
    for (int i = 0; i < 7;  ++i) h += "tau_ext_"    + std::to_string(i) + ";";
    for (int i = 0; i < 16; ++i) h += "O_T_EE_"     + std::to_string(i) + ";";
    for (int i = 0; i < 16; ++i) h += "O_T_EE_cmd_" + std::to_string(i) + ";";
    for (int i = 0; i < 6;  ++i) h += "F_ext_"      + std::to_string(i) + ";";
    h += "gripper_width;";
    h += "gripper_cmd;";
    h += "state\n";
    return h;
}

inline std::string armLogRow(const ArmLogEntry& e) {
    std::string r = std::to_string(e.time) + ";" + std::to_string(e.wall_clock_ns) + ";";
    for (auto v : e.q)          r += std::to_string(v) + ";";
    for (auto v : e.q_cmd)      r += std::to_string(v) + ";";
    for (auto v : e.dq)         r += std::to_string(v) + ";";
    for (auto v : e.tau_J)      r += std::to_string(v) + ";";
    for (auto v : e.tau_ext)    r += std::to_string(v) + ";";
    for (auto v : e.O_T_EE)     r += std::to_string(v) + ";";
    for (auto v : e.O_T_EE_cmd) r += std::to_string(v) + ";";
    for (auto v : e.F_ext)      r += std::to_string(v) + ";";
    r += std::to_string(e.gripper_width) + ";";
    r += std::to_string(e.gripper_cmd) + ";";
    r += std::to_string(static_cast<uint8_t>(e.state)) + "\n";
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

    double   time          = 0.0;  // seconds since logger start (relative)
    uint64_t wall_clock_ns = 0;    // UNIX epoch nanoseconds (system_clock, same source as intention timestamp_arrival_ns)
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

    // Per-object spawn config — constant per episode, repeated each row
    std::array<std::string, MAX_OBJECTS> object_colors;
    std::array<double, MAX_OBJECTS> object_spawn_yaw{};    // Z-rotation at spawn (radians)
    std::array<double, MAX_OBJECTS> object_scale{1.0, 1.0, 1.0, 1.0};  // uniform scale factor

    // Lighting — constant per episode, repeated each row for easy frame-level lookup
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
