#include <atomic>
#include <csignal>
#include <iostream>
#include <memory>
#include <thread>
#include <vector>

#include <gst/gst.h>
#include <yaml-cpp/yaml.h>

// Winsock must be initialised before any socket() call on Windows.
// Include platform_socket.hpp first so winsock2.h beats any windows.h pull-in.
#include "network/platform_socket.hpp"

#include "pipeline/camera_channel.hpp"
#include "pipeline/episode_controller.hpp"

// ---------------------------------------------------------------------------
// Global state (signal handler needs access)
// ---------------------------------------------------------------------------

static std::vector<std::unique_ptr<CameraChannel>> g_channels;
static std::unique_ptr<EpisodeController>           g_episode_ctrl;
static std::atomic<bool>                            g_running{true};

static void onSignal(int) { g_running = false; }

// ---------------------------------------------------------------------------
// YAML parsing helpers
// ---------------------------------------------------------------------------

static StreamerConfig parseStreamConfig(const YAML::Node& n, int default_fps) {
    StreamerConfig c;
    c.host               = n["host"].as<std::string>();
    c.port               = n["port"].as<int>();
    c.feedback_port      = n["feedback_port"].as<int>(5005);
    c.status_port        = n["status_port"].as<int>(5007);
    c.status_interval_ms = n["status_interval_ms"].as<int>(500);
    c.fps                = n["fps"].as<int>(default_fps);
    c.bitrate_kbps       = n["bitrate_kbps"].as<int>(2000);
    c.fec_percentage     = n["fec_percentage"].as<int>(10);
    c.stream_width       = n["width"].as<int>(640);
    c.stream_height      = n["height"].as<int>(480);
    return c;
}

static LoggerConfig parseLogConfig(const YAML::Node& n) {
    LoggerConfig c;
    c.output_dir   = n["output_dir"].as<std::string>("logs");
    c.width        = n["width"].as<int>(224);
    c.height       = n["height"].as<int>(224);
    c.center_crop  = n["center_crop"].as<bool>(false);
    c.crop_x       = n["crop_x"].as<int>(0);
    c.crop_y       = n["crop_y"].as<int>(0);
    c.crop_w       = n["crop_w"].as<int>(0);
    c.crop_h       = n["crop_h"].as<int>(0);
    // camera_name is filled in by CameraChannel from its own name
    return c;
}

static CameraChannelConfig parseCameraConfig(const YAML::Node& n) {
    CameraChannelConfig c;
    c.name             = n["name"].as<std::string>();
    c.source_type      = n["source_type"].as<std::string>("mujoco");
    c.shm_name         = n["shm_name"].as<std::string>("");
    c.realsense_serial = n["realsense_serial"].as<std::string>("");
    c.fps              = n["fps"].as<int>(30);
    c.source_width     = n["source_width"].as<int>(640);
    c.source_height    = n["source_height"].as<int>(480);
    c.stereo_combined    = n["stereo_combined"].as<bool>(false);
    c.stereo_partner_shm = n["stereo_partner_shm"].as<std::string>("");

    if (n["stream"]) {
        c.stream_enabled = n["stream"]["enabled"].as<bool>(false);
        if (c.stream_enabled)
            c.stream = parseStreamConfig(n["stream"], c.fps);
    }

    if (n["log"]) {
        c.log_enabled = n["log"]["enabled"].as<bool>(false);
        if (c.log_enabled)
            c.log = parseLogConfig(n["log"]);
    }

    return c;
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------

int main(int argc, char** argv) {
#ifdef _WIN32
    WSADATA wsa{};
    if (WSAStartup(MAKEWORD(2, 2), &wsa) != 0) {
        std::cerr << "[ERROR] WSAStartup failed" << std::endl;
        return 1;
    }
#endif

    std::string global_config_path = "../config/config.yaml";
    if (argc > 1) global_config_path = argv[1];

    YAML::Node global_cfg;
    try {
        global_cfg = YAML::LoadFile(global_config_path);
    } catch (const std::exception& e) {
        std::cerr << "[ERROR] Failed to load global config: " << e.what() << std::endl;
    #ifdef _WIN32
            WSACleanup();
    #endif
        return 1;
    }

    std::string config_path = global_cfg["streamer_config"].as<std::string>("../config/pipeline_config.yaml");

    YAML::Node cfg;
    try {
        cfg = YAML::LoadFile(config_path);
    } catch (const std::exception& e) {
        std::cerr << "[ERROR] Failed to load streamer config: " << e.what() << std::endl;
    #ifdef _WIN32
            WSACleanup();
    #endif
        return 1;
    }

    // GStreamer must be initialised once before any VideoStreamer is constructed.
    gst_init(nullptr, nullptr);

    // ── Stereo mode flag ──────────────────────────────────────────────────
    bool stereo = cfg["stereo"].as<bool>(false);
    std::cout << "[INFO] Stereo mode: " << (stereo ? "enabled" : "disabled") << std::endl;

    // ── Build camera channels ──────────────────────────────────────────────
    if (!cfg["cameras"] || !cfg["cameras"].IsSequence()) {
        std::cerr << "[ERROR] 'cameras' sequence missing from config" << std::endl;
        return 1;
    }

    try {
        for (const auto& cam_node : cfg["cameras"]) {
            // Filter by eye tag: in mono mode keep "mono" entries (or entries
            // with no eye key); in stereo mode keep "left" and "right" entries.
            std::string eye = cam_node["eye"].as<std::string>("mono");
            // "stereo" = single side-by-side combined stream (new approach).
            // "left"/"right" = legacy two-stream approach (kept for compatibility).
            bool active = stereo ? (eye == "stereo" || eye == "left" || eye == "right")
                                 : (eye == "mono");
            if (!active) {
                std::cout << "[INFO] Skipping camera (eye=" << eye
                          << ", stereo=" << stereo << "): "
                          << cam_node["name"].as<std::string>("?") << std::endl;
                continue;
            }

            auto cam_cfg = parseCameraConfig(cam_node);
            std::cout << "[INFO] Registering camera: " << cam_cfg.name
                      << "  eye=" << eye
                      << "  source=" << cam_cfg.source_type
                      << "  stream=" << cam_cfg.stream_enabled
                      << "  log="    << cam_cfg.log_enabled << std::endl;
            g_channels.push_back(std::make_unique<CameraChannel>(cam_cfg));
        }
    } catch (const std::exception& e) {
        std::cerr << "[ERROR] Camera init: " << e.what() << std::endl;
        return 1;
    }

    // ── Episode controller ─────────────────────────────────────────────────
    int ep_port = cfg["episode_listener_port"].as<int>(7000);
    g_episode_ctrl = std::make_unique<EpisodeController>(ep_port);

    g_episode_ctrl->onEpisodeStart([](const std::string& sess, int idx) {
        for (auto& ch : g_channels) ch->onEpisodeStart(sess, idx);
    });
    g_episode_ctrl->onEpisodeEnd([](const std::string& sess, int idx,
                                    const std::string& reason) {
        for (auto& ch : g_channels) ch->onEpisodeEnd(sess, idx, reason);
    });

    // ── Signal handling ────────────────────────────────────────────────────
    std::signal(SIGINT,  onSignal);
    std::signal(SIGTERM, onSignal);

    // ── Start ──────────────────────────────────────────────────────────────
    try {
        g_episode_ctrl->start();
        for (auto& ch : g_channels) ch->start();

        std::cout << "[INFO] Pipeline running with " << g_channels.size()
                  << " camera(s). Ctrl+C to stop." << std::endl;

        while (g_running)
            std::this_thread::sleep_for(std::chrono::milliseconds(100));

    } catch (const std::exception& e) {
        std::cerr << "[ERROR] " << e.what() << std::endl;
    }

    // ── Shutdown ───────────────────────────────────────────────────────────
    std::cout << "[INFO] Shutting down..." << std::endl;
    for (auto& ch : g_channels) ch->stop();
    g_episode_ctrl->stop();
    g_channels.clear();

    gst_deinit();
#ifdef _WIN32
    WSACleanup();
#endif
    return 0;
}
