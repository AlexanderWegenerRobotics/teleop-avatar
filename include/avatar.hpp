#include <yaml-cpp/yaml.h>
#include <atomic>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

#include "network/platform_socket.hpp"

#include "arm_control.hpp"
#include "head_control.hpp"
#include "network/udp_reliable.hpp"
#ifndef WITH_FRANKA
#include "sim_env/simulation.hpp"
#endif
#include "self_collision_protection.hpp"
#include "data_logger.hpp"
#include "intention/intention_buffer.hpp"
#include "intention/intention_recognizer.hpp"
#include "intention/scene_objects_msg.hpp"
#include "pipeline/episode_msg.hpp"
#include "twin/role.hpp"
#include "twin/telemetry_forward.hpp"
#include "twin/reconciler.hpp"

class Avatar{

public:
    Avatar(const YAML::Node& config, Role role);
    ~Avatar();

    void start();
    void stop();
    bool isRunning() const {return bRunning;}
    SysState getState() const {return state_;}

#ifndef WITH_FRANKA
    std::shared_ptr<Simulation> getSim() const { return sim_; }
#endif

private:
    void updateStateMachine(SysState cmd_state);
    void processRecoveryNotifications();
    bool allInState(SysState state);
    bool anyoneInState(SysState state);
    void requestAllDevices(SysState state);
    void sendDeviceEvent(const std::string& device, const std::string& event);
    ArmControl* getArm(const std::string& name);
    // Arbitrates an authority request: HOLD always wins, otherwise operator outranks orchestrator
    // (so orchestrator can't take POLICY while operator holds HUMAN).
    void applyAuthorityRequest(ArmControl* arm, CommandAuthority requested, const std::string& source);
    // Sends authority_state to the interface only on change (reliable channel), incl. watchdog transitions.
    void publishAuthorityChanges();
    // forwards orchestrator policy_status to the interface
    void relayPolicyStatus();
    void markEpisodeStart();
    void markEpisodeEnd(const std::string& reason);
    void processResetAllCompletion();

private:
    std::vector<ArmControl*> arm_instances;
	std::vector<HeadControl*> head_instances;
    std::unique_ptr<UdpReliable> cmd_channel_;
    std::atomic<SysState> cmd_requested_{SysState::IDLE};
    std::unique_ptr<DataLogger<SceneLogEntry>> scene_logger_;

#ifndef WITH_FRANKA
    std::shared_ptr<Simulation> sim_ = nullptr;
#endif
    std::atomic<bool> bRunning;
    std::atomic<SysState> state_;
    std::shared_ptr<DeviceRegistry> device_registry_;
    std::unordered_map<std::string, DeviceRecord> device_records_;
    std::atomic<bool> reset_all_pending_{false};
    // last authority sent to the interface; absent = never sent
    std::unordered_map<std::string, CommandAuthority> published_authority_;
    // latest orchestrator policy_status, waiting for relay
    std::mutex  policy_status_mtx_;
    std::string policy_status_buf_;
    bool        policy_status_pending_ = false;

    std::string      session_id_;
    std::string      log_base_dir_;
    socket_t         episode_sock_  = kInvalidSocket;
    int              episode_index_ = 0;

    struct ObjectDef {
        std::string name;
        std::string mujoco_body;
        std::string color;
        std::string model_path;
        double fixed_x = 0, fixed_y = 0, fixed_z = 0;
        double fixed_qw = 1, fixed_qx = 0, fixed_qy = 0, fixed_qz = 0;  // w,x,y,z
        bool   spare = false;
    };

    struct SpawnedObject {
        std::string name;
        std::string color;
        std::string model_path;
        double x = 0, y = 0, z = 0;
        double yaw   = 0.0;   // rad, about z
        double scale = 1.0;   // uniform scale
        // spawn orientation (w,x,y,z), overrides yaw when set
        bool   has_quat = false;
        double qw = 1, qx = 0, qy = 0, qz = 0;
    };

    struct EpisodeConfig {
        int                       seed = 0;
        int                       mode = 0;
        std::string               color_bin_mapping;
        std::vector<SpawnedObject> objects;
        std::vector<SpawnedObject> parked;
#ifndef WITH_FRANKA
        LightingConfig            lighting;
#endif
    };

    std::vector<ObjectDef> object_defs_;
    std::vector<ObjectDef> bin_defs_;

    EpisodeConfig current_episode_cfg_{};

    EpisodeConfig requestEpisodeConfig();
    void          startNewEpisodeFolder();
    void          applyEpisodeConfig(const EpisodeConfig& cfg);
#ifndef WITH_FRANKA
    void          writeCameraParams();
#endif

    std::unique_ptr<IntentionBuffer>     intention_buffer_;
    std::unique_ptr<IntentionRecognizer> intention_recognizer_;
    bool intention_static_camera_ = false;

    // pipeline logger episode signaling
    std::string logger_host_;
    int         logger_port_        = 0;
    socket_t    logger_sock_        = kInvalidSocket;
    int         current_episode_idx_    = -1;  // -1 = no active episode
    std::string current_episode_folder_;      // set by startNewEpisodeFolder

    void sendEpisodeEvent(const std::string& type, const std::string& reason);

    // scene objects for the orchestrator
    std::string scene_objects_host_;
    int         scene_objects_port_ = 0;
    // control loop rate = orchestrator tick rate
    double      loop_rate_hz_ = 100.0;
    socket_t    scene_objects_sock_ = kInvalidSocket;

    void sendSceneObjects(const StateSnapshot& snap);

    // twin / reconciler
    Role role_ = Role::Avatar;

    // role avatar: forwards joint telemetry to the twin, no-op if not configured
    std::unique_ptr<TelemetryForwarder> twin_telemetry_;
    void sendTwinTelemetry();

    // role twin only, needs WITH_MUJOCO
    std::unique_ptr<Reconciler> reconciler_;
    // logs Reconciler::getStats() once per tick, twin only
    std::unique_ptr<DataLogger<ReconcilerLogEntry>> reconciler_logger_;
};