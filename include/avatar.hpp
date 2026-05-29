#include <yaml-cpp/yaml.h>
#include <atomic>
#include <unordered_map>

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
#include "pipeline/episode_msg.hpp"

class Avatar{

public:
    Avatar(const YAML::Node& config);
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

    std::string      session_id_;
    std::string      log_base_dir_;
    socket_t         episode_sock_  = kInvalidSocket;
    int              episode_index_ = 0;

    struct EpisodeConfig {
        double pick_x, pick_y, pick_z;
        double place_x, place_y, place_z;
        int    mode;
    };
    EpisodeConfig current_episode_cfg_{};

    EpisodeConfig requestEpisodeConfig();
    void          startNewEpisodeFolder();
    void applyEpisodeConfig(const EpisodeConfig& cfg);

    std::unique_ptr<IntentionBuffer>     intention_buffer_;
    std::unique_ptr<IntentionRecognizer> intention_recognizer_;

    // ── Pipeline logger episode signaling ─────────────────────────────────
    std::string logger_host_;
    int         logger_port_        = 0;
    socket_t    logger_sock_        = kInvalidSocket;
    int         current_episode_idx_ = -1;   // -1 = no active episode

    void sendEpisodeEvent(const std::string& type, const std::string& reason);
};