#include <yaml-cpp/yaml.h>
#include <atomic>
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
    // Arbitrates one authority_request against what an arm already holds, then
    // applies it. The rule, in order:
    //
    //   1. HOLD from any source wins immediately. It is the only request that
    //      cannot make the robot move, so there is never a reason to refuse it.
    //   2. Otherwise the operator outranks the orchestrator.
    //   3. Therefore an orchestrator request for POLICY is refused while the
    //      operator holds HUMAN -- the operator hands the arm back explicitly,
    //      and a policy that has decided it is ready again cannot take it out
    //      from under a hand that is mid-correction.
    //
    // `source` is the requester's own label ("operator" / "orchestrator"), so a
    // misbehaving client could claim to be the operator. That is acceptable:
    // both clients are already trusted to command the arms directly, so this
    // arbitrates cooperating processes rather than being a security boundary.
    void applyAuthorityRequest(ArmControl* arm, CommandAuthority requested, const std::string& source);
    // Sends an authority_state to the interface for any arm whose authority has
    // changed since the last call. Edge-driven rather than per-tick because the
    // interface sits on the RELIABLE channel, where re-asserting at loop rate
    // would be one ack per arm per tick; the orchestrator gets the per-tick
    // re-assert instead, inside SceneObjectsMsg, where a drop costs 10 ms.
    //
    // Driven from the loop rather than from applyAuthorityRequest so it also
    // catches the transitions ArmControl makes by itself -- the staleness
    // watchdog above all, which is exactly the case where the interface must
    // not go on believing it holds the arm.
    void publishAuthorityChanges();
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
    // Last authority published to the interface, per device name. Absent = never
    // published, so the first pass always sends one and the HUD starts correct
    // rather than starting at a guess.
    std::unordered_map<std::string, CommandAuthority> published_authority_;

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
    };

    struct SpawnedObject {
        std::string name;
        std::string color;
        std::string model_path;
        double x = 0, y = 0, z = 0;
        double yaw   = 0.0;   // Z-axis rotation (radians)
        double scale = 1.0;   // uniform size scale factor
    };

    struct EpisodeConfig {
        int                       seed = 0;
        int                       mode = 0;
        std::string               color_bin_mapping;
        std::vector<SpawnedObject> objects;
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

    // ── Pipeline logger episode signaling ─────────────────────────────────
    std::string logger_host_;
    int         logger_port_        = 0;
    socket_t    logger_sock_        = kInvalidSocket;
    int         current_episode_idx_    = -1;  // -1 = no active episode
    std::string current_episode_folder_;      // absolute path set by startNewEpisodeFolder

    void sendEpisodeEvent(const std::string& type, const std::string& reason);

    // ── Scene object geometry, published for external consumers (orchestrator) ──
    std::string scene_objects_host_;
    int         scene_objects_port_ = 0;
    // Avatar control-loop rate, and therefore the orchestrator's tick rate:
    // tick_id is stamped once per iteration and LiveSource paces on it.
    double      loop_rate_hz_ = 100.0;
    socket_t    scene_objects_sock_ = kInvalidSocket;

    void sendSceneObjects(const StateSnapshot& snap);

    // ── Twin / reconciler (docs/twin_concept.md) ────────────────────────────
    Role role_ = Role::Avatar;

    // role == Avatar: forwards real joint telemetry y(t_s) to a paired twin's
    // reconciler. Present-but-disabled (nullptr behavior via enabled()) when
    // no twin_telemetry block is configured, so this is a no-op by default.
    std::unique_ptr<TelemetryForwarder> twin_telemetry_;
    void sendTwinTelemetry();

    // role == Twin: predicts hardware state and reconciles it against
    // buffered avatar telemetry. Only constructed when role == Twin AND the
    // build has WITH_MUJOCO (Reconciler throws in its ctor otherwise --
    // Avatar checks role first so this never fires from a mis-set role on a
    // real-hardware/WITH_FRANKA build, since that build never sees role: twin
    // in practice, but the check is defense-in-depth either way).
    std::unique_ptr<Reconciler> reconciler_;
    // Drains Reconciler::getStats() once per tick. Created alongside the
    // reconciler and only when one exists (role: twin). Without this the
    // stats are computed and thrown away every cycle -- see
    // ReconcilerLogEntry in data_logger.hpp.
    std::unique_ptr<DataLogger<ReconcilerLogEntry>> reconciler_logger_;
};