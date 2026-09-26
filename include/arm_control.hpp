#pragma once

#include <memory>
#include <atomic>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <chrono>
#include <iostream>

#include <yaml-cpp/yaml.h>

#include "sim_env/robot.hpp"
#include "sim_env/gripper.hpp"
#include "MotionGenerator.hpp"
#include "network/udp_stream.hpp"
#include "data_logger.hpp"
#include "self_collision_protection.hpp"
#include "arm_recovery.hpp"
#include "posture_optimizer.hpp"
#include "common.hpp"

class Simulation;

// Low-level control mode selected per arm via the config key "control_mode".
enum class ControlMode {
    CARTESIAN_IMPEDANCE,  // default — unchanged Cartesian impedance + nullspace
    JOINT_IK              // resolved-rate IK → joint impedance tracking q_ref
};

class ArmControl{
public:
    ArmControl(const YAML::Node& device_config, const std::string& session_id);
    ~ArmControl();

    void start();
    void stop();
    bool isRunning() const {return bRunning;}
    std::string getDeviceName() const {return name_;}
    void requestState(SysState state){cmd_state_ = state;}
    SysState getState() const {return state_;}
    Eigen::Isometry3d getTargetPose() const;
    Eigen::Isometry3d getRawTargetPose() const;
    // Raw joint state (q, dq), uniform across sim/real-hardware builds since
    // current_state is populated via the franka::Robot abstraction either
    // way. Used by Avatar to build TwinTelemetryMsg (docs/twin_concept.md).
    void getJointState(Vector7& q, Vector7& dq) const;
    void initSelfCollisionProtection(std::shared_ptr<DeviceRegistry> registry, const SelfCollisionConfig& config) {
        scp_ = std::make_unique<SelfCollisionProtection>(name_, std::move(registry), config);
    }
    void setCollisionImportanceWeight(double weight){scp_state_.weight = weight; 
        std::cout << name_ << " weight parameter set to "  << weight << std::endl;
    }

    ArmRecovery& recovery() { return recovery_; }
    Vector7 getQ0() const { return q0_; }
    void reOrigin();

    // ── Command authority ───────────────────────────────────────────────────
    // Which channel may move THIS arm. See CommandAuthority in common.hpp.
    // Called from the avatar's authority_request handler (cmd_channel_ receive
    // thread); read by the state thread every cycle and by the control thread
    // once per log row, hence the atomic.
    //
    // Entering HUMAN re-origins first and opens the VR gate second, so the
    // operator's first packet is composed against where the policy actually
    // left the arm rather than against where they last let go of it.
    void setAuthority(CommandAuthority requested, const std::string& source);
    CommandAuthority getAuthority() const { return authority_.load(std::memory_order_relaxed); }
    // Commands that arrived on a channel that did not hold authority and were
    // therefore discarded. The only way to tell "the gate is working" from "the
    // sender stopped"; without them every gate test is eyeballed.
    uint64_t getDroppedVrCommands()  const { return dropped_vr_cmds_.load(std::memory_order_relaxed); }
    uint64_t getDroppedAbsCommands() const { return dropped_abs_cmds_.load(std::memory_order_relaxed); }

    void markEpisodeStart() { if (logger_) logger_->markEpisodeStart(); }
    void markEpisodeEnd(const std::string& reason) { if (logger_) logger_->markEpisodeEnd(reason); }
    void restartLogger(const std::string& path);
    void writeEpisodeConfig(int seed, int mode, const std::string& color_bin_mapping);
    
public:
    std::unique_ptr<franka::Robot> robot;
	std::unique_ptr<franka::Gripper> gripper;
	franka::Model* model = nullptr;

private:
    std::thread control_thread;
    std::thread state_thread;
    std::thread gripper_thread;
    std::mutex data_mtx;
    std::atomic<bool> bRunning;
    std::atomic<SysState> state_{SysState::OFFLINE};
    std::atomic<SysState> cmd_state_{SysState::OFFLINE};
    MotionGenerator motion_gen_;
    using ArmStream = UdpStream<ArmCommandMsg, ArmStateMsg>;
    std::unique_ptr<ArmStream> transmission_;
    std::unique_ptr<ArmStream> transmission_absolute_;
    std::unique_ptr<DataLogger<ArmLogEntry>> logger_;
    std::unique_ptr<DataLogger<ArmStateTraceEntry>> state_trace_;
    std::atomic<uint32_t> control_loop_entries_{0};
    std::atomic<uint32_t> fault_streak_{0};
    std::chrono::high_resolution_clock::time_point startTime_;
    ArmRecovery recovery_;
    
private:
    void runControlHandler();
    void runStateHandler();
    ArmLogEntry buildArmLogEntry(const franka::RobotState& rs, const Vector7& tau_cmd, uint8_t log_src);
    void applyOperatorCommand(const ArmCommandMsg& cmd, const ArmCommandMsg& cmd_abs, bool& has_cmd, bool& has_cmd_abs, Eigen::Quaterniond& prev_cmd_quat);
    void waitForCommandOrDeadline(const std::chrono::steady_clock::time_point& deadline);
    void notifyCommandArrived();
    Vector7 jointImpedanceControl(const franka::RobotState& rs);
    Vector7 cartesianImpedanceControl(const franka::RobotState& rs);
    Eigen::Matrix<double, 6, 1> feedforwardWrench(const Eigen::Matrix<double, 6, 1>& v_ref) const;
    Eigen::Matrix<double, 6, 1> filteredReferenceVelocity();
    // Joint friction model (sigmoid form, supervisor's model / Gaz et al. 2019):
    // tau_f(dq) = fp1 / (1 + exp(-fp2 (dq + fp3))) - fp1 / (1 + exp(-fp2 fp3)).
    // Zero at dq = 0, saturating at roughly +-fp1/2. Returns the torque needed
    // to overcome friction, so it is ADDED to the command.
    Vector7 frictionTorque(const Vector7& dq) const;
    void updateStateMachine(SysState cmd_state);
    void updateRecovery();
    bool isHome();
    Eigen::Isometry3d transformCommandToBase(const Eigen::Isometry3d& T_cmd_world) const;
    Eigen::Isometry3d transformBaseToWorld(const Eigen::Isometry3d& T_base) const;
    Eigen::Isometry3d worldAbsoluteToBase(const Eigen::Isometry3d& T_world_abs) const;
    void applySelfCollisionFilter(Eigen::Isometry3d& T_target);
    void validateTargetPose(Eigen::Isometry3d& T_target);
    Vector7 jointLimitAvoidanceTorque(const Vector7& q, const Vector7& dq);
    void latchOriginForEngage(SysState from);
    void resetPostureFromMeasured();
    void applyGripper(bool close);
    void updateGraspConfirmation(double width);
    void updateAuthorityWatchdog();

private:
    std::string name_;
    Eigen::Vector3d base_position_;
    Eigen::Quaterniond base_orientation_;
    Eigen::Matrix3d R_ctrl_to_ee_ = Eigen::Matrix3d::Identity();
    Eigen::Isometry3d T_base_;
    Eigen::Isometry3d target_pose_, target_pose_raw_;
    Vector7 q0_, q_min_, q_max_;
    Vector7 tau_max_;
    Vector7 tau_rate_max_;
    double  torque_rate_margin_{0.9};

    int     rt_control_core_{0};
    int     rt_state_core_{1};
    mutable std::mutex state_mtx;
    franka::RobotState current_state;
    std::atomic<uint64_t> state_sample_ns_{0};
    std::atomic<uint32_t> applied_cmd_seq_{0};
    std::unique_ptr<franka::Model> franka_owned_model_;
    Eigen::Isometry3d T_origin_;
    std::unique_ptr<SelfCollisionProtection> scp_;
    CollisionState scp_state_;
    Vector7 recovery_target_q_ = Vector7::Zero();
    std::chrono::steady_clock::time_point recovery_start_time_;
    bool recovery_deferred_{false};
    std::chrono::steady_clock::time_point recovery_defer_start_;
    std::atomic<double> gripper_width_{0.0};
    std::atomic<bool>   desired_gripper_closed_{false};
    std::atomic<bool>   clutch_active_{true};
    std::atomic<CommandAuthority> authority_{CommandAuthority::UNSET};
    std::atomic<uint64_t> dropped_vr_cmds_{0};
    std::atomic<uint64_t> dropped_abs_cmds_{0};
    std::atomic<uint64_t> authority_last_cmd_ns_{0};
    double              authority_stale_ms_{250.0};
    std::atomic<bool>   grasp_allowed_{false};
    std::atomic<bool>   gripper_busy_{false};
    std::atomic<bool>   gripper_stop_sent_{false};
    bool                gripper_close_applied_{true};
    double              grasp_force_n_{40.0};
    double              grasp_speed_{0.1};

    double              grasp_confirm_tolerance_m_{0.008};
    double              grasp_confirm_time_s_{0.020};
    double              grasp_lost_latch_s_{1.0};
    bool                grasp_track_active_{false};
    double              grasp_track_width_{0.0};
    std::chrono::steady_clock::time_point grasp_track_start_;
    std::chrono::steady_clock::time_point grasp_lost_latch_until_;
    std::atomic<GraspState> grasp_state_{GraspState::OPEN};

    ControlMode control_mode_ = ControlMode::CARTESIAN_IMPEDANCE;
    double state_rate_hz_{200.0};

    // ── Early-wake plumbing ──────────────────────────────────────────────────
    std::mutex              cmd_wake_mtx_;
    std::condition_variable cmd_wake_cv_;
    bool                    cmd_wake_flag_{false};

private:
    Vector7 kp_joint_, kd_joint_, kp_joint_limit_, kd_joint_limit_;
    Vector7 kp_idle_, kd_idle_;
    std::atomic<bool> idle_hold_valid_{false};
    Eigen::Matrix<double, 6, 1> kp_cart_, kd_cart_;
    double eta_lin_{0.0}, eta_rot_{0.0};
    double ff_force_max_{30.0}, ff_torque_max_{8.0};
    double ff_filter_hz_{20.0};

    // ── Joint friction feedforward ───────────────────────────────────────────
    // Config: control.friction (see robot_config_*.yaml). Off unless enabled.
    // Default source is the REFERENCE joint velocity dq_ref = J_pinv * v_ref
    // (the same filtered reference twist the eta feedforward uses), not the
    // measured dq: compensating on measured velocity is positive velocity
    // feedback (moving faster -> more push -> faster), which the wrist, already
    // prone to a limit cycle, does not need. On the reference it can only push
    // along the commanded motion and adds nothing to the closed loop.
    // fp1..fp3 defaults are the Panda identification (Gaz et al. 2019), handed
    // over as the starting point; FR3 friction is similar but not identical,
    // hence scale.
    struct FrictionConfig {
        bool    enabled      = false;
        bool    use_measured = false;   // source: "reference" (default) | "measured"
        double  scale        = 0.5;
        Vector7 mask         = (Vector7() << 1, 1, 1, 1, 0, 0, 0).finished();
        Vector7 max_torque   = (Vector7() << 2.0, 2.0, 2.0, 2.0, 1.0, 1.0, 1.0).finished();
        Vector7 fp1 = (Vector7() << 0.54615, 0.87224, 0.64068, 1.2794, 0.83904, 0.30301, 0.56489).finished();
        Vector7 fp2 = (Vector7() << 5.1181, 9.0657, 10.136, 5.5903, 8.3469, 17.133, 10.336).finished();
        Vector7 fp3 = (Vector7() << 0.039533, 0.025882, -0.04607, 0.036194, 0.026226, -0.021047, 0.0035526).finished();
    } friction_cfg_;
    // Written by the control thread in cartesianImpedanceControl, zeroed every
    // control tick before the state switch, read by buildArmLogEntry on the same
    // thread (log_src 0). Never touched by the state thread.
    Vector7 tau_friction_ = Vector7::Zero();
    Eigen::Matrix<double, 6, 1> v_ref_filt_ = Eigen::Matrix<double, 6, 1>::Zero();
    Vector7 kp_null_, kd_null_;

    // ── Nullspace posture ────────────────────────────────────────────────────
    // State thread writes (update / reset), control thread reads one snapshot
    // per tick into posture_snap_ and uses it for tau_null and the log row.
    PostureConfig     posture_cfg_;
    PostureOptimizer  posture_;
    PostureSnapshot   posture_snap_;
    // RobotState whose F_T_EE / EE_T_K the kinematics functors reuse when
    // evaluating poses at configurations other than the measured one.
    franka::RobotState kin_template_;

private:
    Eigen::Vector3d workspace_min_;
    Eigen::Vector3d workspace_max_;
    double table_height_world_;
    double table_safety_margin_;
    double max_command_velocity_;
    double max_command_angular_velocity_;
    // Second-order bound on the command target (see validateTargetPose). 0 disables.
    double max_command_acceleration_{5.0};          // m/s^2
    double max_command_angular_acceleration_{25.0}; // rad/s^2
    Eigen::Vector3d prev_target_vel_    = Eigen::Vector3d::Zero();
    Eigen::Vector3d prev_target_angvel_ = Eigen::Vector3d::Zero();
    // Furthest the commanded target may sit ahead of the MEASURED pose, in m.
    // Bounds the impedance spring: at kp_cart 1000 N/m, 0.05 m is 50 N. <=0 disables.
    double max_target_lead_{0.0};
    // Rotational twin of the above, in radians. Bounds the rotational spring the
    // same way: at kp_cart 125 Nm/rad, 0.10 rad is 12.5 Nm, which is already the
    // FR3 wrist's per-joint ceiling, so the useful range sits well under that.
    // <=0 disables.
    double max_target_lead_rot_{0.0};
    std::chrono::steady_clock::time_point last_leash_log_time_{};
    std::chrono::steady_clock::time_point last_leash_rot_log_time_{};
    double ee_fingertip_length_;
    double max_tilt_angle_;
    double cmd_dt_;
    double joint_limit_buffer_;
    double joint_limit_torque_frac_;
    bool has_prev_valid_target_{false};
    Eigen::Vector3d prev_valid_target_pos_ = Eigen::Vector3d::Zero();
    Eigen::Quaterniond prev_valid_target_rot_ = Eigen::Quaterniond::Identity();
    // When the last command was accepted, so validateTargetPose can bound the
    // step by MEASURED elapsed time instead of the nominal command period. See
    // the comment there for why cmd_dt_ alone was wrong.
    std::chrono::steady_clock::time_point prev_valid_target_time_{};
    double last_cmd_dt_{0.0};   // measured in validateTargetPose, sizes the plan
    // Last pose actually handed to planCartesian. UdpStream resends the latest
    // command at its own rate, so most packets carry a target the interpolator
    // is already planning to; replanning on those restarts the plan from
    // waypoint 0 and, once the send rate exceeds the plan length, the reference
    // only ever covers a fraction of the remaining distance per replan.
    Eigen::Isometry3d last_planned_target_ = Eigen::Isometry3d::Identity();
    bool              has_planned_target_{false};
};
