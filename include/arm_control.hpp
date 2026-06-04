#pragma once

#include <memory>
#include <atomic>
#include <thread>
#include <mutex>
#include <iostream>

#include <yaml-cpp/yaml.h>

#include "sim_env/robot.hpp"
#include "sim_env/gripper.hpp"
#include "MotionGenerator.hpp"
#include "network/udp_stream.hpp"
#include "data_logger.hpp"
#include "self_collision_protection.hpp"
#include "arm_recovery.hpp"
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
    void initSelfCollisionProtection(std::shared_ptr<DeviceRegistry> registry, const SelfCollisionConfig& config) {
        scp_ = std::make_unique<SelfCollisionProtection>(name_, std::move(registry), config);
    }
    void setCollisionImportanceWeight(double weight){scp_state_.weight = weight; 
        std::cout << name_ << " weight parameter set to "  << weight << std::endl;
    }

    ArmRecovery& recovery() { return recovery_; }
    Vector7 getQ0() const { return q0_; }
    void reOrigin();
    void markEpisodeStart() { if (logger_) logger_->markEpisodeStart(); }
    void markEpisodeEnd(const std::string& reason) { if (logger_) logger_->markEpisodeEnd(reason); }
    void restartLogger(const std::string& path);
    void writeEpisodeConfig(double px, double py, double pz, double gx, double gy, double gz, int mode);
    
public:
    std::unique_ptr<franka::Robot> robot;
	std::unique_ptr<franka::Gripper> gripper;
	franka::Model* model = nullptr;

private:
    std::thread control_thread;
    std::thread state_thread;
    std::mutex data_mtx;
    std::atomic<bool> bRunning;
    std::atomic<SysState> state_{SysState::OFFLINE};
    std::atomic<SysState> cmd_state_{SysState::OFFLINE};
    MotionGenerator motion_gen_;
    using ArmStream = UdpStream<ArmCommandMsg, ArmStateMsg>;
    std::unique_ptr<ArmStream> transmission_;
    std::unique_ptr<DataLogger<ArmLogEntry>> logger_;
    std::chrono::high_resolution_clock::time_point startTime_;
    ArmRecovery recovery_;
    
private:
    void runControlHandler();
    void runStateHandler();
    Vector7 jointImpedanceControl(const franka::RobotState& rs);
    Vector7 cartesianImpedanceControl(const franka::RobotState& rs);
    void updateStateMachine(SysState cmd_state);
    void updateRecovery();
    bool isHome();
    Eigen::Isometry3d transformCommandToBase(const Eigen::Isometry3d& T_cmd_world) const;
    Eigen::Isometry3d transformBaseToWorld(const Eigen::Isometry3d& T_base) const;
    void applySelfCollisionFilter(Eigen::Isometry3d& T_target);
    void validateTargetPose(Eigen::Isometry3d& T_target);
    Vector7 jointLimitAvoidanceTorque(const Vector7& q, const Vector7& dq);
    void applyGripper(bool close);

private:
    std::string name_;
    Eigen::Vector3d base_position_;
    Eigen::Quaterniond base_orientation_;
    // Constant change-of-basis mapping the controller's (protocol-frame) local axes
    // to the EE/flange local axes. Used for body-frame (tool) orientation retargeting.
    // Loaded from config key "controller_axis_map"; defaults to identity. Must be a
    // proper rotation (det = +1) - typically a signed permutation read off the
    // single-axis rotation test.
    Eigen::Matrix3d R_ctrl_to_ee_ = Eigen::Matrix3d::Identity();
    Eigen::Isometry3d T_base_;
    Eigen::Isometry3d target_pose_, target_pose_raw_;
    Vector7 q0_, q_min_, q_max_;
    Vector7 tau_max_;
    Vector7 tau_rate_max_;
    std::mutex state_mtx;
    franka::RobotState current_state;
    // Owns the pinocchio dynamics model when using real Franka (not obtained from loadModel())
    std::unique_ptr<franka::Model> franka_owned_model_;
    Eigen::Isometry3d T_origin_;
    std::unique_ptr<SelfCollisionProtection> scp_;
    CollisionState scp_state_;
    Vector7 recovery_target_q_ = Vector7::Zero();
    std::chrono::steady_clock::time_point recovery_start_time_;
    std::atomic<double> gripper_width_{0.0};
    std::atomic<bool>   desired_gripper_closed_{false};
    std::atomic<bool>   gripper_busy_{false};
    bool                gripper_close_applied_{true};

    ControlMode control_mode_ = ControlMode::CARTESIAN_IMPEDANCE;

private:
    Vector7 kp_joint_, kd_joint_, kp_joint_limit_, kd_joint_limit_;
    Eigen::Matrix<double, 6, 1> kp_cart_, kd_cart_;
    Vector7 kp_null_, kd_null_;

private:
    Eigen::Vector3d workspace_min_;
    Eigen::Vector3d workspace_max_;
    double table_height_world_;
    double table_safety_margin_;
    double max_command_velocity_;
    double max_command_angular_velocity_;
    double ee_fingertip_length_;
    double max_tilt_angle_;
    double cmd_dt_;
    double joint_limit_buffer_;
    double joint_limit_torque_frac_;
    bool has_prev_valid_target_{false};
    Eigen::Vector3d prev_valid_target_pos_ = Eigen::Vector3d::Zero();
    Eigen::Quaterniond prev_valid_target_rot_ = Eigen::Quaterniond::Identity();
};
