#pragma once

#include "sim_env/model.hpp"  // always: pinocchio-based franka::Model

#ifndef WITH_FRANKA

#include <memory>
#include <array>
#include <string>
#include <functional>
#include <atomic>
#include <stdexcept>

#include <Eigen/Dense>
#include <yaml-cpp/yaml.h>

class Simulation;
struct DeviceState;

namespace franka {

// Mirrors libfranka's exception hierarchy (franka/exception.h) closely enough
// that arm_control.cpp can catch franka::ControlException / franka::Exception
// identically in sim and real-robot builds, with no #ifdef at the call site.
class Exception : public std::runtime_error {
public:
    explicit Exception(const std::string& what) : std::runtime_error(what) {}
};

class ControlException : public Exception {
public:
    explicit ControlException(const std::string& what) : Exception(what) {}
};

struct RobotState {
    std::array<double, 7>  q;
    std::array<double, 7>  dq;
    std::array<double, 7>  tau_J;
    std::array<double, 7>  tau_J_d;
    std::array<double, 7>  tau_ext_hat_filtered;
    std::array<double, 6>  O_F_ext_hat_K;
    std::array<double, 16> O_T_EE;

    RobotState() {
        q.fill(0.0);                  dq.fill(0.0);
        tau_J.fill(0.0);              tau_J_d.fill(0.0);
        tau_ext_hat_filtered.fill(0.0);
        O_F_ext_hat_K.fill(0.0);     O_T_EE.fill(0.0);
    }
};

struct Finishable {
    bool motion_finished = false;
};

class Torques : public Finishable {
public:
    Torques(const std::array<double, 7>& torques) noexcept;
    Torques(std::initializer_list<double> torques);
    std::array<double, 7> tau_J{};
};

class Duration {
public:
    Duration() {}
    double time = 1.0 / 1000.0;
};

class Robot {
public:
    Robot();
    ~Robot();

    void set_simulation(Simulation& _sim, const YAML::Node& sim_dev, const YAML::Node& robot_dev);
    Model& loadModel();
    RobotState readOnce();
    void control(std::function<Torques(const RobotState&, Duration)> control_callback);

    // API parity with real libfranka (franka/robot.h) so arm_control.cpp can call
    // these unconditionally, without #ifdef WITH_FRANKA around every call site.
    void setCollisionBehavior(
        const std::array<double, 7>& lower_torque_thresholds,
        const std::array<double, 7>& upper_torque_thresholds,
        const std::array<double, 6>& lower_force_thresholds,
        const std::array<double, 6>& upper_force_thresholds);
    void setJointImpedance(const std::array<double, 7>& K_theta);
    void setCartesianImpedance(const std::array<double, 6>& K_x);
    void automaticErrorRecovery();

private:
    void populateRobotState(const DeviceState& ds, double dt);
    void updateGMO(const std::array<double, 7>& q,
                   const std::array<double, 7>& dq,
                   const std::array<double, 7>& tau_cmd,
                   double dt);
    void checkFrankaErrors(const Vector7& tau_cmd, const Vector7& dq, const Vector7& q);

private:
    Simulation*            sim    = nullptr;
    std::string            name_;
    std::string            ee_frame_name_;
    std::unique_ptr<Model> model_;
    RobotState             robot_state_;
    std::atomic<bool>      bRunning{false};
    Vector7                tau_filtered_;
    Vector7                tau_prev_;

    Vector7 r_;
    Vector7 p_prev_;
    static constexpr double K_GMO = 50.0;

    // Stored for parity with the real API; sim has no separate collision-reflex
    // path yet, so these aren't consumed anywhere (see checkFrankaErrors for the
    // hard torque/velocity/position limits that ARE enforced in sim).
    std::array<double, 7> lower_torque_thresholds_{};
    std::array<double, 7> upper_torque_thresholds_{};
    std::array<double, 6> lower_force_thresholds_{};
    std::array<double, 6> upper_force_thresholds_{};
    std::array<double, 7> joint_impedance_{};
    std::array<double, 6> cartesian_impedance_{};

    // Per-device joint position limits, read from robot_dev["q_min"/"q_max"]
    // in set_simulation() -- the SAME config ArmControl itself plans/brakes
    // against (device_config["q_min"/"q_max"]), so checkFrankaErrors' hard
    // limit check agrees with what the arm's own IK thinks its range is,
    // rather than an independent hardcoded value. Falls back to the FR3
    // factory range if a config omits them (shouldn't happen in practice --
    // ArmControl itself requires these keys).
    std::array<double, 7> q_min_{-2.8973, -1.7628, -2.8973, -3.0718, -2.8973,  0.0175, -2.8973};
    std::array<double, 7> q_max_{ 2.8973,  1.7628,  2.8973, -0.0698,  2.8973,  3.7525,  2.8973};
};

}  // namespace franka

#else  // WITH_FRANKA — use real libfranka; model.hpp above provides franka::Model

#include <franka/robot.h>     // brings in franka::Robot, RobotState, Torques, Duration
#include <franka/exception.h> // franka::Exception, franka::ControlException

#endif  // WITH_FRANKA