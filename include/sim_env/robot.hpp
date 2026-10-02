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

// mirrors libfranka's exception hierarchy so callers catch the same types in sim and real
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
    // same wrench in stiffness frame (K = EE frame here), Cartesian collision thresholds apply to this one
    std::array<double, 6>  K_F_ext_hat_K;
    std::array<double, 16> O_T_EE;

    // lower thresholds -> *_contact, upper thresholds -> *_collision + reflex
    std::array<double, 7>  joint_contact;
    std::array<double, 6>  cartesian_contact;
    std::array<double, 7>  joint_collision;
    std::array<double, 6>  cartesian_collision;

    // mjData::time of the snapshot, sim only
    double                 sim_time = 0.0;

    RobotState() {
        q.fill(0.0);                  dq.fill(0.0);
        tau_J.fill(0.0);              tau_J_d.fill(0.0);
        tau_ext_hat_filtered.fill(0.0);
        O_F_ext_hat_K.fill(0.0);      K_F_ext_hat_K.fill(0.0);
        O_T_EE.fill(0.0);
        joint_contact.fill(0.0);      cartesian_contact.fill(0.0);
        joint_collision.fill(0.0);    cartesian_collision.fill(0.0);
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
    // throws ControlException on upper-threshold crossing, like libfranka's reflex
    void checkCollisionReflex();

private:
    Simulation*            sim    = nullptr;
    std::string            name_;
    std::string            ee_frame_name_;
    std::unique_ptr<Model> model_;
    RobotState             robot_state_;
    std::atomic<bool>      bRunning{false};
    Vector7                tau_filtered_;
    Vector7                tau_prev_;
    // reseeds torque-rate check on entry to control(), tau_prev_ is stale after a pause
    bool                   tau_rate_seed_pending_{true};
    bool                   gmo_seed_pending_{true};
    // latched per joint until back in range, not reset on re-entry to control()
    std::array<bool, 7>    joint_limit_tripped_{};

    Vector7 r_;
    Vector7 p_prev_;
    // <0 = no previous sample
    double  sim_time_prev_ = -1.0;
    static constexpr double K_GMO = 50.0;

    // libfranka default values; torques Nm, forces N (x,y,z) / Nm (R,P,Y) in stiffness frame
    std::array<double, 7> lower_torque_thresholds_{20.0, 20.0, 18.0, 18.0, 16.0, 14.0, 12.0};
    std::array<double, 7> upper_torque_thresholds_{20.0, 20.0, 18.0, 18.0, 16.0, 14.0, 12.0};
    std::array<double, 6> lower_force_thresholds_{20.0, 20.0, 20.0, 25.0, 25.0, 25.0};
    std::array<double, 6> upper_force_thresholds_{20.0, 20.0, 20.0, 25.0, 25.0, 25.0};

    // crossing must persist this many 1 kHz ticks, GMO is too noisy to trip on one sample
    int  collision_persist_ticks_ = 5;
    bool collision_reflex_enabled_ = true;
    std::array<int, 7> joint_reflex_streak_{};
    std::array<int, 6> cart_reflex_streak_{};
    std::array<double, 7> joint_impedance_{};
    std::array<double, 6> cartesian_impedance_{};

    // overwritten from robot_dev q_min/q_max, FR3 factory range as fallback
    std::array<double, 7> q_min_{-2.8973, -1.7628, -2.8973, -3.0718, -2.8973,  0.0175, -2.8973};
    std::array<double, 7> q_max_{ 2.8973,  1.7628,  2.8973, -0.0698,  2.8973,  3.7525,  2.8973};
};

}  // namespace franka

#else  // WITH_FRANKA — use real libfranka; model.hpp above provides franka::Model

#include <franka/robot.h>     // brings in franka::Robot, RobotState, Torques, Duration
#include <franka/exception.h> // franka::Exception, franka::ControlException

#endif  // WITH_FRANKA