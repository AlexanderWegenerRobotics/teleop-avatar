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
    // The same wrench expressed in the stiffness frame. This sim has no separate
    // EE_T_K, so K is the configured EE frame. setCollisionBehavior's Cartesian
    // thresholds are defined on THIS wrench on hardware, not on the base-frame
    // one: checking a per-axis threshold set against O_F_ext_hat_K would mean
    // something different depending on where the wrist happens to be pointing.
    std::array<double, 6>  K_F_ext_hat_K;
    std::array<double, 16> O_T_EE;

    // Collision-behaviour flags, same meaning as libfranka's. The lower
    // thresholds raise *_contact and nothing else; the upper thresholds raise
    // *_collision and trigger the reflex. This is how a caller tells "the arm is
    // touching something" from "the arm has stopped".
    std::array<double, 7>  joint_contact;
    std::array<double, 6>  cartesian_contact;
    std::array<double, 7>  joint_collision;
    std::array<double, 6>  cartesian_collision;

    // mjData::time of the snapshot this state came from (sim build only; 0 on
    // hardware, which has no sim clock). q/dq are consistent in THIS clock.
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
    // Consumes the setCollisionBehavior thresholds against the momentum
    // observer's estimate. Throws ControlException on an upper-threshold
    // crossing, the same way libfranka surfaces cartesian_reflex / joint_reflex.
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
    // Set on every entry to control(). The torque-rate check divides by a fixed
    // 1 ms, but no ticks run while enterFaultAndWaitForReset() holds the loop,
    // so the first tick after a resume differences against a torque from
    // seconds ago and reports a rate that never happened.
    bool                   tau_rate_seed_pending_{true};
    // Same idea for the momentum observer's p_prev_ — see updateGMO.
    bool                   gmo_seed_pending_{true};
    // Per joint: a position-limit violation has been reported and not yet
    // cleared by the joint returning inside its range. Latches so the arm can
    // travel back out of a limit it is already past -- see checkFrankaErrors.
    // Deliberately NOT reset on re-entry to control().
    std::array<bool, 7>    joint_limit_tripped_{};

    Vector7 r_;
    Vector7 p_prev_;
    // mjData::time at the previous GMO update; <0 means "no previous sample".
    double  sim_time_prev_ = -1.0;
    static constexpr double K_GMO = 50.0;

    // Consumed by checkCollisionReflex(). Defaults are libfranka's own
    // setDefaultBehavior example values, so an arm whose config omits the
    // safety.collision_* block still reflexes somewhere sane instead of never.
    // Torques in Nm per joint; forces (x,y,z) in N and (R,P,Y) in Nm, on the
    // stiffness-frame wrench.
    std::array<double, 7> lower_torque_thresholds_{20.0, 20.0, 18.0, 18.0, 16.0, 14.0, 12.0};
    std::array<double, 7> upper_torque_thresholds_{20.0, 20.0, 18.0, 18.0, 16.0, 14.0, 12.0};
    std::array<double, 6> lower_force_thresholds_{20.0, 20.0, 20.0, 25.0, 25.0, 25.0};
    std::array<double, 6> upper_force_thresholds_{20.0, 20.0, 20.0, 25.0, 25.0, 25.0};

    // A momentum observer is noisier than the FR3's internal estimator: this
    // one measures 0.97 N mean / 3.1 N p95 in free motion on arm_left and
    // 2.19 / 4.8 on arm_right (claude/external-wrench-estimator.md). Tripping on
    // a single sample would fault on estimator noise rather than on contact, so
    // a crossing has to persist. At the 1 kHz control tick the default is 5 ms,
    // which is short against any real contact transient.
    int  collision_persist_ticks_ = 5;
    bool collision_reflex_enabled_ = true;
    std::array<int, 7> joint_reflex_streak_{};
    std::array<int, 6> cart_reflex_streak_{};
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