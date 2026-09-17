#pragma once

#include <atomic>
#include <mutex>
#include <vector>
#include <cstddef>
#include <Eigen/Dense>
#include <Eigen/Geometry>

enum class InterpolationSpace {
    JOINT,
    CARTESIAN
};

enum class ProfileType {
    LINEAR,
    TRAPEZOIDAL,
    MINJERK
};

struct InterpolatorConfig {
    int    control_freq;
    int    comm_freq;
    int    n_dof;
    double max_linear_vel;
    double max_angular_vel;

    // Per-joint velocity ceiling, rad/s, one entry per DOF. Empty or the wrong
    // length falls back to max_angular_vel -- which is what computeJointSteps
    // used to do unconditionally, and max_angular_vel is the END-EFFECTOR
    // rotational cap, which says nothing about what a joint can do. With MINJERK
    // peaking at 1.875x the average, a plan timed off 4.0 rad/s reaches
    // 7.5 rad/s on the joint that travels furthest, against FR3 limits of
    // 2.62-5.26. That is what aborted the homing motion on 2026-09-16.
    std::vector<double> max_joint_vel;

    // Fraction of max_joint_vel a plan may reach. The safety check throws at the
    // limit exactly, so a plan aimed straight at it trips on rounding.
    double joint_vel_margin = 0.85;
};

// ── Resolved-rate IK configuration ───────────────────────────────────────────
struct IkConfig {
    Eigen::Vector3d              Kp_p{6.0, 6.0, 6.0};   // task pos gain (1/s)
    double                       Kp_o{4.0};              // task ori gain (1/s)
    double                       v_lin_max{0.5};         // Cartesian linear speed cap (m/s)
    double                       v_ang_max{0.8};         // Cartesian angular speed cap (rad/s)
    double                       lambda{0.01};           // Levenberg–Marquardt damping
    Eigen::Matrix<double,7,1>    Kp_posture = Eigen::Matrix<double,7,1>::Constant(0.0001);  // posture weight (soft, << Wtask)
    Eigen::Matrix<double,7,1>    q0         = Eigen::Matrix<double,7,1>::Zero();
    // FR3. Was Panda's (2.175 / 2.610), like every other limit table in this
    // repo turned out to be.
    Eigen::Matrix<double,7,1>    qd_max     = (Eigen::Matrix<double,7,1>()
                                                << 2.62, 2.62, 2.62, 2.62,
                                                   5.26, 4.18, 5.26).finished();
    Eigen::Matrix<double,7,1>    q_min      = Eigen::Matrix<double,7,1>::Constant(-3.0);
    Eigen::Matrix<double,7,1>    q_max      = Eigen::Matrix<double,7,1>::Constant( 3.0);
    double                       T_brake{0.10}; // position-limit braking horizon (s)
    double                       a_max{10.0};   // per-joint accel cap (rad/s^2)
    double                       gamma{0.05};   // joint-limit safety buffer (rad)
    Eigen::Matrix<double,6,1>    Wtask = Eigen::Matrix<double,6,1>::Ones(); // task weights

};

// ── MotionGenerator ───────────────────────────────────────────────────────────
// Replaces Interpolator (pure rename + new resolved-rate IK additions).
// All existing methods (planJoint, planCartesian, step, getCurrentJoint,
// getCurrentCartesian, isDone, reset) are unchanged and used by
// HOMING / RECOVERING / CARTESIAN_IMPEDANCE paths exactly as before.
class MotionGenerator {
public:
    explicit MotionGenerator(const InterpolatorConfig& config);

    // ── Existing interpolation API (unchanged) ──────────────────────────────
    void planJoint(const Eigen::VectorXd& q_start, const Eigen::VectorXd& q_end,
                   ProfileType profile = ProfileType::TRAPEZOIDAL);

    void planCartesian(const Eigen::Isometry3d& T_start, const Eigen::Isometry3d& T_end,
                       ProfileType profile = ProfileType::TRAPEZOIDAL);

    Eigen::VectorXd   getCurrentJoint()     const;
    Eigen::Isometry3d getCurrentCartesian() const;
    // Base-frame reference twist of the segment step() last traversed; zero once
    // the plan is exhausted. Taken from the waypoints rather than differentiated
    // downstream, so it is exact for any profile and carries no sampling noise.
    Eigen::Matrix<double, 6, 1> getCurrentCartesianVelocity() const;

    // Measured interval between consecutive commands, from whoever is sending:
    // the VR interface on its own port, or the orchestrator on the absolute
    // channel. Sizes the plan so the reference keeps moving across the gap
    // instead of sprinting to the target and waiting.
    void setCommandInterval(double dt_s);

    bool step();
    bool isDone() const;
    void reset();

    // ── Resolved-rate IK API ────────────────────────────────────────────────
    // Configure once (e.g. from constructor of ArmControl).
    void setIkConfig(const IkConfig& c);

    // Call on ENGAGED entry to avoid a jump: seeds q_ref and clears u_prev.
    void seedJointReference(const Eigen::Matrix<double,7,1>& q);

    // Update the Cartesian goal (base frame). Thread-safe; called from state thread.
    void setCartesianGoal(const Eigen::Isometry3d& X_d);
    Eigen::Isometry3d getCartesianGoal() const;

    // One resolved-rate step. Caller supplies fresh q, J (6×7, base frame), x (base frame).
    // Returns updated q_ref (also retrievable via getJointReference).
    // Runs at state-thread rate; the 1 kHz control loop reads getJointReference().
    Eigen::Matrix<double,7,1> stepIk(const Eigen::Matrix<double,7,1>& q,
                                      const Eigen::Matrix<double,6,7>& J,
                                      const Eigen::Isometry3d& x,
                                      double dt);

    // Read q_ref from the control thread (1 kHz).
    Eigen::Matrix<double,7,1> getJointReference()    const;
    // Last IK joint-velocity command — use as feedforward in joint impedance.
    Eigen::Matrix<double,7,1> getVelocityReference() const;

private:
    // Interpolation helpers
    int    computeJointSteps    (const Eigen::VectorXd& q_start, const Eigen::VectorXd& q_end,
                                 ProfileType profile)                                             const;
    // Peak of ds/dt for a profile normalised to unit duration: MINJERK 1.875,
    // TRAPEZOIDAL 1/(1-ramp_fraction), LINEAR 1. Timing a plan on the average
    // rate and then applying a profile that peaks above it is how a trajectory
    // sized to a limit ends up exceeding it.
    static double profilePeakRate(ProfileType profile);
    int    computeCartesianSteps(const Eigen::Isometry3d& T_start, const Eigen::Isometry3d& T_end)  const;
    double applyProfile         (double t, ProfileType profile) const;
    double trapezoidalProfile   (double t) const;
    double linearProfile        (double t) const;
    double minJerkProfile       (double t) const;


private:
    InterpolatorConfig config_;
    std::atomic<int>   min_steps_;
    double             cmd_interval_s_ = 0.0;
    InterpolationSpace space_;
    mutable std::mutex mtx_;

    // Joint and Cartesian plans use independent indices so replanning one
    // space does not corrupt the readout of the other.
    std::vector<Eigen::VectorXd>   joint_waypoints_;
    int                            joint_idx_ = 0;

    std::vector<Eigen::Isometry3d> cartesian_waypoints_;
    int                            cartesian_idx_ = 0;
    Eigen::Matrix<double, 6, 1>    cartesian_vel_ = Eigen::Matrix<double, 6, 1>::Zero();

    // ── IK internal state ────────────────────────────────────────────────────
    mutable std::mutex          ik_mtx_;
    IkConfig                    ik_cfg_;
    Eigen::Matrix<double,7,1>   q_ref_     = Eigen::Matrix<double,7,1>::Zero();
    Eigen::Matrix<double,7,1>   u_prev_    = Eigen::Matrix<double,7,1>::Zero();
    Eigen::Isometry3d           X_goal_    = Eigen::Isometry3d::Identity();
    bool                        ik_seeded_ = false;
};
