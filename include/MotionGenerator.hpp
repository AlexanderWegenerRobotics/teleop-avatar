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

    // rad/s per joint; empty or wrong size falls back to max_angular_vel.
    std::vector<double> max_joint_vel;

    double joint_vel_margin = 0.85;
};

struct IkConfig {
    Eigen::Vector3d              Kp_p{6.0, 6.0, 6.0};   // task pos gain (1/s)
    double                       Kp_o{4.0};              // task ori gain (1/s)
    double                       v_lin_max{0.5};         // Cartesian linear speed cap (m/s)
    double                       v_ang_max{0.8};         // Cartesian angular speed cap (rad/s)
    double                       lambda{0.01};           // Levenberg–Marquardt damping
    Eigen::Matrix<double,7,1>    Kp_posture = Eigen::Matrix<double,7,1>::Constant(0.0001);  // posture weight (soft, << Wtask)
    Eigen::Matrix<double,7,1>    q0         = Eigen::Matrix<double,7,1>::Zero();
    // FR3 limits
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

// Joint/Cartesian trajectory interpolation plus resolved-rate IK.
class MotionGenerator {
public:
    explicit MotionGenerator(const InterpolatorConfig& config);

    void planJoint(const Eigen::VectorXd& q_start, const Eigen::VectorXd& q_end,
                   ProfileType profile = ProfileType::TRAPEZOIDAL);

    void planCartesian(const Eigen::Isometry3d& T_start, const Eigen::Isometry3d& T_end,
                       ProfileType profile = ProfileType::TRAPEZOIDAL);

    Eigen::VectorXd   getCurrentJoint()     const;
    Eigen::Isometry3d getCurrentCartesian() const;
    // Base-frame twist of the last stepped segment, zero when the plan is done.
    Eigen::Matrix<double, 6, 1> getCurrentCartesianVelocity() const;

    // Measured command interval, used to size the plan length.
    void setCommandInterval(double dt_s);

    bool step();
    bool isDone() const;
    // After a joint plan the Cartesian buffer is stale.
    bool isCartesianSpace() const {
        std::lock_guard<std::mutex> lock(mtx_);
        return space_ == InterpolationSpace::CARTESIAN && !cartesian_waypoints_.empty();
    }
    void reset();

    void setIkConfig(const IkConfig& c);

    // Call on ENGAGED entry to avoid a jump.
    void seedJointReference(const Eigen::Matrix<double,7,1>& q);

    void setIkPosture(const Eigen::Matrix<double,7,1>& q_posture);

    // Base frame.
    void setCartesianGoal(const Eigen::Isometry3d& X_d);
    Eigen::Isometry3d getCartesianGoal() const;

    // One resolved-rate step (state thread). J and x in base frame; returns q_ref.
    Eigen::Matrix<double,7,1> stepIk(const Eigen::Matrix<double,7,1>& q,
                                      const Eigen::Matrix<double,6,7>& J,
                                      const Eigen::Isometry3d& x,
                                      double dt);

    Eigen::Matrix<double,7,1> getJointReference()    const;
    Eigen::Matrix<double,7,1> getVelocityReference() const;

private:
    int    computeJointSteps    (const Eigen::VectorXd& q_start, const Eigen::VectorXd& q_end,
                                 ProfileType profile)                                             const;
    // Peak ds/dt of a unit-duration profile.
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

    std::vector<Eigen::VectorXd>   joint_waypoints_;
    int                            joint_idx_ = 0;

    std::vector<Eigen::Isometry3d> cartesian_waypoints_;
    int                            cartesian_idx_ = 0;
    Eigen::Matrix<double, 6, 1>    cartesian_vel_ = Eigen::Matrix<double, 6, 1>::Zero();

    mutable std::mutex          ik_mtx_;
    IkConfig                    ik_cfg_;
    Eigen::Matrix<double,7,1>   q_ref_     = Eigen::Matrix<double,7,1>::Zero();
    Eigen::Matrix<double,7,1>   u_prev_    = Eigen::Matrix<double,7,1>::Zero();
    Eigen::Isometry3d           X_goal_    = Eigen::Isometry3d::Identity();
    bool                        ik_seeded_ = false;
};
