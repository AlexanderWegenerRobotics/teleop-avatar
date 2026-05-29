#pragma once

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
};

class Interpolator {
public:
    explicit Interpolator(const InterpolatorConfig& config);

    void planJoint(const Eigen::VectorXd& q_start, const Eigen::VectorXd& q_end,
                   ProfileType profile = ProfileType::TRAPEZOIDAL);

    void planCartesian(const Eigen::Isometry3d& T_start, const Eigen::Isometry3d& T_end,
                       ProfileType profile = ProfileType::TRAPEZOIDAL);

    Eigen::VectorXd   getCurrentJoint()     const;
    Eigen::Isometry3d getCurrentCartesian() const;

    bool step();
    bool isDone() const;
    void reset();

private:
    int    computeJointSteps(const Eigen::VectorXd& q_start,     const Eigen::VectorXd& q_end)     const;
    int    computeCartesianSteps(const Eigen::Isometry3d& T_start, const Eigen::Isometry3d& T_end) const;
    double applyProfile(double t, ProfileType profile) const;
    double trapezoidalProfile(double t) const;
    double linearProfile(double t)      const;
    double minJerkProfile(double t)     const;

private:
    InterpolatorConfig config_;
    int                min_steps_;
    InterpolationSpace space_;
    mutable std::mutex mtx_;

    // Joint and Cartesian plans maintain independent indices so that
    // replanning one space does not corrupt the readout of the other.
    // This prevents getCurrentJoint() from jumping to waypoint[0] of a
    // stale joint plan when planCartesian() resets the shared index mid-tick.
    std::vector<Eigen::VectorXd>   joint_waypoints_;
    int                            joint_idx_ = 0;

    std::vector<Eigen::Isometry3d> cartesian_waypoints_;
    int                            cartesian_idx_ = 0;
};