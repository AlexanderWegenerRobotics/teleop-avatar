#include "interpolator.hpp"
#include <algorithm>
#include <cmath>
#include <stdexcept>

Interpolator::Interpolator(const InterpolatorConfig& config)
    : config_(config)
    , min_steps_(config.control_freq / config.comm_freq)
    , space_(InterpolationSpace::JOINT)
    , joint_idx_(0)
    , cartesian_idx_(0)
{}

int Interpolator::computeJointSteps(const Eigen::VectorXd& q_start, const Eigen::VectorXd& q_end) const {
    double max_displacement = (q_end - q_start).cwiseAbs().maxCoeff();
    double t_min            = max_displacement / config_.max_angular_vel;
    int    steps            = static_cast<int>(std::ceil(t_min * config_.control_freq));
    return std::max(steps, min_steps_);
}

int Interpolator::computeCartesianSteps(const Eigen::Isometry3d& T_start, const Eigen::Isometry3d& T_end) const {
    double linear_dist  = (T_end.translation() - T_start.translation()).norm();
    Eigen::AngleAxisd aa(T_start.rotation().transpose() * T_end.rotation());
    double angular_dist = std::abs(aa.angle());

    double t_linear  = linear_dist  / config_.max_linear_vel;
    double t_angular = angular_dist / config_.max_angular_vel;
    int    steps     = static_cast<int>(std::ceil(std::max(t_linear, t_angular) * config_.control_freq));
    return std::max(steps, min_steps_);
}

double Interpolator::trapezoidalProfile(double t) const {
    constexpr double ramp_fraction = 0.2;
    if (t <= 0.0) return 0.0;
    if (t >= 1.0) return 1.0;
    double t_ramp = ramp_fraction;
    double v_max  = 1.0 / (1.0 - ramp_fraction);
    double pos    = 0.0;
    if (t < t_ramp) {
        pos = 0.5 * (v_max / t_ramp) * t * t;
    } else if (t < 1.0 - t_ramp) {
        pos = 0.5 * v_max * t_ramp + v_max * (t - t_ramp);
    } else {
        double dt = t - (1.0 - t_ramp);
        pos = 1.0 - 0.5 * (v_max / t_ramp) * (t_ramp - dt) * (t_ramp - dt);
    }
    return std::clamp(pos, 0.0, 1.0);
}

double Interpolator::linearProfile(double t) const {
    if (t <= 0.0) return 0.0;
    if (t >= 1.0) return 1.0;
    return t;
}

double Interpolator::minJerkProfile(double t) const {
    if (t <= 0.0) return 0.0;
    if (t >= 1.0) return 1.0;
    return 10.0*t*t*t - 15.0*t*t*t*t + 6.0*t*t*t*t*t;
}

double Interpolator::applyProfile(double t, ProfileType profile) const {
    switch (profile) {
        case ProfileType::LINEAR:      return linearProfile(t);
        case ProfileType::MINJERK:     return minJerkProfile(t);
        case ProfileType::TRAPEZOIDAL:
        default:                       return trapezoidalProfile(t);
    }
}

void Interpolator::planJoint(const Eigen::VectorXd& q_start, const Eigen::VectorXd& q_end,
                              ProfileType profile) {
    if (q_start.size() != config_.n_dof || q_end.size() != config_.n_dof)
        throw std::invalid_argument("Joint vector size mismatch");

    int n_steps = computeJointSteps(q_start, q_end);
    std::vector<Eigen::VectorXd> waypoints(n_steps);

    for (int i = 0; i < n_steps; ++i) {
        double t = (n_steps > 1) ? static_cast<double>(i) / (n_steps - 1) : 1.0;
        double s = applyProfile(t, profile);
        waypoints[i] = q_start + s * (q_end - q_start);
    }

    std::lock_guard<std::mutex> lock(mtx_);
    space_           = InterpolationSpace::JOINT;
    joint_waypoints_ = std::move(waypoints);
    joint_idx_       = 0;
}

void Interpolator::planCartesian(const Eigen::Isometry3d& T_start, const Eigen::Isometry3d& T_end, ProfileType profile) {
    int n_steps = computeCartesianSteps(T_start, T_end);
    std::vector<Eigen::Isometry3d> waypoints(n_steps);

    Eigen::Quaterniond q_start(T_start.rotation());
    Eigen::Quaterniond q_end(T_end.rotation());

    for (int i = 0; i < n_steps; ++i) {
        double t = (n_steps > 1) ? static_cast<double>(i) / (n_steps - 1) : 1.0;
        double s = applyProfile(t, profile);

        Eigen::Isometry3d T = Eigen::Isometry3d::Identity();
        T.translation() = T_start.translation() + s * (T_end.translation() - T_start.translation());
        T.linear()      = q_start.slerp(s, q_end).toRotationMatrix();
        waypoints[i]    = T;
    }
    std::lock_guard<std::mutex> lock(mtx_);
    space_               = InterpolationSpace::CARTESIAN;
    cartesian_waypoints_ = std::move(waypoints);
    cartesian_idx_       = 0;
}

Eigen::VectorXd Interpolator::getCurrentJoint() const {
    std::lock_guard<std::mutex> lock(mtx_);
    if (joint_waypoints_.empty()) return Eigen::VectorXd::Zero(config_.n_dof);
    int idx = std::min(joint_idx_, (int)joint_waypoints_.size() - 1);
    return joint_waypoints_[idx];
}

Eigen::Isometry3d Interpolator::getCurrentCartesian() const {
    std::lock_guard<std::mutex> lock(mtx_);
    if (cartesian_waypoints_.empty()) return Eigen::Isometry3d::Identity();
    int idx = std::min(cartesian_idx_, (int)cartesian_waypoints_.size() - 1);
    return cartesian_waypoints_[idx];
}

bool Interpolator::step() {
    std::lock_guard<std::mutex> lock(mtx_);
    if (space_ == InterpolationSpace::JOINT) {
        if (joint_idx_ < (int)joint_waypoints_.size() - 1) {
            ++joint_idx_;
            return true;
        }
    } else {
        if (cartesian_idx_ < (int)cartesian_waypoints_.size() - 1) {
            ++cartesian_idx_;
            return true;
        }
    }
    return false;
}

bool Interpolator::isDone() const {
    std::lock_guard<std::mutex> lock(mtx_);
    if (space_ == InterpolationSpace::JOINT)
        return joint_idx_ >= (int)joint_waypoints_.size() - 1;
    return cartesian_idx_ >= (int)cartesian_waypoints_.size() - 1;
}

void Interpolator::reset() {
    std::lock_guard<std::mutex> lock(mtx_);
    joint_idx_     = 0;
    cartesian_idx_ = 0;
}