#include "MotionGenerator.hpp"
#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <array>

// ─────────────────────────────────────────────────────────────────────────────
//  Construction
// ─────────────────────────────────────────────────────────────────────────────

MotionGenerator::MotionGenerator(const InterpolatorConfig& config)
    : config_(config)
    // Floor of 1. Integer division makes this 0 as soon as comm_freq exceeds
    // control_freq, and then a zero-distance command yields n_steps = 0, an
    // empty waypoint vector, and getCurrentCartesian() falling back to
    // Isometry3d::Identity() -- i.e. an impedance target at the robot base.
    // Harmless at 200 Hz, latent the moment the command rate is raised.
    , min_steps_(std::max(1, config.control_freq / config.comm_freq))   // until the first measurement
    , space_(InterpolationSpace::JOINT)
    , joint_idx_(0)
    , cartesian_idx_(0)
{}

// ─────────────────────────────────────────────────────────────────────────────
//  Interpolation helpers
// ─────────────────────────────────────────────────────────────────────────────

void MotionGenerator::setCommandInterval(double dt_s) {
    if (!(dt_s > 0.0)) return;

    constexpr double kEma    = 0.2;
    constexpr double kShrink = 0.9;
    constexpr int    kMax    = 100;

    cmd_interval_s_ = (cmd_interval_s_ <= 0.0) ? dt_s
                                               : (1.0 - kEma) * cmd_interval_s_ + kEma * dt_s;

    // Deliberately biased short. A plan LONGER than the command interval is
    // superseded before it finishes, and since each replan only re-ramps the
    // remaining distance, the reference converges geometrically and never
    // arrives. A plan shorter than the interval just completes and holds --
    // the old behaviour, which is merely suboptimal rather than divergent.
    const int steps = static_cast<int>(std::lround(kShrink * cmd_interval_s_ * config_.control_freq));
    min_steps_.store(std::clamp(steps, 1, kMax), std::memory_order_relaxed);
}

int MotionGenerator::computeJointSteps(const Eigen::VectorXd& q_start, const Eigen::VectorXd& q_end) const {
    double max_displacement = (q_end - q_start).cwiseAbs().maxCoeff();
    double t_min            = max_displacement / config_.max_angular_vel;
    int    steps            = static_cast<int>(std::ceil(t_min * config_.control_freq));
    return std::max(steps, min_steps_.load(std::memory_order_relaxed));
}

int MotionGenerator::computeCartesianSteps(const Eigen::Isometry3d& T_start, const Eigen::Isometry3d& T_end) const {
    double linear_dist  = (T_end.translation() - T_start.translation()).norm();
    Eigen::AngleAxisd aa(T_start.rotation().transpose() * T_end.rotation());
    double angular_dist = std::abs(aa.angle());

    double t_linear  = linear_dist  / config_.max_linear_vel;
    double t_angular = angular_dist / config_.max_angular_vel;
    int    steps     = static_cast<int>(std::ceil(std::max(t_linear, t_angular) * config_.control_freq));
    return std::max(steps, min_steps_.load(std::memory_order_relaxed));
}

double MotionGenerator::trapezoidalProfile(double t) const {
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

double MotionGenerator::linearProfile(double t) const {
    if (t <= 0.0) return 0.0;
    if (t >= 1.0) return 1.0;
    return t;
}

double MotionGenerator::minJerkProfile(double t) const {
    if (t <= 0.0) return 0.0;
    if (t >= 1.0) return 1.0;
    return 10.0*t*t*t - 15.0*t*t*t*t + 6.0*t*t*t*t*t;
}

double MotionGenerator::applyProfile(double t, ProfileType profile) const {
    switch (profile) {
        case ProfileType::LINEAR:      return linearProfile(t);
        case ProfileType::MINJERK:     return minJerkProfile(t);
        case ProfileType::TRAPEZOIDAL:
        default:                       return trapezoidalProfile(t);
    }
}

// ─────────────────────────────────────────────────────────────────────────────
//  Interpolation API
// ─────────────────────────────────────────────────────────────────────────────

void MotionGenerator::planJoint(const Eigen::VectorXd& q_start, const Eigen::VectorXd& q_end,
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

void MotionGenerator::planCartesian(const Eigen::Isometry3d& T_start, const Eigen::Isometry3d& T_end,
                                     ProfileType profile) {
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

Eigen::VectorXd MotionGenerator::getCurrentJoint() const {
    std::lock_guard<std::mutex> lock(mtx_);
    if (joint_waypoints_.empty()) return Eigen::VectorXd::Zero(config_.n_dof);
    int idx = std::min(joint_idx_, (int)joint_waypoints_.size() - 1);
    return joint_waypoints_[idx];
}

Eigen::Isometry3d MotionGenerator::getCurrentCartesian() const {
    std::lock_guard<std::mutex> lock(mtx_);
    if (cartesian_waypoints_.empty()) return Eigen::Isometry3d::Identity();
    int idx = std::min(cartesian_idx_, (int)cartesian_waypoints_.size() - 1);
    return cartesian_waypoints_[idx];
}

bool MotionGenerator::step() {
    std::lock_guard<std::mutex> lock(mtx_);
    if (space_ == InterpolationSpace::JOINT) {
        cartesian_vel_.setZero();
        if (joint_idx_ < (int)joint_waypoints_.size() - 1) {
            ++joint_idx_;
            return true;
        }
        return false;
    }

    if (cartesian_idx_ < (int)cartesian_waypoints_.size() - 1) {
        const Eigen::Isometry3d& a = cartesian_waypoints_[cartesian_idx_];
        ++cartesian_idx_;
        const Eigen::Isometry3d& b = cartesian_waypoints_[cartesian_idx_];

        const double f = static_cast<double>(config_.control_freq);
        Eigen::AngleAxisd aa(Eigen::Quaterniond(b.rotation()) *
                             Eigen::Quaterniond(a.rotation()).conjugate());
        cartesian_vel_.head<3>() = (b.translation() - a.translation()) * f;
        cartesian_vel_.tail<3>() = aa.axis() * aa.angle() * f;

        // f is the ASSUMED loop rate. A loop running slower than control_freq
        // would scale this up, and it feeds a force, not a target. The plan was
        // built under these caps, so exceeding them means the assumption is
        // wrong; clamp rather than propagate it.
        const double vl = cartesian_vel_.head<3>().norm();
        if (vl > config_.max_linear_vel)
            cartesian_vel_.head<3>() *= config_.max_linear_vel / vl;
        const double va = cartesian_vel_.tail<3>().norm();
        if (va > config_.max_angular_vel)
            cartesian_vel_.tail<3>() *= config_.max_angular_vel / va;
        return true;
    }

    cartesian_vel_.setZero();
    return false;
}

Eigen::Matrix<double, 6, 1> MotionGenerator::getCurrentCartesianVelocity() const {
    std::lock_guard<std::mutex> lock(mtx_);
    return cartesian_vel_;
}

bool MotionGenerator::isDone() const {
    std::lock_guard<std::mutex> lock(mtx_);
    if (space_ == InterpolationSpace::JOINT)
        return joint_idx_ >= (int)joint_waypoints_.size() - 1;
    return cartesian_idx_ >= (int)cartesian_waypoints_.size() - 1;
}

void MotionGenerator::reset() {
    std::lock_guard<std::mutex> lock(mtx_);
    joint_idx_     = 0;
    cartesian_idx_ = 0;
    cartesian_vel_.setZero();
}

// ─────────────────────────────────────────────────────────────────────────────
//  Resolved-rate IK — configuration & seeding
// ─────────────────────────────────────────────────────────────────────────────

void MotionGenerator::setIkConfig(const IkConfig& c) {
    std::lock_guard<std::mutex> lock(ik_mtx_);
    ik_cfg_ = c;
}

void MotionGenerator::seedJointReference(const Eigen::Matrix<double,7,1>& q) {
    std::lock_guard<std::mutex> lock(ik_mtx_);
    q_ref_     = q;
    u_prev_    = Eigen::Matrix<double,7,1>::Zero();
    ik_seeded_ = true;
}

void MotionGenerator::setCartesianGoal(const Eigen::Isometry3d& X_d) {
    std::lock_guard<std::mutex> lock(ik_mtx_);
    X_goal_ = X_d;
}

Eigen::Isometry3d MotionGenerator::getCartesianGoal() const {
    std::lock_guard<std::mutex> lock(ik_mtx_);
    return X_goal_;
}

Eigen::Matrix<double,7,1> MotionGenerator::getJointReference() const {
    std::lock_guard<std::mutex> lock(ik_mtx_);
    return q_ref_;
}

Eigen::Matrix<double,7,1> MotionGenerator::getVelocityReference() const {
    std::lock_guard<std::mutex> lock(ik_mtx_);
    return u_prev_;
}

// ─────────────────────────────────────────────────────────────────────────────
//  stepIk — one resolved-rate IK tick
//
//  Inputs  (caller provides fresh values each tick):
//    q   current joint positions (7)
//    J   base-frame geometric Jacobian (6×7), same convention as
//        model->zeroJacobian(kEndEffector, rs)
//    x   current EE pose in base frame (from rs.O_T_EE)
//    dt  state-thread period (1/500 s)
//
//  Computes joint velocity u, integrates into q_ref_, stores u as u_prev_.
//  Returns q_ref_ (same as getJointReference()).
// ─────────────────────────────────────────────────────────────────────────────
Eigen::Matrix<double,7,1> MotionGenerator::stepIk(
    const Eigen::Matrix<double,7,1>& q,
    const Eigen::Matrix<double,6,7>& J,
    const Eigen::Isometry3d& x,
    double dt)
{
    std::lock_guard<std::mutex> lock(ik_mtx_);

    // Fallback seed: should not happen if seedJointReference() was called on
    // ENGAGED entry, but guards against races at first tick.
    if (!ik_seeded_) {
        q_ref_     = q;
        u_prev_    = Eigen::Matrix<double,7,1>::Zero();
        ik_seeded_ = true;
    }

    const IkConfig& c = ik_cfg_;

    // ── Task-space error ──────────────────────────────────────────────────────
    // x MUST be the reference pose FK(q_ref), NOT the measured EE pose. The caller
    // evaluates FK and the Jacobian at q_ref so this IK is a pure feedforward
    // reference generator, decoupled from the robot. Passing the measured pose here
    // closes an integrator around the compliant joint-impedance loop and yields an
    // undamped Cartesian oscillation.
    Eigen::Vector3d e_p = X_goal_.translation() - x.translation();

    // Orientation error — same convention as cartesianImpedanceControl:
    //   q_err = q_d * q_cur^{-1},  error axis = vec(q_err)
    Eigen::Quaterniond q_d(X_goal_.rotation());
    Eigen::Quaterniond q_cur(x.rotation());
    if (q_d.dot(q_cur) < 0.0) q_d.coeffs() *= -1.0;
    // Full rotation vector, matching cartesianImpedanceControl -- see the long
    // comment there. vec(q_err) is half the rotation vector and saturates past
    // 180 deg. ik.kp_o is halved in config alongside this so the commanded task
    // angular velocity is unchanged.
    Eigen::Quaterniond q_err = (q_d * q_cur.inverse()).normalized();
    Eigen::AngleAxisd  aa_err(q_err);
    Eigen::Vector3d    e_o = aa_err.axis() * aa_err.angle();

    // Proportional task velocity, capped — gives a smooth, moderate command the
    // acceleration limit can track (a full Newton step / dt is bang-bang and stalls).
    Eigen::Matrix<double,6,1> v_des;
    v_des.head<3>() = c.Kp_p.cwiseProduct(e_p);
    v_des.tail<3>() = c.Kp_o * e_o;
    {
        double vlin = v_des.head<3>().norm();
        if (vlin > c.v_lin_max && vlin > 1e-9) v_des.head<3>() *= c.v_lin_max / vlin;
        double vang = v_des.tail<3>().norm();
        if (vang > c.v_ang_max && vang > 1e-9) v_des.tail<3>() *= c.v_ang_max / vang;
    }

    // Damped weighted resolved-rate with a soft posture pull toward q0 (no null-space):
    //   min || J u - v_des ||^2_Wtask + Kp_posture || u - (q0 - q_ref) ||^2 + lambda||u||^2
    Eigen::Matrix<double,7,1> e_post = c.q0 - q_ref_;
    Eigen::Matrix<double,7,7> A = J.transpose() * c.Wtask.asDiagonal() * J;
    A.diagonal()         += c.Kp_posture;
    A.diagonal().array() += c.lambda;
    Eigen::Matrix<double,7,1> b = J.transpose() * c.Wtask.asDiagonal() * v_des
                                  + c.Kp_posture.cwiseProduct(e_post);
    Eigen::Matrix<double,7,1> u = A.ldlt().solve(b);

    // ── Per-joint box bounds ─────────────────────────────────────────────────
    // Each bound folds three safety constraints:
    //   term 1: velocity limit        (±qd_max_i)
    //   term 2: position-limit braking (don't command a velocity that would
    //           breach q_min/q_max within horizon T_brake)
    //   term 3: acceleration limit    (u_prev ± a_max*dt)
    Eigen::Matrix<double,7,1> L, U;
    for (int i = 0; i < 7; ++i) {
        const double ul_vel =  c.qd_max(i);
        // Position-limit braking uses q_ref_ (not actual q) because we are
        // commanding q_ref_; this prevents q_ref_ itself from approaching the
        // joint limit regardless of where actual q currently is.
        const double ul_pos = (c.q_max(i) - c.gamma - q_ref_(i)) / c.T_brake;
        const double ll_pos = (c.q_min(i) + c.gamma - q_ref_(i)) / c.T_brake;
        const double ul_acc =  u_prev_(i) + c.a_max * dt;
        const double ll_acc =  u_prev_(i) - c.a_max * dt;

        U(i) = std::min({ul_vel,  ul_pos, ul_acc});
        L(i) = std::max({-ul_vel, ll_pos, ll_acc});

        // Numerical safeguard: if already past a limit L may exceed U
        if (L(i) > U(i)) {
            double mid = 0.5 * (L(i) + U(i));
            L(i) = U(i) = mid;
        }
    }

    u = u.cwiseMax(L).cwiseMin(U);

    // ── Integrate joint reference ─────────────────────────────────────────────
    q_ref_ += u * dt;

    // Hard-clamp q_ref to joint limits (belt-and-suspenders)
    for (int i = 0; i < 7; ++i)
        q_ref_(i) = std::clamp(q_ref_(i), c.q_min(i) + c.gamma, c.q_max(i) - c.gamma);

    u_prev_ = u;
    return q_ref_;
}
