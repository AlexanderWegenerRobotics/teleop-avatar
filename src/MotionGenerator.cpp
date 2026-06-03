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
    , min_steps_(config.control_freq / config.comm_freq)
    , space_(InterpolationSpace::JOINT)
    , joint_idx_(0)
    , cartesian_idx_(0)
{}

// ─────────────────────────────────────────────────────────────────────────────
//  Interpolation helpers
// ─────────────────────────────────────────────────────────────────────────────

int MotionGenerator::computeJointSteps(const Eigen::VectorXd& q_start, const Eigen::VectorXd& q_end) const {
    double max_displacement = (q_end - q_start).cwiseAbs().maxCoeff();
    double t_min            = max_displacement / config_.max_angular_vel;
    int    steps            = static_cast<int>(std::ceil(t_min * config_.control_freq));
    return std::max(steps, min_steps_);
}

int MotionGenerator::computeCartesianSteps(const Eigen::Isometry3d& T_start, const Eigen::Isometry3d& T_end) const {
    double linear_dist  = (T_end.translation() - T_start.translation()).norm();
    Eigen::AngleAxisd aa(T_start.rotation().transpose() * T_end.rotation());
    double angular_dist = std::abs(aa.angle());

    double t_linear  = linear_dist  / config_.max_linear_vel;
    double t_angular = angular_dist / config_.max_angular_vel;
    int    steps     = static_cast<int>(std::ceil(std::max(t_linear, t_angular) * config_.control_freq));
    return std::max(steps, min_steps_);
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

Eigen::Matrix<double,7,1> MotionGenerator::getJointReference() const {
    std::lock_guard<std::mutex> lock(ik_mtx_);
    return q_ref_;
}

Eigen::Matrix<double,7,1> MotionGenerator::getVelocityReference() const {
    std::lock_guard<std::mutex> lock(ik_mtx_);
    return u_prev_;
}

// ─────────────────────────────────────────────────────────────────────────────
//  Box-constrained LS solver (active-set, ≤7 iterations, RT-safe)
//
//  Solves:  min_{u} || A u - b ||^2   s.t.  L_i ≤ u_i ≤ U_i
//  where A is already the 7×7 SPD normal-equation matrix and b is the
//  corresponding right-hand side (built by the caller).
//
//  Algorithm:
//    Start with all joints free.  Each iteration:
//      1. Build the reduced (free×free) system with clamped joints moved to RHS.
//      2. Solve via LDLT.
//      3. Find the worst bound violation among free joints.
//      4. If any: fix that joint at its violated bound and repeat.
//      5. Else: done.
//    At most 7 joints can be clamped → at most 7 iterations.  The 7×7 LDLT
//    solve is a few dozen FLOPs; the whole routine runs in single-digit µs.
// ─────────────────────────────────────────────────────────────────────────────
Eigen::Matrix<double,7,1> MotionGenerator::solveBoxConstrainedLS(
    const Eigen::Matrix<double,7,7>& A,
    const Eigen::Matrix<double,7,1>& b,
    const Eigen::Matrix<double,7,1>& L,
    const Eigen::Matrix<double,7,1>& U) const
{
    constexpr int N = 7;
    Eigen::Matrix<double,7,1> u = Eigen::Matrix<double,7,1>::Zero();

    bool   clamped[N] = {};
    double bound[N]   = {};

    for (int iter = 0; iter <= N; ++iter) {
        // Collect free indices
        int free_idx[N];
        int nf = 0;
        for (int i = 0; i < N; ++i)
            if (!clamped[i]) free_idx[nf++] = i;

        if (nf == 0) break;

        // Build RHS for free joints: subtract contribution of clamped joints
        Eigen::Matrix<double, N, 1> b_red = Eigen::Matrix<double, N, 1>::Zero();
        for (int fi = 0; fi < nf; ++fi) {
            int i = free_idx[fi];
            b_red(fi) = b(i);
            for (int j = 0; j < N; ++j)
                if (clamped[j]) b_red(fi) -= A(i, j) * bound[j];
        }

        // Build reduced A (free × free)
        Eigen::MatrixXd A_red(nf, nf);
        for (int fi = 0; fi < nf; ++fi)
            for (int fj = 0; fj < nf; ++fj)
                A_red(fi, fj) = A(free_idx[fi], free_idx[fj]);

        // Solve reduced system
        Eigen::VectorXd u_free = A_red.ldlt().solve(b_red.head(nf));

        // Write back
        for (int fi = 0; fi < nf; ++fi)
            u(free_idx[fi]) = u_free(fi);
        for (int j = 0; j < N; ++j)
            if (clamped[j]) u(j) = bound[j];

        // Find worst bound violator among free joints
        int    worst      = -1;
        double worst_viol = 0.0;
        for (int fi = 0; fi < nf; ++fi) {
            int    i    = free_idx[fi];
            double viol = 0.0;
            if      (u(i) < L(i)) viol = L(i) - u(i);
            else if (u(i) > U(i)) viol = u(i) - U(i);
            if (viol > worst_viol) { worst_viol = viol; worst = i; }
        }

        if (worst < 0) break; // all free joints within bounds → done

        // Clamp worst violator
        clamped[worst] = true;
        bound[worst]   = (u(worst) < L(worst)) ? L(worst) : U(worst);
    }

    // Hard clamp as safety net (catches edge cases where L > U after numerical issues)
    return u.cwiseMax(L).cwiseMin(U);
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
    Eigen::Quaterniond q_err = q_d * q_cur.inverse();
    Eigen::Vector3d    e_o(q_err.x(), q_err.y(), q_err.z());

    // Desired task velocity
    Eigen::Matrix<double,6,1> v_des;
    v_des.head<3>() = c.Kp_p.cwiseProduct(e_p);
    v_des.tail<3>() = c.Kp_o * e_o;

    // Cartesian speed caps
    {
        double vlin = v_des.head<3>().norm();
        if (vlin > c.v_lin_max && vlin > 1e-9)
            v_des.head<3>() *= c.v_lin_max / vlin;
        double vang = v_des.tail<3>().norm();
        if (vang > c.v_ang_max && vang > 1e-9)
            v_des.tail<3>() *= c.v_ang_max / vang;
    }

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

    // ── Normal equations ─────────────────────────────────────────────────────
    // Objective: min || J u - v_des ||²_W  +  λ||u||²  +  μ||u - u_post||²
    // Normal eq: (J^T W J + (λ+μ)I) u = J^T W v_des + μ u_post
    Eigen::Matrix<double,7,1> u_post = c.Kp_posture.cwiseProduct(c.q0 - q);

    Eigen::Matrix<double,7,6> JtW = J.transpose() * c.Wtask.asDiagonal();
    Eigen::Matrix<double,7,7> A   = JtW * J + (c.lambda + c.mu) * Eigen::Matrix<double,7,7>::Identity();
    Eigen::Matrix<double,7,1> b   = JtW * v_des + c.mu * u_post;

    // ── Solve box-constrained LS ─────────────────────────────────────────────
    Eigen::Matrix<double,7,1> u = solveBoxConstrainedLS(A, b, L, U);

    // ── Integrate joint reference ─────────────────────────────────────────────
    q_ref_ += u * dt;

    // Hard-clamp q_ref to joint limits (belt-and-suspenders)
    for (int i = 0; i < 7; ++i)
        q_ref_(i) = std::clamp(q_ref_(i), c.q_min(i) + c.gamma, c.q_max(i) - c.gamma);

    u_prev_ = u;
    return q_ref_;
}
