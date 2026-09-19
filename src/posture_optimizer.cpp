#include "posture_optimizer.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <vector>

namespace {

constexpr double kPi = 3.14159265358979323846;

Eigen::Matrix<double, 6, 1> poseError(const Eigen::Isometry3d& x, const Eigen::Isometry3d& x_ref) {
    Eigen::Matrix<double, 6, 1> e;
    e.head<3>() = x.translation() - x_ref.translation();
    Eigen::AngleAxisd aa(x.rotation() * x_ref.rotation().transpose());
    e.tail<3>() = aa.axis() * aa.angle();
    return e;
}

}  // namespace

void PostureOptimizer::init(const PostureConfig& cfg, const Vector7& q_min, const Vector7& q_max,
                            const Vector7& q0, const Eigen::Matrix3d& R_base, PostureKinematics kin) {
    cfg_    = cfg;
    if (cfg_.samples < 3) cfg_.samples = 3;
    if (cfg_.samples % 2 == 0) ++cfg_.samples;
    q_min_  = q_min;
    q_max_  = q_max;
    R_base_ = R_base;
    g_base_ = R_base.transpose() * Eigen::Vector3d(0, 0, -1);
    kin_    = std::move(kin);

    ArmGeometry g = geometry(q0);
    if (!g.ok) {
        initialised_ = false;
        return;
    }
    phi_home_ = std::atan2(g.e.dot(g.t), g.e.dot(g.r));
    Eigen::Vector3d p_w_world = R_base_ * g.p_w;
    z_home_ = p_w_world.z();
    y_home_ = p_w_world.y();
    initialised_ = true;
    reset(q0);
}

void PostureOptimizer::reset(const Vector7& q) {
    std::lock_guard<std::mutex> lock(mtx_);
    have_v_prev_ = false;
    s_filt_      = 0.0;
    snap_        = PostureSnapshot{};
    snap_.q_ref  = q;
    snap_.valid  = initialised_ && cfg_.enabled;
}

PostureSnapshot PostureOptimizer::snapshot() const {
    std::lock_guard<std::mutex> lock(mtx_);
    return snap_;
}

double PostureOptimizer::swivelAngle(const Vector7& q) const {
    ArmGeometry g = geometry(q);
    if (!g.ok) return 0.0;
    return std::atan2(g.e.dot(g.t), g.e.dot(g.r));
}

// ---------------------------------------------------------------------------

PostureOptimizer::ArmGeometry PostureOptimizer::geometry(const Vector7& q) const {
    ArmGeometry g;
    if (!kin_.pose) return g;
    g.p_s = kin_.pose(franka::Frame::kJoint2, q).translation();
    g.p_e = kin_.pose(franka::Frame::kJoint4, q).translation();
    g.p_w = kin_.pose(franka::Frame::kEndEffector, q).translation();

    Eigen::Vector3d sw = g.p_w - g.p_s;
    if (sw.norm() < 1e-6) return g;
    g.u = sw.normalized();

    Eigen::Vector3d se = g.p_e - g.p_s;
    Eigen::Vector3d e_raw = se - se.dot(g.u) * g.u;
    if (e_raw.norm() < 1e-6) return g;
    g.e = e_raw.normalized();

    Eigen::Vector3d r_raw = g_base_ - g_base_.dot(g.u) * g.u;
    if (r_raw.norm() < 1e-3) {
        Eigen::Vector3d x_base = Eigen::Vector3d::UnitX();
        r_raw = x_base - x_base.dot(g.u) * g.u;
    }
    g.r = r_raw.normalized();
    g.t = g.u.cross(g.r);
    g.ok = true;
    return g;
}

double PostureOptimizer::swivelTarget(const ArmGeometry& g) const {
    Eigen::Vector3d p_w_world = R_base_ * g.p_w;
    double phi = phi_home_ + cfg_.swivel_offset_deg * kPi / 180.0
               + cfg_.k_height  * (p_w_world.z() - z_home_)
               + cfg_.k_lateral * (p_w_world.y() - y_home_);
    return phi;
}

double PostureOptimizer::cost(const ArmGeometry& g, double phi_target) const {
    Eigen::Vector3d e_target = std::cos(phi_target) * g.r + std::sin(phi_target) * g.t;
    return 1.0 - g.e.dot(e_target);
}

double PostureOptimizer::jointMargin(const Vector7& q) const {
    double m = std::numeric_limits<double>::infinity();
    for (int i = 0; i < 7; ++i)
        m = std::min({m, q(i) - q_min_(i), q_max_(i) - q(i)});
    return m;
}

double PostureOptimizer::manipulability(const Matrix6x7& J) {
    Eigen::Matrix<double, 6, 6> JJt = J * J.transpose();
    double det = JJt.determinant();
    return det > 0.0 ? std::sqrt(det) : 0.0;
}

Vector7 PostureOptimizer::nullspaceTangent(const Matrix6x7& J) {
    Eigen::JacobiSVD<Matrix6x7> svd(J, Eigen::ComputeFullV);
    Vector7 v = svd.matrixV().col(6);
    if (have_v_prev_ && v.dot(v_prev_) < 0.0) v = -v;
    v_prev_      = v;
    have_v_prev_ = true;
    return v;
}

// Samples q + s*v leave the self-motion manifold at second order in s; one
// damped Gauss-Newton step on the EE pose error puts them back so the joint
// margins and the elbow direction are evaluated at the wrist pose that will
// actually be held.
Vector7 PostureOptimizer::projectToManifold(const Vector7& q_s, const Eigen::Isometry3d& x_ref) const {
    Vector7 q = q_s;
    constexpr double kLambdaSq = 1e-4;
    for (int it = 0; it < 2; ++it) {
        Eigen::Isometry3d x = kin_.pose(franka::Frame::kEndEffector, q);
        Eigen::Matrix<double, 6, 1> err = poseError(x, x_ref);
        if (err.norm() < 1e-6) break;
        Matrix6x7 J = kin_.jacobian(q);
        Eigen::Matrix<double, 6, 6> A = J * J.transpose();
        A.diagonal().array() += kLambdaSq;
        q -= J.transpose() * A.ldlt().solve(err);
    }
    return q;
}

// ---------------------------------------------------------------------------

Vector7 PostureOptimizer::update(const Vector7& q, double dt) {
    if (!enabled()) {
        std::lock_guard<std::mutex> lock(mtx_);
        snap_.q_ref = q;
        snap_.valid = false;
        return q;
    }

    const int    n  = cfg_.samples;
    const int    c  = n / 2;
    const double ds = cfg_.window_rad / c;

    Matrix6x7 J0 = kin_.jacobian(q);
    Vector7   v  = nullspaceTangent(J0);
    Eigen::Isometry3d x_ref = kin_.pose(franka::Frame::kEndEffector, q);

    std::vector<double>  s(n), cst(n), mrg(n), mnp(n);
    std::vector<bool>    feas(n);
    std::vector<Vector7> qs(n);
    double phi_now = 0.0, phi_target = 0.0;

    for (int k = 0; k < n; ++k) {
        s[k]  = (k - c) * ds;
        qs[k] = (k == c) ? q : projectToManifold(q + s[k] * v, x_ref);
        ArmGeometry g = geometry(qs[k]);
        mrg[k]  = jointMargin(qs[k]);
        mnp[k]  = (cfg_.manip_floor > 0.0 || k == c) ? manipulability(kin_.jacobian(qs[k])) : 0.0;
        feas[k] = g.ok && mrg[k] >= cfg_.margin_rad
                       && (cfg_.manip_floor <= 0.0 || mnp[k] >= cfg_.manip_floor);
        if (g.ok) {
            double phi_t = swivelTarget(g);
            cst[k] = cost(g, phi_t);
            if (k == c) {
                phi_now    = std::atan2(g.e.dot(g.t), g.e.dot(g.r));
                phi_target = phi_t;
            }
        } else {
            cst[k] = 2.0;
        }
    }

    int    best = c;
    double s_opt = 0.0;

    if (feas[c]) {
        int lo = c, hi = c;
        while (lo > 0     && feas[lo - 1]) --lo;
        while (hi < n - 1 && feas[hi + 1]) ++hi;
        for (int k = lo; k <= hi; ++k)
            if (cst[k] < cst[best]) best = k;
        s_opt = s[best];
        if (best > lo && best < hi) {
            double y0 = cst[best - 1], y1 = cst[best], y2 = cst[best + 1];
            double denom = y0 - 2.0 * y1 + y2;
            if (denom > 1e-12) {
                double off = 0.5 * (y0 - y2) / denom;
                s_opt += std::clamp(off, -1.0, 1.0) * ds;
            }
        }
        s_opt = std::clamp(s_opt, s[lo], s[hi]);
    } else {
        // Already inside the margin: move toward the sample that regains the
        // most margin, preferring the nearer one when tied.
        for (int k = 0; k < n; ++k) {
            if (mrg[k] > mrg[best] + 1e-9 ||
                (std::abs(mrg[k] - mrg[best]) <= 1e-9 && std::abs(s[k]) < std::abs(s[best])))
                best = k;
        }
        s_opt = s[best];
    }

    if (best != c && cfg_.manip_floor <= 0.0)
        mnp[best] = manipulability(kin_.jacobian(qs[best]));

    double alpha = (cfg_.filter_tau_s > 0.0) ? std::clamp(dt / cfg_.filter_tau_s, 0.0, 1.0) : 1.0;
    s_filt_ += alpha * (s_opt - s_filt_);
    double s_cmd = std::clamp(s_filt_, -cfg_.lead_rad, cfg_.lead_rad);

    Vector7 q_ref = q + s_cmd * v;

    std::lock_guard<std::mutex> lock(mtx_);
    snap_.q_ref         = q_ref;
    snap_.valid         = true;
    snap_.s_opt         = s_cmd;
    snap_.cost          = cst[best];
    snap_.margin        = mrg[best];
    snap_.manip         = mnp[best];
    snap_.feasible      = feas[best];
    snap_.swivel        = phi_now;
    snap_.swivel_target = phi_target;
    return q_ref;
}
