#pragma once

// Nullspace posture reference for a 7-DOF arm under Cartesian impedance.
//
// At a fixed wrist pose the FR3 has one degree of redundancy: the elbow orbit.
// Each tick this class walks a short window of that orbit, discards the part
// that violates joint-limit margins (and optionally a manipulability floor),
// and inside what remains picks the elbow direction closest to a heuristic
// human-like swivel angle. The result is a joint reference q_ref that
// cartesianImpedanceControl's nullspace spring pulls toward instead of the
// fixed q0. It is a pure function of (q, basin) -- no task phase, no clock --
// so it runs the same in AWAITING and ENGAGED and is reproducible offline.
//
// Kinematics come in through PostureKinematics so the same code runs against
// libfranka's franka::Model on hardware and the Pinocchio-backed sim Model.

#include <array>
#include <functional>
#include <mutex>

#include <Eigen/Dense>

#include "common.hpp"
#include "sim_env/model.hpp"   // franka::Frame on both builds

struct PostureConfig {
    bool   enabled           = true;
    double window_rad        = 0.30;   // half-width of the orbit search window
    int    samples           = 7;      // odd; the centre sample is s = 0
    double lead_rad          = 0.15;   // |q_ref - q| cap: bounds the nullspace torque, hence the elbow speed
    double filter_tau_s      = 0.20;   // first-order filter on the optimum
    double margin_rad        = 0.30;   // posture keeps this far from q_min / q_max
    double manip_floor       = 0.0;    // sqrt(det(J J^T)) floor, 0 disables
    double swivel_offset_deg = 0.0;    // added to the swivel angle of q0
    double k_height          = 0.0;    // rad of swivel per m of wrist rise above its q0 height (world z)
    double k_lateral         = 0.0;    // rad of swivel per m of wrist lateral offset from q0 (world y)
};

struct PostureKinematics {
    // Base-frame pose of a robot frame at q.
    std::function<Eigen::Isometry3d(franka::Frame, const Vector7&)> pose;
    // Base-frame end-effector Jacobian at q.
    std::function<Matrix6x7(const Vector7&)> jacobian;
};

struct PostureSnapshot {
    Vector7 q_ref    = Vector7::Zero();
    bool    valid    = false;
    double  s_opt    = 0.0;   // filtered orbit displacement handed to the spring (rad)
    double  cost     = 0.0;   // 1 - e.e* at the chosen sample
    double  margin   = 0.0;   // min joint-limit margin at the chosen sample (rad)
    double  manip    = 0.0;   // sqrt(det(J J^T)) at the chosen sample
    bool    feasible = false; // chosen sample satisfies the margins
    double  swivel   = 0.0;   // current swivel angle (rad)
    double  swivel_target = 0.0;
};

class PostureOptimizer {
public:
    void init(const PostureConfig& cfg, const Vector7& q_min, const Vector7& q_max,
              const Vector7& q0, const Eigen::Matrix3d& R_base, PostureKinematics kin);

    bool enabled() const { return cfg_.enabled && initialised_; }

    // Re-seed on every entry into a state that uses the reference.
    void reset(const Vector7& q);

    // State-thread rate. Returns the new reference.
    Vector7 update(const Vector7& q, double dt);

    // Control-thread read; one lock.
    PostureSnapshot snapshot() const;

    // Swivel angle of the elbow about the shoulder-wrist axis, measured from
    // "world down". Public so offline tools can report it.
    double swivelAngle(const Vector7& q) const;

private:
    struct ArmGeometry {
        Eigen::Vector3d p_s, p_e, p_w;
        Eigen::Vector3d u;   // shoulder -> wrist, unit
        Eigen::Vector3d e;   // elbow direction perpendicular to u, unit
        Eigen::Vector3d r, t;// swivel reference frame in the plane perpendicular to u
        bool ok = false;
    };
    ArmGeometry geometry(const Vector7& q) const;
    double swivelTarget(const ArmGeometry& g) const;
    double cost(const ArmGeometry& g, double phi_target) const;
    double jointMargin(const Vector7& q) const;
    static double manipulability(const Matrix6x7& J);
    Vector7 nullspaceTangent(const Matrix6x7& J);
    Vector7 projectToManifold(const Vector7& q_s, const Eigen::Isometry3d& x_ref) const;

    PostureConfig     cfg_;
    PostureKinematics kin_;
    Vector7 q_min_ = Vector7::Zero(), q_max_ = Vector7::Zero();
    Eigen::Vector3d g_base_ = Eigen::Vector3d(0, 0, -1);   // world down, in base frame
    Eigen::Matrix3d R_base_ = Eigen::Matrix3d::Identity();
    double          phi_home_ = 0.0;
    double          z_home_   = 0.0, y_home_ = 0.0;         // world-frame wrist height / lateral at q0
    bool            initialised_ = false;

    Vector7 v_prev_ = Vector7::Zero();
    bool    have_v_prev_ = false;
    double  s_filt_ = 0.0;

    mutable std::mutex mtx_;
    PostureSnapshot snap_;
};
