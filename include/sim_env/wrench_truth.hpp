#pragma once

// MuJoCo ground truth for the momentum observer, for validation only.
//
// This file is compiled ONLY in the simulation build (see the sim_env block in
// CMakeLists). Nothing here is visible to arm_control.cpp, nothing reaches
// franka::RobotState, and no value computed here is ever fed back into the
// controller. It exists to write a CSV that can be joined against arm.csv on
// mjData::time afterwards.
//
// What it records, and why that rather than a force/torque sensor:
//
// A MuJoCo <force>/<torque> site sensor reports the load TRANSMITTED through
// its site -- gravity and inertia of every body distal to it, plus contact.
// The observer estimates EXTERNAL wrench only, so comparing the two needs the
// distal dynamics subtracted, which is a second model-based estimate with its
// own errors. Contact forces need no such compensation: MuJoCo knows them
// exactly.
//
// The primary comparison is in JOINT SPACE (tau_contact vs the observer's r_),
// because that is what the observer actually estimates. O_F_ext_hat_K is r_
// pushed through a DAMPED least-squares inverse of J^T, which discards the
// nullspace component and adds its own bias -- comparing Cartesian wrenches
// conflates observer error with projection error. Joint torques also have no
// frame, so the base-vs-world question does not arise for the main metric.

#include <array>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include <Eigen/Dense>
#include <mujoco/mujoco.h>
#include <yaml-cpp/yaml.h>

#include "data_logger.hpp"

// One row per configured arm per sample.
struct WrenchTruthEntry {
    double      sim_time = 0.0;   // mjData::time -- the join key against arm.csv
    std::string device;

    // Joint-space external torque from contacts alone, on this arm's 7 DOFs.
    // THE comparison against ArmLogEntry::tau_ext.
    std::array<double, 7> tau_contact{};

    // The other things MuJoCo solves as constraints, separated out. Without
    // this split qfrc_constraint is unusable as ground truth: it mixes contact
    // with dry friction and joint limits, and the observer models friction
    // internally rather than seeing it as external.
    std::array<double, 7> tau_friction{};
    std::array<double, 7> tau_limit{};
    // qfrc_constraint verbatim. Reconciliation check: contact + friction +
    // limit should account for essentially all of it. What is left over is
    // equality constraints and anything this decomposition missed.
    std::array<double, 7> tau_constraint{};

    // Net contact wrench on the arm, taken about ref_point_world.
    // F_world is in the MuJoCo world frame; F_base is the same wrench rotated
    // into THIS arm's base frame, which is the frame O_F_ext_hat_K lives in.
    // They are different frames and the two arms are mounted differently -- see
    // the R_wb note in the .cpp.
    std::array<double, 6> F_world{};
    std::array<double, 6> F_base{};
    std::array<double, 3> ref_point_world{};

    // MuJoCo's EE body pose expressed in the arm base frame, for the kinematic
    // cross-check against ArmLogEntry::O_T_EE. The hand is defined TWICE and
    // independently -- in the URDF pinocchio loads, and via attach_to /
    // attach_offset in sim_config -- so if these disagree, every wrench
    // comparison downstream is meaningless.
    std::array<double, 3> ee_pos_base{};
    std::array<double, 4> ee_quat_base{1.0, 0.0, 0.0, 0.0};   // w x y z

    int  ncon_arm    = 0;      // contacts contributing to this arm
    bool cart_valid  = false;  // false when no EE body resolved: Cartesian columns are NaN
};

class WrenchTruth {
public:
    // Returns nullptr when the config block is absent or disabled, when no
    // configured device resolves, or on any setup failure -- every one of those
    // is logged once and is not fatal. The caller simply does not sample.
    static std::unique_ptr<WrenchTruth> create(
        const mjModel* m,
        const YAML::Node& sim_config,
        const YAML::Node& robot_config,
        const std::unordered_map<std::string, std::vector<int>>& joint_ids,
        const std::string& session_id);

    ~WrenchTruth();

    // Call from the sim thread with the live mjData, after mj_step. Decimates
    // internally to the configured rate off mjData::time, so the caller can
    // call it every step.
    void sample(const mjData* d);

private:
    struct Arm {
        std::string      name;
        std::vector<int> dof;          // this arm's DOF indices, in joint order
        std::vector<int> jnt;          // matching joint ids, for the limit rows
        int              root_body = -1;   // subtree test anchor
        int              ee_body   = -1;   // -1 = Cartesian columns disabled
        Eigen::Matrix3d  R_wb = Eigen::Matrix3d::Identity();  // base -> world
        Eigen::Vector3d  t_wb = Eigen::Vector3d::Zero();
        // One logger per arm, not one shared. DataLogger::write() keeps a single
        // latest-value slot that a background thread polls, so two arms writing
        // back to back in the same sample would lose the first row.
        std::unique_ptr<DataLogger<WrenchTruthEntry>> logger;
    };

    WrenchTruth(const mjModel* m, double rate_hz, std::vector<Arm> arms);

    bool isInSubtree(int body, int root) const;

    const mjModel* m_ = nullptr;
    double         period_ = 0.005;
    double         next_sample_time_ = 0.0;
    std::vector<Arm> arms_;

    // mj_jac scratch, sized nv once so sample() allocates nothing.
    std::vector<mjtNum> jacp1_, jacr1_, jacp2_, jacr2_;
};

std::string wrenchTruthHeader();
std::string wrenchTruthRow(const WrenchTruthEntry& e);
