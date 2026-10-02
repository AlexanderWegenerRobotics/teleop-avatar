#pragma once

// MuJoCo contact ground truth for validating the momentum observer, sim only, logged to CSV

#include <array>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include <Eigen/Dense>
#include <mujoco/mujoco.h>
#include <yaml-cpp/yaml.h>

#include "data_logger.hpp"

struct WrenchTruthEntry {
    double      sim_time = 0.0;   // join key against arm.csv
    std::string device;

    // joint torque from contacts only, compare against tau_ext
    std::array<double, 7> tau_contact{};

    std::array<double, 7> tau_friction{};
    std::array<double, 7> tau_limit{};
    // qfrc_constraint as is, should be ~ contact + friction + limit
    std::array<double, 7> tau_constraint{};

    // net contact wrench about ref_point_world, F_base is in the arm base frame (like O_F_ext_hat_K)
    std::array<double, 6> F_world{};
    std::array<double, 6> F_base{};
    std::array<double, 3> ref_point_world{};

    // EE pose in base frame, cross-check against O_T_EE
    std::array<double, 3> ee_pos_base{};
    std::array<double, 4> ee_quat_base{1.0, 0.0, 0.0, 0.0};   // w x y z

    int  ncon_arm    = 0;      // contacts contributing to this arm
    bool cart_valid  = false;  // false when no EE body resolved: Cartesian columns are NaN
};

class WrenchTruth {
public:
    // nullptr when disabled or setup fails (logged, not fatal)
    static std::unique_ptr<WrenchTruth> create(
        const mjModel* m,
        const YAML::Node& sim_config,
        const YAML::Node& robot_config,
        const std::unordered_map<std::string, std::vector<int>>& joint_ids,
        const std::string& session_id);

    ~WrenchTruth();

    // call every step after mj_step, decimates internally
    void sample(const mjData* d);

    void restartLoggers(const std::string& folder);

private:
    struct Arm {
        std::string      name;
        std::vector<int> dof;          // this arm's DOF indices, in joint order
        std::vector<int> jnt;          // matching joint ids, for the limit rows
        int              root_body = -1;   // subtree test anchor
        int              ee_body   = -1;   // -1 = Cartesian columns disabled
        Eigen::Matrix3d  R_wb = Eigen::Matrix3d::Identity();  // base -> world
        Eigen::Vector3d  t_wb = Eigen::Vector3d::Zero();
        // one logger per arm, DataLogger only keeps the latest value
        std::unique_ptr<DataLogger<WrenchTruthEntry>> logger;
    };

    WrenchTruth(const mjModel* m, double rate_hz, std::vector<Arm> arms);

    bool isInSubtree(int body, int root) const;

    const mjModel* m_ = nullptr;
    double         period_ = 0.005;
    double         next_sample_time_ = 0.0;
    std::vector<Arm> arms_;

    // preallocated so sample() doesn't allocate
    std::vector<mjtNum> jacp1_, jacr1_, jacp2_, jacr2_;
};

std::string wrenchTruthHeader();
std::string wrenchTruthRow(const WrenchTruthEntry& e);
